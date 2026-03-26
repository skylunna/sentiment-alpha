# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.sentiment_alpha.scraper import NewsScraper
from src.sentiment_alpha.analyzer import SentimentAnalyzer
import yfinance as yf
from datetime import datetime, timedelta

# 页面配置
st.set_page_config(
    page_title="Market Sentiment Alpha",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Market Sentiment Alpha")
st.markdown("**AI-Powered News Sentiment Analysis for Stocks**")

# 侧边栏
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
num_articles = st.sidebar.slider("Number of Articles", 5, 50, 20)

# 初始化组件
@st.cache_resource
def get_analyzer():
    return SentimentAnalyzer()

scraper = NewsScraper()
analyzer = get_analyzer()

# 主逻辑
if st.sidebar.button("🔍 Analyze Sentiment"):
    with st.spinner("Fetching news and analyzing sentiment..."):
        # 1. 获取新闻
        news_df = scraper.fetch_yahoo_finance_news(ticker, num_articles)
        
        if news_df.empty:
            st.warning(f"⚠️ No news found for {ticker}. Try another ticker or check your network.")
            st.stop()
        
        # 2. 情感分析
        analyzed_df = analyzer.analyze_dataframe(news_df, 'title')
        
        # 3. 获取股价数据
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        
        # 显示结果
        st.success(f"✅ Analyzed {len(analyzed_df)} articles")
        
        # 情绪统计
        col1, col2, col3 = st.columns(3)
        
        pos_count = len(analyzed_df[analyzed_df['label'] == 'POSITIVE'])
        neg_count = len(analyzed_df[analyzed_df['label'] == 'NEGATIVE'])
        neu_count = len(analyzed_df[analyzed_df['label'] == 'NEUTRAL'])
        
        avg_sentiment = analyzed_df['sentiment_score'].mean()
        
        col1.metric("Positive", f"{pos_count} ({pos_count/len(analyzed_df)*100:.1f}%)")
        col2.metric("Negative", f"{neg_count} ({neg_count/len(analyzed_df)*100:.1f}%)")
        col3.metric("Avg Sentiment", f"{avg_sentiment:.2f}", 
                    delta="Bullish 📈" if avg_sentiment > 0 else "Bearish 📉")
        
        # 情绪分布图
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Positive', 'Negative', 'Neutral'],
            values=[pos_count, neg_count, neu_count],
            hole=.3
        )])
        fig_pie.update_layout(title="Sentiment Distribution")
        
        # 股价与情绪时间线
        fig_timeline = go.Figure()
        
        # 添加新闻点
        for idx, row in analyzed_df.iterrows():
            color = 'green' if row['sentiment_score'] > 0 else 'red' if row['sentiment_score'] < 0 else 'gray'
            size = abs(row['sentiment_score']) * 15 + 5
            
            fig_timeline.add_trace(go.Scatter(
                x=[row['published_at']],
                y=[hist['Close'].iloc[-1] if not hist.empty else 0],
                mode='markers',
                marker=dict(size=size, color=color, opacity=0.6),
                text=f"{row['title']}<br>Sentiment: {row['sentiment_score']:.2f}",
                hoverinfo='text',
                name=row['label']
            ))
        
        fig_timeline.update_layout(
            title="News Sentiment Timeline",
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=False
        )
        
        # 显示图表
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_chart2:
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # 新闻列表
        st.subheader("📰 Analyzed News")
        for idx, row in analyzed_df.iterrows():
            emoji = "🟢" if row['sentiment_score'] > 0 else "🔴" if row['sentiment_score'] < 0 else "⚪"
            st.markdown(f"{emoji} **{row['title']}**")
            st.markdown(f"*Source: {row['publisher']} | Score: {row['sentiment_score']:.2f}*")
            st.markdown("---")