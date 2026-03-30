import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.sentiment_alpha.scraper import NewsScraper
from src.sentiment_alpha.analyzer import SentimentAnalyzer
import yfinance as yf
from src.sentiment_alpha.insights import (
    calculate_daily_sentiment_index,
    align_sentiment_with_price,
    generate_sentiment_summary
)
from src.sentiment_alpha.visualizer import (
    create_sentiment_price_chart,
    create_correlation_heatmap
)

# =============================================================================
# 页面配置
# =============================================================================
st.set_page_config(
    page_title="Market Sentiment Alpha",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Market Sentiment Alpha")
st.markdown("**AI-Powered News Sentiment Analysis for Stocks**")

# =============================================================================
# 侧边栏配置
# =============================================================================
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
num_articles = st.sidebar.slider("Number of Articles", 5, 50, 20)
show_advanced = st.sidebar.checkbox("🔬 Show Advanced Analysis", value=True)

# =============================================================================
# 初始化组件（带缓存）
# =============================================================================
@st.cache_resource
def get_analyzer():
    return SentimentAnalyzer()

@st.cache_data(ttl=3600)  # 缓存1小时
def get_stock_data(ticker: str, period: str = "3mo"):
    """缓存股价数据，避免重复请求"""
    return yf.Ticker(ticker).history(period=period)

def interpret_sentiment(pos_ratio: float, avg_score: float, article_count: int) -> str:
    """提供情绪数据的专业解读（非投资建议）"""
    if article_count < 5:
        return "⚠️ 新闻样本较少，情绪指标参考性有限"
    if avg_score > 0.5 and pos_ratio > 0.6:
        return "🟢 情绪显著偏多：市场讨论氛围乐观，但需结合技术面确认"
    elif avg_score < -0.5 and pos_ratio < 0.3:
        return "🔴 情绪显著偏空：市场讨论氛围谨慎，注意风险控制"
    elif abs(avg_score) < 0.2:
        return "⚪ 情绪中性：市场讨论无明显倾向，观望为主"
    else:
        return "🟡 情绪温和：存在一定倾向，但信号强度一般"

scraper = NewsScraper()
analyzer = get_analyzer()

# =============================================================================
# 主逻辑
# =============================================================================
if st.sidebar.button("🔍 Analyze Sentiment"):
    with st.spinner("Fetching news and analyzing sentiment..."):
        
        # 🔹 1. 获取新闻
        news_df = scraper.fetch_yahoo_finance_news(ticker, num_articles)
        
        # 🔍 调试面板（默认折叠）
        with st.expander("🔍 Debug: Raw News Data", expanded=False):
            st.write(f"**News count:** {len(news_df)}")
            if not news_df.empty:
                st.write("**Columns:**", news_df.columns.tolist())
                st.write("**First row:**")
                st.json(news_df.iloc[0].to_dict())
            else:
                st.warning("No news data")

        if news_df.empty:
            st.warning(f"⚠️ No news found for `{ticker}`. Try another ticker or check your network.")
            st.stop()
        
        # 🔹 2. 情感分析
        analyzed_df = analyzer.analyze_dataframe(news_df, 'title')
        
        # 🔹 3. 获取股价数据（只获取一次，缓存复用）
        price_hist = get_stock_data(ticker, period="3mo")
        
        # 🔹 4. 显示基础分析结果
        st.success(f"✅ Analyzed {len(analyzed_df)} articles")
        
        # 情绪统计卡片
        col1, col2, col3 = st.columns(3)
        
        pos_count = len(analyzed_df[analyzed_df['label'] == 'POSITIVE'])
        neg_count = len(analyzed_df[analyzed_df['label'] == 'NEGATIVE'])
        neu_count = len(analyzed_df[analyzed_df['label'] == 'NEUTRAL'])
        total = len(analyzed_df)
        
        avg_sentiment = analyzed_df['sentiment_score'].mean()
        
        col1.metric("😊 Positive", f"{pos_count} ({pos_count/total*100:.1f}%)")
        col2.metric("😞 Negative", f"{neg_count} ({neg_count/total*100:.1f}%)")
        col3.metric("📊 Avg Score", f"{avg_sentiment:.2f}", 
                    delta="📈 Bullish" if avg_sentiment > 0.1 else "📉 Bearish" if avg_sentiment < -0.1 else "➡️ Neutral")
        
        # 情绪解读
        interpretation = interpret_sentiment(
            pos_ratio=pos_count/total if total > 0 else 0,
            avg_score=avg_sentiment,
            article_count=total
        )
        st.info(f"💡 {interpretation}")
        
        # 🔹 5. 可视化：情绪分布 + 新闻时间线
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # 饼图
            fig_pie = go.Figure(data=[go.Pie(
                labels=['😊 Positive', '😞 Negative', '😐 Neutral'],
                values=[pos_count, neg_count, neu_count],
                hole=0.4,
                marker=dict(colors=['#2ca02c', '#d62728', '#7f7f7f'])
            )])
            fig_pie.update_layout(title="📊 Sentiment Distribution", height=300, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, width="stretch")
        
        with col_chart2:
            # 新闻时间线（散点图）
            fig_timeline = go.Figure()
            
            for _, row in analyzed_df.iterrows():
                # 确保日期是 datetime
                pub_date = pd.to_datetime(row['published_at'], errors='coerce')
                if pd.isna(pub_date):
                    continue
                    
                color = '#2ca02c' if row['sentiment_score'] > 0.1 else '#d62728' if row['sentiment_score'] < -0.1 else '#7f7f7f'
                size = abs(row['sentiment_score']) * 12 + 6
                
                fig_timeline.add_trace(go.Scatter(
                    x=[pub_date],
                    y=[row['sentiment_score']],
                    mode='markers',
                    marker=dict(size=size, color=color, opacity=0.7, line=dict(width=1, color='white')),
                    text=f"<b>{row['title']}</b><br>📰 {row['publisher']}<br>🎯 Score: {row['sentiment_score']:.2f}",
                    hoverinfo='text',
                    name=row['label'],
                    showlegend=False
                ))
            
            fig_timeline.update_layout(
                title="📰 News Sentiment Timeline",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                yaxis_range=[-1.2, 1.2],
                height=300,
                margin=dict(t=30, b=10, l=10, r=10),
                hovermode='closest'
            )
            st.plotly_chart(fig_timeline, width="stretch")
        
        # 🔹 6. 新闻列表（可滚动）
        with st.expander(f"📰 View Analyzed News ({len(analyzed_df)} articles)", expanded=False):
            for idx, row in analyzed_df.iterrows():
                emoji = "🟢" if row['sentiment_score'] > 0.1 else "🔴" if row['sentiment_score'] < -0.1 else "⚪"
                score_color = "green" if row['sentiment_score'] > 0.1 else "red" if row['sentiment_score'] < -0.1 else "gray"
                
                st.markdown(f"""
                {emoji} **{row['title']}**  
                *📰 {row['publisher']} | 🎯 <span style="color:{score_color}">{row['sentiment_score']:+.2f}</span>*
                """, unsafe_allow_html=True)
                if row.get('summary') and len(str(row['summary'])) > 20:
                    st.caption(f"_{row['summary'][:150]}..._")
                st.divider()

        # =====================================================================
        # 🔬 高级分析：情绪指数与股价关联（可选）
        # =====================================================================
        if show_advanced:
            st.divider()
            st.subheader("🔬 Advanced: Sentiment-Price Correlation")
            
            # 计算每日情绪指数
            daily_sentiment = calculate_daily_sentiment_index(analyzed_df)
            
            if daily_sentiment.empty:
                st.warning("⚠️ Could not calculate daily sentiment index (insufficient data)")
            elif price_hist.empty:
                st.warning(f"⚠️ Could not fetch price history for `{ticker}`")
            else:
                # 🔍 调试：显示日期范围
                with st.expander("🔍 Debug: Date Alignment", expanded=False):
                    sentiment_dates = pd.to_datetime(daily_sentiment['date']).dt.date.astype(str).unique()
                    price_dates = pd.to_datetime(price_hist.index).date.astype(str)
                    overlap = set(sentiment_dates) & set(price_dates)
                    
                    col_d1, col_d2, col_d3 = st.columns(3)
                    col_d1.metric("Sentiment Dates", f"{len(sentiment_dates)} days")
                    col_d2.metric("Price Dates", f"{len(price_dates)} days") 
                    col_d3.metric("✅ Overlapping", f"{len(overlap)} days", 
                                 delta="Good!" if len(overlap) >= 3 else "Limited")
                    
                    if overlap:
                        st.write("**Sample overlapping dates:**", sorted(list(overlap))[:5])
                
                # 执行对齐分析
                analysis_result = align_sentiment_with_price(
                    daily_sentiment,
                    price_hist,
                    max_lag_days=3
                )
                
                if not analysis_result or 'aligned_data' not in analysis_result:
                    st.warning("⚠️ Could not align sentiment with price data. This may happen if:")
                    st.markdown("""
                    - News dates don't overlap with trading days (weekends/holidays)
                    - Price data fetch failed
                    - Date format mismatch (try a different ticker)
                    """)
                else:
                    aligned_df = analysis_result['aligned_data']
                    summary = generate_sentiment_summary(daily_sentiment)
                    
                    # 📊 关键指标卡片
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric(
                        "📈 Avg Sentiment", 
                        f"{summary.get('avg_sentiment', 0):+.2f}",
                        delta="🔼 Rising" if summary.get('sentiment_trend') == 'rising' else "🔽 Falling"
                    )
                    
                    best_corr = analysis_result.get('best_correlation')
                    col2.metric(
                        "🎯 Best Correlation",
                        f"{best_corr:.3f}" if best_corr and pd.notna(best_corr) else "N/A",
                        delta=analysis_result.get('interpretation', '').split(',')[0] if analysis_result.get('interpretation') else ""
                    )
                    
                    best_lag = analysis_result.get('best_lag_days', 0)
                    col3.metric(
                        "⏱️ Optimal Lag",
                        f"{best_lag}d" if best_lag != 0 else "Same day",
                        delta="🔮 Leads" if best_lag > 0 else "🔙 Lags" if best_lag < 0 else "⚡ Sync"
                    )
                    
                    col4.metric(
                        "📰 Total Articles",
                        f"{summary.get('total_articles', 0)}"
                    )
                    
                    # 📈 双轴图表：股价 + 情绪指数
                    if not aligned_df.empty and 'date' in aligned_df.columns:
                        st.plotly_chart(
                            create_sentiment_price_chart(aligned_df, title=f"{ticker}: Price vs Sentiment Index"),
                            width="stretch"
                        )
                        
                        # 🔥 相关性热图
                        correlations = analysis_result.get('correlations', {})
                        if correlations:
                            st.plotly_chart(
                                create_correlation_heatmap(correlations),
                                width="stretch"
                            )
                            st.caption("💡 **Tip**: Positive lag = sentiment leads price | Negative lag = sentiment lags price")
                        
                        # 📋 原始数据表格
                        with st.expander("📋 View Aligned Data"):
                            display_cols = [c for c in ['date', 'sentiment_index', 'Close', 'price_return'] if c in aligned_df.columns]
                            st.dataframe(aligned_df[display_cols].round(4), use_container_width=True)
                    
                    # 💡 解读建议
                    st.success("""
                    **📚 How to interpret**:
                    - **Correlation > 0.3**: Moderate relationship between sentiment and price
                    - **Positive lag**: News sentiment may *predict* future price movements
                    - **Negative lag**: Price movements may *drive* news sentiment
                    - ⚠️ This is for research only, not financial advice!
                    """)

# =============================================================================
# 页脚
# =============================================================================
st.divider()
st.caption("""
🔬 **Market Sentiment Alpha** | Built with ❤️ using Python, Streamlit & Hugging Face  
⚠️ **Disclaimer**: For educational/research purposes only. Not financial advice.  
📦 **GitHub**: [YourRepo](https://github.com/yourusername/sentiment-alpha) | ⭐ Star if you find it useful!
""")