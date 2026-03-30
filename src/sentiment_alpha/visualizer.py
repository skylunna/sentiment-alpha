import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional

# src/sentiment_alpha/visualizer.py

def create_sentiment_price_chart(
    aligned_df: pd.DataFrame,
    price_column: str = 'Close',
    title: str = "Sentiment vs Price"
) -> go.Figure:
    """创建双轴图表"""
    if aligned_df.empty or 'date' not in aligned_df.columns:
        print(f"⚠️ Cannot create chart: empty data or missing 'date' column")
        print(f"   Available columns: {aligned_df.columns.tolist() if not aligned_df.empty else 'N/A'}")
        return go.Figure()
    
    fig = go.Figure()
    
    # 确保日期是 datetime 类型
    dates = pd.to_datetime(aligned_df['date'], errors='coerce')
    
    # 左轴：股价
    fig.add_trace(go.Scatter(
        x=dates,
        y=aligned_df[price_column],
        name='Stock Price',
        yaxis='y',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # 右轴：情绪指数
    if 'sentiment_index' in aligned_df.columns:
        fig.add_trace(go.Scatter(
            x=dates,
            y=aligned_df['sentiment_index'],
            name='Sentiment Index',
            yaxis='y2',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            hovertemplate='<b>Sentiment</b><br>Date: %{x}<br>Index: %{y:.3f}<extra></extra>'
        ))
    
    # 布局
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis=dict(
            title=f"Price ({price_column})",
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
            side="left"
        ),
        yaxis2=dict(
            title="Sentiment Index (-1 to 1)",
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            overlaying="y",
            side="right",
            range=[-1.2, 1.2]
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_correlation_heatmap(correlations: dict) -> go.Figure:
    """
    创建相关性热图：显示不同滞后期的情绪-股价相关性
    """
    # 准备数据
    lags = sorted(correlations.keys())
    corr_values = [correlations[lag] for lag in lags]
    
    fig = go.Figure(data=go.Heatmap(
        z=[corr_values],
        x=lags,
        y=['Correlation'],
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=[[f"{v:.3f}" if pd.notna(v) else "N/A" for v in corr_values]],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Sentiment-Price Correlation by Lag (days)",
        xaxis_title="Lag (days) - Positive: Sentiment leads, Negative: Sentiment lags",
        yaxis_title="",
        height=200,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # 添加零线
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def create_sentiment_distribution_chart(news_df: pd.DataFrame) -> go.Figure:
    """
    创建情感分布饼图（增强版）
    """
    if news_df.empty or 'label' not in news_df.columns:
        return go.Figure()
    
    # 统计
    counts = news_df['label'].value_counts()
    labels = ['Positive', 'Neutral', 'Negative']
    values = [
        counts.get('POSITIVE', 0),
        counts.get('NEUTRAL', 0), 
        counts.get('NEGATIVE', 0)
    ]
    colors = ['#2ca02c', '#7f7f7f', '#d62728']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='percent+label',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title="News Sentiment Distribution",
        height=350,
        margin=dict(t=30, b=10, l=10, r=10)
    )
    
    return fig