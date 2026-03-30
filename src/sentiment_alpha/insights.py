"""
Sentiment Analysis Insights Module
计算情绪指数、关联分析、相关性解读等
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
# ✅ 已移除未使用的: List, timedelta


def calculate_daily_sentiment_index(
    news_df: pd.DataFrame, 
    date_column: str = 'published_at',
    sentiment_column: str = 'sentiment_score'
) -> pd.DataFrame:
    """
    计算每日情绪指数
    
    Args:
        news_df: 包含情感分析结果的新闻DataFrame
        date_column: 日期列名
        sentiment_column: 情感分数列名
    
    Returns:
        DataFrame with columns: ['date', 'sentiment_index', 'article_count', 'sentiment_std']
    """
    if news_df.empty or date_column not in news_df.columns:
        return pd.DataFrame()
    
    df = news_df.copy()
    
    # 确保日期是 datetime 类型
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    
    if df.empty:
        return pd.DataFrame()
    
    # 按日期分组聚合
    daily = df.groupby(df[date_column].dt.date).agg({
        sentiment_column: ['mean', 'std', 'count'],
        'title': 'first'
    }).reset_index()
    
    # 扁平化列名
    daily.columns = ['date', 'sentiment_index', 'sentiment_std', 'article_count', 'sample_title']
    
    # 填充标准差
    daily['sentiment_std'] = daily['sentiment_std'].fillna(0)
    
    # 计算置信度
    daily['confidence'] = np.minimum(daily['article_count'] / 10, 1.0)
    daily['weighted_sentiment'] = daily['sentiment_index'] * daily['confidence']
    
    return daily.sort_values('date')


def _interpret_correlation(lag: int, corr: Optional[float]) -> str:
    """
    解读相关性结果（内部辅助函数）
    """
    if corr is None or pd.isna(corr) or abs(corr) < 0.1:
        return "😐 情绪与股价无明显相关性"
    
    strength = "强" if abs(corr) > 0.5 else "中等" if abs(corr) > 0.3 else "弱"
    direction = "正相关" if corr > 0 else "负相关"
    
    if lag == 0:
        timing = "同步"
    elif lag > 0:
        timing = f"情绪领先{lag}天"
    else:
        timing = f"情绪滞后{abs(lag)}天"
    
    return f"{strength}{direction} ({timing}), 相关系数: {corr:.3f}"


def align_sentiment_with_price(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
    price_column: str = 'Close',
    max_lag_days: int = 5
) -> Dict:
    """
    将情绪指数与股价数据对齐，计算不同滞后期的相关性
    """
    if sentiment_df.empty or price_df.empty:
        return {}
    
    # 🔑 关键：统一使用 "日期字符串" 作为合并键
    def normalize_date(dt) -> Optional[str]:
        """将任何日期格式转换为 'YYYY-MM-DD' 字符串"""
        if pd.isna(dt):
            return None
        try:
            dt_parsed = pd.to_datetime(dt, errors='coerce')
            if pd.isna(dt_parsed):
                return None
            # 移除时区，只保留日期
            if hasattr(dt_parsed, 'tzinfo') and dt_parsed.tzinfo is not None:
                dt_parsed = dt_parsed.tz_localize(None)
            return dt_parsed.date().isoformat()
        except Exception:
            return None
    
    # 1. 处理情绪数据
    sentiment = sentiment_df.copy()
    sentiment['date_key'] = sentiment['date'].apply(normalize_date)
    sentiment = sentiment.dropna(subset=['date_key'])
    
    if sentiment.empty:
        return {}
    
    # 2. 处理股价数据（yfinance 格式：index=日期）
    price = price_df.copy()
    
    # 确保价格列存在
    if price_column not in price.columns:
        available = [c for c in price.columns if c.lower() in ['close', 'price', 'adj close']]
        if available:
            price_column = available[0]
        else:
            return {}
    
    # 重置索引并提取日期
    price_reset = price.reset_index()
    
    # 自动检测日期列
    date_col = None
    for col in ['Date', 'date', 'Datetime', 'datetime']:
        if col in price_reset.columns:
            date_col = col
            break
    if date_col is None:
        date_col = price_reset.columns[0]
    
    # 转换日期并创建合并键
    price_reset['date_key'] = price_reset[date_col].apply(normalize_date)
    price_reset = price_reset.dropna(subset=['date_key'])
    
    if price_reset.empty:
        return {}
    
    # 🔍 调试输出
    sentiment_dates = set(sentiment['date_key'].unique())
    price_dates = set(price_reset['date_key'].unique())
    overlap = sentiment_dates & price_dates
    
    if not overlap:
        return {}
    
    # 3. 按 date_key 合并
    merge_cols = ['date_key', price_column]
    if 'Volume' in price_reset.columns:
        merge_cols.append('Volume')
    
    merged = pd.merge(
        sentiment,
        price_reset[merge_cols],
        on='date_key',
        how='inner'
    ).sort_values('date')
    
    if merged.empty:
        return {}
    
    # 4. 转换 date_key 回 datetime 用于绘图
    merged['date'] = pd.to_datetime(merged['date_key'])
    
    # 5. 计算收益率
    merged['price_return'] = merged[price_column].pct_change()
    
    # 6. 计算不同滞后的相关性
    correlations = {}
    
    for lag in range(-max_lag_days, max_lag_days + 1):
        if lag == 0:
            shifted = merged['price_return']
        else:
            shifted = merged['price_return'].shift(-lag)
        
        corr = merged['sentiment_index'].corr(shifted)
        if pd.notna(corr):
            correlations[lag] = corr
    
    # 7. 找出最佳滞后
    if correlations:
        best_lag = max(correlations, key=lambda x: abs(correlations[x]))
        best_corr = correlations[best_lag]
    else:
        best_lag, best_corr = 0, None
    
    # 8. 准备返回数据
    result_cols = ['date', 'sentiment_index', 'sentiment_std', 'article_count', price_column]
    if 'price_return' in merged.columns:
        result_cols.append('price_return')
    
    result_df = merged[result_cols].copy()
    
    return {
        'aligned_data': result_df,
        'correlations': correlations,
        'best_lag_days': best_lag,
        'best_correlation': best_corr,
        'interpretation': _interpret_correlation(best_lag, best_corr),
        'data_points': len(result_df),
        'overlap_count': len(overlap)
    }


def generate_sentiment_summary(sentiment_df: pd.DataFrame) -> Dict:
    """
    生成情绪摘要统计
    
    Args:
        sentiment_df: 包含情绪指数的DataFrame
    
    Returns:
        dict with summary statistics
    """
    if sentiment_df.empty or 'sentiment_index' not in sentiment_df.columns:
        return {
            'avg_sentiment': 0.0,
            'sentiment_trend': 'neutral',
            'volatility': 0.0,
            'total_articles': 0,
            'bullish_days': 0,
            'bearish_days': 0,
        }
    
    # 计算趋势：最近3天 vs 之前7天
    if len(sentiment_df) >= 10:
        recent = sentiment_df.tail(3)['sentiment_index'].mean()
        older = sentiment_df.tail(10).head(7)['sentiment_index'].mean()
        trend = 'rising' if recent > older else 'falling'
    elif len(sentiment_df) >= 3:
        trend = 'rising' if sentiment_df['sentiment_index'].iloc[-1] > sentiment_df['sentiment_index'].iloc[0] else 'falling'
    else:
        trend = 'neutral'
    
    return {
        'avg_sentiment': float(sentiment_df['sentiment_index'].mean()),
        'sentiment_trend': trend,
        'volatility': float(sentiment_df['sentiment_std'].mean()) if 'sentiment_std' in sentiment_df.columns else 0.0,
        'total_articles': int(sentiment_df['article_count'].sum()) if 'article_count' in sentiment_df.columns else len(sentiment_df),
        'bullish_days': int(len(sentiment_df[sentiment_df['sentiment_index'] > 0.1])),
        'bearish_days': int(len(sentiment_df[sentiment_df['sentiment_index'] < -0.1])),
    }


# ✅ 导出公共接口
__all__ = [
    'calculate_daily_sentiment_index',
    'align_sentiment_with_price', 
    'generate_sentiment_summary',
    '_interpret_correlation',  # 内部函数，可选导出
]