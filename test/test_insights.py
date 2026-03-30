"""集成测试：测试完整的新闻→情感→分析流程"""
import pytest
import pandas as pd

# 只在有网络时运行这些测试
pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_full_pipeline_with_mock_news():
    """测试完整流程（使用模拟新闻数据，避免网络依赖）"""
    from sentiment_alpha.scraper import NewsScraper
    from sentiment_alpha.analyzer import SentimentAnalyzer
    from sentiment_alpha.insights import calculate_daily_sentiment_index
    
    # 使用模拟数据代替真实爬虫（避免网络不稳定）
    mock_news = pd.DataFrame({
        'title': [
            'Apple stock rises on strong earnings',
            'Tech sector faces headwinds',
            'Market outlook remains positive'
        ],
        'publisher': ['Yahoo Finance'] * 3,
        'link': ['http://example.com'] * 3,
        'published_at': pd.date_range('2026-03-01', periods=3, freq='D'),
        'ticker': ['AAPL'] * 3,
        'summary': [''] * 3
    })
    
    # 情感分析
    analyzer = SentimentAnalyzer()
    analyzed = analyzer.analyze_dataframe(mock_news, 'title')
    
    # 验证分析结果
    assert 'sentiment_score' in analyzed.columns
    assert len(analyzed) == 3
    
    # 计算情绪指数
    daily = calculate_daily_sentiment_index(analyzed)
    
    assert not daily.empty
    assert 'sentiment_index' in daily.columns