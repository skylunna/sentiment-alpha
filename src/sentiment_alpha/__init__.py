__version__ = "0.1.0"

from .scraper import NewsScraper
from .analyzer import SentimentAnalyzer
from .insights import (
    calculate_daily_sentiment_index,
    align_sentiment_with_price,
    generate_sentiment_summary,
)
from .visualizer import (
    create_sentiment_price_chart,
    create_correlation_heatmap,
    create_sentiment_distribution_chart,
)

__all__ = [
    # 版本
    "__version__",
    # 爬虫
    "NewsScraper",
    # 分析器
    "SentimentAnalyzer",
    # 洞察分析
    "calculate_daily_sentiment_index",
    "align_sentiment_with_price",
    "generate_sentiment_summary",
    # 可视化
    "create_sentiment_price_chart",
    "create_correlation_heatmap",
    "create_sentiment_distribution_chart",
]