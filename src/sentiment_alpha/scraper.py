import yfinance as yf
import pandas as pd
from typing import List, Optional
import time

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

def create_session_with_retries():
    """创建带重试机制的 requests session"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504),
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

class NewsScraper:
    """
    金融新闻获取器
    使用 yfinance 内置 API，更稳定可靠
    """
    
    def __init__(self):
        pass
    
    def fetch_yahoo_finance_news(self, ticker: str, max_articles: int = 20) -> pd.DataFrame:
        """
        从 yfinance 获取新闻数据
        
        Args:
            ticker: 股票代码 (如 'AAPL', '600519.SS')
            max_articles: 最大文章数
        
        Returns:
            DataFrame with columns: ['title', 'publisher', 'link', 'published_at', 'ticker']
        """
        try:
            stock = yf.Ticker(ticker)
            news_list = stock.news
            
            if not news_list:
                print(f"⚠️  No news found for {ticker}")
                return pd.DataFrame(columns=['title', 'publisher', 'link', 'published_at', 'ticker'])
            
            # 解析新闻数据
            articles = []
            for item in news_list[:max_articles]:
                articles.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', 'Unknown'),
                    'link': item.get('link', ''),
                    'published_at': pd.to_datetime(item.get('providerPublishTime', 0), unit='s', errors='coerce'),
                    'ticker': ticker,
                    'thumbnail': item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '')
                })
            
            df = pd.DataFrame(articles)
            
            # 按时间排序
            if not df.empty:
                df = df.sort_values('published_at', ascending=False).reset_index(drop=True)
            
            print(f"✅ Fetched {len(df)} news articles for {ticker}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching news for {ticker}: {e}")
            return pd.DataFrame(columns=['title', 'publisher', 'link', 'published_at', 'ticker'])
    
    def fetch_multiple_tickers(self, tickers: List[str], max_per_ticker: int = 10) -> pd.DataFrame:
        """
        批量获取多个股票的新闻
        """
        all_news = []
        
        for ticker in tickers:
            print(f"📰 Fetching news for {ticker}...")
            df = self.fetch_yahoo_finance_news(ticker, max_per_ticker)
            all_news.append(df)
            time.sleep(0.5)  # 友好速率限制
        
        if all_news:
            return pd.concat(all_news, ignore_index=True)
        return pd.DataFrame()