import yfinance as yf
import pandas as pd
from typing import List, Optional
import time
from datetime import datetime

class NewsScraper:
    """金融新闻获取器 - 适配 yfinance 新数据结构"""
    
    def fetch_yahoo_finance_news(self, ticker: str, max_articles: int = 20) -> pd.DataFrame:
        """
        从 yfinance 获取新闻（适配嵌套 content 结构）
        """
        try:
            stock = yf.Ticker(ticker)
            news_list = stock.news
            
            if not news_list:
                print(f"⚠️ No news found for {ticker}")
                return pd.DataFrame(columns=['title', 'publisher', 'link', 'published_at', 'ticker', 'summary'])
            
            articles = []
            for item in news_list[:max_articles]:
                # 从 content 嵌套对象中提取字段
                content = item.get('content', {}) if isinstance(item, dict) else item
                
                # 提取标题
                title = content.get('title', '')
                if not title or title.strip() == '':
                    title = item.get('title', 'No Title')  # 回退到顶层
                
                # 提取发布商
                provider = content.get('provider', {})
                publisher = provider.get('displayName', '') if isinstance(provider, dict) else ''
                if not publisher:
                    publisher = item.get('publisher', content.get('publisher', 'Unknown'))
                
                # 提取链接
                canonical_url = content.get('canonicalUrl', {})
                link = canonical_url.get('url', '') if isinstance(canonical_url, dict) else ''
                if not link:
                    link = content.get('link', item.get('link', ''))
                
                # 提取时间（支持多种格式）
                pub_date = content.get('pubDate', '') or content.get('published_at', '') or content.get('providerPublishTime', 0)
                
                if isinstance(pub_date, (int, float)):
                    # 时间戳（秒）
                    published_at = pd.to_datetime(pub_date, unit='s', errors='coerce')
                elif isinstance(pub_date, str):
                    # ISO 格式字符串: '2026-03-24T20:57:00Z'
                    published_at = pd.to_datetime(pub_date, errors='coerce')
                else:
                    published_at = pd.NaT
                
                # 提取摘要
                summary = content.get('summary', content.get('description', ''))
                
                articles.append({
                    'title': str(title).strip(),
                    'publisher': str(publisher).strip(),
                    'link': str(link).strip(),
                    'published_at': published_at,
                    'ticker': ticker,
                    'summary': str(summary).strip()
                })
            
            df = pd.DataFrame(articles)
            
            # 过滤空标题的行
            if not df.empty:
                df = df[df['title'].str.len() > 3].reset_index(drop=True)
                df = df.sort_values('published_at', ascending=False).reset_index(drop=True)
                print(f"✅ Fetched {len(df)} news articles for {ticker}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching news for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=['title', 'publisher', 'link', 'published_at', 'ticker', 'summary'])
    
    def fetch_multiple_tickers(self, tickers: List[str], max_per_ticker: int = 10) -> pd.DataFrame:
        """批量获取多个股票的新闻"""
        all_news = []
        for ticker in tickers:
            print(f"📰 Fetching news for {ticker}...")
            df = self.fetch_yahoo_finance_news(ticker, max_per_ticker)
            all_news.append(df)
            time.sleep(0.5)
        
        if all_news:
            return pd.concat(all_news, ignore_index=True)
        return pd.DataFrame()