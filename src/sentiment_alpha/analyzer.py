from transformers import pipeline
import pandas as pd
from typing import Dict, List, Optional
import torch

class SentimentAnalyzer:
    """
    基于 Hugging Face Transformers 的情感分析引擎
    """
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        初始化情感分析器

        Args:
            model_name: Hugging Face 模型名称
        """
        self.device = 0 if torch.cuda.is_available() else -1

        try:
            
            self.analyzer = pipeline( 
                "sentiment-analysis", # type: ignore[arg-type]
                model=model_name,
                device=self.device,
                truncation=True
            )
            print(f"✅ Loaded model: {model_name}")
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print("Using fallback analyzer...")
            self.analyzer = None
    
    def analyze_text(self, text: str) -> Dict:
        """
        分析单条文本的情感

        Returns:
            Dict with keys: ['label', 'score', 'sentiment_score']
            sentiment_score: -1 (negative) to 1 (positive)
        """
        if not text or len(text.strip()) < 5:
            return {
                'label': 'NEUTRAL',
                'score': 1.0,
                'sentiment_score': 0.0
            }
        
        if self.analyzer is None:
            # 简单规则-based 回退方案
            return self._rule_based_analysis(text)
        
        try:
            result = self.analyzer(text[:512])[0]   # 限制长度

            # 转换为 -1 到 1 的分数
            label = result['label']
            score = result['score']

            if label == 'POSITIVE':
                sentiment_score = score
            elif label == 'NEGATIVE':
                sentiment_score = - score
            else:
                sentiment_score = 0.0
            
            return {
                'label': label,
                'score': score,
                'sentiment_score': sentiment_score
            }
        
        except Exception as e:
            print(f"Error analyzing text: {e}")

            return {
                'label': 'ERROR',
                'score': 0.0,
                'sentiment_score': 0.0
            }
        
    def _rule_based_analysis(self, text: str) -> Dict:
        """
        简单的规则 -based 情感分析 (回退方案)
        """

        positive_words = ['up', 'gain', 'grow', 'profit', 'beat', 'surge', 'rally', 'bullish']
        negative_words = ['down', 'loss', 'fall', 'drop', 'miss', 'decline', 'bearish', 'crash']

        text_lower = text.lower()

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        total = pos_count + neg_count

        if total == 0:
            return {'label': 'NEUTRAL', 'score': 1.0, 'sentiment_score': 0.0}
        
        sentiment_score = (pos_count - neg_count) / total

        if sentiment_score > 0:
            label = 'POSITIVE'
        elif sentiment_score < 0:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'

        return {
            'label': label,
            'score': abs(sentiment_score),
            'sentiment_score': sentiment_score
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'title') -> pd.DataFrame:
        """
        批量分析 DataFrame 中的文本
        """
        results = []

        for idx, row in df.iterrows():
            text = row.get(text_column, '')
            sentiment = self.analyze_text(text)
            results.append(sentiment)
        
        result_df = pd.DataFrame(results)

        # 合并原始数据和分析结果
        return pd.concat([df.reset_index(drop=True), result_df], axis=1)