from transformers import pipeline
import pandas as pd
from typing import Dict, List, Optional
import torch

class SentimentAnalyzer:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        使用 Twitter 情感模型，更小更快，适合新闻短文本
        模型大小: ~500MB
        """
        self.device = 0 if torch.cuda.is_available() else -1
        
        # 模型列表
        models = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",  # Twitter 情感，适合短文本
            "distilbert-base-uncased-finetuned-sst-2-english",   # 原模型
            "nlptown/bert-base-multilingual-uncased-sentiment",  # 多语言支持
        ]
        
        for model in models:
            try:
                print(f"🔄 Trying to load: {model}")
                self.analyzer = pipeline(
                    "sentiment-analysis", # type: ignore[arg-type]
                    model=model,
                    device=self.device,
                    truncation=True,
                    max_length=512
                )
                print(f"✅ Loaded model: {model}")
                return
            except Exception as e:
                print(f"⚠️  Failed to load {model}: {e}")
                continue
        
        # 全部失败，使用回退方案
        print("❌ All models failed, using rule-based fallback")
        self.analyzer = None
    
    def analyze_text(self, title: str, summary: str = "") -> Dict:
        """
        分析新闻情感 - 适配 Twitter-RoBERTa 标签格式
        """
        # 组合文本：标题 + 摘要（如果有）
        if summary and len(summary.strip()) > 20:
            text = f"{title}. {summary}"
        else:
            text = str(title).strip()
        
        if not text or len(text) < 5:
            return {'label': 'NEUTRAL', 'score': 1.0, 'sentiment_score': 0.0}
        
        # 如果使用回退方案
        if self.analyzer is None:
            return self._rule_based_analysis(text)
        
        try:
            result = self.analyzer(text[:512])[0]
            
            label_raw = result['label']  # 可能是: LABEL_0, LABEL_1, LABEL_2 或 POSITIVE, NEGATIVE
            score = result['score']
            
            # 统一标签映射
            label_mapping = {
                # Twitter-RoBERTa 格式
                'LABEL_0': 'NEGATIVE',
                'LABEL_1': 'NEUTRAL', 
                'LABEL_2': 'POSITIVE',
                # 其他常见格式
                'negative': 'NEGATIVE',
                'neg': 'NEGATIVE',
                'neutral': 'NEUTRAL',
                'neu': 'NEUTRAL',
                'positive': 'POSITIVE',
                'pos': 'POSITIVE',
                # 星級格式
                '1 star': 'NEGATIVE',
                '2 stars': 'NEGATIVE',
                '3 stars': 'NEUTRAL',
                '4 stars': 'POSITIVE',
                '5 stars': 'POSITIVE',
            }
            
            label = label_mapping.get(label_raw.upper(), label_raw.upper())
            
            # 计算 -1 到 1 的情感分数
            if label == 'POSITIVE':
                sentiment_score = score
            elif label == 'NEGATIVE':
                sentiment_score = -score
            else:  # NEUTRAL
                sentiment_score = 0.0
            
            return {
                'label': label,
                'score': round(score, 3),
                'sentiment_score': round(sentiment_score, 3)
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing text: {e}")
            return self._rule_based_analysis(text)
        
        
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