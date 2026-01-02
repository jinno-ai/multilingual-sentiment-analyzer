"""
Multilingual Sentiment Analyzer

Cross-lingual sentiment analysis with fine-tuned transformers.
"""

from src.models.sentiment_model import MultilingualSentimentAnalyzer
from src.preprocessing.text_processor import TextProcessor, JapaneseProcessor, ChineseProcessor

__version__ = "0.1.0"
__author__ = "Jinno"

__all__ = [
    'MultilingualSentimentAnalyzer',
    'TextProcessor',
    'JapaneseProcessor',
    'ChineseProcessor'
]
