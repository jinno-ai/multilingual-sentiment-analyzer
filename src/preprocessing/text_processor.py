"""
Text Preprocessing for Sentiment Analysis

Handles text cleaning, normalization, and language detection.
"""

import re
from typing import List, Dict, Optional, Tuple
import unicodedata


class TextProcessor:
    """Text preprocessing for multilingual sentiment analysis"""
    
    def __init__(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
    
    def clean_text(
        self,
        text: str,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_emojis: bool = False,
        lowercase: bool = False,
        normalize_unicode: bool = True
    ) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_emojis: Remove emojis
            lowercase: Convert to lowercase
            normalize_unicode: Normalize unicode characters
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize unicode
        if normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove mentions
        if remove_mentions:
            text = self.mention_pattern.sub('', text)
        
        # Remove hashtags (keep the word, remove #)
        if remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        else:
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove emojis
        if remove_emojis:
            text = self.emoji_pattern.sub('', text)
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            from langdetect import detect_langs
            
            results = detect_langs(text)
            if results:
                top_result = results[0]
                return top_result.lang, top_result.prob
            return 'unknown', 0.0
        
        except ImportError:
            # Fallback: simple heuristic
            return self._simple_language_detect(text)
        except Exception:
            return 'unknown', 0.0
    
    def _simple_language_detect(self, text: str) -> Tuple[str, float]:
        """Simple language detection based on character ranges"""
        # Count character types
        japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))
        chinese_chars = len(re.findall(r'[\u4E00-\u9FFF]', text))
        korean_chars = len(re.findall(r'[\uAC00-\uD7AF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total = japanese_chars + chinese_chars + korean_chars + latin_chars
        
        if total == 0:
            return 'unknown', 0.0
        
        # Determine language
        if japanese_chars > total * 0.3:
            return 'ja', japanese_chars / total
        elif korean_chars > total * 0.3:
            return 'ko', korean_chars / total
        elif chinese_chars > total * 0.3 and japanese_chars < total * 0.1:
            return 'zh', chinese_chars / total
        else:
            return 'en', latin_chars / total
    
    def extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text"""
        return self.emoji_pattern.findall(text)
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        return [tag[1:] for tag in self.hashtag_pattern.findall(text)]
    
    def extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text"""
        return [mention[1:] for mention in self.mention_pattern.findall(text)]
    
    def tokenize_simple(self, text: str) -> List[str]:
        """Simple whitespace tokenization"""
        return text.split()
    
    def batch_clean(
        self,
        texts: List[str],
        **kwargs
    ) -> List[str]:
        """Clean multiple texts"""
        return [self.clean_text(text, **kwargs) for text in texts]


class JapaneseProcessor(TextProcessor):
    """Specialized processor for Japanese text"""
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None
    
    def _get_tokenizer(self):
        """Lazy load Japanese tokenizer"""
        if self.tokenizer is None:
            try:
                import fugashi
                self.tokenizer = fugashi.Tagger()
            except ImportError:
                pass
        return self.tokenizer
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Japanese text"""
        tokenizer = self._get_tokenizer()
        
        if tokenizer:
            return [word.surface for word in tokenizer(text)]
        else:
            # Fallback: character-based
            return list(text.replace(' ', ''))
    
    def normalize_japanese(self, text: str) -> str:
        """Normalize Japanese text"""
        # Convert full-width to half-width for numbers and letters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove repeated characters (e.g., すごーーーい → すごーい)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text


class ChineseProcessor(TextProcessor):
    """Specialized processor for Chinese text"""
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None
    
    def _get_tokenizer(self):
        """Lazy load Chinese tokenizer"""
        if self.tokenizer is None:
            try:
                import jieba
                self.tokenizer = jieba
            except ImportError:
                pass
        return self.tokenizer
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Chinese text"""
        tokenizer = self._get_tokenizer()
        
        if tokenizer:
            return list(tokenizer.cut(text))
        else:
            # Fallback: character-based
            return list(text.replace(' ', ''))
    
    def convert_traditional_to_simplified(self, text: str) -> str:
        """Convert traditional Chinese to simplified"""
        try:
            from hanziconv import HanziConv
            return HanziConv.toSimplified(text)
        except ImportError:
            return text
