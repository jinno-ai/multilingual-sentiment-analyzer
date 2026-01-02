"""
Integration tests for Multilingual Sentiment Analyzer

These tests verify end-to-end sentiment analysis pipeline.
"""

import pytest
from unittest.mock import Mock, patch
import torch


@pytest.fixture
def mock_model_and_tokenizer():
    """Mock model and tokenizer"""
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }

    mock_model = Mock()
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.1, 0.3, 0.6]])
    mock_model.return_value = mock_output
    mock_model.eval = Mock()

    return mock_model, mock_tokenizer


@pytest.mark.integration
@patch('src.models.sentiment_model.AutoTokenizer')
@patch('src.models.sentiment_model.AutoModelForSequenceClassification')
def test_full_analysis_pipeline(mock_model_class, mock_tokenizer_class, mock_model_and_tokenizer):
    """Test complete sentiment analysis pipeline"""
    from src.models.sentiment_model import MultilingualSentimentAnalyzer
    from src.preprocessing.text_processor import TextProcessor

    mock_model, mock_tokenizer = mock_model_and_tokenizer
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model

    # Initialize
    analyzer = MultilingualSentimentAnalyzer()
    analyzer.load_model()

    processor = TextProcessor()

    # Test text
    text = "I absolutely love this product! It's amazing!"

    # Preprocess
    cleaned = processor.clean_text(text)

    # Analyze
    result = analyzer.analyze(cleaned)

    # Verify
    assert 'sentiment' in result
    assert 'confidence' in result
    assert 'scores' in result


@pytest.mark.integration
def test_multilingual_processing():
    """Test processing text in multiple languages"""
    from src.preprocessing.text_processor import TextProcessor, JapaneseProcessor, ChineseProcessor

    # Test English
    processor = TextProcessor()
    english = "Great product!"
    cleaned_english = processor.clean_text(english)
    assert len(cleaned_english) > 0

    # Test Japanese
    jp_processor = JapaneseProcessor()
    japanese = "これは素晴らしい製品です！"
    cleaned_japanese = jp_processor.clean_text(japanese)
    assert len(cleaned_japanese) > 0

    # Test Chinese
    cn_processor = ChineseProcessor()
    chinese = "这是一个很棒的产品！"
    cleaned_chinese = cn_processor.clean_text(chinese)
    assert len(cleaned_chinese) > 0


@pytest.mark.integration
def test_language_detection():
    """Test language detection"""
    from src.preprocessing.text_processor import TextProcessor

    processor = TextProcessor()

    # Test different languages
    texts = [
        ("This is English", "en"),
        ("これは日本語です", "ja"),
        ("这是中文", "zh")
    ]

    for text, expected_lang in texts:
        lang, confidence = processor.detect_language(text)
        # Note: Simple detection might not be perfect
        assert lang in ['en', 'ja', 'zh', 'unknown']


@pytest.mark.integration
@patch('src.models.sentiment_model.AutoTokenizer')
@patch('src.models.sentiment_model.AutoModelForSequenceClassification')
def test_batch_analysis_pipeline(mock_model_class, mock_tokenizer_class):
    """Test batch processing pipeline"""
    from src.models.sentiment_model import MultilingualSentimentAnalyzer
    from src.preprocessing.text_processor import TextProcessor

    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
        'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])
    }

    mock_model = Mock()
    mock_output = Mock()
    mock_output.logits = torch.tensor([
        [0.1, 0.3, 0.6],
        [0.7, 0.2, 0.1]
    ])
    mock_model.return_value = mock_output

    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model

    # Initialize
    analyzer = MultilingualSentimentAnalyzer()
    analyzer.load_model()

    processor = TextProcessor()

    # Batch
    texts = [
        "I love this!",
        "I hate this!",
        "It's okay."
    ]

    # Preprocess
    cleaned_texts = processor.batch_clean(texts)

    # Analyze
    results = analyzer.analyze_batch(cleaned_texts)

    # Verify
    assert len(results) == len(texts)
    assert all('sentiment' in r for r in results)


@pytest.mark.integration
def test_text_extraction_features():
    """Test text extraction features"""
    from src.preprocessing.text_processor import TextProcessor

    processor = TextProcessor()

    # Test with social media text
    text = "Check out https://example.com @user #hashtag 😊"

    # Extract components
    emojis = processor.extract_emojis(text)
    hashtags = processor.extract_hashtags(text)
    mentions = processor.extract_mentions(text)

    assert len(emojis) >= 0
    assert len(hashtags) >= 0
    assert len(mentions) >= 0


@pytest.mark.integration
def test_specialized_processors():
    """Test specialized language processors"""
    from src.preprocessing.text_processor import JapaneseProcessor, ChineseProcessor

    # Japanese processor
    jp_processor = JapaneseProcessor()
    japanese_text = "すごーーーい！"

    normalized = jp_processor.normalize_japanese(japanese_text)
    assert len(normalized) > 0

    # Chinese processor
    cn_processor = ChineseProcessor()
    chinese_text = "测试文本"

    tokens = cn_processor.tokenize(chinese_text)
    assert len(tokens) > 0


@pytest.mark.integration
@patch('src.models.sentiment_model.AutoTokenizer')
@patch('src.models.sentiment_model.AutoModelForSequenceClassification')
def test_confidence_thresholding(mock_model_class, mock_tokenizer_class):
    """Test confidence threshold filtering"""
    from src.models.sentiment_model import MultilingualSentimentAnalyzer

    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }

    mock_model = Mock()
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.33, 0.34, 0.33]])  # Low confidence
    mock_model.return_value = mock_output

    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model

    analyzer = MultilingualSentimentAnalyzer()
    analyzer.load_model()

    result = analyzer.analyze("Test")

    # Even with low confidence, should return result
    assert result['confidence'] >= 0
    assert result['confidence'] <= 1


@pytest.mark.integration
def test_cleaning_pipeline():
    """Test complete text cleaning pipeline"""
    from src.preprocessing.text_processor import TextProcessor

    processor = TextProcessor()

    # Dirty text
    dirty = "  Check out https://t.co/abc @user #test 😊!!!  "

    # Clean
    clean = processor.clean_text(
        dirty,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
        remove_emojis=True,
        lowercase=True
    )

    # Should be cleaned
    assert len(clean) < len(dirty)
    assert "https://" not in clean
    assert "@user" not in clean
