"""
Unit tests for Sentiment Model
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from src.models.sentiment_model import MultilingualSentimentAnalyzer


@pytest.fixture
def analyzer():
    """Create sentiment analyzer"""
    return MultilingualSentimentAnalyzer()


def test_analyzer_initialization(analyzer):
    """Test analyzer initialization"""
    assert analyzer.model_name == "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    assert analyzer.model is None
    assert analyzer.tokenizer is None
    assert analyzer.labels == ["negative", "neutral", "positive"]


@patch('src.models.sentiment_model.AutoTokenizer')
@patch('src.models.sentiment_model.AutoModelForSequenceClassification')
def test_model_loading(mock_model_class, mock_tokenizer_class, analyzer):
    """Test model loading"""
    mock_tokenizer = Mock()
    mock_model = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model

    analyzer.load_model()

    assert analyzer.tokenizer == mock_tokenizer
    assert analyzer.model == mock_model
    mock_model.to.assert_called_once()
    mock_model.eval.assert_called_once()


def test_analyze_without_loading(analyzer):
    """Test analyzing without loading model first"""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        analyzer.analyze("Test text")


@patch('src.models.sentiment_model.AutoTokenizer')
@patch('src.models.sentiment_model.AutoModelForSequenceClassification')
def test_analyze_single_text(mock_model_class, mock_tokenizer_class, analyzer):
    """Test analyzing single text"""
    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = Mock()
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.1, 0.3, 0.6]])
    mock_model.return_value = mock_output
    mock_model_class.from_pretrained.return_value = mock_model

    analyzer.load_model()

    result = analyzer.analyze("This is great!")

    assert result['sentiment'] in ['negative', 'neutral', 'positive']
    assert 'confidence' in result
    assert 'scores' in result
    assert 0 <= result['confidence'] <= 1.0


@patch('src.models.sentiment_model.AutoTokenizer')
@patch('src.models.sentiment_model.AutoModelForSequenceClassification')
def test_analyze_batch(mock_model_class, mock_tokenizer_class, analyzer):
    """Test batch analysis"""
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
        'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])
    }
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = Mock()
    mock_output = Mock()
    mock_output.logits = torch.tensor([
        [0.1, 0.3, 0.6],
        [0.7, 0.2, 0.1]
    ])
    mock_model.return_value = mock_output
    mock_model_class.from_pretrained.return_value = mock_model

    analyzer.load_model()

    texts = ["Great!", "Terrible!"]
    results = analyzer.analyze(texts)

    assert len(results) == 2
    assert all('sentiment' in r for r in results)
    assert all('confidence' in r for r in results)


@patch('src.models.sentiment_model.AutoTokenizer')
@patch('src.models.sentiment_model.AutoModelForSequenceClassification')
def test_sentiment_scores(mock_model_class, mock_tokenizer_class, analyzer):
    """Test that sentiment scores are returned"""
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = Mock()
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.1, 0.3, 0.6]])
    mock_model.return_value = mock_output
    mock_model_class.from_pretrained.return_value = mock_model

    analyzer.load_model()

    result = analyzer.analyze("Test")

    assert 'scores' in result
    assert 'negative' in result['scores']
    assert 'neutral' in result['scores']
    assert 'positive' in result['scores']


@patch('src.models.sentiment_model.AutoTokenizer')
@patch('src.models.sentiment_model.AutoModelForSequenceClassification')
def test_multilingual_support(mock_model_class, mock_tokenizer_class, analyzer):
    """Test multilingual text support"""
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = Mock()
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.1, 0.3, 0.6]])
    mock_model.return_value = mock_output
    mock_model_class.from_pretrained.return_value = mock_model

    analyzer.load_model()

    # Test different languages
    english_result = analyzer.analyze("This is great")
    japanese_result = analyzer.analyze("これは素晴らしい")
    spanish_result = analyzer.analyze("Esto es genial")

    # All should return results
    for result in [english_result, japanese_result, spanish_result]:
        assert 'sentiment' in result
        assert 'confidence' in result
