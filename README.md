# Multilingual Sentiment Analyzer

Cross-lingual sentiment analysis with fine-tuned transformers. Supports 100+ languages including English, Japanese, Chinese, Korean, and more.

## Features

- 🌍 **Multi-language Support**: Analyze sentiment in 100+ languages
- 🎯 **High Accuracy**: Fine-tuned XLM-RoBERTa model
- ⚡ **Batch Processing**: Efficient batch analysis
- 🔧 **Preprocessing**: Built-in text cleaning and normalization
- 🚀 **REST API**: FastAPI server included

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### CLI Usage

```bash
# Analyze single text
python run.py analyze "I love this product!"

# Analyze with language detection
python run.py analyze "これは素晴らしい！" --detect-language

# Batch analysis from file
python run.py batch reviews.txt

# Interactive mode
python run.py interactive

# Start API server
python run.py server
```

### Python API

```python
from src.models.sentiment_model import MultilingualSentimentAnalyzer

# Initialize
analyzer = MultilingualSentimentAnalyzer()
analyzer.load_model()

# Analyze
result = analyzer.analyze("This is amazing!")
print(result['sentiment'])  # positive
print(result['confidence'])  # 0.95
```

### REST API

```bash
# Start server
python run.py server

# Analyze via API
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!"}'
```

## Supported Languages

English, Japanese, Chinese, Korean, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Arabic, and 90+ more.

## License

MIT
