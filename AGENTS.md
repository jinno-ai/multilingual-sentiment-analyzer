# AI Agent Guidelines for Multilingual Sentiment Analyzer

## Project Overview

This is a cross-lingual sentiment analysis system using fine-tuned transformer models. The project supports 100+ languages and provides both CLI tools and REST API interfaces.

## Technology Stack

- **ML Framework**: PyTorch 2.1.0, Transformers 4.36.0
- **API Framework**: FastAPI 0.109.0, Uvicorn 0.27.0
- **Testing**: Pytest 7.4.3, httpx 0.26.0
- **Model**: XLM-RoBERTa (Twitter sentiment)

## Core Components

### Model Architecture (`src/models/sentiment_model.py`)
- MultilingualSentimentAnalyzer class
- Model loading and inference
- Batch processing capabilities
- GPU/CPU automatic detection

### API Server (`src/api/main.py`)
- FastAPI application
- Single text analysis endpoint (`/analyze`)
- Batch analysis endpoint (`/analyze/batch`)
- Health check endpoints

### Text Preprocessing (`src/preprocessing/text_processor.py`)
- Text cleaning and normalization
- Language detection capabilities

## Development Guidelines

### Code Quality
- Follow PEP 8 style guidelines
- Maintain type hints for all functions
- Write docstrings for all public methods
- Keep functions focused and modular

### Testing Requirements
- Unit tests for all new functionality
- Integration tests for API endpoints
- Test coverage > 80%
- Use pytest for testing framework

### Performance Considerations
- Optimize batch processing for throughput
- Monitor GPU memory usage
- Implement proper error handling
- Use async operations for I/O bound tasks

## API Design Principles

1. **RESTful Conventions**: Use appropriate HTTP methods and status codes
2. **Error Handling**: Provide clear error messages and proper status codes
3. **Validation**: Use Pydantic models for request validation
4. **Documentation**: Maintain OpenAPI documentation

## Model Management

### Model Versioning
- Track model versions in configuration
- Support model rollback capabilities
- Document model performance metrics

### Deployment Considerations
- Model loading on startup
- Graceful shutdown handling
- Health check endpoints
- Monitoring and logging

## Development Workflow

1. **Feature Development**: Create feature branches from main
2. **Testing**: Write tests before implementation (TDD when appropriate)
3. **Code Review**: Ensure code meets quality standards
4. **Documentation**: Update relevant documentation
5. **Deployment**: Follow proper deployment procedures

## Common Tasks

### Adding New Language Support
- Verify model supports target language
- Add language-specific preprocessing if needed
- Update documentation
- Test with sample texts

### Model Updates
- Test new model version thoroughly
- Compare performance metrics
- Update model configuration
- Document breaking changes

### API Endpoint Addition
- Follow FastAPI patterns
- Add proper error handling
- Include request validation
- Update API documentation

## Dependencies Management

- Pin exact versions in requirements.txt
- Regularly update dependencies
- Test thoroughly after dependency updates
- Document any breaking changes

## Performance Optimization

- Use batch processing when possible
- Implement caching for repeated requests
- Monitor and optimize memory usage
- Profile code for bottlenecks

## Security Considerations

- Input validation and sanitization
- Rate limiting for API endpoints
- Proper error message handling (don't leak sensitive info)
- Keep dependencies updated