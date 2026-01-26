# Project-Specific Instructions for AI Assistants

## Project Context

You are working on the **Multilingual Sentiment Analyzer**, a production-ready sentiment analysis system supporting 100+ languages using transformer models.

## Project Goals

1. **High Accuracy**: Maintain F1-Score > 88% across supported languages
2. **Performance**: Target < 50ms inference time per text
3. **Scalability**: Support 10K+ requests per minute
4. **Usability**: Provide intuitive CLI and API interfaces

## Key Technical Decisions

### Model Choice
- **Primary Model**: XLM-RoBERTa (Twitter sentiment fine-tuned)
- **Rationale**: Excellent cross-lingual performance, efficient inference
- **Alternatives Considered**: mBERT, mT5 (kept for future evaluation)

### Architecture
- **API Framework**: FastAPI (async, modern, automatic OpenAPI docs)
- **ML Framework**: PyTorch (industry standard for transformers)
- **Deployment**: Docker-ready, supports GPU acceleration

## Coding Standards

### Python Conventions
- Follow PEP 8 strictly
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use f-strings for string formatting

### Code Organization
```
src/
├── models/          # ML model implementations
├── api/             # API endpoints and routing
├── preprocessing/   # Text processing utilities
└── utils/           # Helper functions
tests/
├── unit/           # Unit tests
└── integration/    # Integration tests
```

### Error Handling
- Use specific exception types
- Provide actionable error messages
- Log errors with context
- Never expose sensitive data in error messages

## Testing Requirements

### Test Coverage
- Minimum 80% code coverage
- Unit tests for all business logic
- Integration tests for API endpoints
- Performance tests for critical paths

### Test Organization
- Use pytest framework
- Group tests by functionality
- Use fixtures for common setup
- Mock external dependencies

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_sentiment_model.py
```

## API Development Guidelines

### Endpoint Design
- Use noun-based resource naming
- Implement proper HTTP methods (GET, POST, PUT, DELETE)
- Return appropriate status codes
- Include request validation

### Response Format
```python
# Success response
{
    "text": "input text",
    "sentiment": "positive",
    "confidence": 0.95,
    "scores": {
        "positive": 0.95,
        "neutral": 0.03,
        "negative": 0.02
    }
}

# Error response
{
    "detail": "Error message description"
}
```

### Performance Optimization
- Implement async I/O operations
- Use connection pooling
- Enable response compression
- Cache frequently accessed data

## Model Development Guidelines

### Model Updates
- Test new models thoroughly before deployment
- Compare against baseline metrics
- Document performance differences
- Maintain model version history

### Performance Metrics
- Track F1-score, accuracy, precision, recall
- Monitor inference time (p50, p95, p99)
- Measure throughput (requests/second)
- Profile memory usage

### Language Support
- Verify language coverage for new models
- Test with edge cases (mixed scripts, code-switching)
- Document any limitations
- Provide language detection fallback

## Common Development Tasks

### Adding New Features
1. Create feature branch from main
2. Write tests first (TDD when appropriate)
3. Implement feature
4. Run tests and linting
5. Update documentation
6. Create pull request

### Debugging
- Use Python debugger (pdb) for code issues
- Check model logs for ML problems
- Monitor API logs for endpoint issues
- Use profiling tools for performance

### Code Review Checklist
- [ ] Code follows project conventions
- [ ] Tests cover new functionality
- [ ] Documentation is updated
- [ ] No sensitive data in code
- [ ] Performance impact considered
- [ ] Error handling is proper

## Dependency Management

### Adding Dependencies
- Add to requirements.txt with specific version
- Test thoroughly with new dependency
- Update this document if needed
- Check for security vulnerabilities

### Version Pinning
- Pin exact versions in requirements.txt
- Document reason for version constraints
- Test compatibility after updates

## Deployment Considerations

### Environment Setup
- Use virtual environments
- Set appropriate environment variables
- Configure logging levels
- Enable monitoring

### Performance Tuning
- Adjust batch sizes based on hardware
- Configure GPU memory allocation
- Set appropriate worker counts
- Enable request caching

## Security Best Practices

- Never commit API keys or secrets
- Validate all user inputs
- Sanitize text data before processing
- Implement rate limiting
- Use HTTPS in production
- Keep dependencies updated

## Communication Style

- Be concise and direct in responses
- Provide code examples when helpful
- Explain technical decisions clearly
- Ask clarifying questions when needed
- Focus on actionable feedback

## Project-Specific Patterns

### Model Loading
```python
analyzer = MultilingualSentimentAnalyzer()
analyzer.load_model()  # Always call before analyze
```

### Error Handling
```python
try:
    result = analyzer.analyze(text)
except RuntimeError as e:
    # Handle model-specific errors
    logger.error(f"Model error: {e}")
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
```

### Batch Processing
```python
# Always use batch processing for multiple texts
results = analyzer.analyze_batch(texts, batch_size=32)
```

## Continuous Improvement

- Monitor performance metrics regularly
- Gather user feedback
- Track error rates and patterns
- Plan incremental improvements
- Document lessons learned

## Getting Help

- Check project documentation first
- Review existing code patterns
- Consult team members
- Refer to framework documentation
- Use debugging tools systematically