"""
FastAPI Application for Sentiment Analysis

RESTful API for multilingual sentiment analysis.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union
import uvicorn

from src.models.sentiment_model import MultilingualSentimentAnalyzer

# Create app
app = FastAPI(
    title="Multilingual Sentiment Analyzer",
    description="Cross-lingual sentiment analysis API",
    version="1.0.0"
)

# Global model instance
analyzer = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global analyzer
    analyzer = MultilingualSentimentAnalyzer()
    analyzer.load_model()


class TextRequest(BaseModel):
    """Request model for single text"""
    text: str = Field(..., description="Text to analyze")


class BatchRequest(BaseModel):
    """Request model for batch"""
    texts: List[str] = Field(..., description="List of texts to analyze")
    batch_size: int = Field(32, description="Batch size for processing")


@app.post("/analyze")
async def analyze_text(request: TextRequest):
    """Analyze sentiment of single text"""
    try:
        result = analyzer.analyze(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch")
async def analyze_batch(request: BatchRequest):
    """Analyze sentiment of multiple texts"""
    try:
        results = analyzer.analyze_batch(request.texts, request.batch_size)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Multilingual Sentiment Analyzer",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
