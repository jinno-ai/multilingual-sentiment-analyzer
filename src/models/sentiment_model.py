"""
Multilingual Sentiment Analysis Model

Cross-lingual sentiment analysis using XLM-RoBERTa.
"""

from typing import Dict, List, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class MultilingualSentimentAnalyzer:
    """Sentiment analyzer supporting multiple languages"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = ["negative", "neutral", "positive"]
    
    def load_model(self) -> None:
        """Load pretrained model and tokenizer"""
        print(f"🧠 Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
    
    def analyze(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Analyze sentiment of text(s)"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Handle single text
        if isinstance(text, str):
            return self._analyze_single(text)
        
        # Handle batch
        return [self._analyze_single(t) for t in text]
    
    def _analyze_single(self, text: str) -> Dict:
        """Analyze sentiment of single text"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predictions
        scores_np = scores.cpu().numpy()[0]
        predicted_label = self.labels[np.argmax(scores_np)]
        confidence = float(np.max(scores_np))
        
        return {
            "text": text,
            "sentiment": predicted_label,
            "confidence": confidence,
            "scores": {
                label: float(score)
                for label, score in zip(self.labels, scores_np)
            }
        }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Analyze sentiment in batches for efficiency"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            scores_np = scores.cpu().numpy()
            for j, text in enumerate(batch):
                predicted_label = self.labels[np.argmax(scores_np[j])]
                confidence = float(np.max(scores_np[j]))
                
                results.append({
                    "text": text,
                    "sentiment": predicted_label,
                    "confidence": confidence,
                    "scores": {
                        label: float(score)
                        for label, score in zip(self.labels, scores_np[j])
                    }
                })
        
        return results
