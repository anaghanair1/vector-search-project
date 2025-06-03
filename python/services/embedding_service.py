"""
Embedding service using sentence-transformers for local processing
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
from tqdm import tqdm
import torch

class EmbeddingService:
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        """
        Initialize embedding service with specified model
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"Loading model: {self.model_name}")
            
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            
            # Load model
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # Get embedding dimension
            test_embedding = self.model.encode("test")
            self.embedding_dimension = len(test_embedding)
            
            print(f"Model loaded successfully!")
            print(f"Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Create embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
            
        except Exception as e:
            raise RuntimeError(f"Failed to create embedding: {e}")
    
    def create_batch_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        Create embeddings for multiple texts efficiently
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if not texts:
            return []
        
        try:
            print(f"Creating embeddings for {len(texts)} texts...")
            print(f"Batch size: {batch_size}")
            
            # Process all texts at once
            if show_progress:
                embeddings = self.model.encode(
                    texts, 
                    convert_to_numpy=True, 
                    show_progress_bar=True,
                    batch_size=batch_size
                )
            else:
                embeddings = self.model.encode(
                    texts, 
                    convert_to_numpy=True, 
                    batch_size=batch_size
                )
            
            # Convert to list of lists
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            print(f"Successfully created {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            raise RuntimeError(f"Failed to create batch embeddings: {e}")
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'device': str(self.model.device) if self.model else None,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown') if self.model else None
        }

# Test the service if run directly
if __name__ == "__main__":
    service = EmbeddingService()
    
    # Test single embedding
    test_text = "This is a test sentence for embedding."
    embedding = service.create_embedding(test_text)
    print(f"Single embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch embeddings
    test_texts = [
        "Great food and excellent service!",
        "Terrible experience, would not recommend.",
        "Average restaurant with decent prices."
    ]
    
    batch_embeddings = service.create_batch_embeddings(test_texts)
    print(f"Batch embeddings created: {len(batch_embeddings)}")
    
    # Print model info
    print("Model info:", service.get_model_info())