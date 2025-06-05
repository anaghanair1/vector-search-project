"""
Embedding service using sentence transformers
Tried a few different models but this one seems to work best for our use case

sentence-transformers Documentation: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
all-MiniLM-L6-v2 Model Card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
Hugging Face Transformers: https://huggingface.co/docs/transformers/index
PyTorch Device Selection: https://pytorch.org/docs/stable/notes/cuda.html
Batch Processing: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html#computing-embeddings
Device Management: https://pytorch.org/tutorials/recipes/recipes/changing_default_device.html
Progress Tracking: https://github.com/tqdm/tqdm

"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm
import torch

class EmbeddingService:
    def __init__(self, model_name='all-MiniLM-L6-v2'):  # 384 dimensions
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self.load_model()
    
    def load_model(self):
        """Load the sentence transformer model - takes a few seconds first time"""
        try:
            print(f"Loading model: {self.model_name}")
            
            # check if we have gpu available
            if torch.cuda.is_available():
                device = 'cuda'
                print("Using GPU acceleration")
            else:
                device = 'cpu'
                print("Using CPU (slower but works)")
            
            # load the model
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # figure out embedding dimension
            test_text = "test"
            test_emb = self.model.encode(test_text)
            self.embedding_dim = len(test_emb)
            
            print(f"Model loaded! Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load {self.model_name}: {e}")
    
    def create_embedding(self, text):
        """Create embedding for single text"""
        if not self.model:
            raise RuntimeError("Model not loaded yet")
        
        try:
            # create the embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()  # convert to regular python list
            
        except Exception as e:
            print(f"Embedding creation failed: {e}")
            raise RuntimeError(f"Failed to create embedding: {e}")
    
    def create_batch_embeddings(self, texts, batch_size=32, show_progress=True):
        """Create embeddings for multiple texts - much faster than one by one"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if not texts:
            return []
        
        try:
            print(f"Creating embeddings for {len(texts)} texts...")
            
            # process all at once - sentence-transformers handles batching internally
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
            
            # convert to list format
            result = []
            for emb in embeddings:
                result.append(emb.tolist())
            
            print(f"Created {len(result)} embeddings successfully")
            return result
            
        except Exception as e:
            print(f"Batch embedding failed: {e}")
            raise RuntimeError(f"Batch processing error: {e}")
    
    @property
    def embedding_dimension(self):
        return self.embedding_dim
    
    def get_model_info(self):
        """Get info about loaded model"""
        info = {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': str(self.model.device) if self.model else None,
        }
        
        # try to get max sequence length if available
        if self.model and hasattr(self.model, 'max_seq_length'):
            info['max_seq_length'] = self.model.max_seq_length
        
        return info

# quick test if running directly
if __name__ == "__main__":
    service = EmbeddingService()
    
    # test single embedding
    test_text = "This restaurant has great food"
    emb = service.create_embedding(test_text)
    print(f"Single embedding size: {len(emb)}")
    print(f"First few values: {emb[:5]}")
    
    # test batch
    texts = [
        "Amazing food and service!",
        "Terrible experience, would not go back",
        "Pretty good overall, decent prices"
    ]
    
    batch_embs = service.create_batch_embeddings(texts)
    print(f"Batch size: {len(batch_embs)}")
    
    print("Model info:", service.get_model_info())