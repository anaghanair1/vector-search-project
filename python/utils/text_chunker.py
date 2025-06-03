"""
Text chunking utilities for processing long reviews into smaller segments
Optimized for embedding models and semantic search
"""
import re
from typing import List, Dict, Any

class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):  # Larger chunks
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks with smart sentence boundaries
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean and normalize the text
        cleaned_text = self._clean_text(text)
        
        # If text is short enough, return as single chunk
        if len(cleaned_text) <= self.chunk_size:
            return [cleaned_text]
        
        # Split into sentences first
        sentences = self._split_into_sentences(cleaned_text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk if it's meaningful
                if len(current_chunk.strip()) > 50:  # Minimum chunk size
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if chunks and self.overlap > 0:
                    # Take last few words for overlap
                    words = current_chunk.split()
                    overlap_words = words[-10:] if len(words) > 10 else words
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 50:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might confuse embeddings
        cleaned = re.sub(r'[^\w\s.,!?;:()\-\'"]+', ' ', cleaned)
        
        # Normalize quotes
        cleaned = cleaned.replace('"', '"').replace('"', '"')
        cleaned = cleaned.replace(''', "'").replace(''', "'")
        
        return cleaned.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only keep meaningful sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def chunk_review(self, review: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single review into multiple segments
        
        Args:
            review: Dictionary with 'text', 'review_id', 'stars', etc.
            
        Returns:
            List of chunk dictionaries
        """
        review_text = review.get('text', '')
        chunks = self.chunk_text(review_text)
        
        chunked_review = []
        for i, chunk_text in enumerate(chunks):
            chunk_data = {
                'review_id': review.get('review_id', f"review_{hash(review_text)}"),
                'chunk_text': chunk_text,
                'chunk_index': i,
                'stars': review.get('stars', 0),
                'original_review': review
            }
            chunked_review.append(chunk_data)
        
        return chunked_review
    
    def chunk_reviews(self, reviews: List[Dict[str, Any]], show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Chunk multiple reviews efficiently
        
        Args:
            reviews: List of review dictionaries
            show_progress: Whether to show progress
            
        Returns:
            List of all chunks from all reviews
        """
        all_chunks = []
        
        if show_progress:
            print(f"Chunking {len(reviews)} reviews...")
        
        for i, review in enumerate(reviews):
            if show_progress and i % 50 == 0:
                print(f"  Processing review {i + 1}/{len(reviews)}")
            
            review_chunks = self.chunk_review(review)
            all_chunks.extend(review_chunks)
        
        if show_progress:
            avg_chunks = len(all_chunks) / len(reviews) if reviews else 0
            print(f"Created {len(all_chunks)} chunks from {len(reviews)} reviews")
            print(f"Average {avg_chunks:.1f} chunks per review")
        
        return all_chunks
    
    def get_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk['chunk_text']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_characters': sum(chunk_lengths)
        }