"""
Text chunking utilities for processing long reviews into smaller segments
Optimized for embedding models and semantic search
"""
import re
from typing import List, Dict, Any

class TextChunker:
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
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
        
        chunks = []
        start = 0
        
        while start < len(cleaned_text):
            end = min(start + self.chunk_size, len(cleaned_text))
            
            # Try to find a good breaking point (sentence boundary)
            if end < len(cleaned_text):
                end = self._find_sentence_boundary(cleaned_text, start, end)
            
            # Extract chunk
            chunk = cleaned_text[start:end].strip()
            
            if chunk and len(chunk) > 10:  # Only add meaningful chunks
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(end - self.overlap, start + 1)
            
            # Prevent infinite loops
            if start >= len(cleaned_text):
                break
        
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
    
    def _find_sentence_boundary(self, text: str, start: int, preferred_end: int) -> int:
        """
        Find the best sentence boundary within a reasonable range
        
        Args:
            text: Full text
            start: Start position
            preferred_end: Preferred end position
            
        Returns:
            Best end position
        """
        # Look for sentence endings within the last 100 characters
        search_start = max(preferred_end - 100, start)
        search_text = text[search_start:preferred_end]
        
        # Find sentence endings (., !, ?)
        sentence_endings = []
        for match in re.finditer(r'[.!?]\s+', search_text):
            sentence_endings.append(search_start + match.end())
        
        if sentence_endings:
            # Return the last sentence ending
            return sentence_endings[-1]
        
        # If no sentence ending found, look for other boundaries
        # Comma, semicolon, or colon
        for pattern in [r'[,;:]\s+', r'\s+and\s+', r'\s+but\s+', r'\s+or\s+']:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                return search_start + matches[-1].end()
        
        # If no good boundary found, use preferred end
        return preferred_end
    
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
            print(f"📝 Chunking {len(reviews)} reviews...")
        
        for i, review in enumerate(reviews):
            if show_progress and i % 50 == 0:
                print(f"  Processing review {i + 1}/{len(reviews)}")
            
            review_chunks = self.chunk_review(review)
            all_chunks.extend(review_chunks)
        
        if show_progress:
            avg_chunks = len(all_chunks) / len(reviews) if reviews else 0
            print(f"✅ Created {len(all_chunks)} chunks from {len(reviews)} reviews")
            print(f"📊 Average {avg_chunks:.1f} chunks per review")
        
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

# Example usage and testing
if __name__ == "__main__":
    # Test the chunker
    chunker = TextChunker(chunk_size=200, overlap=30)
    
    # Test with sample review
    sample_review = {
        'review_id': 'test_001',
        'text': """This restaurant was absolutely amazing! The food was delicious, 
                   the service was excellent, and the atmosphere was perfect. I especially 
                   loved the pasta dish - it was cooked to perfection with a rich, flavorful 
                   sauce. The staff was very attentive and friendly throughout our entire meal. 
                   The prices were reasonable for the quality of food and service we received. 
                   We will definitely be coming back here again and would highly recommend 
                   this place to anyone looking for a great dining experience.""",
        'stars': 5
    }
    
    # Chunk the review
    chunks = chunker.chunk_review(sample_review)
    
    print(f"Original text length: {len(sample_review['text'])}")
    print(f"Number of chunks: {len(chunks)}")
    print()
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Length: {len(chunk['chunk_text'])}")
        print(f"  Text: {chunk['chunk_text']}")
        print()
    
    # Get statistics
    stats = chunker.get_stats(chunks)
    print("Chunking Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")