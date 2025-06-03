"""
Working Semantic Search - Fixed version
"""
import sys
import os
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import db_config
from services.embedding_service import EmbeddingService

class WorkingSemanticSearch:
    def __init__(self):
        """Initialize search system"""
        self.client = db_config.get_client()
        self.table_name = 'review_chunks'
        
        # Initialize embedding service
        try:
            self.embedding_service = EmbeddingService()
            print("Embedding service loaded successfully")
        except Exception as e:
            print(f"❌ Could not load embedding service: {e}")
            self.embedding_service = None
        
        # Check data
        count = self.get_total_chunks()
        print(f"Available chunks: {count}")
    
    def get_total_chunks(self) -> int:
        """Get total chunks"""
        try:
            result = self.client.table(self.table_name).select('id', count='exact').execute()
            return result.count or 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return 0
    
    def parse_embedding(self, embedding_data) -> List[float]:
        """
        Parse embedding from database (handles different formats)
        """
        try:
            if isinstance(embedding_data, list):
                # Already a list
                return [float(x) for x in embedding_data]
            elif isinstance(embedding_data, str):
                # String representation - try to parse
                import ast
                parsed = ast.literal_eval(embedding_data)
                return [float(x) for x in parsed]
            else:
                print(f"❌ Unknown embedding format: {type(embedding_data)}")
                return []
        except Exception as e:
            print(f"❌ Error parsing embedding: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        try:
            if not vec1 or not vec2:
                return 0.0
                
            # Ensure same dimensions
            if len(vec1) != len(vec2):
                print(f"❌ Dimension mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
            
            # Convert to numpy arrays
            a = np.array(vec1, dtype=float)
            b = np.array(vec2, dtype=float)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return max(0, min(100, similarity * 100))
            
        except Exception as e:
            print(f"❌ Cosine similarity error: {e}")
            return 0.0
    
    def search_with_supabase(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search using Supabase vector function"""
        if not self.embedding_service:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedding_service.create_embedding(query)
            print(f"Query embedding created: {len(query_embedding)} dimensions")
            
            # Use Supabase function
            result = self.client.rpc('search_similar_reviews', {
                'query_embedding': query_embedding,
                'match_threshold': 0.1,  # Low threshold to get more results
                'match_count': limit
            }).execute()
            
            results = result.data or []
            
            # Convert similarity to percentage
            for item in results:
                if 'similarity' in item:
                    item['similarity_score'] = item['similarity'] * 100
            
            return results
            
        except Exception as e:
            print(f"❌ Supabase search failed: {e}")
            return []
    
    def manual_vector_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Manual vector search with proper embedding parsing"""
        if not self.embedding_service:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedding_service.create_embedding(query)
            print(f"Query embedding: {len(query_embedding)} dimensions")
            
            # Get chunks with embeddings
            result = self.client.table(self.table_name)\
                .select('*')\
                .limit(50)\
                .execute()  # Limit to 50 for testing
            
            chunks = result.data or []
            print(f"Testing with {len(chunks)} chunks")
            
            scored_results = []
            
            for i, chunk in enumerate(chunks):
                if i % 10 == 0:
                    print(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Parse the embedding
                chunk_embedding = self.parse_embedding(chunk.get('embedding'))
                
                if not chunk_embedding:
                    continue
                
                # Calculate similarity
                similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                
                if similarity > 20:  # Only include decent matches
                    chunk['similarity_score'] = similarity
                    scored_results.append(chunk)
            
            # Sort by similarity
            scored_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return scored_results[:limit]
            
        except Exception as e:
            print(f"❌ Manual search failed: {e}")
            return []
    
    def display_results(self, results: List[Dict[str, Any]], query: str):
        """Display search results"""
        if not results:
            print("No results found")
            return
        
        print(f"\nFound {len(results)} results for '{query}':")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            text = result.get('chunk_text', '')
            stars = result.get('stars', 0)
            review_id = result.get('review_id', 'unknown')
            similarity = result.get('similarity_score', 0)
            
            print(f"\nResult {i}:")
            print(f"   Similarity Score: {similarity:.1f}%")
            print(f"   Star Rating: {stars}/5 stars")
            print(f"   Review ID: {review_id}")
            print(f"   Text: {text[:150]}...")
            print("-" * 80)
    
    def test_search(self, query: str):
        """Test both search methods"""
        print(f"\nTesting search for: '{query}'")
        print("=" * 50)
        
        # Try Supabase function first
        print("\nMethod 1: Supabase Function")
        results1 = self.search_with_supabase(query, limit=3)
        if results1:
            self.display_results(results1, query)
        else:
            print("Supabase function failed")
        
        # Try manual search
        print("\nMethod 2: Manual Vector Search")
        results2 = self.manual_vector_search(query, limit=3)
        if results2:
            self.display_results(results2, query)
        else:
            print("Manual search failed")

def main():
    """Main function"""
    print("Working Semantic Search Test")
    print("=" * 40)
    
    try:
        searcher = WorkingSemanticSearch()
        
        # Test with one query first
        searcher.test_search("great food excellent service")
        
        print("\n" + "="*40)
        print("If this worked, your semantic search is ready!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)