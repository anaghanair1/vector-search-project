"""
Simple Semantic Search - Efficient Version
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import db_config
from services.embedding_service import EmbeddingService

class SemanticSearcher:
    def __init__(self):
        """Initialize searcher with persistent embedding service"""
        self.client = db_config.get_client()
        self.embedding_service = EmbeddingService()  # Load once
        print(f"Embedding service ready - {self.embedding_service.embedding_dimension} dimensions")
    
    def search_reviews(self, query: str, limit: int = 5):
        """Search for similar reviews"""
        print(f"\nSearching for: '{query}'")
        print("=" * 50)
        
        try:
            # Create query embedding (no reloading)
            query_embedding = self.embedding_service.create_embedding(query)
            
            # Search using Supabase function
            result = self.client.rpc('search_similar_reviews', {
                'query_embedding': query_embedding,
                'match_threshold': 0.2,  # Lower threshold
                'match_count': limit
            }).execute()
            
            if not result.data:
                print("No results found")
                return
            
            # Display results
            print(f"Found {len(result.data)} results:")
            print("=" * 60)
            
            for i, match in enumerate(result.data, 1):
                similarity_score = match['similarity'] * 100
                
                print(f"\nResult {i}:")
                print(f"   Similarity Score: {similarity_score:.1f}%")
                print(f"   Star Rating: {match['stars']}/5")
                print(f"   Review ID: {match['review_id']}")
                print(f"   Text: {match['chunk_text'][:200]}...")  # Limit text display
                print("-" * 60)
            
        except Exception as e:
            print(f"Search failed: {e}")

def main():
    """Main function"""
    print("Semantic Search Test")
    print("=" * 30)
    
    # Initialize searcher once
    searcher = SemanticSearcher()
    
    # Test queries
    test_queries = [
        "delicious food amazing taste",
        "horrible service rude staff", 
        "good value reasonable prices",
        "romantic atmosphere perfect date"
    ]
    
    for query in test_queries:
        searcher.search_reviews(query, limit=3)
    
    print("\nSearch complete!")

if __name__ == "__main__":
    main()