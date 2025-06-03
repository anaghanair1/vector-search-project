"""
Vector store service for Supabase operations
Handles all database interactions for review chunks and embeddings
"""
from typing import List, Dict, Any, Optional
from config.database import db_config

class VectorStore:
    def __init__(self):
        self.client = db_config.get_client()
        self.table_name = 'review_chunks'
    
    def insert_chunk(self, review_id: str, chunk_text: str, chunk_index: int, 
                    embedding: List[float], stars: int) -> bool:
        """
        Insert a single review chunk with embedding
        
        Args:
            review_id: Unique identifier for the review
            chunk_text: The text content of the chunk
            chunk_index: Position of chunk within the original review
            embedding: Vector embedding of the chunk
            stars: Star rating of the original review
            
        Returns:
            bool: True if successful
        """
        try:
            data = {
                'review_id': review_id,
                'chunk_text': chunk_text,
                'chunk_index': chunk_index,
                'embedding': embedding,
                'stars': stars
            }
            
            result = self.client.table(self.table_name).insert(data).execute()
            return True
            
        except Exception as e:
            print(f"Error inserting chunk: {e}")
            return False
    
    def insert_batch_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Insert multiple chunks at once (much faster)
        
        Args:
            chunks: List of chunk dictionaries with keys:
                   review_id, chunk_text, chunk_index, embedding, stars
                   
        Returns:
            int: Number of chunks successfully inserted
        """
        if not chunks:
            return 0
        
        try:
            # Supabase can handle batch inserts efficiently
            result = self.client.table(self.table_name).insert(chunks).execute()
            
            inserted_count = len(result.data) if result.data else len(chunks)
            print(f"✅ Successfully inserted {inserted_count} chunks")
            return inserted_count
            
        except Exception as e:
            print(f"❌ Error inserting batch chunks: {e}")
            return 0
    
    def search_similar(self, query_embedding: List[float], 
                      match_threshold: float = 0.7, 
                      match_count: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Vector to search for
            match_threshold: Minimum similarity score (0-1)
            match_count: Maximum number of results
            
        Returns:
            List of matching chunks with similarity scores
        """
        try:
            # Use the PostgreSQL function we created earlier
            result = self.client.rpc('search_similar_reviews', {
                'query_embedding': query_embedding,
                'match_threshold': match_threshold,
                'match_count': match_count
            }).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"❌ Error searching similar chunks: {e}")
            return []
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks in database"""
        try:
            result = self.client.table(self.table_name).select('id', count='exact').execute()
            return result.count if result.count is not None else 0
            
        except Exception as e:
            print(f"❌ Error getting chunk count: {e}")
            return 0
    
    def get_chunks_by_review_id(self, review_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific review"""
        try:
            result = self.client.table(self.table_name)\
                .select('*')\
                .eq('review_id', review_id)\
                .order('chunk_index')\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"❌ Error getting chunks for review {review_id}: {e}")
            return []
    
    def get_sample_chunks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample chunks for testing"""
        try:
            result = self.client.table(self.table_name)\
                .select('*')\
                .limit(limit)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"❌ Error getting sample chunks: {e}")
            return []
    
    def delete_all_chunks(self) -> bool:
        """Delete all chunks (use with caution!)"""
        try:
            result = self.client.table(self.table_name)\
                .delete()\
                .neq('id', 0)\
                .execute()  # Delete all records
            
            print("🗑️ All chunks deleted successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting chunks: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            # Get total count
            total_count = self.get_chunk_count()
            
            # Get unique review count
            result = self.client.table(self.table_name)\
                .select('review_id')\
                .execute()
            
            unique_reviews = len(set(row['review_id'] for row in result.data)) if result.data else 0
            
            # Get star distribution
            star_result = self.client.table(self.table_name)\
                .select('stars')\
                .execute()
            
            stars_data = [row['stars'] for row in star_result.data] if star_result.data else []
            star_distribution = {}
            for star in stars_data:
                star_distribution[star] = star_distribution.get(star, 0) + 1
            
            return {
                'total_chunks': total_count,
                'unique_reviews': unique_reviews,
                'avg_chunks_per_review': round(total_count / unique_reviews, 2) if unique_reviews > 0 else 0,
                'star_distribution': star_distribution
            }
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Test the vector store
    store = VectorStore()
    
    # Get current stats
    print("📊 Database Statistics:")
    stats = store.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get sample chunks
    print("\n🔍 Sample chunks:")
    samples = store.get_sample_chunks(3)
    for i, chunk in enumerate(samples, 1):
        print(f"  {i}. Review: {chunk['review_id'][:20]}...")
        print(f"     Text: {chunk['chunk_text'][:100]}...")
        print(f"     Stars: {chunk['stars']}")
        print()