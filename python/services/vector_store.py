"""
Vector store for handling database operations
Handles the review chunks and their embeddings

Repository Pattern: https://martinfowler.com/eaaCatalog/repository.html
Data Access Object Pattern: https://en.wikipedia.org/wiki/Data_access_object
Supabase Python CRUD: https://supabase.com/docs/reference/python/ins
Batch Insert Optimization: https://supabase.com/docs/reference/python/upsert
Error Handling: https://supabase.com/docs/reference/python/handling-errors
Connection Management: https://github.com/supabase/supabase-py/blob/main/examples/

"""
from typing import List, Dict, Any
from config.database import db_config

class VectorStore:
    def __init__(self):
        self.client = db_config.get_client()
        self.table = 'review_chunks'  # main table for storing chunks
    
    def insert_chunk(self, review_id, chunk_text, chunk_index, embedding, stars):
        """Insert a single chunk into database"""
        try:
            data = {
                'review_id': review_id,
                'chunk_text': chunk_text,
                'chunk_index': chunk_index,
                'embedding': embedding,
                'stars': stars
            }
            
            result = self.client.table(self.table).insert(data).execute()
            return True
            
        except Exception as e:
            print(f"Insert failed: {e}")
            return False
    
    def insert_batch_chunks(self, chunks_data):
        """Insert multiple chunks at once - way faster than individual inserts"""
        if not chunks_data:
            return 0
        
        try:
            result = self.client.table(self.table).insert(chunks_data).execute()
            
            # count successful inserts
            if result.data:
                count = len(result.data)
            else:
                count = len(chunks_data)  # assume all succeeded if no error
            
            print(f"Inserted {count} chunks")
            return count        
        except Exception as e:
            print(f"Batch insert error: {e}")
            return 0
    
    def search_similar(self, query_embedding, match_threshold=0.7, match_count=10):
        """Search for similar chunks using vector similarity"""
        try:
            # use the database function we created
            result = self.client.rpc('search_similar_reviews', {
                'query_embedding': query_embedding,
                'match_threshold': match_threshold,
                'match_count': match_count
            }).execute()
            
            if result.data:
                return result.data
            else:
                return []
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_chunk_count(self):
        """Get total number of chunks"""
        try:
            result = self.client.table(self.table).select('id', count='exact').execute()
            if result.count is not None:
                return result.count
            else:
                return 0
            
        except Exception as e:
            print(f"Error counting chunks: {e}")
            return 0
    
    def get_chunks_by_review(self, review_id):
        """Get all chunks for specific review"""
        try:
            result = self.client.table(self.table)\
                .select('*')\
                .eq('review_id', review_id)\
                .order('chunk_index')\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error getting review chunks: {e}")
            return []
    
    def get_sample_chunks(self, limit=5):
        """Get some sample chunks for testing"""
        try:
            result = self.client.table(self.table)\
                .select('*')\
                .limit(limit)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error getting samples: {e}")
            return []
    
    def delete_all_chunks(self):
        """Delete everything - be careful with this!"""
        try:
            # delete all records
            result = self.client.table(self.table)\
                .delete()\
                .neq('id', 0)\
                .execute()
            
            print("All chunks deleted")
            return True
            
        except Exception as e:
            print(f"Delete failed: {e}")
            return False
    
    def get_database_stats(self):
        """Get various stats about the database"""
        try:
            total_chunks = self.get_chunk_count()
            
            # get unique reviews
            result = self.client.table(self.table)\
                .select('review_id')\
                .execute()
            
            if result.data:
                unique_reviews = len(set(row['review_id'] for row in result.data))
            else:
                unique_reviews = 0
            
            # star rating distribution
            star_result = self.client.table(self.table)\
                .select('stars')\
                .execute()
            
            star_counts = {}
            if star_result.data:
                for row in star_result.data:
                    stars = row['stars']
                    star_counts[stars] = star_counts.get(stars, 0) + 1
            
            stats = {
                'total_chunks': total_chunks,
                'unique_reviews': unique_reviews,
                'star_distribution': star_counts
            }
            
            # calculate average chunks per review
            if unique_reviews > 0:
                stats['avg_chunks_per_review'] = round(total_chunks / unique_reviews, 2)
            else:
                stats['avg_chunks_per_review'] = 0
            
            return stats
            
        except Exception as e:
            print(f"Stats error: {e}")
            return {}

# test the store if run directly
if __name__ == "__main__":
    store = VectorStore()
    
    print("Current database stats:")
    stats = store.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample chunks:")
    samples = store.get_sample_chunks(3)
    for i, chunk in enumerate(samples, 1):
        print(f"  {i}. Review: {chunk.get('review_id', 'unknown')}")
        text = chunk.get('chunk_text', '')
        print(f"     Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"     Stars: {chunk.get('stars', 'unknown')}")
        print()