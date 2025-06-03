"""
Dataset processing script for Yelp reviews
Downloads, chunks, embeds, and stores reviews in Supabase vector database
"""
import sys
import os
import requests
from typing import List, Dict, Any
import time

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from utils.text_chunker import TextChunker

def download_yelp_dataset(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Download Yelp review data from Hugging Face datasets API
    
    Args:
        limit: Maximum number of reviews to download
        
    Returns:
        List of review dictionaries
    """
    print(f"📥 Downloading Yelp dataset (limit: {limit})...")
    
    try:
        # Hugging Face datasets server API
        url = f"https://datasets-server.huggingface.co/rows"
        params = {
            'dataset': 'Yelp/yelp_review_full',
            'config': 'yelp_review_full', 
            'split': 'train',
            'offset': 0,
            'length': min(limit, 100)  # API limit per request
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'rows' not in data or not data['rows']:
            raise ValueError("No data found in API response")
        
        # Transform to our format
        reviews = []
        for i, row in enumerate(data['rows']):
            review_data = row['row']
            review = {
                'review_id': f"yelp_{i:04d}",
                'text': review_data.get('text', ''),
                'stars': review_data.get('label', 0) + 1,  # Convert 0-4 to 1-5
                'original_data': review_data
            }
            reviews.append(review)
        
        print(f"✅ Downloaded {len(reviews)} reviews successfully")
        return reviews[:limit]
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("🔄 Using fallback sample data...")
        return get_sample_data()

def get_sample_data() -> List[Dict[str, Any]]:
    """Fallback sample data for testing"""
    return [
        {
            'review_id': 'sample_001',
            'text': 'This restaurant was absolutely amazing! The food was delicious, the service was excellent, and the atmosphere was perfect. I especially loved the pasta dish - it was cooked to perfection with a rich, flavorful sauce. The staff was very attentive and friendly throughout our entire meal.',
            'stars': 5
        },
        {
            'review_id': 'sample_002', 
            'text': 'Had a terrible experience here. The food was cold and tasteless, the service was incredibly slow, and the place was dirty. My burger was undercooked and the fries were soggy. The waiter seemed annoyed when I asked questions about the menu. Would definitely not recommend this place to anyone.',
            'stars': 1
        },
        {
            'review_id': 'sample_003',
            'text': 'Pretty decent restaurant overall. The food was okay, nothing spectacular but satisfying enough. Service was average - took a while to get our orders but the staff was polite when they did come around. The prices are reasonable for the portion sizes you get.',
            'stars': 3
        },
        {
            'review_id': 'sample_004',
            'text': 'Great little place! Love the cozy atmosphere and friendly staff. The pizza was really good with fresh ingredients and crispy crust. Only downside was the wait time - took about 45 minutes to get our food. But overall a positive experience.',
            'stars': 4
        },
        {
            'review_id': 'sample_005',
            'text': 'Disappointing visit. The restaurant looked nice from outside but food quality was poor. My salad had wilted lettuce and the dressing was bland. The chicken was dry and overcooked. For the price we paid, expected much better quality.',
            'stars': 2
        }
    ]

def process_reviews_batch(chunks: List[Dict[str, Any]], 
                         embedding_service: EmbeddingService,
                         vector_store: VectorStore,
                         batch_size: int = 50) -> int:
    """
    Process a batch of review chunks - create embeddings and store
    
    Args:
        chunks: List of chunk dictionaries
        embedding_service: Embedding service instance
        vector_store: Vector store instance
        batch_size: Number of chunks to process at once
        
    Returns:
        Number of chunks successfully processed
    """
    total_processed = 0
    
    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        try:
            # Extract texts for embedding
            texts = [chunk['chunk_text'] for chunk in batch]
            
            # Create embeddings (this is much faster than the Node.js API version!)
            print("🤖 Creating embeddings...")
            embeddings = embedding_service.create_batch_embeddings(
                texts, 
                batch_size=32,  # Internal batch size for the model
                show_progress=True
            )
            
            # Prepare data for database insertion
            db_data = []
            for chunk, embedding in zip(batch, embeddings):
                db_record = {
                    'review_id': chunk['review_id'],
                    'chunk_text': chunk['chunk_text'],
                    'chunk_index': chunk['chunk_index'],
                    'embedding': embedding,
                    'stars': chunk['stars']
                }
                db_data.append(db_record)
            
            # Insert into database
            print("💾 Storing in database...")
            inserted_count = vector_store.insert_batch_chunks(db_data)
            
            if inserted_count == len(batch):
                total_processed += inserted_count
                print(f"✅ Batch {batch_num} completed successfully!")
            else:
                print(f"⚠️ Batch {batch_num} partially completed: {inserted_count}/{len(batch)}")
                total_processed += inserted_count
            
            # Small delay between batches to be nice to the database
            if i + batch_size < len(chunks):
                time.sleep(0.5)
                
        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            print("🔄 Continuing with next batch...")
            continue
    
    return total_processed

def main():
    """Main processing function"""
    print("🚀 Python Vector Search - Dataset Processing")
    print("=" * 60)
    
    try:
        # Initialize services
        print("🔧 Initializing services...")
        embedding_service = EmbeddingService()
        vector_store = VectorStore()
        chunker = TextChunker(chunk_size=300, overlap=50)
        
        # Show model info
        model_info = embedding_service.get_model_info()
        print(f"🤖 Using model: {model_info['model_name']}")
        print(f"📐 Embedding dimension: {model_info['embedding_dimension']}")
        
        # Check existing data
        existing_count = vector_store.get_chunk_count()
        print(f"📊 Existing chunks in database: {existing_count}")
        
        # Download new data
        new_reviews = download_yelp_dataset(limit=50)  # Start with 50 reviews
        
        if not new_reviews:
            print("❌ No reviews to process")
            return
        
        # Chunk the reviews
        print(f"\n📝 Chunking {len(new_reviews)} reviews...")
        chunks = chunker.chunk_reviews(new_reviews, show_progress=True)
        
        if not chunks:
            print("❌ No chunks created")
            return
        
        # Show chunking statistics
        stats = chunker.get_stats(chunks)
        print(f"\n📊 Chunking Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")
        
        # Process the chunks
        print(f"\n🔄 Processing {len(chunks)} chunks...")
        processed_count = process_reviews_batch(
            chunks, 
            embedding_service, 
            vector_store,
            batch_size=20  # Smaller batches for better error handling
        )
        
        # Final results
        print(f"\n" + "=" * 60)
        print("🎉 PROCESSING COMPLETE!")
        print(f"✅ Successfully processed: {processed_count}/{len(chunks)} chunks")
        
        # Updated database stats
        final_count = vector_store.get_chunk_count()
        print(f"📊 Total chunks in database: {final_count}")
        print(f"📈 New chunks added: {final_count - existing_count}")
        
        # Show database statistics
        db_stats = vector_store.get_database_stats()
        print(f"\n📋 Database Statistics:")
        for key, value in db_stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n🎯 Ready for semantic search!")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)