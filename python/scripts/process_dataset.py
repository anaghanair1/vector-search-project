"""
Dataset processing script
Downloads yelp reviews, chunks them, creates embeddings, stores in database
"""
import sys
import os
import requests
import time

# add parent dir to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from utils.text_chunker import TextChunker

def download_yelp_data(limit=100):
    """Download yelp reviews from huggingface datasets"""
    print(f"📥 Downloading yelp data (limit: {limit})...")
    
    try:
        # huggingface datasets api
        url = "https://datasets-server.huggingface.co/rows"
        params = {
            'dataset': 'Yelp/yelp_review_full',
            'config': 'yelp_review_full', 
            'split': 'train',
            'offset': 0,
            'length': min(limit, 100)  # api has limits
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'rows' not in data:
            raise ValueError("No data in response")
        
        # convert to our format
        reviews = []
        for i, row in enumerate(data['rows']):
            review_data = row['row']
            review = {
                'review_id': f"yelp_{i:04d}",
                'text': review_data.get('text', ''),
                'stars': review_data.get('label', 0) + 1,  # convert 0-4 to 1-5 stars
                'raw_data': review_data
            }
            reviews.append(review)
        
        print(f"Got {len(reviews)} reviews")
        return reviews[:limit]
        
    except Exception as e:
        print(f"Download failed: {e}")
        print("Using sample data instead...")
        return get_sample_reviews()

def get_sample_reviews():
    """Fallback sample data when download fails"""
    return [
        {
            'review_id': 'sample_001',
            'text': 'This place is absolutely incredible! The food was amazing, service was top notch, and the atmosphere was perfect. I had the pasta special and it was cooked perfectly with such a rich and flavorful sauce. Our server was attentive and friendly throughout the entire meal. Will definitely be coming back soon!',
            'stars': 5
        },
        {
            'review_id': 'sample_002', 
            'text': 'What a disappointment. The food was cold when it arrived, completely tasteless, and the service was incredibly slow. My burger was undercooked and the fries were soggy and gross. The waiter seemed annoyed every time we asked a question. Definitely will not be returning to this place.',
            'stars': 1
        },
        {
            'review_id': 'sample_003',
            'text': 'Pretty average restaurant overall. The food was decent, nothing to write home about but it was satisfying. Service was okay - took a while to get our orders but the staff was polite when they came around. Prices are reasonable for what you get.',
            'stars': 3
        },
        {
            'review_id': 'sample_004',
            'text': 'Really nice little spot! Love the cozy vibe and the staff is super friendly. The pizza was really good with fresh toppings and a nice crispy crust. Only complaint is the wait time - took about 45 minutes to get our food. But overall a great experience.',
            'stars': 4
        },
        {
            'review_id': 'sample_005',
            'text': 'Went here for dinner last week and was pretty disappointed. The place looked nice from the outside but the food quality was just poor. My salad had wilted lettuce and the dressing was really bland. The chicken was dry and overcooked. For what we paid, expected much better.',
            'stars': 2
        }
    ]

def process_batch(chunks, embedding_service, vector_store, batch_size=50):
    """Process a batch of chunks - create embeddings and store them"""
    total_processed = 0
    
    # process in smaller batches to avoid memory issues
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        try:
            # get texts for embedding
            texts = [chunk['chunk_text'] for chunk in batch]
            
            # create embeddings - this is the slow part
            print("Creating embeddings...")
            embeddings = embedding_service.create_batch_embeddings(
                texts, 
                batch_size=32,
                show_progress=True
            )
            
            # prepare data for database
            db_records = []
            for chunk, embedding in zip(batch, embeddings):
                record = {
                    'review_id': chunk['review_id'],
                    'chunk_text': chunk['chunk_text'],
                    'chunk_index': chunk['chunk_index'],
                    'embedding': embedding,
                    'stars': chunk['stars']
                }
                db_records.append(record)
            
            # insert into database
            print("Saving to database...")
            inserted = vector_store.insert_batch_chunks(db_records)
            
            if inserted == len(batch):
                total_processed += inserted
                print(f"Batch {batch_num} done!")
            else:
                print(f"Batch {batch_num} partially done: {inserted}/{len(batch)}")
                total_processed += inserted
            
            # small delay to be nice to the database
            if i + batch_size < len(chunks):
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Batch {batch_num} failed: {e}")
            continue  # try next batch
    
    return total_processed

def main():
    """Main processing function"""
    print("Vector Search Dataset Processing")
    print("=" * 50)
    
    try:
        # initialize our services
        print("Setting up services...")
        embedding_service = EmbeddingService()
        vector_store = VectorStore()
        chunker = TextChunker(chunk_size=300, overlap=50)  # smaller chunks work better
        
        # show what model we're using
        model_info = embedding_service.get_model_info()
        print(f"Model: {model_info['model_name']}")
        print(f"Dimensions: {model_info['embedding_dimension']}")
        
        # check current database state
        existing = vector_store.get_chunk_count()
        print(f"Current chunks in database: {existing}")
        
        # download new reviews
        reviews = download_yelp_data(limit=50)  # start small
        
        if not reviews:
            print("No reviews to process")
            return
        
        # chunk the reviews
        print(f"\nChunking {len(reviews)} reviews...")
        chunks = chunker.chunk_reviews(reviews, show_progress=True)
        
        if not chunks:
            print("No chunks created")
            return
        
        # show chunking stats
        stats = chunker.get_stats(chunks)
        print(f"\nChunking stats:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")
        
        # process all the chunks
        print(f"\nProcessing {len(chunks)} chunks...")
        processed = process_batch(
            chunks, 
            embedding_service, 
            vector_store,
            batch_size=20  # smaller batches = more stable
        )
        
        # show final results
        print(f"\n" + "=" * 50)
        print("Processing finished!")
        print(f"Processed: {processed}/{len(chunks)} chunks")
        
        # updated stats
        final_count = vector_store.get_chunk_count()
        print(f"Total chunks now: {final_count}")
        print(f"Added: {final_count - existing} new chunks")
        
        # database stats
        db_stats = vector_store.get_database_stats()
        print(f"\nDatabase stats:")
        for key, value in db_stats.items():
            print(f"  {key}: {value}")
        
        print(f"\nReady for searching!")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)