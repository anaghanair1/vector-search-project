"""
Quick test script to check if everything works.
Tests database connection, existing data, embedding service, etc.

Database Health Checks: https://docs.docker.com/engine/reference/builder/#healthcheck
Connection Testing: Standard database testing practices
System Validation: https://pytest.org/

"""
from config.database import db_config
from services.vector_store import VectorStore
from services.embedding_service import EmbeddingService

def test_db_connection():
    """Check if we can connect to database"""
    print("Testing database connection...")
    print("=" * 40)
    
    try:
        success = db_config.test_connection()
        
        if success:
            print("Database works!")
        else:
            print("Database connection failed!")
            return False
            
        return True
        
    except Exception as e:
        print(f"Connection test error: {e}")
        return False

def test_existing_data():
    """Check what data we have in database"""
    print("\nChecking existing data...")
    print("=" * 40)
    
    try:
        store = VectorStore()
        
        # get stats
        stats = store.get_database_stats()
        
        print("Database stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        total_chunks = stats.get('total_chunks', 0)
        if total_chunks > 0:
            print(f"\nFound {total_chunks} chunks in database!")
            
            # show some samples
            print("\nSample data:")
            samples = store.get_sample_chunks(3)
            
            for i, chunk in enumerate(samples, 1):
                print(f"\n  Sample {i}:")
                print(f"    ID: {chunk['review_id']}")
                print(f"    Stars: {chunk['stars']}")
                text = chunk['chunk_text']
                if len(text) > 120:
                    text = text[:120] + "..."
                print(f"    Text: {text}")
                print(f"    Embedding size: {len(chunk['embedding'])}")
                
            return True
        else:
            print("No data found in database")
            return False
            
    except Exception as e:
        print(f"Data test failed: {e}")
        return False

def test_embeddings():
    """Test embedding service"""
    print("\nTesting embedding service...")
    print("=" * 40)
    
    try:
        service = EmbeddingService()
        
        # show model info
        info = service.get_model_info()
        print("Model info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # test creating an embedding
        test_text = "This restaurant has excellent food and great service."
        embedding = service.create_embedding(test_text)
        
        print(f"\nTest embedding created:")
        print(f"  Size: {len(embedding)}")
        print(f"  First few values: {embedding[:5]}")
        print(f"  Type: {type(embedding[0])}")
        
        return True
        
    except Exception as e:
        print(f"Embedding test failed: {e}")
        return False

def test_search():
    """Test actual search functionality"""
    print("\nTesting search...")
    print("=" * 40)
    
    try:
        store = VectorStore()
        service = EmbeddingService()
        
        # check if we have data to search
        count = store.get_chunk_count()
        if count == 0:
            print("No data to search - skipping")
            return True
        
        # try a search
        query = "excellent food great service"
        print(f"Testing query: '{query}'")
        
        # create embedding
        query_emb = service.create_embedding(query)
        
        # search
        results = store.search_similar(query_emb, match_threshold=0.5, match_count=3)
        
        if results:
            print(f"Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                similarity = result.get('similarity', 0)
                text = result.get('chunk_text', '')
                stars = result.get('stars', 0)
                
                print(f"\n  Result {i}:")
                print(f"    Similarity: {similarity:.3f}")
                print(f"    Stars: {stars}")
                if len(text) > 80:
                    text = text[:80] + "..."
                print(f"    Text: {text}")
        else:
            print("No results found (maybe try lower threshold)")
        
        return True
        
    except Exception as e:
        print(f"Search test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Vector Search Connection Test")
    print("=" * 50)
    
    # run the tests
    tests = [
        ("Database Connection", test_db_connection),
        ("Existing Data", test_existing_data),
        ("Embedding Service", test_embeddings),
        ("Search Function", test_search),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            results.append((test_name, False))
    
    # summary
    print("\n" + "=" * 50)
    print("Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("Everything works! Ready to go!")
    else:
        print("Some issues found - check errors above")

if __name__ == "__main__":
    main()