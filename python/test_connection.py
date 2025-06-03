"""
Test script to verify database connection and existing data
"""
from config.database import db_config
from services.vector_store import VectorStore
from services.embedding_service import EmbeddingService

def test_database_connection():
    """Test basic database connectivity"""
    print("🔍 Testing Database Connection...")
    print("=" * 50)
    
    try:
        # Test connection
        success = db_config.test_connection()
        
        if success:
            print("✅ Database connection working!")
        else:
            print("❌ Database connection failed!")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def test_existing_data():
    """Test access to existing review chunks"""
    print("\n📊 Testing Existing Data...")
    print("=" * 50)
    
    try:
        store = VectorStore()
        
        # Get database statistics
        stats = store.get_database_stats()
        
        print("Database Statistics:")
        for key, value in stats.items():
            print(f"  📈 {key}: {value}")
        
        if stats.get('total_chunks', 0) > 0:
            print(f"\n✅ Found {stats['total_chunks']} existing chunks!")
            
            # Show sample data
            print("\n🔍 Sample chunks:")
            samples = store.get_sample_chunks(3)
            
            for i, chunk in enumerate(samples, 1):
                print(f"\n  Sample {i}:")
                print(f"    🆔 Review ID: {chunk['review_id']}")
                print(f"    ⭐ Stars: {chunk['stars']}")
                print(f"    📝 Text: {chunk['chunk_text'][:150]}...")
                print(f"    🔢 Embedding dimension: {len(chunk['embedding'])}")
                
            return True
        else:
            print("⚠️ No existing data found in database")
            return False
            
    except Exception as e:
        print(f"❌ Data test failed: {e}")
        return False

def test_embedding_service():
    """Test the local embedding service"""
    print("\n🤖 Testing Embedding Service...")
    print("=" * 50)
    
    try:
        # Initialize service
        service = EmbeddingService()
        
        # Show model info
        info = service.get_model_info()
        print("Model Information:")
        for key, value in info.items():
            print(f"  🔧 {key}: {value}")
        
        # Test single embedding
        test_text = "This is a test review for the restaurant."
        embedding = service.create_embedding(test_text)
        
        print(f"\n✅ Created test embedding:")
        print(f"  📐 Dimension: {len(embedding)}")
        print(f"  🔢 First 5 values: {embedding[:5]}")
        print(f"  📊 Data type: {type(embedding[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding service test failed: {e}")
        return False

def test_search_functionality():
    """Test search with existing data"""
    print("\n🔍 Testing Search Functionality...")
    print("=" * 50)
    
    try:
        store = VectorStore()
        service = EmbeddingService()
        
        # Check if we have data to search
        count = store.get_chunk_count()
        if count == 0:
            print("⚠️ No data to search - skipping search test")
            return True
        
        # Create a test query
        query = "great food and excellent service"
        print(f"🔍 Searching for: '{query}'")
        
        # Create query embedding
        query_embedding = service.create_embedding(query)
        
        # Search for similar chunks
        results = store.search_similar(query_embedding, match_threshold=0.5, match_count=3)
        
        if results:
            print(f"✅ Found {len(results)} similar chunks:")
            
            for i, result in enumerate(results, 1):
                similarity = result.get('similarity', 0)
                text = result.get('chunk_text', '')
                stars = result.get('stars', 0)
                
                print(f"\n  Result {i}:")
                print(f"    🎯 Similarity: {similarity:.3f}")
                print(f"    ⭐ Stars: {stars}")
                print(f"    📝 Text: {text[:100]}...")
        else:
            print("⚠️ No similar chunks found (try lowering match_threshold)")
        
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Python Vector Search Project - Connection Test")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Database Connection", test_database_connection),
        ("Existing Data", test_existing_data),
        ("Embedding Service", test_embedding_service),
        ("Search Functionality", test_search_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! Your Python setup is ready!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()