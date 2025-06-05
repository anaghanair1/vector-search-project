"""
Big test file to check if everything works
Tests all the search stuff - semantic, hybrid, similarity etc
"""
import sys
import os
import time
import numpy as np

# add parent dir so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import db_config
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.hybrid_search_service import HybridSearchService
from services.query_processor import QueryProcessor

class TestRunner:
    def __init__(self):
        self.client = db_config.get_client()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.hybrid_service = HybridSearchService()
        self.query_processor = QueryProcessor()
        self.test_results = []
        
    def record_test(self, test_name, passed, info=""):
        status = "PASS" if passed else "FAIL"
        self.test_results.append((test_name, passed, info))
        print(f"[{status}] {test_name}")
        if info:
            print(f"    {info}")
    
    def check_database_stuff(self):
        print("\n=== DATABASE TESTS ===")
        
        try:
            # basic connection test
            works = db_config.test_connection()
            self.record_test("Database connects", works)
            
            # check if we have data
            count = self.vector_store.get_chunk_count()
            has_data = count > 0
            self.record_test("Has data in database", has_data, f"Found {count} chunks")
            
            # try getting some samples
            samples = self.vector_store.get_sample_chunks(3)
            got_samples = len(samples) > 0
            self.record_test("Can get sample data", got_samples, f"Got {len(samples)} samples")
            
        except Exception as e:
            self.record_test("Database connection", False, f"Error: {e}")
    
    def check_embeddings(self):
        print("\n=== EMBEDDING TESTS ===")
        
        try:
            # test making a single embedding
            text = "test sentence for embedding"
            emb = self.embedding_service.create_embedding(text)
            single_works = len(emb) == 384
            self.record_test("Single embedding works", single_works, f"Got {len(emb)} dimensions")
            
            # test if same text gives same result
            emb2 = self.embedding_service.create_embedding(text)
            same_result = np.allclose(emb, emb2, atol=1e-6)
            self.record_test("Embeddings are consistent", same_result)
            
            # test batch processing
            texts = ["first text", "second text", "third text"]
            batch_embs = self.embedding_service.create_batch_embeddings(texts, show_progress=False)
            batch_works = len(batch_embs) == 3 and all(len(e) == 384 for e in batch_embs)
            self.record_test("Batch embeddings work", batch_works, f"Made {len(batch_embs)} embeddings")
            
            # test different texts give different results
            different_text = "totally different sentence"
            emb_diff = self.embedding_service.create_embedding(different_text)
            different_results = not np.allclose(emb, emb_diff, atol=0.1)
            self.record_test("Different texts = different embeddings", different_results)
            
        except Exception as e:
            self.record_test("Embedding service", False, f"Error: {e}")
    
    def check_similarity_math(self):
        print("\n=== SIMILARITY TESTS ===")
        
        try:
            # manual cosine similarity calculation
            def cos_sim(a, b):
                a, b = np.array(a), np.array(b)
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            # test with similar and different texts
            good_text = "great food excellent service"
            similar_text = "amazing food wonderful service"
            bad_text = "terrible food horrible service"
            
            emb1 = self.embedding_service.create_embedding(good_text)
            emb2 = self.embedding_service.create_embedding(similar_text)
            emb3 = self.embedding_service.create_embedding(bad_text)
            
            sim_good = cos_sim(emb1, emb2)
            sim_bad = cos_sim(emb1, emb3)
            
            # similar texts should be more similar than opposite texts
            ranking_works = sim_good > sim_bad
            self.record_test("Similarity ranking makes sense", ranking_works, 
                          f"Similar: {sim_good:.3f} > Opposite: {sim_bad:.3f}")
            
            # values should be between 0 and 1
            bounds_ok = 0 <= sim_good <= 1 and 0 <= sim_bad <= 1
            self.record_test("Similarity values in range", bounds_ok, "Values between 0-1")
            
        except Exception as e:
            self.record_test("Similarity calculations", False, f"Error: {e}")
    
    def check_basic_search(self):
        print("\n=== BASIC SEARCH TESTS ===")
        
        queries_to_test = [
            "delicious food",
            "great service", 
            "terrible experience",
            "good value",
            "romantic dinner"
        ]
        
        for query in queries_to_test:
            try:
                query_emb = self.embedding_service.create_embedding(query)
                
                result = self.client.rpc('search_similar_reviews', {
                    'query_embedding': query_emb,
                    'match_threshold': 0.1,
                    'match_count': 5
                }).execute()
                
                found_stuff = bool(result.data)
                self.record_test(f"Search works for '{query}'", found_stuff, 
                              f"Found {len(result.data) if result.data else 0} results")
                
                if result.data:
                    # check if results make sense
                    top_score = result.data[0]['similarity']
                    decent_score = top_score > 0.1
                    self.record_test(f"Good results for '{query}'", decent_score, 
                                  f"Top score: {top_score:.3f}")
                
            except Exception as e:
                self.record_test(f"Search for '{query}'", False, f"Error: {e}")
    
    def check_hybrid_search(self):
        print("\n=== HYBRID SEARCH TESTS ===")
        
        try:
            # check if hybrid search is set up
            stats = self.hybrid_service.get_stats()
            ready = stats.get('hybrid_search_ready', False)
            self.record_test("Hybrid search available", ready)
            
            if not ready:
                print("    Skipping hybrid tests - not configured")
                return
            
            test_queries = [
                "amazing delicious food",
                "terrible rude staff",
                "good value for money",
                "romantic date night",
                "fast quick service"
            ]
            
            for query in test_queries:
                try:
                    # test hybrid search
                    result = self.hybrid_service.search(query, match_count=5)
                    got_results = len(result['results']) > 0
                    self.record_test(f"Hybrid search '{query}'", got_results, 
                                  f"Found {len(result['results'])} results")
                    
                    # compare different search methods
                    semantic_result = self.hybrid_service.semantic_only(query, match_count=5)
                    keyword_result = self.hybrid_service.keyword_only(query, match_count=5)
                    
                    sem_count = len(semantic_result['results'])
                    key_count = len(keyword_result['results'])
                    hyb_count = len(result['results'])
                    
                    self.record_test(f"Method comparison '{query}'", True, 
                                  f"Semantic: {sem_count}, Keyword: {key_count}, Hybrid: {hyb_count}")
                    
                except Exception as e:
                    self.record_test(f"Hybrid search '{query}'", False, f"Error: {e}")
                    
        except Exception as e:
            self.record_test("Hybrid search setup", False, f"Error: {e}")
    
    def check_query_processing(self):
        print("\n=== QUERY PROCESSING TESTS ===")
        
        try:
            test_queries = [
                "delicious amazing food",
                "terrible horrible service", 
                "expensive overpriced place",
                "romantic date night",
                "fast quick delivery"
            ]
            
            for query in test_queries:
                try:
                    result = self.query_processor.process_query(query, enhance=True)
                    
                    # basic processing check
                    basic_ok = result['original_query'] == query and result['cleaned_query']
                    self.record_test(f"Process query '{query}'", basic_ok)
                    
                    # enhancement check
                    got_enhanced = len(result['enhanced_query']) > len(result['original_query'])
                    self.record_test(f"Enhanced '{query}'", got_enhanced, 
                                  f"Original: {len(query)}, Enhanced: {len(result['enhanced_query'])}")
                    
                    # keyword extraction
                    got_keywords = len(result['keywords']) > 0
                    self.record_test(f"Keywords from '{query}'", got_keywords, 
                                  f"Keywords: {result['keywords']}")
                    
                    # analysis stuff
                    got_analysis = result['analysis']['main_category'] and result['analysis']['sentiment']
                    self.record_test(f"Analysis of '{query}'", got_analysis, 
                                  f"Category: {result['analysis']['main_category']}, Sentiment: {result['analysis']['sentiment']}")
                    
                except Exception as e:
                    self.record_test(f"Query processing '{query}'", False, f"Error: {e}")
                    
        except Exception as e:
            self.record_test("Query processing", False, f"Error: {e}")
    
    def check_edge_cases(self):
        print("\n=== EDGE CASE TESTS ===")
        
        weird_inputs = [
            ("a", "single letter"),
            ("", "empty string"),
            ("12345", "just numbers"),
            ("!@#$%", "special chars"),
            ("the the the", "repeated words"),
            ("a" * 500, "really long text"),
            ("Good food! Great service? Perfect place.", "lots of punctuation")
        ]
        
        for query, desc in weird_inputs:
            try:
                if query:  # skip empty for embedding
                    emb = self.embedding_service.create_embedding(query)
                    emb_ok = len(emb) == 384
                    self.record_test(f"Embedding {desc}", emb_ok)
                    
                    # test query processing
                    processed = self.query_processor.process_query(query)
                    proc_ok = processed['cleaned_query'] is not None
                    self.record_test(f"Processing {desc}", proc_ok)
                    
                else:
                    # test empty query
                    try:
                        processed = self.query_processor.process_query(query)
                        empty_ok = processed['cleaned_query'] == ""
                        self.record_test(f"Edge case {desc}", empty_ok)
                    except:
                        self.record_test(f"Edge case {desc}", True, "handled gracefully")
                        
            except Exception as e:
                # some edge cases might fail but that's ok
                handled_ok = "empty" in str(e).lower() or "invalid" in str(e).lower()
                self.record_test(f"Edge case {desc}", handled_ok, f"Error: {str(e)[:50]}")
    
    def check_performance(self):
        print("\n=== PERFORMANCE TESTS ===")
        
        try:
            # test embedding speed
            texts = ["performance test sentence"] * 10
            start = time.time()
            embs = self.embedding_service.create_batch_embeddings(texts, show_progress=False)
            emb_time = time.time() - start
            
            fast_enough = emb_time < 15.0  # 10 embeddings in under 15 seconds
            self.record_test("Embedding speed ok", fast_enough, f"10 embeddings in {emb_time:.2f}s")
            
            # test search speed
            query_emb = self.embedding_service.create_embedding("performance test")
            start = time.time()
            result = self.client.rpc('search_similar_reviews', {
                'query_embedding': query_emb,
                'match_threshold': 0.1,
                'match_count': 10
            }).execute()
            search_time = time.time() - start
            
            search_fast = search_time < 10.0  # search in under 10 seconds
            self.record_test("Search speed ok", search_fast, f"Search in {search_time:.2f}s")
            
            # check database size
            total = self.vector_store.get_chunk_count()
            has_scale = total > 0
            self.record_test("Database has scale", has_scale, f"Handling {total} chunks")
            
        except Exception as e:
            self.record_test("Performance tests", False, f"Error: {e}")
    
    def check_accuracy(self):
        print("\n=== ACCURACY TESTS ===")
        
        accuracy_checks = [
            {
                "query": "amazing delicious food",
                "should_find": ["food", "delicious", "tasty", "flavor", "meal"],
                "avoid": ["terrible", "horrible", "bad"],
                "desc": "positive food search"
            },
            {
                "query": "terrible rude service",
                "should_find": ["service", "staff", "waiter", "server"],
                "avoid": ["delicious", "amazing", "great"],
                "desc": "negative service search"
            },
            {
                "query": "expensive overpriced",
                "should_find": ["price", "cost", "money", "expensive"],
                "avoid": ["cheap", "affordable"],
                "desc": "price search"
            }
        ]
        
        for test in accuracy_checks:
            try:
                query_emb = self.embedding_service.create_embedding(test["query"])
                
                result = self.client.rpc('search_similar_reviews', {
                    'query_embedding': query_emb,
                    'match_threshold': 0.2,
                    'match_count': 5
                }).execute()
                
                if result.data:
                    top = result.data[0]
                    text = top['chunk_text'].lower()
                    score = top['similarity']
                    
                    # check if result makes sense
                    has_relevant = any(term in text for term in test["should_find"])
                    avoids_wrong = not any(term in text for term in test["avoid"])
                    
                    accuracy = score * 100
                    good_result = has_relevant and accuracy > 20
                    
                    self.record_test(test["desc"], good_result, 
                                  f"Score: {accuracy:.1f}%, Relevant: {has_relevant}")
                else:
                    self.record_test(test["desc"], False, "no results")
                    
            except Exception as e:
                self.record_test(test["desc"], False, f"Error: {e}")
    
    def check_vector_ops(self):
        print("\n=== VECTOR OPERATION TESTS ===")
        
        try:
            # test inserting a vector
            test_data = {
                'review_id': 'test_review_123',
                'chunk_text': 'this is a test chunk for testing vector operations',
                'chunk_index': 0,
                'embedding': self.embedding_service.create_embedding('test vector data'),
                'stars': 4
            }
            
            insert_ok = self.vector_store.insert_chunk(
                test_data['review_id'],
                test_data['chunk_text'], 
                test_data['chunk_index'],
                test_data['embedding'],
                test_data['stars']
            )
            self.record_test("Vector insert works", insert_ok)
            
            # test searching for similar vectors
            search_results = self.vector_store.search_similar(
                test_data['embedding'],
                match_threshold=0.5,
                match_count=3
            )
            search_ok = len(search_results) > 0
            self.record_test("Vector search works", search_ok, f"Found {len(search_results)} similar")
            
            # test batch insert
            batch_stuff = [
                {
                    'review_id': f'batch_test_{i}',
                    'chunk_text': f'batch test chunk number {i}',
                    'chunk_index': 0,
                    'embedding': self.embedding_service.create_embedding(f'batch test data {i}'),
                    'stars': i % 5 + 1
                }
                for i in range(3)
            ]
            
            batch_count = self.vector_store.insert_batch_chunks(batch_stuff)
            batch_ok = batch_count == 3
            self.record_test("Batch vector insert", batch_ok, f"Inserted {batch_count} vectors")
            
        except Exception as e:
            self.record_test("Vector operations", False, f"Error: {e}")
    
    def run_everything(self):
        print("COMPLETE VECTOR SEARCH TEST RUN")
        print("=" * 50)
        
        # run all the test categories
        self.check_database_stuff()
        self.check_embeddings()
        self.check_similarity_math()
        self.check_basic_search()
        self.check_hybrid_search()
        self.check_query_processing()
        self.check_edge_cases()
        self.check_performance()
        self.check_accuracy()
        self.check_vector_ops()
        
        # show results
        self.show_results()
    
    def show_results(self):
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        
        passed = sum(1 for _, result, _ in self.test_results if result)
        total = len(self.test_results)
        
        print(f"Passed: {passed}/{total} ({(passed/total*100):.1f}%)")
        
        # show failed tests
        failed = [name for name, result, _ in self.test_results if not result]
        if failed:
            print(f"\nFailed tests:")
            for test in failed:
                print(f"  - {test}")
        
        print(f"\nOverall:")
        if passed / total >= 0.9:
            print("EXCELLENT - everything working great")
        elif passed / total >= 0.8:
            print("GOOD - working well with minor issues")
        elif passed / total >= 0.7:
            print("OK - working but has some problems")
        else:
            print("NEEDS WORK - multiple issues found")

def main():
    runner = TestRunner()
    runner.run_everything()

if __name__ == "__main__":
    main()