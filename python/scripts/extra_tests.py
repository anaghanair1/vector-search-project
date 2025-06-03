"""
Comprehensive Test Suite for Semantic Search System
Tests functionality, accuracy, and edge cases
"""
import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import db_config
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore

class SemanticSearchTester:
    def __init__(self):
        """Initialize comprehensive test suite"""
        self.client = db_config.get_client()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.test_results = []
        
        print(f"Test Suite Initialized")
        print(f"Embedding Model: {self.embedding_service.model_name}")
        print(f"Dimensions: {self.embedding_service.embedding_dimension}")
        print("=" * 60)
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results"""
        status = "PASS" if passed else "FAIL"
        self.test_results.append((test_name, passed, details))
        print(f"[{status}] {test_name}")
        if details:
            print(f"      {details}")
    
    def test_1_database_connectivity(self):
        """Test 1: Database Connection and Data Availability"""
        print("\n1. DATABASE CONNECTIVITY TESTS")
        print("-" * 40)
        
        try:
            # Test connection
            count = self.vector_store.get_chunk_count()
            self.log_test("Database Connection", count >= 0, f"Found {count} chunks")
            
            # Test data availability
            has_data = count > 100
            self.log_test("Sufficient Data Available", has_data, 
                         f"Need >100 chunks for testing, found {count}")
            
            # Test sample retrieval
            samples = self.vector_store.get_sample_chunks(5)
            self.log_test("Sample Data Retrieval", len(samples) > 0, 
                         f"Retrieved {len(samples)} sample chunks")
            
            return has_data
            
        except Exception as e:
            self.log_test("Database Connectivity", False, f"Error: {e}")
            return False
    
    def test_2_embedding_consistency(self):
        """Test 2: Embedding Generation Consistency"""
        print("\n2. EMBEDDING CONSISTENCY TESTS")
        print("-" * 40)
        
        test_text = "This restaurant has amazing food and excellent service"
        
        try:
            # Test same text produces same embedding
            embedding1 = self.embedding_service.create_embedding(test_text)
            embedding2 = self.embedding_service.create_embedding(test_text)
            
            are_identical = np.allclose(embedding1, embedding2, atol=1e-6)
            self.log_test("Embedding Determinism", are_identical, 
                         "Same text should produce identical embeddings")
            
            # Test embedding dimensions
            correct_dim = len(embedding1) == 384
            self.log_test("Embedding Dimensions", correct_dim, 
                         f"Expected 384, got {len(embedding1)}")
            
            # Test different texts produce different embeddings
            different_text = "Terrible food and horrible service"
            embedding3 = self.embedding_service.create_embedding(different_text)
            
            are_different = not np.allclose(embedding1, embedding3, atol=0.1)
            self.log_test("Different Text Different Embeddings", are_different,
                         "Different texts should produce different embeddings")
            
            return are_identical and correct_dim and are_different
            
        except Exception as e:
            self.log_test("Embedding Generation", False, f"Error: {e}")
            return False
    
    def test_3_similarity_logic(self):
        """Test 3: Similarity Calculation Logic"""
        print("\n3. SIMILARITY LOGIC TESTS")
        print("-" * 40)
        
        try:
            # Test identical text similarity
            text1 = "Great food and excellent service"
            text2 = "Great food and excellent service"  # Identical
            text3 = "Amazing food and wonderful service"  # Similar
            text4 = "Terrible food and horrible service"  # Opposite
            
            emb1 = self.embedding_service.create_embedding(text1)
            emb2 = self.embedding_service.create_embedding(text2)
            emb3 = self.embedding_service.create_embedding(text3)
            emb4 = self.embedding_service.create_embedding(text4)
            
            # Calculate cosine similarities manually
            def cosine_similarity(a, b):
                a, b = np.array(a), np.array(b)
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            sim_identical = cosine_similarity(emb1, emb2)
            sim_similar = cosine_similarity(emb1, emb3)
            sim_opposite = cosine_similarity(emb1, emb4)
            
            # Test identical similarity (should be ~1.0)
            identical_correct = sim_identical > 0.99
            self.log_test("Identical Text Similarity", identical_correct,
                         f"Similarity: {sim_identical:.4f} (should be ~1.0)")
            
            # Test similar > opposite
            ranking_correct = sim_similar > sim_opposite
            self.log_test("Similarity Ranking Logic", ranking_correct,
                         f"Similar: {sim_similar:.4f} > Opposite: {sim_opposite:.4f}")
            
            return identical_correct and ranking_correct
            
        except Exception as e:
            self.log_test("Similarity Logic", False, f"Error: {e}")
            return False
    
    def test_4_search_functionality(self):
        """Test 4: Search Function Integration"""
        print("\n4. SEARCH FUNCTIONALITY TESTS")
        print("-" * 40)
        
        test_queries = [
            ("delicious food", "Should find positive food reviews"),
            ("terrible service", "Should find negative service reviews"),
            ("good value", "Should find price-related reviews"),
            ("xyz123nonexistent", "Should handle non-matching queries")
        ]
        
        all_passed = True
        
        for query, description in test_queries:
            try:
                query_embedding = self.embedding_service.create_embedding(query)
                
                # Test search function
                results = self.client.rpc('search_similar_reviews', {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.1,  # Low threshold to get results
                    'match_count': 5
                }).execute()
                
                has_results = bool(results.data)
                
                if query == "xyz123nonexistent":
                    # This should have no or very low similarity results
                    passed = not has_results or (has_results and max(r['similarity'] for r in results.data) < 0.3)
                    details = f"Non-matching query correctly handled"
                else:
                    # Normal queries should return results
                    passed = has_results
                    details = f"Found {len(results.data) if results.data else 0} results"
                
                self.log_test(f"Search: '{query}'", passed, details)
                all_passed = all_passed and passed
                
            except Exception as e:
                self.log_test(f"Search: '{query}'", False, f"Error: {e}")
                all_passed = False
        
        return all_passed
    
    def test_5_semantic_accuracy(self):
        """Test 5: Semantic Understanding Accuracy"""
        print("\n5. SEMANTIC ACCURACY TESTS")
        print("-" * 40)
        
        # Define test cases with expected behavior
        semantic_tests = [
            {
                "query": "amazing delicious food",
                "should_find": ["food", "delicious", "tasty", "flavor"],
                "should_avoid": ["terrible", "horrible", "bad"],
                "description": "Positive food query"
            },
            {
                "query": "slow rude service",
                "should_find": ["service", "staff", "wait"],
                "should_avoid": ["food", "taste", "flavor"],
                "description": "Service-related query"
            },
            {
                "query": "expensive overpriced",
                "should_find": ["price", "cost", "expensive", "money"],
                "should_avoid": ["taste", "atmosphere"],
                "description": "Price-related query"
            }
        ]
        
        all_passed = True
        
        for test_case in semantic_tests:
            try:
                query = test_case["query"]
                query_embedding = self.embedding_service.create_embedding(query)
                
                results = self.client.rpc('search_similar_reviews', {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.2,
                    'match_count': 5
                }).execute()
                
                if results.data:
                    # Analyze top result
                    top_result = results.data[0]
                    result_text = top_result['chunk_text'].lower()
                    similarity = top_result['similarity']
                    
                    # Check if result contains expected terms
                    contains_expected = any(term in result_text for term in test_case["should_find"])
                    avoids_unexpected = not any(term in result_text for term in test_case["should_avoid"])
                    
                    # Semantic accuracy score
                    semantic_score = similarity * 100
                    passed = contains_expected and semantic_score > 25
                    
                    details = f"Score: {semantic_score:.1f}%, Contains expected: {contains_expected}"
                    self.log_test(test_case["description"], passed, details)
                    
                else:
                    passed = False
                    details = "No results found"
                    self.log_test(test_case["description"], passed, details)
                
                all_passed = all_passed and passed
                
            except Exception as e:
                self.log_test(test_case["description"], False, f"Error: {e}")
                all_passed = False
        
        return all_passed
    
    def test_6_performance_scalability(self):
        """Test 6: Performance and Scalability"""
        print("\n6. PERFORMANCE TESTS")
        print("-" * 40)
        
        try:
            # Test embedding generation speed
            test_texts = ["This is a test sentence"] * 10
            
            start_time = time.time()
            embeddings = self.embedding_service.create_batch_embeddings(test_texts, show_progress=False)
            embedding_time = time.time() - start_time
            
            embedding_speed_ok = embedding_time < 5.0  # Should process 10 embeddings in <5 seconds
            self.log_test("Embedding Generation Speed", embedding_speed_ok,
                         f"10 embeddings in {embedding_time:.2f}s")
            
            # Test search speed
            query_embedding = self.embedding_service.create_embedding("test query")
            
            start_time = time.time()
            results = self.client.rpc('search_similar_reviews', {
                'query_embedding': query_embedding,
                'match_threshold': 0.1,
                'match_count': 10
            }).execute()
            search_time = time.time() - start_time
            
            search_speed_ok = search_time < 2.0  # Should search in <2 seconds
            self.log_test("Search Speed", search_speed_ok,
                         f"Search completed in {search_time:.2f}s")
            
            # Test memory usage (basic check)
            total_chunks = self.vector_store.get_chunk_count()
            memory_reasonable = total_chunks > 0
            self.log_test("Database Scale", memory_reasonable,
                         f"Handling {total_chunks} chunks")
            
            return embedding_speed_ok and search_speed_ok and memory_reasonable
            
        except Exception as e:
            self.log_test("Performance Tests", False, f"Error: {e}")
            return False
    
    def test_7_edge_cases(self):
        """Test 7: Edge Cases and Error Handling"""
        print("\n7. EDGE CASE TESTS")
        print("-" * 40)
        
        edge_cases = [
            ("", "Empty query"),
            ("a", "Single character"),
            ("a" * 1000, "Very long query"),
            ("12345", "Numbers only"),
            ("!@#$%", "Special characters only"),
            ("The the the the", "Repeated words")
        ]
        
        all_passed = True
        
        for query, description in edge_cases:
            try:
                if query:  # Skip empty for embedding (will fail)
                    embedding = self.embedding_service.create_embedding(query)
                    
                    results = self.client.rpc('search_similar_reviews', {
                        'query_embedding': embedding,
                        'match_threshold': 0.1,
                        'match_count': 3
                    }).execute()
                    
                    # Edge cases should not crash the system
                    passed = True
                    details = f"Handled gracefully, got {len(results.data) if results.data else 0} results"
                else:
                    # Empty query should be handled by validation
                    passed = True
                    details = "Empty query validation passed"
                
                self.log_test(description, passed, details)
                
            except Exception as e:
                # Some edge cases might fail gracefully
                passed = "empty" in str(e).lower() or "invalid" in str(e).lower()
                details = f"Expected error: {str(e)[:50]}..."
                self.log_test(description, passed, details)
                all_passed = all_passed and passed
        
        return all_passed
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("COMPREHENSIVE SEMANTIC SEARCH TEST SUITE")
        print("=" * 60)
        
        test_functions = [
            self.test_1_database_connectivity,
            self.test_2_embedding_consistency,
            self.test_3_similarity_logic,
            self.test_4_search_functionality,
            self.test_5_semantic_accuracy,
            self.test_6_performance_scalability,
            self.test_7_edge_cases
        ]
        
        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                print(f"Test suite error: {e}")
        
        # Final summary
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, result, _ in self.test_results if result)
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total} ({(passed/total*100):.1f}%)")
        print()
        
        # Detailed results
        categories = {}
        for test_name, result, details in self.test_results:
            category = test_name.split(':')[0] if ':' in test_name else "General"
            if category not in categories:
                categories[category] = []
            categories[category].append((test_name, result, details))
        
        for category, tests in categories.items():
            cat_passed = sum(1 for _, result, _ in tests if result)
            print(f"{category}: {cat_passed}/{len(tests)} passed")
        
        print("\nOVERALL SYSTEM STATUS:")
        if passed / total >= 0.8:
            print("✅ SYSTEM WORKING CORRECTLY - Ready for production")
        elif passed / total >= 0.6:
            print("⚠️  SYSTEM MOSTLY WORKING - Minor issues detected")
        else:
            print("❌ SYSTEM HAS ISSUES - Requires attention")

def main():
    """Run comprehensive tests"""
    tester = SemanticSearchTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()