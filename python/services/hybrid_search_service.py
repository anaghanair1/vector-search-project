"""
Hybrid search service - combines semantic and keyword search
Works better than either approach alone
"""
import time
from typing import List, Dict
from config.database import db_config
from services.embedding_service import EmbeddingService
from services.query_processor import QueryProcessor

class HybridSearchService:
    def __init__(self):
        """Setup hybrid search system"""
        self.client = db_config.get_client()
        self.embedding_service = EmbeddingService()
        self.query_processor = QueryProcessor()
        
        print("Hybrid search ready")
        print(f"   Model: {self.embedding_service.model_name}")
        print(f"   Dimensions: {self.embedding_service.embedding_dimension}")
    
    def search(self, query, semantic_weight=0.6, keyword_weight=0.4, 
               match_threshold=0.1, match_count=10, enhance_query=True):
        """
        Main hybrid search function
        
        Args:
            query: what to search for
            semantic_weight: how much to weight semantic similarity (0-1)
            keyword_weight: how much to weight keyword matching (0-1)  
            match_threshold: minimum similarity score to include
            match_count: max results to return
            enhance_query: whether to expand query with synonyms
        """
        start_time = time.time()
        
        # make sure weights add up to 1
        if abs(semantic_weight + keyword_weight - 1.0) > 0.001:
            raise ValueError("Weights must add up to 1.0")
        
        # process the query
        processed_query = self.query_processor.process_query(query, enhance=enhance_query)
        
        # create embedding for semantic search
        emb_start = time.time()
        query_embedding = self.embedding_service.create_embedding(processed_query['enhanced_query'])
        emb_time = time.time() - emb_start
        
        # do the hybrid search
        search_start = time.time()
        results = self._run_hybrid_search(
            query_embedding=query_embedding,
            query_text=processed_query['keyword_query'],
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            match_threshold=match_threshold,
            match_count=match_count
        )
        search_time = time.time() - search_start
        
        total_time = time.time() - start_time
        
        # package up the response
        response = {
            'query': {
                'original': query,
                'processed': processed_query,
                'embedding_size': len(query_embedding)
            },
            'settings': {
                'semantic_weight': semantic_weight,
                'keyword_weight': keyword_weight,
                'match_threshold': match_threshold,
                'match_count': match_count
            },
            'results': results,
            'timing': {
                'total_results': len(results),
                'embedding_time_ms': round(emb_time * 1000, 2),
                'search_time_ms': round(search_time * 1000, 2),
                'total_time_ms': round(total_time * 1000, 2),
                'has_semantic': any(r['semantic_similarity'] > match_threshold for r in results),
                'has_keywords': any(r['keyword_rank'] > 0 for r in results)
            }
        }
        
        return response
    
    def _run_hybrid_search(self, query_embedding, query_text, semantic_weight, 
                          keyword_weight, match_threshold, match_count):
        """Actually execute the hybrid search in database"""
        try:
            result = self.client.rpc('hybrid_search_reviews', {
                'query_embedding': query_embedding,
                'query_text': query_text,
                'semantic_weight': semantic_weight,
                'keyword_weight': keyword_weight,
                'match_threshold': match_threshold,
                'match_count': match_count
            }).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Hybrid search failed: {e}")
            return []
    
    def semantic_only(self, query, **kwargs):
        """Search using only semantic similarity"""
        return self.search(
            query=query,
            semantic_weight=1.0,
            keyword_weight=0.0,
            **kwargs
        )
    
    def keyword_only(self, query, **kwargs):
        """Search using only keyword matching"""
        return self.search(
            query=query,
            semantic_weight=0.0,
            keyword_weight=1.0,
            **kwargs
        )
    
    def compare_methods(self, query, **kwargs):
        """Compare different search approaches"""
        methods = {
            'hybrid': self.search(query, **kwargs),
            'semantic_only': self.semantic_only(query, **kwargs),
            'keyword_only': self.keyword_only(query, **kwargs)
        }
        
        return {
            'query': query,
            'methods': methods,
            'comparison': self._analyze_differences(methods)
        }
    
    def _analyze_differences(self, methods):
        """Analyze how different methods perform"""
        analysis = {
            'result_counts': {},
            'unique_results': {},
            'overlap': {}
        }
        
        # count results per method
        for method_name, method_results in methods.items():
            results = method_results['results']
            analysis['result_counts'][method_name] = len(results)
            analysis['unique_results'][method_name] = set(r['id'] for r in results)
        
        # check overlaps
        hybrid_ids = analysis['unique_results'].get('hybrid', set())
        semantic_ids = analysis['unique_results'].get('semantic_only', set())
        keyword_ids = analysis['unique_results'].get('keyword_only', set())
        
        analysis['overlap'] = {
            'hybrid_semantic': len(hybrid_ids & semantic_ids),
            'hybrid_keyword': len(hybrid_ids & keyword_ids),
            'semantic_keyword': len(semantic_ids & keyword_ids),
            'all_three': len(hybrid_ids & semantic_ids & keyword_ids)
        }
        
        return analysis
    
    def get_stats(self):
        """Get stats about searchable database"""
        try:
            # total chunks
            total_result = self.client.table('review_chunks').select('id', count='exact').execute()
            total_chunks = total_result.count or 0
            
            # chunks with text search vectors (needed for keyword search)
            vector_result = self.client.table('review_chunks')\
                .select('id', count='exact')\
                .not_.is_('text_search_vector', 'null')\
                .execute()
            chunks_with_vectors = vector_result.count or 0
            
            return {
                'total_chunks': total_chunks,
                'chunks_with_text_vectors': chunks_with_vectors,
                'text_search_coverage': round((chunks_with_vectors / total_chunks) * 100, 2) if total_chunks > 0 else 0,
                'hybrid_search_ready': chunks_with_vectors > 0
            }
            
        except Exception as e:
            print(f"Stats error: {e}")
            return {}
    
    def find_optimal_weights(self, query, test_count=5):
        """
        Test different weight combinations to find what works best
        Useful for understanding query types
        """
        combinations = []
        for i in range(test_count + 1):
            semantic_w = i / test_count
            keyword_w = 1 - semantic_w
            combinations.append((semantic_w, keyword_w))
        
        results = {}
        for sem_w, key_w in combinations:
            key = f"s{sem_w:.1f}_k{key_w:.1f}"
            try:
                search_result = self.search(
                    query=query,
                    semantic_weight=sem_w,
                    keyword_weight=key_w,
                    match_count=5
                )
                results[key] = {
                    'weights': {'semantic': sem_w, 'keyword': key_w},
                    'result_count': len(search_result['results']),
                    'avg_score': sum(r['hybrid_score'] for r in search_result['results']) / len(search_result['results']) if search_result['results'] else 0,
                    'has_both_signals': search_result['timing']['has_semantic'] and search_result['timing']['has_keywords']
                }
            except Exception as e:
                results[key] = {'error': str(e)}
        
        # find best combination
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_key = max(valid_results.keys(), 
                          key=lambda k: valid_results[k]['avg_score'])
            best_weights = valid_results[best_key]['weights']
        else:
            best_weights = {'semantic': 0.6, 'keyword': 0.4}  # fallback
        
        return {
            'query': query,
            'tested_combinations': results,
            'optimal_weights': best_weights,
            'recommendation': f"Use semantic={best_weights['semantic']:.1f}, keyword={best_weights['keyword']:.1f}"
        }

# test the service
if __name__ == "__main__":
    service = HybridSearchService()
    
    # get stats
    print("Search system stats:")
    stats = service.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # test search
    test_query = "delicious food excellent service"
    print(f"\nTesting: '{test_query}'")
    
    result = service.search(test_query, match_count=3)
    print(f"Found {result['timing']['total_results']} results")
    print(f"Search time: {result['timing']['total_time_ms']}ms")