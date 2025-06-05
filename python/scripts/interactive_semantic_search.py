"""
Interactive demo for semantic search
Shows what the system can do with different types of queries.

Python Input/Output: https://docs.python.org/3/tutorial/inputoutput.html
Command Line Interfaces: https://realpython.com/python-command-line-arguments/
Progress Tracking: https://github.com/tqdm/tqdm

"""
import sys
import os
import time

# add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import db_config
from services.embedding_service import EmbeddingService

class InteractiveDemo:
    def __init__(self):
        """Setup the demo system"""
        print("Setting up semantic search demo...")
        print("=" * 50)
        
        self.client = db_config.get_client()
        self.embedding_service = EmbeddingService()
        
        # get some basic stats
        result = self.client.table('review_chunks').select('id', count='exact').execute()
        self.total_chunks = result.count or 0
        
        print(f"Model: {self.embedding_service.model_name}")
        print(f"Embedding size: {self.embedding_service.embedding_dimension}")
        print(f"Database chunks: {self.total_chunks:,}")
        print(f"Demo ready!")
        print("=" * 50)
    
    def search_and_show(self, query, explain=True):
        """Do a search and show results nicely"""
        if explain:
            print(f"\nQuery: '{query}'")
            print("=" * 40)
            print("Steps:")
            print("  1. Converting to vector...")
        
        start_time = time.time()
        
        # create embedding
        query_emb = self.embedding_service.create_embedding(query)
        emb_time = time.time() - start_time
        
        if explain:
            print(f"  2. Embedding done in {emb_time:.3f}s")
            print(f"  3. Searching {self.total_chunks:,} chunks...")
        
        search_start = time.time()
        
        # search
        result = self.client.rpc('search_similar_reviews', {
            'query_embedding': query_emb,
            'match_threshold': 0.15,  # lower for demo
            'match_count': 5
        }).execute()
        
        search_time = time.time() - search_start
        total_time = time.time() - start_time
        
        if explain:
            print(f"  4. Search done in {search_time:.3f}s")
            print(f"  5. Total time: {total_time:.3f}s")
        
        if not result.data:
            print("\nNo results found")
            return []
        
        # show results
        print(f"\nFound {len(result.data)} results:")
        print("=" * 50)
        
        for i, match in enumerate(result.data, 1):
            score = match['similarity'] * 100
            stars = "★" * match['stars'] + "☆" * (5 - match['stars'])
            
            print(f"\nResult {i}:")
            print(f"   Similarity: {score:.1f}%")
            print(f"   Rating: {stars} ({match['stars']}/5)")
            print(f"   Review: {match['review_id']}")
            text = match['chunk_text']
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"   Text: {text}")
            print("-" * 50)
        
        return result.data
    
    def demo_categories(self):
        """Show different types of semantic understanding"""
        
        categories = [
            {
                "name": "FOOD QUALITY",
                "desc": "Understanding food-related concepts",
                "queries": [
                    "delicious amazing food",
                    "tasty flavorful dishes", 
                    "terrible bland food",
                    "fresh ingredients",
                    "overcooked burnt"
                ]
            },
            {
                "name": "SERVICE QUALITY", 
                "desc": "Service and staff experiences",
                "queries": [
                    "excellent friendly service",
                    "helpful attentive staff",
                    "rude slow service",
                    "ignored by waitress",
                    "professional team"
                ]
            },
            {
                "name": "PRICE & VALUE",
                "desc": "Cost and value perceptions", 
                "queries": [
                    "expensive overpriced",
                    "great value reasonable",
                    "cheap affordable",
                    "worth the money",
                    "too costly"
                ]
            },
            {
                "name": "ATMOSPHERE",
                "desc": "Environment and mood",
                "queries": [
                    "romantic cozy atmosphere",
                    "noisy crowded place", 
                    "peaceful quiet",
                    "lively energetic",
                    "beautiful decor"
                ]
            }
        ]
        
        print("\nSEMANTIC SEARCH DEMO")
        print("=" * 50)
        print("Press Enter after each result...")
        print("=" * 50)
        
        for cat_num, category in enumerate(categories, 1):
            print(f"\n[{cat_num}/{len(categories)}] {category['name']}")
            print(f"Description: {category['desc']}")
            print("-" * 30)
            
            for query_num, query in enumerate(category['queries'], 1):
                print(f"\nQuery {query_num}/{len(category['queries'])}")
                
                results = self.search_and_show(query, explain=True)
                
                if results:
                    print(f"\nSemantic analysis:")
                    print(f"  Found content related to '{query}'")
                    print(f"  even without exact word matches!")
                
                user_input = input(f"\nPress Enter to continue (or 'skip'): ").strip().lower()
                if user_input == 'skip':
                    break
                elif user_input in ['quit', 'exit']:
                    return
            
            if cat_num < len(categories):
                input(f"\nDone with {category['name']}. Press Enter for next...")
    
    def interactive_mode(self):
        """Let user try their own searches"""
        print("\n" + "=" * 50)
        print("INTERACTIVE MODE")
        print("=" * 50)
        print("Try your own restaurant search queries.")
        print("Type 'demo' to go back, 'quit' to exit.")
        print("=" * 50)
        
        while True:
            query = input("\nEnter search query: ").strip()
            
            if not query:
                continue
            elif query.lower() in ['quit', 'exit']:
                break
            elif query.lower() == 'demo':
                return 'demo'
            elif query.lower() == 'help':
                print("\nTry these examples:")
                print("  - 'spicy hot food'")
                print("  - 'family friendly'") 
                print("  - 'date night romantic'")
                print("  - 'fast casual'")
                print("  - 'vegetarian options'")
                continue
            
            self.search_and_show(query, explain=True)
            
            print(f"\nNotice the semantic matching!")
        
        return 'quit'
    
    def performance_test(self):
        """Quick performance demo"""
        print("\n" + "=" * 50)
        print("PERFORMANCE TEST")
        print("=" * 50)
        
        queries = [
            "excellent food quality",
            "poor service", 
            "reasonable prices",
            "romantic atmosphere",
            "fast delivery"
        ]
        
        print(f"Testing {len(queries)} queries...")
        print(f"Database: {self.total_chunks:,} chunks")
        
        total_start = time.time()
        
        for i, query in enumerate(queries, 1):
            start = time.time()
            
            query_emb = self.embedding_service.create_embedding(query)
            emb_time = time.time() - start
            
            search_start = time.time()
            result = self.client.rpc('search_similar_reviews', {
                'query_embedding': query_emb,
                'match_threshold': 0.2,
                'match_count': 5
            }).execute()
            search_time = time.time() - search_start
            
            total_time = time.time() - start
            results_count = len(result.data) if result.data else 0
            
            print(f"Query {i}: '{query}'")
            print(f"  Embedding: {emb_time:.3f}s | Search: {search_time:.3f}s | Total: {total_time:.3f}s")
            print(f"  Results: {results_count}")
        
        total_time = time.time() - total_start
        avg_time = total_time / len(queries)
        
        print(f"\nPerformance summary:")
        print(f"  Total: {total_time:.3f}s")
        print(f"  Average: {avg_time:.3f}s per query")
        print(f"  Speed: {len(queries)/total_time:.1f} queries/second")
        print(f"  System handles real-time loads!")
        
        input("\nPress Enter to continue...")
    
    def run_demo(self):
        """Main demo function"""
        print("\nWELCOME TO SEMANTIC SEARCH DEMO")
        print("=" * 50)
        print("Choose what you want to see:")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("  1. Guided demo - see examples")
            print("  2. Interactive search - try your own") 
            print("  3. Performance test")
            print("  4. Exit")
            
            choice = input("\nChoice (1-4): ").strip()
            
            if choice == '1':
                self.demo_categories()
            elif choice == '2':
                result = self.interactive_mode()
                if result == 'demo':
                    self.demo_categories()
            elif choice == '3':
                self.performance_test()
            elif choice == '4':
                break
            else:
                print("Please enter 1, 2, 3, or 4")
        
        print("\nThanks for trying the demo!")

def main():
    """Run the demo"""
    try:
        demo = InteractiveDemo()
        demo.run_demo()
    except Exception as e:
        print(f"Demo error: {e}")
        print("Make sure database connection works")

if __name__ == "__main__":
    main()