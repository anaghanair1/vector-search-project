"""
Interactive Semantic Search Demo 
"""
import sys
import os
import time
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import db_config
from services.embedding_service import EmbeddingService

class InteractiveSemanticDemo:
    def __init__(self):
        """Initialize interactive demo system"""
        print("Initializing Semantic Search Demo...")
        print("=" * 60)
        
        self.client = db_config.get_client()
        self.embedding_service = EmbeddingService()
        
        # Get database stats
        result = self.client.table('review_chunks').select('id', count='exact').execute()
        self.total_chunks = result.count or 0
        
        print(f"Model: {self.embedding_service.model_name}")
        print(f"Embedding Dimensions: {self.embedding_service.embedding_dimension}")
        print(f"Database Chunks: {self.total_chunks:,}")
        print(f"Ready for demonstration!")
        print("=" * 60)
    
    def search_and_display(self, query: str, show_explanation: bool = True) -> List[Dict]:
        """Perform search and display results with explanations"""
        if show_explanation:
            print(f"\nQUERY: '{query}'")
            print("=" * 50)
            print("Processing:")
            print("  1. Converting query to 384-dimensional vector...")
        
        start_time = time.time()
        
        # Create query embedding
        query_embedding = self.embedding_service.create_embedding(query)
        embedding_time = time.time() - start_time
        
        if show_explanation:
            print(f"  2. Embedding created in {embedding_time:.3f}s")
            print(f"  3. Searching {self.total_chunks:,} chunks using cosine similarity...")
        
        search_start = time.time()
        
        # Search using Supabase function
        result = self.client.rpc('search_similar_reviews', {
            'query_embedding': query_embedding,
            'match_threshold': 0.15,  # Lower threshold for demo
            'match_count': 5
        }).execute()
        
        search_time = time.time() - search_start
        total_time = time.time() - start_time
        
        if show_explanation:
            print(f"  4. Search completed in {search_time:.3f}s")
            print(f"  5. Total processing time: {total_time:.3f}s")
        
        if not result.data:
            print("\nNo results found (similarity too low)")
            return []
        
        # Display results
        print(f"\nFOUND {len(result.data)} RESULTS:")
        print("=" * 60)
        
        for i, match in enumerate(result.data, 1):
            similarity_score = match['similarity'] * 100
            stars = "★" * match['stars'] + "☆" * (5 - match['stars'])
            
            print(f"\nResult {i}:")
            print(f"   Similarity Score: {similarity_score:.1f}%")
            print(f"   Star Rating: {stars} ({match['stars']}/5)")
            print(f"   Review ID: {match['review_id']}")
            print(f"   Text: {match['chunk_text']}")
            print("-" * 60)
        
        return result.data
    
    def demonstrate_semantic_power(self):
        """Show various types of semantic understanding"""
        
        demo_categories = [
            {
                "title": "FOOD QUALITY DETECTION",
                "description": "Shows how the system understands food-related concepts",
                "queries": [
                    "delicious amazing food",
                    "tasty flavorful meal", 
                    "mouth-watering cuisine",
                    "bland tasteless food",
                    "overcooked burnt meal"
                ]
            },
            {
                "title": "SERVICE QUALITY ANALYSIS", 
                "description": "Demonstrates understanding of service-related experiences",
                "queries": [
                    "excellent friendly service",
                    "attentive helpful staff",
                    "rude slow service",
                    "ignored by waitress",
                    "professional courteous team"
                ]
            },
            {
                "title": "PRICE & VALUE PERCEPTION",
                "description": "Shows understanding of cost and value concepts", 
                "queries": [
                    "expensive overpriced",
                    "great value reasonable price",
                    "cheap affordable meal",
                    "worth every penny",
                    "too costly for quality"
                ]
            },
            {
                "title": "ATMOSPHERE & AMBIANCE",
                "description": "Captures environmental and mood-related descriptions",
                "queries": [
                    "romantic cozy atmosphere",
                    "noisy crowded restaurant", 
                    "peaceful quiet dining",
                    "lively energetic vibe",
                    "beautiful elegant decor"
                ]
            },
            {
                "title": "SPECIFIC DISHES & CUISINE",
                "description": "Finds specific food items and cooking styles",
                "queries": [
                    "pizza crispy crust",
                    "pasta sauce rich",
                    "burger juicy beef",
                    "salad fresh vegetables", 
                    "dessert sweet chocolate"
                ]
            },
            {
                "title": "EMOTIONAL EXPERIENCES",
                "description": "Understands emotional responses and overall experiences",
                "queries": [
                    "disappointed upset experience",
                    "happy satisfied customer",
                    "frustrated angry visit",
                    "amazed impressed meal",
                    "regret waste money"
                ]
            },
            {
                "title": "TIME & CONVENIENCE FACTORS",
                "description": "Captures timing and convenience aspects",
                "queries": [
                    "quick fast service",
                    "long wait time",
                    "convenient location parking",
                    "rushed hurried meal",
                    "relaxed leisurely dining"
                ]
            },
            {
                "title": "RECOMMENDATION PATTERNS",
                "description": "Finds recommendation and return visit patterns", 
                "queries": [
                    "highly recommend friends",
                    "never coming back",
                    "definitely return soon",
                    "tell everyone about",
                    "avoid this place"
                ]
            }
        ]
        
        print("\nSEMANTIC SEARCH POWER DEMONSTRATION")
        print("=" * 60)
        print("\nPress Enter after each result to continue...")
        print("=" * 60)
        
        total_categories = len(demo_categories)
        
        for cat_num, category in enumerate(demo_categories, 1):
            print(f"\n[Category {cat_num}/{total_categories}] {category['title']}")
            print(f"Description: {category['description']}")
            print("-" * 40)
            
            for query_num, query in enumerate(category['queries'], 1):
                print(f"\nQuery {query_num}/{len(category['queries'])}")
                
                # Perform search
                results = self.search_and_display(query, show_explanation=True)
                
                if results:
                    # Show semantic understanding
                    print(f"\nSEMANTIC ANALYSIS:")
                    print(f"  The system found content related to '{query}'")
                    print(f"  even if the exact words don't match!")
                    print(f"  Similarity scores show confidence levels.")
                
                # Wait for user input
                user_input = input(f"\nPress Enter to continue (or 'skip' to next category): ").strip().lower()
                if user_input == 'skip':
                    break
                elif user_input in ['quit', 'exit', 'stop']:
                    return
            
            if cat_num < total_categories:
                input(f"\nFinished {category['title']}. Press Enter for next category...")
    
    def interactive_search_mode(self):
        """Allow supervisor to try their own queries"""
        print("\n" + "=" * 60)
        print("INTERACTIVE SEARCH MODE")
        print("=" * 60)
        print("Enter any search term related to restaurant experiences.")
        print("Type 'demo' to return to guided demo, or 'quit' to exit.")
        print("=" * 60)
        
        while True:
            query = input("\nEnter your search query: ").strip()
            
            if not query:
                continue
            elif query.lower() in ['quit', 'exit', 'stop']:
                break
            elif query.lower() == 'demo':
                return 'demo'
            elif query.lower() == 'help':
                print("\nSample queries you can try:")
                print("  - 'spicy hot food'")
                print("  - 'family friendly restaurant'") 
                print("  - 'date night romantic'")
                print("  - 'fast casual dining'")
                print("  - 'vegetarian healthy options'")
                continue
            
            # Perform the search
            self.search_and_display(query, show_explanation=True)
            
            print(f"\nNotice how the system found semantically related content!")
        
        return 'quit'
    
    def show_technical_explanation(self):
        """Explain the technical concepts for supervisor"""
        print("\n" + "=" * 60)
        print("TECHNICAL EXPLANATION")
        print("=" * 60)
        
        explanations = [
            {
                "concept": "Vector Embeddings",
                "explanation": """
Each piece of text is converted into a 384-dimensional vector (list of numbers).
Similar meanings produce similar vectors. For example:
  "delicious food" → [0.1, -0.3, 0.7, ..., 0.2]
  "tasty meal"     → [0.09, -0.29, 0.68, ..., 0.19]  (similar numbers)
  "terrible service" → [-0.2, 0.4, -0.1, ..., -0.3]  (different numbers)
"""
            },
            {
                "concept": "Cosine Similarity", 
                "explanation": """
We measure similarity using the angle between vectors in 384D space.
  - Similar concepts: Small angle (high similarity score)
  - Different concepts: Large angle (low similarity score)
  - Formula: cosine(angle) = (A·B) / (|A|×|B|)
  - Result: 0-100% similarity score
"""
            },
            {
                "concept": "Semantic Understanding",
                "explanation": """
The AI model was trained on millions of text examples to learn that:
  - "great" and "excellent" have similar meanings
  - "food" and "meal" refer to similar concepts  
  - "terrible" and "horrible" express similar sentiments
This allows matching by MEANING rather than exact words.
"""
            },
            {
                "concept": "Database Search",
                "explanation": """
The database contains:
  - {total_chunks} text chunks from restaurant reviews
  - Each chunk has its 384D vector stored
  - Search compares query vector to all stored vectors
  - Returns top matches ranked by similarity score
  - Entire search happens in milliseconds
""".format(total_chunks=self.total_chunks)
            }
        ]
        
        for i, item in enumerate(explanations, 1):
            print(f"\n{i}. {item['concept']}")
            print("-" * 30)
            print(item['explanation'])
            
            if i < len(explanations):
                input("Press Enter to continue...")
        
        input("\nPress Enter to return to demo...")
    
    def run_supervisor_demo(self):
        """Main demo function for supervisor"""
        print("\nWELCOME TO SEMANTIC SEARCH DEMONSTRATION")
        print("=" * 60)
        print("\nChoose your demonstration path:")
        print("=" * 60)
        
        while True:
            print("\nOptions:")
            print("  1. Guided Demo - See pre-selected examples")
            print("  2. Interactive Search - Try your own queries") 
            print("  3. Technical Explanation - How it works")
            print("  4. Quick Performance Test")
            print("  5. Exit Demo")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                self.demonstrate_semantic_power()
            elif choice == '2':
                result = self.interactive_search_mode()
                if result == 'demo':
                    self.demonstrate_semantic_power()
            elif choice == '3':
                self.show_technical_explanation()
            elif choice == '4':
                self.quick_performance_test()
            elif choice == '5':
                break
            else:
                print("Please enter 1, 2, 3, 4, or 5")
        
        print("\nThank you for the demonstration!")
    
    def quick_performance_test(self):
        """Show system performance metrics"""
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST")
        print("=" * 60)
        
        test_queries = [
            "excellent food quality",
            "poor customer service", 
            "reasonable prices",
            "romantic atmosphere",
            "fast delivery"
        ]
        
        print(f"Testing search speed with {len(test_queries)} queries...")
        print(f"Database size: {self.total_chunks:,} chunks")
        
        total_start = time.time()
        
        for i, query in enumerate(test_queries, 1):
            start = time.time()
            
            query_embedding = self.embedding_service.create_embedding(query)
            embedding_time = time.time() - start
            
            search_start = time.time()
            result = self.client.rpc('search_similar_reviews', {
                'query_embedding': query_embedding,
                'match_threshold': 0.2,
                'match_count': 5
            }).execute()
            search_time = time.time() - search_start
            
            total_time = time.time() - start
            results_count = len(result.data) if result.data else 0
            
            print(f"Query {i}: '{query}'")
            print(f"  Embedding: {embedding_time:.3f}s | Search: {search_time:.3f}s | Total: {total_time:.3f}s")
            print(f"  Results: {results_count}")
        
        total_duration = time.time() - total_start
        avg_time = total_duration / len(test_queries)
        
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"  Total time: {total_duration:.3f}s")
        print(f"  Average per query: {avg_time:.3f}s")
        print(f"  Queries per second: {len(test_queries)/total_duration:.1f}")
        print(f"  System can handle real-time search loads!")
        
        input("\nPress Enter to return to menu...")

def main():
    """Run the supervisor demonstration"""
    try:
        demo = InteractiveSemanticDemo()
        demo.run_supervisor_demo()
    except Exception as e:
        print(f"Demo error: {e}")
        print("Please ensure your database connection is working.")

if __name__ == "__main__":
    main()