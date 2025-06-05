"""
Query processor for hybrid search
Handles query analysis, keyword extraction, query enhancement etc.
"""
import re
from typing import List, Dict

class QueryProcessor:
    def __init__(self):
        """Setup processor with restaurant domain knowledge"""
        # synonyms for common restaurant terms
        self.synonyms = {
            'delicious': ['tasty', 'flavorful', 'amazing', 'excellent'],
            'terrible': ['horrible', 'awful', 'disgusting', 'bad'],
            'good': ['great', 'nice', 'decent', 'solid'],
            'fast': ['quick', 'speedy', 'prompt'],
            'slow': ['sluggish', 'delayed', 'lengthy'],
            'expensive': ['costly', 'pricey', 'overpriced'],
            'cheap': ['affordable', 'inexpensive', 'budget'],
            'fresh': ['crisp', 'new', 'vibrant'],
            'spicy': ['hot', 'fiery', 'zesty'],
            'friendly': ['nice', 'kind', 'helpful', 'polite'],
            'rude': ['impolite', 'unfriendly', 'hostile']
        }
        
        # restaurant categories - helps with understanding context
        self.categories = {
            'food': ['taste', 'flavor', 'delicious', 'fresh', 'cooking', 'meal'],
            'service': ['staff', 'waiter', 'waitress', 'server', 'friendly'],
            'atmosphere': ['ambiance', 'mood', 'decor', 'music', 'lighting'],
            'price': ['cost', 'expensive', 'cheap', 'value', 'money'],
            'location': ['parking', 'convenient', 'accessible'],
            'timing': ['fast', 'slow', 'quick', 'wait', 'time']
        }
        
        # sentiment words
        self.positive_words = ['amazing', 'excellent', 'wonderful', 'great', 'fantastic', 
                              'perfect', 'love', 'best', 'awesome', 'incredible']
        self.negative_words = ['terrible', 'horrible', 'awful', 'worst', 'hate', 
                              'disgusting', 'disappointing', 'poor', 'bad']
        
        print("📝 Query processor ready with restaurant knowledge")
    
    def process_query(self, query, enhance=True):
        """Main query processing function"""
        # clean up the query
        cleaned = self.clean_query(query)
        
        # analyze what the query is about
        analysis = self.analyze_query(cleaned)
        
        # extract keywords
        keywords = self.extract_keywords(cleaned)
        
        # enhance if requested
        if enhance:
            enhanced = self.enhance_query(cleaned, analysis)
            keyword_query = self.build_keyword_query(keywords, analysis)
        else:
            enhanced = cleaned
            keyword_query = cleaned
        
        return {
            'original_query': query,
            'cleaned_query': cleaned,
            'enhanced_query': enhanced,
            'keyword_query': keyword_query,
            'keywords': keywords,
            'analysis': analysis,
            'enhanced': enhance
        }
    
    def clean_query(self, query):
        """Clean up the query text"""
        # basic cleanup
        cleaned = query.lower().strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # normalize spaces
        cleaned = re.sub(r'[^\w\s\-\'\"]+', ' ', cleaned)  # remove weird chars
        
        # handle common abbreviations
        replacements = {
            'w/': 'with',
            'w/o': 'without',
            'gt': 'great',
            'thru': 'through'
        }
        
        for abbrev, full in replacements.items():
            cleaned = cleaned.replace(abbrev, full)
        
        return cleaned.strip()
    
    def analyze_query(self, query):
        """Analyze what the query is asking for"""
        words = query.split()
        
        # figure out main category
        category_scores = {}
        for cat, cat_words in self.categories.items():
            score = sum(1 for word in words if word in cat_words)
            if score > 0:
                category_scores[cat] = score
        
        main_category = max(category_scores, key=category_scores.get) if category_scores else 'general'
        
        # detect sentiment
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count > negative_count:
            sentiment = 'positive'
        elif negative_count > positive_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # guess intent
        if any(word in words for word in ['recommend', 'best']):
            intent = 'seeking_recommendation'
        elif any(word in words for word in ['avoid', 'worst']):
            intent = 'seeking_warnings'
        else:
            intent = 'general_search'
        
        return {
            'main_category': main_category,
            'sentiment': sentiment,
            'intent': intent,
            'category_scores': category_scores
        }
    
    def extract_keywords(self, query):
        """Extract important keywords"""
        # words to ignore
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'me', 'my', 'we'
        }
        
        words = query.split()
        keywords = []
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                keywords.append(word)
        
        # remove duplicates but keep order
        seen = set()
        unique_keywords = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                unique_keywords.append(word)
        
        return unique_keywords
    
    def enhance_query(self, query, analysis):
        """Add synonyms and related terms to improve semantic search"""
        words = query.split()
        enhanced_words = list(words)  # start with original
        
        # add synonyms for known words
        for word in words:
            if word in self.synonyms:
                # add a couple synonyms, not too many
                synonyms = self.synonyms[word][:2]
                enhanced_words.extend(synonyms)
        
        # add category-specific terms if we detected a category
        if analysis['main_category'] in self.categories:
            cat_terms = self.categories[analysis['main_category']]
            # add relevant terms not already in query
            for term in cat_terms[:2]:  # just a couple
                if term not in query:
                    enhanced_words.append(term)
        
        return ' '.join(enhanced_words)
    
    def build_keyword_query(self, keywords, analysis):
        """Build optimized query for keyword search"""
        query_parts = list(keywords)
        
        # add sentiment words if detected
        if analysis['sentiment'] == 'positive':
            query_parts.extend(['excellent', 'great'])
        elif analysis['sentiment'] == 'negative':
            query_parts.extend(['terrible', 'bad'])
        
        return ' '.join(query_parts)
    
    def get_suggestions(self, partial_query):
        """Generate suggestions for partial queries"""
        suggestions = []
        
        # common patterns
        common_patterns = [
            "delicious food",
            "excellent service", 
            "reasonable prices",
            "romantic atmosphere",
            "family friendly",
            "quick service",
            "good value",
            "fresh ingredients"
        ]
        
        partial_lower = partial_query.lower()
        
        for pattern in common_patterns:
            if pattern.startswith(partial_lower) or partial_lower in pattern:
                suggestions.append(pattern)
        
        return suggestions[:5]  # limit suggestions

# test if run directly
if __name__ == "__main__":
    processor = QueryProcessor()
    
    test_queries = [
        "delicious food great service",
        "terrible slow service",
        "expensive overpriced restaurant",
        "romantic date night"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = processor.process_query(query)
        print(f"  Cleaned: {result['cleaned_query']}")
        print(f"  Enhanced: {result['enhanced_query']}")
        print(f"  Keywords: {result['keywords']}")
        print(f"  Category: {result['analysis']['main_category']}")
        print(f"  Sentiment: {result['analysis']['sentiment']}")