"""
Text chunking utilities
Splits long reviews into smaller pieces for better embedding/search

LangChain RecursiveCharacterTextSplitter: https://python.langchain.com/docs/how_to/recursive_text_splitter/
LangChain Text Splitters Concepts: https://python.langchain.com/docs/concepts/text_splitters/
NLTK Text Processing: https://www.nltk.org/book/ch03.html
spaCy Sentence Segmentation: https://spacy.io/usage/linguistic-features#sbd
Regex Patterns: https://docs.python.org/3/library/re.html
Text Preprocessing: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
Overlap Strategies: https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846
Query Enhancement Techniques: https://en.wikipedia.org/wiki/Query_expansion
Text Classification: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
Stop Words Lists: https://www.ranks.nl/stopwords
Sentiment Analysis: https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
Keyword Extraction: https://pypi.org/project/yake/

"""
import re
from typing import List, Dict

class TextChunker:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # might want to make these configurable later
    
    def chunk_text(self, text):
        """Split text into chunks with some overlap"""
        if not text or not text.strip():
            return []
        
        # clean up the text first
        clean_text = self.clean_text(text)
        
        # if it's short enough, just return as-is
        if len(clean_text) <= self.chunk_size:
            return [clean_text]
        
        # split into sentences first - better than just character splitting
        sentences = self.split_sentences(clean_text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # check if adding this sentence would make chunk too big
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # save current chunk if it's long enough
                if len(current_chunk.strip()) > 50:  # minimum useful size
                    chunks.append(current_chunk.strip())
                
                # start new chunk, maybe with some overlap
                if chunks and self.overlap > 0:
                    # take some words from end of previous chunk
                    words = current_chunk.split()
                    if len(words) > 10:
                        overlap_words = words[-10:]  # last 10 words
                        current_chunk = " ".join(overlap_words) + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                # add sentence to current chunk
                if current_chunk:
                    current_chunk = current_chunk + " " + sentence
                else:
                    current_chunk = sentence
        
        # don't forget the last chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 50:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def clean_text(self, text):
        """Clean up text - remove weird characters, normalize spacing"""
        # normalize whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # remove some problematic characters but keep punctuation
        cleaned = re.sub(r'[^\w\s.,!?;:()\-\'"]+', ' ', cleaned)
        
        # fix quotes
        cleaned = cleaned.replace('"', '"').replace('"', '"')
        cleaned = cleaned.replace(''', "'").replace(''', "'")
        
        return cleaned.strip()
    
    def split_sentences(self, text):
        """Basic sentence splitting - could be improved"""
        # simple approach using regex
        sentences = re.split(r'[.!?]+\s+', text)
        
        # filter out very short "sentences"
        good_sentences = []
        for s in sentences:
            s = s.strip()
            if len(s) > 10:  # ignore tiny fragments
                good_sentences.append(s)
        
        return good_sentences
    
    def chunk_review(self, review):
        """Chunk a single review into multiple pieces"""
        text = review.get('text', '')
        chunks = self.chunk_text(text)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_data = {
                'review_id': review.get('review_id', f"review_{hash(text)}"),
                'chunk_text': chunk_text,
                'chunk_index': i,
                'stars': review.get('stars', 0),
                'original_review': review  # keep reference to original
            }
            result.append(chunk_data)
        
        return result
    
    def chunk_reviews(self, reviews, show_progress=True):
        """Process multiple reviews and chunk them all"""
        all_chunks = []
        
        if show_progress:
            print(f"Chunking {len(reviews)} reviews...")
        
        for i, review in enumerate(reviews):
            # show progress occasionally
            if show_progress and i % 50 == 0:
                print(f"  Processing review {i + 1}/{len(reviews)}")
            
            review_chunks = self.chunk_review(review)
            all_chunks.extend(review_chunks)
        
        if show_progress:
            avg_chunks = len(all_chunks) / len(reviews) if reviews else 0
            print(f"Created {len(all_chunks)} total chunks from {len(reviews)} reviews")
            print(f"Average {avg_chunks:.1f} chunks per review")
        
        return all_chunks
    
    def get_stats(self, chunks):
        """Calculate some basic statistics about chunks"""
        if not chunks:
            return {}
        
        lengths = [len(chunk['chunk_text']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(lengths) / len(lengths),
            'min_chunk_length': min(lengths),
            'max_chunk_length': max(lengths),
            'total_characters': sum(lengths)
        }