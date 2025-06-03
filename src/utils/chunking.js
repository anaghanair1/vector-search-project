export class TextChunker {
    constructor(chunkSize = 500, overlap = 50) {
      this.chunkSize = chunkSize;
      this.overlap = overlap;
    }
  
    chunkText(text) {
      if (!text || text.length === 0) {
        return [];
      }
  
      // Clean the text
      const cleanedText = text.trim().replace(/\s+/g, ' ');
      
      if (cleanedText.length <= this.chunkSize) {
        return [cleanedText];
      }
  
      const chunks = [];
      let start = 0;
  
      while (start < cleanedText.length) {
        let end = start + this.chunkSize;
        
        // If we're not at the end of the text, try to find a sentence boundary
        if (end < cleanedText.length) {
          // Look for sentence endings within the last 100 characters of the chunk
          const searchStart = Math.max(end - 100, start);
          const possibleEnd = cleanedText.substring(searchStart, end);
          const sentenceEndMatch = possibleEnd.match(/[.!?]\s/g);
          
          if (sentenceEndMatch) {
            const lastSentenceEnd = possibleEnd.lastIndexOf(sentenceEndMatch[sentenceEndMatch.length - 1]);
            if (lastSentenceEnd !== -1) {
              end = searchStart + lastSentenceEnd + 1;
            }
          }
        }
  
        const chunk = cleanedText.substring(start, end).trim();
        if (chunk.length > 0) {
          chunks.push(chunk);
        }
  
        // Move start position considering overlap
        start = end - this.overlap;
        
        // Prevent infinite loop
        if (start >= cleanedText.length) {
          break;
        }
      }
  
      return chunks;
    }
  
    chunkReview(review) {
      const reviewText = review.text || '';
      const chunks = this.chunkText(reviewText);
      
      return chunks.map((chunk, index) => ({
        reviewId: review.review_id || `review_${Date.now()}_${Math.random()}`,
        text: chunk,
        chunkIndex: index,
        stars: review.stars || 0,
        originalReview: review
      }));
    }
  
    chunkReviews(reviews, logProgress = true) {
      const allChunks = [];
      
      reviews.forEach((review, index) => {
        if (logProgress && index % 100 === 0) {
          console.log(`Chunking review ${index + 1}/${reviews.length}`);
        }
        
        const reviewChunks = this.chunkReview(review);
        allChunks.push(...reviewChunks);
      });
  
      if (logProgress) {
        console.log(`Total chunks created: ${allChunks.length}`);
      }
  
      return allChunks;
    }
  }