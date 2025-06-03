import { supabase } from '../config/supabase.js';

export class VectorStore {
  async insertChunk(reviewId, chunkText, chunkIndex, embedding, stars) {
    try {
      const { data, error } = await supabase
        .from('review_chunks')
        .insert([{
          review_id: reviewId,
          chunk_text: chunkText,
          chunk_index: chunkIndex,
          embedding: embedding,
          stars: stars
        }]);

      if (error) {
        throw error;
      }

      return data;
    } catch (error) {
      console.error('Error inserting chunk:', error);
      throw error;
    }
  }

  async insertBatchChunks(chunks) {
    try {
      const { data, error } = await supabase
        .from('review_chunks')
        .insert(chunks);

      if (error) {
        throw error;
      }

      return data;
    } catch (error) {
      console.error('Error inserting batch chunks:', error);
      throw error;
    }
  }

  async searchSimilar(queryEmbedding, matchThreshold = 0.7, matchCount = 10) {
    try {
      const { data, error } = await supabase
        .rpc('search_similar_reviews', {
          query_embedding: queryEmbedding,
          match_threshold: matchThreshold,
          match_count: matchCount
        });

      if (error) {
        throw error;
      }

      return data;
    } catch (error) {
      console.error('Error searching similar reviews:', error);
      throw error;
    }
  }

  async getReviewCount() {
    try {
      const { count, error } = await supabase
        .from('review_chunks')
        .select('*', { count: 'exact', head: true });

      if (error) {
        throw error;
      }

      return count;
    } catch (error) {
      console.error('Error getting review count:', error);
      throw error;
    }
  }

  async clearAllData() {
    try {
      const { error } = await supabase
        .from('review_chunks')
        .delete()
        .neq('id', 0); // Delete all records

      if (error) {
        throw error;
      }

      console.log('All data cleared successfully');
    } catch (error) {
      console.error('Error clearing data:', error);
      throw error;
    }
  }
}