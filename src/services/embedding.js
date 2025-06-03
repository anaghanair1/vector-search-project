import fetch from 'node-fetch';
import dotenv from 'dotenv';

dotenv.config();

const HF_API_KEY = process.env.HUGGINGFACE_API_KEY;
// WORKING MODEL: BAAI/bge-small-en-v1.5 (384 dimensions)
const MODEL_URL = 'https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5';

export class EmbeddingService {
  constructor() {
    if (!HF_API_KEY) {
      throw new Error('Missing Hugging Face API key');
    }
  }

  async createEmbedding(text) {
    try {
      const response = await fetch(MODEL_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${HF_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          inputs: text,
          options: { 
            wait_for_model: true
          }
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const embedding = await response.json();
      
      // The working model returns array directly
      if (Array.isArray(embedding) && embedding.length === 384) {
        return embedding;
      } else {
        console.log('Unexpected response format:', embedding);
        throw new Error(`Invalid embedding response format. Expected array of 384 numbers, got: ${typeof embedding}`);
      }
    } catch (error) {
      console.error('Error creating embedding:', error);
      throw error;
    }
  }

  async createBatchEmbeddings(texts, batchSize = 5) {
    const embeddings = [];
    
    console.log(`Creating embeddings for ${texts.length} texts using BAAI/bge-small-en-v1.5...`);
    
    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      console.log(`Processing text ${i + 1}/${texts.length} (${Math.round((i/texts.length)*100)}%)`);
      
      try {
        const embedding = await this.createEmbedding(text);
        embeddings.push(embedding);
        
        // Wait 1 second between requests to be nice to the API
        if (i < texts.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
      } catch (error) {
        console.error(`Error processing text ${i + 1}:`, error);
        
        // Wait longer on error before retrying
        console.log('Waiting 3 seconds before retry...');
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // Try once more
        try {
          console.log(`Retrying text ${i + 1}...`);
          const embedding = await this.createEmbedding(text);
          embeddings.push(embedding);
        } catch (retryError) {
          console.error(`Failed again on text ${i + 1}, skipping this text...`);
          // Continue without this embedding - don't add anything to maintain correct indexing
          throw retryError; // This will cause the batch to be marked as failed
        }
      }
    }
    
    console.log(`✅ Successfully created ${embeddings.length} embeddings`);
    return embeddings;
  }
}