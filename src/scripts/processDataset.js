import fetch from 'node-fetch';
import { EmbeddingService } from '../services/embedding.js';
import { VectorStore } from '../services/vectorStore.js';
import { TextChunker } from '../utils/chunking.js';

async function downloadYelpDataset(limit = 100) {
  console.log('Downloading Yelp dataset...');
  
  try {
    // Using Hugging Face datasets API to get a sample
    const response = await fetch('https://datasets-server.huggingface.co/rows?dataset=Yelp%2Fyelp_review_full&config=yelp_review_full&split=train&offset=0&length=100');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.rows && data.rows.length > 0) {
      // Transform the data to match our expected format
      const reviews = data.rows.map((row, index) => ({
        review_id: `yelp_${index}`,
        text: row.row.text,
        stars: row.row.label + 1, // Labels are 0-4, convert to 1-5 stars
      }));
      
      console.log(`Downloaded ${reviews.length} reviews`);
      return reviews.slice(0, limit);
    } else {
      throw new Error('No data found in response');
    }
  } catch (error) {
    console.error('Error downloading dataset:', error);
    
    // Fallback: return sample data for testing
    console.log('Using sample data for testing...');
    return [
      {
        review_id: 'sample_1',
        text: 'This restaurant was absolutely amazing! The food was delicious, the service was excellent, and the atmosphere was perfect. I especially loved the pasta dish - it was cooked to perfection with a rich, flavorful sauce. The staff was very attentive and friendly. Will definitely be coming back here again!',
        stars: 5
      },
      {
        review_id: 'sample_2',
        text: 'Had a terrible experience here. The food was cold, the service was slow, and the place was dirty. My burger was undercooked and the fries were soggy. The waiter seemed annoyed when I asked questions about the menu. Would not recommend this place to anyone.',
        stars: 1
      },
      {
        review_id: 'sample_3',
        text: 'Pretty good restaurant overall. The food was decent, nothing spectacular but satisfying. Service was okay, took a while to get our orders but the staff was polite. The prices are reasonable for the portion sizes. It\'s a solid choice if you\'re in the area.',
        stars: 3
      }
    ];
  }
}

async function processDataset() {
  try {
    console.log('Starting dataset processing...');
    
    // Initialize services
    const embeddingService = new EmbeddingService();
    const vectorStore = new VectorStore();
    const chunker = new TextChunker(300, 50); // Smaller chunks for better embeddings
    
    // Download dataset
    const reviews = await downloadYelpDataset(50); // Process 50 reviews for testing
    console.log(`Processing ${reviews.length} reviews`);
    
    // Chunk the reviews
    console.log('Chunking reviews...');
    const chunks = chunker.chunkReviews(reviews);
    console.log(`Created ${chunks.length} chunks from ${reviews.length} reviews`);
    
    // Process chunks in batches
    const batchSize = 10;
    let processedCount = 0;
    
    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      console.log(`Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(chunks.length / batchSize)}`);
      
      try {
        // Create embeddings for the batch
        const texts = batch.map(chunk => chunk.text);
        const embeddings = await embeddingService.createBatchEmbeddings(texts, 5);
        
        // Prepare data for insertion
        const insertData = batch.map((chunk, index) => ({
          review_id: chunk.reviewId,
          chunk_text: chunk.text,
          chunk_index: chunk.chunkIndex,
          embedding: embeddings[index],
          stars: chunk.stars
        }));
        
        // Insert into vector store
        await vectorStore.insertBatchChunks(insertData);
        processedCount += batch.length;
        
        console.log(`Successfully processed ${processedCount}/${chunks.length} chunks`);
        
        // Add delay between batches to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 1000));
        
      } catch (error) {
        console.error(`Error processing batch starting at index ${i}:`, error);
        // Continue with next batch instead of failing completely
        continue;
      }
    }
    
    console.log('\n✅ Dataset processing completed successfully!');
    console.log(`Total chunks processed: ${processedCount}`);
    
    // Verify the data was inserted
    const totalCount = await vectorStore.getReviewCount();
    console.log(`Total chunks in database: ${totalCount}`);
    
  } catch (error) {
    console.error('Error processing dataset:', error);
    process.exit(1);
  }
}

// Run if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  processDataset();
}