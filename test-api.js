import fetch from 'node-fetch';
import dotenv from 'dotenv';

dotenv.config();

const HF_API_KEY = process.env.HUGGINGFACE_API_KEY;

// List of models to try (feature extraction models)
const MODELS_TO_TRY = [
  'https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2',
  'https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5',
  'https://api-inference.huggingface.co/models/thenlper/gte-small',
  'https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2'
];

async function testModel(modelUrl, modelName) {
  console.log(`\n🧪 Testing: ${modelName}`);
  console.log(`URL: ${modelUrl}`);
  
  try {
    const response = await fetch(modelUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${HF_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        inputs: "This is a test sentence for embedding.",
        options: { 
          wait_for_model: true
        }
      })
    });

    console.log(`Status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.log(`❌ Error: ${errorText}`);
      return false;
    }

    const result = await response.json();
    
    if (Array.isArray(result) && result.length > 0) {
      console.log('✅ SUCCESS!');
      console.log(`- Embedding dimensions: ${result.length}`);
      console.log(`- First 5 values: [${result.slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
      console.log(`- Type: ${typeof result[0]}`);
      return { url: modelUrl, name: modelName, dimensions: result.length };
    } else {
      console.log(`❌ Unexpected format: ${typeof result}`);
      console.log(`Response: ${JSON.stringify(result).substring(0, 200)}...`);
      return false;
    }
    
  } catch (error) {
    console.log(`❌ Network error: ${error.message}`);
    return false;
  }
}

async function findWorkingModel() {
  console.log('🔍 Finding a working embedding model...');
  console.log(`API Key: ${HF_API_KEY ? HF_API_KEY.substring(0, 10) + '...' : 'NOT FOUND'}`);
  
  if (!HF_API_KEY) {
    console.log('❌ No API key found!');
    return;
  }

  for (let i = 0; i < MODELS_TO_TRY.length; i++) {
    const modelUrl = MODELS_TO_TRY[i];
    const modelName = modelUrl.split('/').pop();
    
    const result = await testModel(modelUrl, modelName);
    
    if (result) {
      console.log('\n🎉 FOUND WORKING MODEL!');
      console.log(`Model: ${result.name}`);
      console.log(`URL: ${result.url}`);
      console.log(`Dimensions: ${result.dimensions}`);
      
      console.log('\n📝 Update your embedding.js with this URL:');
      console.log(`const MODEL_URL = '${result.url}';`);
      
      return result;
    }
    
    // Wait between attempts
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  
  console.log('\n❌ No working models found. This might be an API key issue.');
  console.log('Try:');
  console.log('1. Check if your API key is valid at https://huggingface.co/settings/tokens');
  console.log('2. Make sure your account has access to Inference API');
  console.log('3. Try a different model manually');
}

findWorkingModel();