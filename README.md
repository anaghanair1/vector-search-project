# Restaurant Review Vector Search

This is a search system for finding restaurant reviews using embeddings. It lets you search by meaning instead of just keywords.

## What it does

Instead of searching for exact words, this finds reviews that mean similar things. So if you search for "delicious food" it might also find reviews mentioning "tasty meals" or "amazing cuisine" even though they don't use the exact same words.

The system combines two types of search:
- **Semantic search** - understands the meaning behind your query
- **Keyword search** - traditional exact word matching  
- **Hybrid search** - uses both methods together for better results

### Database
- Using Supabase (PostgreSQL) with the pgvector extension
- Stores text chunks with their vector embeddings
- Has about 187 review chunks from 50 different reviews currently

### AI Model
- Uses the `all-MiniLM-L6-v2` model from sentence-transformers
- Converts text into 384-dimensional vectors
- Runs locally so no API costs

### Architecture
The code is organized into different services:
- `embedding_service.py` - handles the AI model and creating vectors
- `vector_store.py` - manages database operations  
- `hybrid_search_service.py` - combines semantic and keyword search
- `query_processor.py` - analyzes and enhances search queries

## Getting started

### Prerequisites
- Python 3.8+
- Supabase account (free tier works fine)

### Installation

1. Clone this repo and navigate to the python directory
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install supabase python-dotenv sentence-transformers transformers torch "numpy<2" pandas datasets requests tqdm
   ```

4. Set up your environment variables by creating a `.env` file:
   ```
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

### Database setup

In your Supabase SQL editor, run the queries in database/schema.sql, and run the function in database/functions.sql, to create the tables and functions.

### Testing everything

Run the comprehensive test suite:
```bash
python all_tests.py
```

This will test all functionality - database, embeddings, search, etc. You should see mostly PASS results.

## Usage examples

### Interactive demo
```bash
python scripts/interactive_semantic_search.py
```

### Add your own data
```bash
python scripts/process_dataset.py
```

### Test everything
```bash
python test_connection.py
```

## Performance

On my setup (MacBook Pro), the system handles:
- ~0.04 seconds to generate 10 embeddings
- ~0.44 seconds to search 187 chunks
- Can process about 20 chunks per batch when adding new data

Currently managing 187 review chunks from 50 reviews with 98.6% test pass rate.

## Troubleshooting

**NumPy version issues**: Make sure to install `numpy<2` as newer versions have compatibility issues with some ML libraries.

**Model loading fails**: The sentence-transformers model downloads automatically on first use. Make sure you have internet connection and about 90MB of space.

**Database connection errors**: Double check your Supabase credentials in the .env file and that the pgvector extension is enabled.

**Slow performance**: The first time you run embeddings it downloads the model. Subsequent runs should be much faster.

## Dependencies

Main libraries used:
- `supabase` - Database client
- `sentence-transformers` - AI embeddings
- `numpy` - Vector operations  
- `pandas` - Data processing
- `transformers` - ML model support
- `torch` - Neural network backend

## License

The Yelp dataset has its own licensing terms.

## Notes

The code isn't perfect but it works well for the intended use case. The restaurant review domain was chosen because it has clear semantic relationships (good/bad, food/service, etc.).