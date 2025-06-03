"""
Database configuration for Supabase connection
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

class DatabaseConfig:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables")
        
        self.client = self._create_client()
    
    def _create_client(self) -> Client:
        """Create and return Supabase client"""
        try:
            client = create_client(self.supabase_url, self.supabase_key)
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Supabase: {e}")
    
    def get_client(self) -> Client:
        """Get the Supabase client"""
        return self.client
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            # Try a simple query to test connection
            result = self.client.table('review_chunks').select('id').limit(1).execute()
            print(f"✅ Database connection successful!")
            print(f"📊 Found {len(result.data)} sample record(s)")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

# Global database instance
db_config = DatabaseConfig()