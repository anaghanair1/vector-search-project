"""
Database config for supabase stuff
TODO: maybe add connection pooling later?
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()  # load the env vars

class DatabaseConfig:
    def __init__(self):
        # get credentials from environment 
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        # basic validation - probably should add more checks
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing supabase credentials in .env file")
        
        self.client = self._get_client()
    
    def _get_client(self):
        """Creates supabase client connection"""
        try:
            client = create_client(self.supabase_url, self.supabase_key)
            return client
        except Exception as e:
            print(f"Failed to connect: {e}")
            raise ConnectionError(f"Supabase connection failed: {e}")
    
    def get_client(self):
        return self.client
    
    def test_connection(self):
        """Quick test to see if db works"""
        try:
            # just try to query something simple
            result = self.client.table('review_chunks').select('id').limit(1).execute()
            print("✅ Database connection works!")
            if result.data:
                print(f"Found {len(result.data)} test record")
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

# create global instance - probably not the best pattern but works for now
db_config = DatabaseConfig()