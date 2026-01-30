from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

from src.vector_db import PineconeClient

def main():
    print("Checking Pinecone stats...")
    try:
        client = PineconeClient()
        stats = client.index.describe_index_stats()
        print(f"Index Stats: {stats}")
        
        namespace = client.config.namespace
        ns_stats = stats.get('namespaces', {}).get(namespace, {})
        print(f"Namespace '{namespace}' vector count: {ns_stats.get('vector_count', 0)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
