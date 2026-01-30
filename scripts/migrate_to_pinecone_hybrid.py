import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load Environment
load_dotenv()
from src.config import config, DATA_DIR, INDEX_DIR
from src.vector_db import PineconeClient

# Dataset Path
DATA_FILE = DATA_DIR / "processed" / "processed_patents_10k.json"

async def generate_embeddings(client, texts, model="text-embedding-3-small"):
    """Batch generate embeddings."""
    try:
        resp = await client.embeddings.create(input=texts, model=model)
        return [data.embedding for data in resp.data]
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

async def main():
    print("[INFO] Starting Pinecone Hybrid Migration...")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not found.")
        return

    if not os.environ.get("PINECONE_API_KEY"):
        print("[ERROR] PINECONE_API_KEY not found.")
        return

    # 1. Load Data
    print(f"[INFO] Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        patents = json.load(f)
    
    print(f"[INFO] Loaded {len(patents)} patents.")
    
    # 2. Initialize Pinecone Client
    # This will load default BM25Encoder or existing one
    pc_client = PineconeClient()
    
    # 3. Fit BM25 Encoder Globally
    print("[INFO] Fitting BM25 Encoder on full corpus...")
    # Combine abstract and claims for rich representation
    corpus_texts = []
    
    for p in patents:
        # Use abstract + claims if available, else just abstract
        text = f"{p.get('abstract', '')} {p.get('claims', '')}".strip()
        if not text:
            text = p.get('title', '')
        corpus_texts.append(text)
        
    # Fit and Save
    pc_client.bm25_encoder.fit(corpus_texts)
    pc_client.bm25_encoder.dump(str(pc_client.bm25_params_path))
    print(f"[INFO] BM25 Params saved to {pc_client.bm25_params_path}")
    
    # 4. Generate Embeddings & Upsert (Batch)
    openai_client = AsyncOpenAI()
    batch_size = 50  # Pinecone recommendation
    
    total_upserted = 0
    
    # Process in batches
    for i in tqdm(range(0, len(patents), batch_size), desc="Indexing"):
        batch_patents = patents[i : i + batch_size]
        batch_texts = corpus_texts[i : i + batch_size]
        
        # Generate Embeddings
        embeddings = await generate_embeddings(openai_client, batch_texts)
        if not embeddings:
            continue
            
        embeddings_np = np.array(embeddings)
        
        # Prepare Metadata
        metadata_list = []
        for p, text in zip(batch_patents, batch_texts):
            # Parse IPC
            ipc_code = p.get("ipc_code", "")
            if isinstance(ipc_code, list) and ipc_code:
                ipc_code = ipc_code[0]
            
            meta = {
                "chunk_id": f"pat_{p.get('publication_number', 'unknown')}",
                "patent_id": p.get("publication_number", "unknown"),
                "title": p.get("title", ""),
                "content": text,
                "ipc_codes": [ipc_code] if ipc_code else []
            }
            metadata_list.append(meta)
            
        # Upsert
        # Note: add_vectors internally generates sparse vectors using the fitted encoder we just set
        try:
            pc_client.add_vectors(embeddings_np, metadata_list)
            total_upserted += len(metadata_list)
        except Exception as e:
            print(f"[ERROR] Upsert failed at batch {i}: {e}")
            
    print(f"\n[SUCCESS] Migration Complete! Total upserted: {total_upserted}")
    print("[INFO] You can now use Pinecone Hybrid search without local pickle files.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
