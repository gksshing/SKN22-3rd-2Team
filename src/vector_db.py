"""
Short-Cut v3.0 - FAISS + BM25 Hybrid Vector Database
========================================================
In-memory vector database with Hybrid Search (RRF Fusion).

Features:
- FAISS IndexFlatIP for dense vector search
- BM25 for sparse keyword search
- RRF (Reciprocal Rank Fusion) for result merging
- Zero-latency startup with pre-computed index

Author: Team 뀨💕
License: MIT
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from src.config import config, FaissConfig, PineconeConfig, EMBEDDINGS_DIR, INDEX_DIR


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchResult:
    """Result from vector similarity search."""
    chunk_id: str
    patent_id: str
    score: float
    content: str
    content_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Hybrid search fields
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0


@dataclass
class InsertResult:
    """Result from inserting vectors."""
    success: bool
    inserted_count: int
    index_path: str
    error_message: Optional[str] = None


# =============================================================================
# BM25 Search Engine
# =============================================================================

class BM25SearchEngine:
    """
    BM25-based sparse keyword search.
    
    Used in combination with FAISS for hybrid search.
    """
    
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.chunk_ids = []
        self.metadata_list = []
        self._initialized = False
    
    def build_index(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "content",
        id_key: str = "chunk_id",
    ) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document dicts
            text_key: Key for text content
            id_key: Key for document ID
        """
        if not BM25_AVAILABLE:
            logger.warning("rank_bm25 not available. Install with: pip install rank_bm25")
            return
        
        self.corpus = []
        self.chunk_ids = []
        self.metadata_list = []
        
        for doc in documents:
            text = doc.get(text_key, "")
            # Tokenize: lowercase and split
            tokens = self._tokenize(text)
            self.corpus.append(tokens)
            self.chunk_ids.append(doc.get(id_key, ""))
            self.metadata_list.append(doc)
        
        self.bm25 = BM25Okapi(self.corpus)
        self._initialized = True
        
        logger.info(f"Built BM25 index for {len(self.corpus)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search using BM25.
        
        Returns:
            List of (chunk_id, score, metadata) tuples
        """
        if not self._initialized or self.bm25 is None:
            logger.warning("BM25 index not initialized")
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((
                    self.chunk_ids[idx],
                    float(scores[idx]),
                    self.metadata_list[idx],
                ))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split, filter."""
        if not text:
            return []
        
        # Lowercase and split
        tokens = text.lower().split()
        
        # Remove very short tokens and punctuation-only tokens
        tokens = [t for t in tokens if len(t) > 2 and re.search(r'[a-z0-9]', t)]
        
        return tokens
    
    def save_local(self, path: Path) -> None:
        """Save BM25 index to disk."""
        data = {
            "corpus": self.corpus,
            "chunk_ids": self.chunk_ids,
            "metadata_list": self.metadata_list,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index to {path}")
    
    def load_local(self, path: Path) -> bool:
        """Load BM25 index from disk."""
        if not path.exists():
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.corpus = data["corpus"]
            self.chunk_ids = data["chunk_ids"]
            self.metadata_list = data["metadata_list"]
            
            if BM25_AVAILABLE:
                self.bm25 = BM25Okapi(self.corpus)
                self._initialized = True
                logger.info(f"Loaded BM25 index from {path} ({len(self.corpus)} docs)")
                return True
            else:
                logger.warning("rank_bm25 not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False


# =============================================================================
# FAISS Client with Hybrid Search
# =============================================================================

class FaissClient:
    """
    FAISS-based in-memory vector database with Hybrid Search.
    
    Features:
    - Dense search using FAISS IndexFlatIP
    - Sparse search using BM25
    - RRF (Reciprocal Rank Fusion) for combining results
    """
    
    def __init__(
        self,
        faiss_config: FaissConfig = None,
        embedding_dim: int = None,
    ):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        
        self.config = faiss_config or config.faiss
        self.embedding_dim = embedding_dim or config.embedding.embedding_dim
        
        # FAISS index
        self.index = None
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self._loaded = False
        
        # BM25 engine for hybrid search
        self.bm25_engine = BM25SearchEngine()
        self.bm25_path = self.config.index_path.parent / "bm25_index.pkl"
        
        logger.info(f"FAISS Client initialized (dim={self.embedding_dim})")
    
    def create_index(self, use_cosine: bool = True) -> None:
        """
        Create a new FAISS index.
        """
        if use_cosine:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info(f"Created IndexFlatIP (cosine similarity) with dim={self.embedding_dim}")
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created IndexFlatL2 with dim={self.embedding_dim}")
        
        self.metadata = {}
        self.id_to_idx = {}
        self._loaded = True
    
    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        normalize: bool = True,
    ) -> int:
        """
        Add vectors to the index with metadata.
        Also builds BM25 index for hybrid search.
        """
        if self.index is None:
            self.create_index()
        
        embeddings = embeddings.astype(np.float32)
        
        if normalize:
            faiss.normalize_L2(embeddings)
        
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        for i, meta in enumerate(metadata_list):
            idx = start_idx + i
            self.metadata[idx] = meta
            chunk_id = meta.get("chunk_id", f"chunk_{idx}")
            self.id_to_idx[chunk_id] = idx
        
        # Build BM25 index for hybrid search
        self.bm25_engine.build_index(metadata_list, text_key="content", id_key="chunk_id")
        
        logger.info(f"Added {len(embeddings)} vectors to index (total: {self.index.ntotal})")
        
        return len(embeddings)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        normalize: bool = True,
    ) -> List[SearchResult]:
        """
        Search for similar vectors (dense search only).
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        if normalize:
            faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            
            meta = self.metadata.get(int(idx), {})
            
            results.append(SearchResult(
                chunk_id=meta.get("chunk_id", f"chunk_{idx}"),
                patent_id=meta.get("patent_id", ""),
                score=float(score),
                content=meta.get("content", ""),
                content_type=meta.get("content_type", ""),
                dense_score=float(score),
                metadata={
                    "ipc_code": meta.get("ipc_code", ""),
                    "importance_score": meta.get("importance_score", 0.0),
                    "weight": meta.get("weight", 1.0),
                    "title": meta.get("title", ""),
                    "abstract": meta.get("abstract", ""),
                    "claims": meta.get("claims", ""),
                },
            ))
        
        return results
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        normalize: bool = True,
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense (FAISS) and sparse (BM25) results using RRF.
        
        Args:
            query_embedding: Dense query vector
            query_text: Original query text for BM25
            top_k: Number of results to return
            dense_weight: Weight for dense search in RRF
            sparse_weight: Weight for sparse search in RRF
            rrf_k: RRF constant (default 60)
            normalize: Normalize query embedding
            
        Returns:
            List of SearchResult objects sorted by RRF score
        """
        # Dense search
        dense_results = self.search(query_embedding, top_k=top_k * 2, normalize=normalize)
        
        # Sparse search
        sparse_raw = self.bm25_engine.search(query_text, top_k=top_k * 2)
        
        # RRF Fusion
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_data: Dict[str, SearchResult] = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            rrf_scores[result.chunk_id] += dense_weight / (rrf_k + rank + 1)
            result.dense_score = result.score
            chunk_data[result.chunk_id] = result
        
        # Process sparse results
        for rank, (chunk_id, score, meta) in enumerate(sparse_raw):
            rrf_scores[chunk_id] += sparse_weight / (rrf_k + rank + 1)
            
            if chunk_id not in chunk_data:
                # Create SearchResult from BM25 result
                chunk_data[chunk_id] = SearchResult(
                    chunk_id=chunk_id,
                    patent_id=meta.get("patent_id", ""),
                    score=0.0,
                    content=meta.get("content", ""),
                    content_type=meta.get("content_type", ""),
                    sparse_score=score,
                    metadata={
                        "ipc_code": meta.get("ipc_code", ""),
                        "importance_score": meta.get("importance_score", 0.0),
                        "title": meta.get("title", ""),
                        "abstract": meta.get("abstract", ""),
                        "claims": meta.get("claims", ""),
                    },
                )
            else:
                chunk_data[chunk_id].sparse_score = score
        
        # Sort by RRF score and update results
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for chunk_id, rrf_score in sorted_ids[:top_k]:
            if chunk_id in chunk_data:
                result = chunk_data[chunk_id]
                result.rrf_score = rrf_score
                result.score = rrf_score  # Use RRF score as primary score
                final_results.append(result)
        
        logger.info(f"Hybrid search: {len(dense_results)} dense + {len(sparse_raw)} sparse -> {len(final_results)} fused")
        
        return final_results
    
    async def async_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        normalize: bool = True,
    ) -> List[SearchResult]:
        """Async wrapper for dense search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(query_embedding, top_k, normalize)
        )
    
    async def async_hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[SearchResult]:
        """Async wrapper for hybrid search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.hybrid_search(
                query_embedding, query_text, top_k, dense_weight, sparse_weight
            )
        )
    
    def save_local(self, index_path: Path = None, metadata_path: Path = None) -> None:
        """Save FAISS index, metadata, and BM25 index to disk."""
        index_path = index_path or self.config.index_path
        metadata_path = metadata_path or self.config.metadata_path
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path} ({self.index.ntotal} vectors)")
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                "metadata": self.metadata,
                "id_to_idx": self.id_to_idx,
            }, f)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save BM25 index
        self.bm25_engine.save_local(self.bm25_path)
    
    def load_local(self, index_path: Path = None, metadata_path: Path = None) -> bool:
        """Load FAISS index, metadata, and BM25 index from disk."""
        index_path = index_path or self.config.index_path
        metadata_path = metadata_path or self.config.metadata_path
        
        if not index_path.exists():
            logger.warning(f"Index file not found: {index_path}")
            return False
        
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index from {index_path} ({self.index.ntotal} vectors)")
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get("metadata", {})
                self.id_to_idx = data.get("id_to_idx", {})
            logger.info(f"Loaded metadata from {metadata_path}")
            
            # Load BM25 index
            if self.bm25_path.exists():
                self.bm25_engine.load_local(self.bm25_path)
            else:
                # Rebuild BM25 from metadata
                docs = list(self.metadata.values())
                if docs:
                    self.bm25_engine.build_index(docs, text_key="content", id_key="chunk_id")
            
            self._loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.index is None:
            return {"initialized": False, "total_vectors": 0}
        
        return {
            "initialized": True,
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": type(self.index).__name__,
            "metadata_count": len(self.metadata),
            "bm25_initialized": self.bm25_engine._initialized,
            "bm25_docs": len(self.bm25_engine.corpus) if self.bm25_engine._initialized else 0,
        }


# =============================================================================
# Pinecone Client with Hybrid Search
# =============================================================================

class PineconeClient:
    """
    Pinecone-based vector database (Serverless) with Hybrid Search.
    
    Features:
    - Dense search using Pinecone (Serverless)
    - Sparse search using local BM25
    - RRF Fusion for result merging
    - Batch upsert logic
    """
    
    def __init__(
        self,
        pinecone_config: PineconeConfig = None,
        embedding_dim: int = None,
    ):
        if not PINECONE_AVAILABLE:
            raise ImportError("pinecone is required. Install with: pip install pinecone>=3.0.0")
        
        self.config = pinecone_config or config.pinecone
        self.embedding_dim = embedding_dim or config.embedding.embedding_dim
        
        # Initialize Pinecone
        if not self.config.api_key:
            raise ValueError("PINECONE_API_KEY not set")
            
        self.pc = Pinecone(api_key=self.config.api_key)
        
        # Check/Create Index
        self._ensure_index_exists()
        
        self.index = self.pc.Index(self.config.index_name)
        
        # Setup Local BM25
        self.bm25_engine = BM25SearchEngine()
        # We reuse FAISS metadata path for consistency or define a new one?
        # Let's use a specific path to avoid conflict/overwrite of FAISS data if running side-by-side
        self.metadata_path = INDEX_DIR / "pinecone_metadata.pkl"
        self.bm25_path = INDEX_DIR / "pinecone_bm25.pkl"
        
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        
        logger.info(f"Pinecone Client initialized (index={self.config.index_name})")

    def _ensure_index_exists(self):
        """Check if index exists, create if not (Serverless)."""
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        
        if self.config.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index '{self.config.index_name}'...")
            try:
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=ServerlessSpec(
                        cloud=self.config.cloud,
                        region=self.config.region
                    )
                )
                logger.info(f"Index '{self.config.index_name}' created successfully")
            except Exception as e:
                logger.error(f"Failed to create Pinecone index: {e}")
                raise

    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        normalize: bool = False,  # Pinecone cosine metric usually handles normalized vectors better, but check metric
    ) -> int:
        """
        Batch add vectors to Pinecone and update local BM25.
        """
        if normalize and self.config.metric == 'cosine':
             # Normalize embeddings to unit length for cosine similarity
             # (Though Pinecone 'cosine' does normalization automatically, it's safe to do valid L2 norm)
             pass 
             # Only strictly needed if metric is dotproduct acting as cosine
        
        total = len(embeddings)
        batch_size = self.config.batch_size
        
        logger.info(f"Upserting {total} vectors to Pinecone (batch_size={batch_size})...")
        
        # Prepare for BM25
        # We can update self.metadata incrementally or rebuild. 
        # Here we assume batch/initial load pattern.
        
        upsert_count = 0
        
        # Use tqdm for progress visibility
        from tqdm import tqdm
        for i in tqdm(range(0, total, batch_size), desc="Upserting to Pinecone", unit="batch"):
            batch_vectors = embeddings[i : i + batch_size]
            batch_meta = metadata_list[i : i + batch_size]
            
            vectors_to_upsert = []
            for j, (vec, meta) in enumerate(zip(batch_vectors, batch_meta)):
                # chunk_id should be unique. 
                chunk_id = meta.get("chunk_id", f"chk_{i+j}")
                
                # Update local metadata
                self.metadata[chunk_id] = meta
                
                # Prepare Pinecone vector
                # Flatten metadata for Pinecone (limitations on nested dicts?)
                # Pinecone supports basic types. Ensure robust types.
                
                # Metadata size limit check (40KB per vector)
                content_text = meta.get("content", "")
                # Safe truncate to ~30KB to allow room for other fields
                if len(content_text.encode('utf-8')) > 30000:
                    # Simple truncation by character may not match byte size perfectly but good enough approximation
                    # 10000 chars is usually safe even with multi-byte chars
                    content_text = content_text[:10000]

                flat_meta = {
                    "text": content_text,
                    "title": (meta.get("title", "") or "")[:1000],  # Truncate title
                    "patent_id": meta.get("patent_id", ""),
                    "ipc_code": (meta.get("ipc_codes") or [""])[0] if isinstance(meta.get("ipc_codes"), list) else str(meta.get("ipc_codes", "")),
                    # Add other fields as needed, keep it lightweight
                }
                
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": vec.tolist(),
                    "metadata": flat_meta
                })
            
            # Upsert batch
            try:
                self.index.upsert(vectors=vectors_to_upsert, namespace=self.config.namespace)
                upsert_count += len(vectors_to_upsert)
            except Exception as e:
                logger.error(f"Pinecone upsert failed at batch {i}: {e}")
                # Continue or raise? Usually raise to ensure integrity
                raise e
        
        logger.info(f"Upserted {upsert_count} vectors to Pinecone")
        
        # Build/Update BM25
        # Note: If we are adding incrementally, rebuilding BM25 every time is slow.
        # But assuming build_index_from_patents uses this once.
        docs = list(self.metadata.values())
        self.bm25_engine.build_index(docs, text_key="content", id_key="chunk_id")
        
        return upsert_count

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        normalize: bool = False, # Handled by Pinecone usually
    ) -> List[SearchResult]:
        """
        Dense search using Pinecone.
        """
        if query_embedding.ndim > 1:
            query_embedding = query_embedding[0] # Take first if batch
            
        try:
            response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                namespace=self.config.namespace
            )
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []
            
        results = []
        for match in response['matches']:
            meta = match['metadata'] if match.get('metadata') else {}
            chunk_id = match['id']
            score = match['score']
            
            # Local metadata might have more details (claims, etc)
            local_meta = self.metadata.get(chunk_id, {})
            
            # Prefer local metadata for full context if avail, else fallback to Pinecone meta
            content = local_meta.get("content") or meta.get("text", "")
            patent_id = local_meta.get("patent_id") or meta.get("patent_id", "")
            
            results.append(SearchResult(
                chunk_id=chunk_id,
                patent_id=patent_id,
                score=score,
                content=content,
                content_type=local_meta.get("content_type", "unknown"),
                dense_score=score,
                metadata=local_meta or meta
            ))
            
        return results

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        normalize: bool = True,
    ) -> List[SearchResult]:
        """
        Hybrid search (Pinecone Dense + Local BM25) with RRF.
        Identical logic to FaissClient.hybrid_search.
        """
        # Dense search (Pinecone)
        dense_results = self.search(query_embedding, top_k=top_k * 2)
        
        # Sparse search (Local BM25)
        sparse_raw = self.bm25_engine.search(query_text, top_k=top_k * 2)
        
        # RRF Fusion
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_data: Dict[str, SearchResult] = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            rrf_scores[result.chunk_id] += dense_weight / (rrf_k + rank + 1)
            result.dense_score = result.score
            chunk_data[result.chunk_id] = result
        
        # Process sparse results
        for rank, (chunk_id, score, meta) in enumerate(sparse_raw):
            rrf_scores[chunk_id] += sparse_weight / (rrf_k + rank + 1)
            
            if chunk_id not in chunk_data:
                # Create SearchResult from BM25 result
                chunk_data[chunk_id] = SearchResult(
                    chunk_id=chunk_id,
                    patent_id=meta.get("patent_id", ""),
                    score=0.0,
                    content=meta.get("content", ""),
                    content_type=meta.get("content_type", ""),
                    sparse_score=score,
                    metadata={
                        "ipc_code": meta.get("ipc_code", ""),
                        "importance_score": meta.get("importance_score", 0.0),
                        "title": meta.get("title", ""),
                        "abstract": meta.get("abstract", ""),
                        "claims": meta.get("claims", ""),
                    },
                )
            else:
                chunk_data[chunk_id].sparse_score = score
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for chunk_id, rrf_score in sorted_ids[:top_k]:
            if chunk_id in chunk_data:
                result = chunk_data[chunk_id]
                result.rrf_score = rrf_score
                result.score = rrf_score
                final_results.append(result)
        
        logger.info(f"Pinecone Hybrid: {len(dense_results)} dense + {len(sparse_raw)} sparse -> {len(final_results)} fused")
        
        return final_results

    async def async_search(self, *args, **kwargs):
        """Async wrapper."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search(*args, **kwargs))

    async def async_hybrid_search(self, *args, **kwargs):
        """Async wrapper."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.hybrid_search(*args, **kwargs))

    def save_local(self) -> None:
        """Save local metadata and BM25 index."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved Pinecone metadata cache to {self.metadata_path}")
        
        self.bm25_engine.save_local(self.bm25_path)

    def load_local(self) -> bool:
        """Load local metadata and BM25 index."""
        if not self.metadata_path.exists():
            return False
            
        try:
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded Pinecone metadata cache ({len(self.metadata)} items)")
            
            if self.bm25_path.exists():
                self.bm25_engine.load_local(self.bm25_path)
            
            self._loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load Pinecone local cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get index stats."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "type": "pinecone", 
                "total_vectors": stats.get('total_vector_count', 0),
                "bm25_docs": len(self.bm25_engine.corpus)
            }
        except:
            return {"type": "pinecone", "error": "stats_failed"}


# =============================================================================
# Keyword Extractor
# =============================================================================

class KeywordExtractor:
    """
    Extract keywords from query text for BM25 search.
    """
    
    # Common stop words to filter out
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'whose',
    }
    
    # Technical terms to boost
    TECHNICAL_TERMS = {
        'method', 'system', 'apparatus', 'device', 'process', 'machine',
        'algorithm', 'model', 'network', 'layer', 'module', 'component',
        'database', 'index', 'vector', 'embedding', 'retrieval', 'search',
        'query', 'document', 'text', 'language', 'neural', 'learning',
        'training', 'inference', 'classification', 'clustering', 'ranking',
        'generation', 'processing', 'analysis', 'extraction', 'recognition',
    }
    
    @classmethod
    def extract(cls, text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords sorted by importance
        """
        if not text:
            return []
        
        # Tokenize
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text.lower())
        
        # Filter stop words and short words
        filtered = [w for w in words if w not in cls.STOP_WORDS and len(w) > 2]
        
        # Count frequency
        word_freq = defaultdict(int)
        for word in filtered:
            word_freq[word] += 1
        
        # Score words (frequency + technical term boost)
        scored = []
        for word, freq in word_freq.items():
            score = freq
            if word in cls.TECHNICAL_TERMS:
                score *= 2  # Boost technical terms
            scored.append((word, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in scored[:max_keywords]]


# =============================================================================
# High-Level Operations
# =============================================================================

async def build_index_from_patents(
    processed_patents: List[Dict[str, Any]],
    embedder,
    faiss_client: FaissClient = None,
    save_to_disk: bool = True,
) -> InsertResult:
    """Build FAISS + BM25 hybrid index from processed patents."""
    if faiss_client is None:
        faiss_client = FaissClient()
    
    logger.info(f"Building hybrid index from {len(processed_patents)} patents...")
    
    all_chunks = []
    for patent in tqdm(processed_patents, desc="Extracting chunks"):
        patent_metadata = {
            "patent_id": patent.get("publication_number", ""),
            "title": patent.get("title", ""),
            "abstract": patent.get("abstract", ""),
            "ipc_codes": patent.get("ipc_codes", []),
            "importance_score": patent.get("importance_score", 0.0),
        }
        
        claims = patent.get("claims", [])
        claims_text = ""
        if claims and isinstance(claims[0], dict):
            claims_text = claims[0].get("claim_text", "")
        
        for chunk in patent.get("chunks", []):
            chunk_data = {
                "chunk_id": chunk.get("chunk_id", ""),
                "patent_id": patent_metadata["patent_id"],
                "content": chunk.get("content", ""),
                "content_type": chunk.get("chunk_type", "description"),
                "ipc_code": (patent.get("ipc_codes") or [""])[0][:20],
                "importance_score": patent_metadata["importance_score"],
                "weight": 1.0,
                "title": patent_metadata["title"],
                "abstract": patent_metadata["abstract"][:500] if patent_metadata["abstract"] else "",
                "claims": claims_text[:1000] if claims_text else "",
            }
            all_chunks.append(chunk_data)
    
    logger.info(f"Total chunks to index: {len(all_chunks)}")
    
    if not all_chunks:
        return InsertResult(
            success=False,
            inserted_count=0,
            index_path=str(faiss_client.config.index_path),
            error_message="No chunks to index",
        )
    
    try:
        # Generate embeddings
        logger.info("Generating embeddings...")
        embedding_results = await embedder.embed_patent_chunks(all_chunks)
        
        embeddings = np.array([r.embedding for r in embedding_results])
        
        for i, result in enumerate(embedding_results):
            all_chunks[i]["weight"] = result.weight
        
        # Create and populate index (FAISS + BM25)
        faiss_client.create_index(use_cosine=True)
        faiss_client.add_vectors(embeddings, all_chunks)
        
        if save_to_disk:
            faiss_client.save_local()
        
        return InsertResult(
            success=True,
            inserted_count=len(all_chunks),
            index_path=str(faiss_client.config.index_path),
        )
        
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        return InsertResult(
            success=False,
            inserted_count=0,
            index_path=str(faiss_client.config.index_path),
            error_message=str(e),
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Test hybrid search operations."""
    logging.basicConfig(
        level=logging.INFO,
        format=config.logging.log_format,
    )
    
    print("\n" + "=" * 70)
    print("⚡ 쇼특허 (Short-Cut) v3.0 - Hybrid Search Test")
    print("=" * 70)
    
    if not FAISS_AVAILABLE:
        print("❌ faiss-cpu not installed")
        return
    
    if not BM25_AVAILABLE:
        print("❌ rank_bm25 not installed. Install with: pip install rank_bm25")
        return
    
    # Initialize client
    client = FaissClient()
    
    # Test data
    print("\n📦 Creating test index...")
    client.create_index(use_cosine=True)
    
    n_vectors = 100
    dim = config.embedding.embedding_dim
    test_embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
    
    test_metadata = [
        {
            "chunk_id": f"test_chunk_{i}",
            "patent_id": f"US-{1000000 + i}-A",
            "content": f"Method for neural network based retrieval system using vector embedding and semantic search technology claim {i}",
            "content_type": "claim" if i % 3 == 0 else "abstract",
            "ipc_code": "G06N3",
            "importance_score": float(i % 10),
            "title": f"Neural Network Retrieval System Patent {i}",
        }
        for i in range(n_vectors)
    ]
    
    print(f"📥 Adding {n_vectors} test vectors...")
    client.add_vectors(test_embeddings, test_metadata)
    
    stats = client.get_stats()
    print(f"📊 Index stats: {stats}")
    
    # Test hybrid search
    print("\n🔍 Testing hybrid search...")
    query_embedding = test_embeddings[0]
    query_text = "neural network semantic search retrieval"
    
    results = client.hybrid_search(query_embedding, query_text, top_k=5)
    
    print(f"   Found: {len(results)} results")
    for r in results[:5]:
        print(f"   - {r.patent_id}: RRF={r.rrf_score:.4f} (dense={r.dense_score:.4f}, sparse={r.sparse_score:.4f})")
    
    # Test keyword extraction
    print("\n🔑 Testing keyword extraction...")
    keywords = KeywordExtractor.extract(query_text)
    print(f"   Keywords: {keywords}")
    
    print("\n" + "=" * 70)
    print("✅ Hybrid search test complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
