"""
Chroma-backed vector memory store.

Production-shaped persistent vector DB with multi-tenancy.
Same interface as SQLiteVectorStore / VectorMemoryStore so the agent
can swap backends without code changes.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import chromadb
from chromadb.config import Settings

from .vector_store import get_embedder, SimpleEmbedder


class ChromaVectorStore:
    """Persistent Chroma collection for companion memories."""

    def __init__(
        self,
        persist_dir: str = "memory_store/chroma",
        collection_name: str = "companion_memories",
        embedder: Optional[Any] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_embedder()

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as e:
            print(f"[chroma] PersistentClient failed ({e}). Falling back to in-memory.")
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _make_id(self, user_id: str, character_id: str, text: str) -> str:
        raw = f"{user_id}:{character_id}:{text[:120]}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def add(
        self,
        text: str,
        user_id: str,
        character_id: str,
        kind: str = "fact",
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> str:
        text = text.strip()
        if not text:
            return ""

        item_id = self._make_id(user_id, character_id, text)
        embedding = self.embedder.embed([text])[0]

        meta = {
            "user_id": user_id,
            "character_id": character_id,
            "kind": kind,
            "importance": float(importance),
            "created_at": datetime.utcnow().isoformat(),
        }
        if metadata:
            # Chroma metadata values must be primitive
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v

        self.collection.upsert(
            ids=[item_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )
        return item_id

    def search(
        self,
        query: str,
        user_id: str,
        character_id: str,
        limit: int = 6,
        min_score: float = 0.15,
        kind_filter: Optional[str] = None,
    ) -> List[Dict]:
        if not query:
            return self.get_all(user_id, character_id, limit=limit)

        query_emb = self.embedder.embed_query(query)

        where = {
            "$and": [
                {"user_id": user_id},
                {"character_id": character_id},
            ]
        }
        if kind_filter:
            where["$and"].append({"kind": kind_filter})

        try:
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma cosine distance → similarity
            sim = 1.0 - float(dist) if dist is not None else 0.0
            importance = float(meta.get("importance", 0.5)) if meta else 0.5
            combined = 0.75 * sim + 0.25 * importance
            if combined >= min_score:
                hits.append({
                    "text": doc,
                    "score": round(combined, 4),
                    "kind": meta.get("kind", "fact") if meta else "fact",
                    "importance": importance,
                })
        return hits

    def get_all(self, user_id: str, character_id: str, limit: int = 50) -> List[str]:
        try:
            results = self.collection.get(
                where={"$and": [{"user_id": user_id}, {"character_id": character_id}]},
                include=["documents"],
                limit=limit,
            )
            return results.get("documents", []) or []
        except Exception:
            return []

    def clear(self, user_id: str, character_id: str):
        try:
            results = self.collection.get(
                where={"$and": [{"user_id": user_id}, {"character_id": character_id}]},
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
        except Exception:
            pass
