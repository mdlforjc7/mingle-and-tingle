"""SQLite-backed vector memory store (persistent)."""

from __future__ import annotations
import json, hashlib, sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Dict
import numpy as np
from scipy.spatial.distance import cosine
from .vector_store import get_embedder

class SQLiteVectorStore:
    def __init__(self, db_path="memory_store/memories.db", embedder=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_embedder()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()
    def _init(self):
        self._conn.execute("""CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY, user_id TEXT, character_id TEXT, text TEXT,
            kind TEXT, importance REAL, embedding BLOB, metadata TEXT,
            created_at TEXT, last_accessed TEXT)""")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_uc ON memories(user_id, character_id)")
        self._conn.commit()
    def add(self, text, user_id, character_id, kind="fact", importance=0.5, metadata=None):
        text = text.strip()
        if not text: return ""
        iid = hashlib.md5(f"{user_id}:{character_id}:{text[:80]}".encode()).hexdigest()[:16]
        emb = self.embedder.embed([text])[0]
        now = datetime.utcnow().isoformat()
        self._conn.execute("""INSERT OR REPLACE INTO memories
            (id, user_id, character_id, text, kind, importance, embedding, metadata, created_at, last_accessed)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (iid, user_id, character_id, text, kind, importance, json.dumps(emb), json.dumps(metadata or {}), now, now))
        self._conn.commit()
        return iid
    def search(self, query, user_id, character_id, limit=6, min_score=0.12):
        rows = self._conn.execute("SELECT * FROM memories WHERE user_id=? AND character_id=?", (user_id, character_id)).fetchall()
        if not rows: return []
        qv = np.array(self.embedder.embed_query(query), dtype=np.float32)
        scored = []
        for r in rows:
            vec = np.array(json.loads(r["embedding"]), dtype=np.float32)
            sim = float(1.0 - cosine(qv, vec))
            comb = 0.8*sim + 0.2*r["importance"]
            if comb >= min_score: scored.append((comb, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text":r["text"], "score":round(s,4), "kind":r["kind"], "importance":r["importance"]} for s,r in scored[:limit]]
    def get_all(self, user_id, character_id, limit=50):
        rows = self._conn.execute("SELECT text FROM memories WHERE user_id=? AND character_id=? ORDER BY created_at DESC LIMIT ?", (user_id, character_id, limit)).fetchall()
        return [r["text"] for r in rows]
