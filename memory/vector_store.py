"""Improved offline TF-IDF vector store + optional OpenAI embeddings."""

from __future__ import annotations
import json, hashlib, math, re
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from scipy.spatial.distance import cosine

@dataclass
class MemoryItem:
    id: str
    text: str
    user_id: str
    character_id: str
    kind: str = "fact"
    importance: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, data): return cls(**{k:v for k,v in data.items() if k in cls.__dataclass_fields__})

class SimpleEmbedder:
    def __init__(self, dims: int = 256):
        self.dims = dims
        self.doc_freq = Counter()
        self.n_docs = 0
    def _tokenize(self, text: str):
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        stop = {"a","an","the","is","are","was","were","be","been","and","or","but","in","on","at","to","for","of","with","my","i","me","you","he","she","it","we","they","this","that","am","do","does","did","have","has","had","not","no","yes","so","if","as"}
        return [t for t in tokens if t not in stop and len(t)>1]
    def _hash_idx(self, tok): return int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % self.dims
    def _vectorize(self, tokens):
        tf = Counter(tokens)
        vec = np.zeros(self.dims, dtype=np.float32)
        for tok, c in tf.items():
            idx = self._hash_idx(tok)
            w = 1.0 + math.log(c)
            df = self.doc_freq.get(tok, 0)
            idf = math.log((1+self.n_docs)/(1+df)) + 1.0
            vec[idx] += w * idf
        return vec / (np.linalg.norm(vec) + 1e-9)
    def embed(self, texts):
        out = []
        for t in texts:
            toks = self._tokenize(t)
            for tok in set(toks): self.doc_freq[tok] += 1
            self.n_docs += 1
            out.append(self._vectorize(toks).tolist())
        return out
    def embed_query(self, text): return self._vectorize(self._tokenize(text)).tolist()

def get_embedder(prefer_real=True):
    return SimpleEmbedder()

class VectorMemoryStore:
    def __init__(self, embedder=None, storage_path="memory_store/vector_memories.json"):
        self.embedder = embedder or get_embedder()
        self.storage_path = Path(storage_path) if storage_path else None
        self.items = {}
        self._load()
    def _load(self):
        if not self.storage_path or not self.storage_path.exists(): return
        try:
            data = json.loads(self.storage_path.read_text())
            for d in data.get("items", []):
                item = MemoryItem.from_dict(d)
                self.items[item.id] = item
        except Exception: pass
    def _save(self):
        if not self.storage_path: return
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"items": [i.to_dict() for i in self.items.values()]}
            self.storage_path.write_text(json.dumps(payload, indent=2))
        except Exception: pass
    def _make_id(self, u, c, t): return hashlib.md5(f"{u}:{c}:{t[:80]}".encode()).hexdigest()[:16]
    def add(self, text, user_id, character_id, kind="fact", importance=0.5, metadata=None):
        text = text.strip()
        if not text: return ""
        iid = self._make_id(user_id, character_id, text)
        if iid in self.items:
            self.items[iid].importance = max(self.items[iid].importance, importance)
            self._save()
            return iid
        emb = self.embedder.embed([text])[0]
        item = MemoryItem(id=iid, text=text, user_id=user_id, character_id=character_id, kind=kind, importance=importance, embedding=emb, metadata=metadata or {})
        self.items[iid] = item
        self._save()
        return iid
    def search(self, query, user_id, character_id, limit=6, min_score=0.12, kind_filter=None):
        cands = [i for i in self.items.values() if i.user_id==user_id and i.character_id==character_id and i.embedding]
        if not cands: return []
        qv = np.array(self.embedder.embed_query(query), dtype=np.float32)
        scored = []
        for it in cands:
            sim = float(1.0 - cosine(qv, np.array(it.embedding, dtype=np.float32)))
            combined = 0.8*sim + 0.2*it.importance
            if combined >= min_score: scored.append((combined, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text":it.text, "score":round(s,4), "kind":it.kind, "importance":it.importance} for s,it in scored[:limit]]
    def get_all(self, user_id, character_id, limit=50):
        items = [i for i in self.items.values() if i.user_id==user_id and i.character_id==character_id]
        items.sort(key=lambda x: x.created_at, reverse=True)
        return [i.text for i in items[:limit]]
    def clear(self, user_id, character_id):
        for iid in [k for k,v in self.items.items() if v.user_id==user_id and v.character_id==character_id]:
            del self.items[iid]
        self._save()
