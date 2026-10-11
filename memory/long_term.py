"""Thin wrapper around the vector store for long-term memory."""

from __future__ import annotations
from .vector_store import VectorMemoryStore

class LongTermMemory:
    def __init__(self, storage_dir: str = "memory_store"):
        self.store = VectorMemoryStore(storage_path=f"{storage_dir}/vector_memories.json")

    def add_memory(self, user_id, character_id, text, kind="fact", importance=0.5):
        return self.store.add(text, user_id, character_id, kind=kind, importance=importance)

    def get_relevant(self, user_id, character_id, query="", limit=6):
        hits = self.store.search(query or "", user_id, character_id, limit=limit)
        return [h["text"] for h in hits]

    def get_all(self, user_id, character_id, limit=50):
        return self.store.get_all(user_id, character_id, limit=limit)
