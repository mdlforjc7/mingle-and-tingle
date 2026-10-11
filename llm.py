"""LLM layer with rich offline MockLLM + optional OpenAI-compatible backend."""

from __future__ import annotations
import os
from typing import List, Dict, Optional

class BaseLLM:
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.8) -> str:
        raise NotImplementedError

class MockLLM(BaseLLM):
    def chat(self, messages, temperature=0.8):
        system = ""
        last_user = ""
        for m in messages:
            role = m.get("role", "")
            if role == "system":
                system = m.get("content", "")
            elif role == "user":
                last_user = m.get("content", "")
        memories = self._extract_memories(system)
        if "You are Elara" in system:
            return self._elara(last_user, memories)
        if "You are Sofia" in system:
            return self._sofia(last_user, memories)
        if "You are Kai" in system:
            return self._kai(last_user, memories)
        if "You are Marcus" in system:
            return self._marcus(last_user, memories)
        return "I'm here. Tell me more."

    def _extract_memories(self, system: str) -> List[str]:
        facts = []
        in_mem = False
        for line in system.splitlines():
            line = line.strip()
            if any(h in line.lower() for h in ("key facts about the user", "relevant long-term memories", "long-term memories", "core memory")):
                in_mem = True
                continue
            if in_mem and line.endswith(":") and not line.startswith("-"):
                in_mem = False
            if in_mem and line.startswith("- ") and len(line) > 4:
                f = line[2:].strip()
                if f and f != "(none yet)":
                    facts.append(f)
        return facts[:8]

    def _mentions(self, text, *words):
        lower = text.lower()
        return any(w in lower for w in words)

    def _has_memory_about(self, memories, *keywords):
        for m in memories:
            if any(k in m.lower() for k in keywords):
                return m
        return None

    def _elara(self, user, memories):
        dog = self._has_memory_about(memories, "dog", "baxter", "pet", "cat")
        if dog and self._mentions(user, "dog", "baxter", "pet", "how is", "how's"):
            return "Baxter, right? You mentioned him. Hope he’s been keeping you company."
        if self._mentions(user, "my dog", "named baxter", "dog is named"):
            return "A dog named Baxter. Got it. Golden retriever energy fits."
        if self._mentions(user, "hi", "hello", "hey"):
            return "Hey. Didn’t expect you this early. What’s going on?"
        if self._mentions(user, "tired", "exhausted", "long day"):
            return "Yeah… I know that feeling. Want to just sit with it for a bit?"
        if "?" in user:
            return "Hmm. Fair question. Give me a second to actually think about it."
        return "Okay. I’m still here. Don’t stop on my account."

    def _sofia(self, user, memories):
        dog = self._has_memory_about(memories, "dog", "baxter", "pet")
        if dog and self._mentions(user, "dog", "baxter", "pet", "how is"):
            return "Baxter!! I’ve been thinking about that golden fluff. How’s he doing?"
        if self._mentions(user, "my dog", "named baxter"):
            return "Wait — Baxter? That’s such a good name. I already like him."
        if self._mentions(user, "hi", "hello", "hey"):
            return "Hey you! Perfect timing. How’s your day treating you?"
        if self._mentions(user, "sad", "rough", "bad day"):
            return "Hey… come here. You don’t have to carry that alone right now."
        return "That sounds kind of amazing. Tell me more."

    def _kai(self, user, memories):
        if self._mentions(user, "how are you", "how's it going"):
            return "Functional. You?"
        if self._mentions(user, "miss", "love", "like you"):
            return "You’re really going for it today. …I’m still here."
        if self._mentions(user, "hi", "hello"):
            return "You’re here. Talk."
        return "Noted. Continue."

    def _marcus(self, user, memories):
        if self._mentions(user, "lost", "confused", "stuck"):
            return "That’s an honest place to be. What’s the part that feels heaviest right now?"
        if self._mentions(user, "hi", "hello"):
            return "Good to see you. How are you actually doing?"
        if "?" in user:
            return "Good question. Let me answer it properly."
        return "I’m listening. Take your time."

class OpenAICompatibleLLM(BaseLLM):
    def __init__(self):
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not key:
            raise ValueError("No API key")
        self.client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    def chat(self, messages, temperature=0.85):
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature, max_tokens=400)
        return resp.choices[0].message.content.strip()

def get_llm():
    if os.getenv("OPENAI_API_KEY") or os.getenv("GROK_API_KEY"):
        try:
            return OpenAICompatibleLLM()
        except Exception as e:
            print(f"[llm] real backend failed: {e}")
    return MockLLM()
