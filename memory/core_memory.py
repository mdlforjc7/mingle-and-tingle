"""Core (always-in-context) memory for a companion relationship."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class CoreMemory:
    character: dict
    user_id: str
    relationship_summary: str = "Just started talking. No strong dynamic established yet."
    key_user_facts: List[str] = field(default_factory=list)

    def add_user_fact(self, fact: str):
        fact = fact.strip()
        if fact and fact not in self.key_user_facts:
            self.key_user_facts.append(fact)
            if len(self.key_user_facts) > 12:
                self.key_user_facts = self.key_user_facts[-12:]

    def update_relationship_summary(self, summary: str):
        self.relationship_summary = summary.strip()

    def to_prompt_block(self) -> str:
        lines = [
            f"Character: {self.character['name']} ({self.character.get('gender','')}, {self.character.get('age','')})",
            f"Relationship summary: {self.relationship_summary}",
        ]
        if self.key_user_facts:
            lines.append("Key facts about the user:")
            for f in self.key_user_facts:
                lines.append(f"- {f}")
        else:
            lines.append("Key facts about the user: (none yet)")
        return "\n".join(lines)
