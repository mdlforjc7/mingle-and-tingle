"""
LangGraph StateGraph implementation of Chaterbait.

Enhanced flow:
  START → retrieve_memories → generate_response → route
                                                      ├─ (has new personal info) → extract_facts → update_relationship → END
                                                      └─ (otherwise)            → update_relationship → END

Uses the existing personality cards, CoreMemory, and VectorMemoryStore.
"""

from __future__ import annotations

from typing import Annotated, TypedDict, List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import re

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from characters import get_character, list_characters
from memory import CoreMemory, LongTermMemory
from memory.sqlite_store import SQLiteVectorStore
from memory.chroma_store import ChromaVectorStore
from llm import get_llm, BaseLLM


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CompanionState(TypedDict):
    """Shared state that travels through the graph."""
    messages: Annotated[list, add_messages]   # conversation history
    user_id: str
    character_id: str
    retrieved_memories: list                  # semantic hits this turn
    core_summary: str                         # short relationship blurb
    last_user_message: str                    # convenience for nodes
    turn_count: int                           # how many user turns so far
    should_extract: bool                      # set by router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PERSONAL_INFO_PATTERNS = [
    r"\bmy name is\b",
    r"\bi live\b",
    r"\bi work\b",
    r"\bi have a\b",
    r"\bmy dog\b",
    r"\bmy cat\b",
    r"\bmy pet\b",
    r"\bi'?m from\b",
    r"\bi am from\b",
    r"\bi like\b",
    r"\bi love\b",
    r"\bi hate\b",
    r"\bmy job\b",
    r"\bmy girlfriend\b",
    r"\bmy boyfriend\b",
    r"\bmy partner\b",
    r"\bi'?m a\b",
    r"\bi am a\b",
    r"\bmy age\b",
    r"\bi'?m \d+\b",
    r"\bi study\b",
    r"\bi graduated\b",
    r"\bmy birthday\b",
    r"\bmy favorite\b",
    r"\bi prefer\b",
    r"\bi used to\b",
    r"\bmy family\b",
    r"\bmy mom\b",
    r"\bmy dad\b",
    r"\bbreakup\b",
    r"\bex-\b",
]

def looks_like_personal_info(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in PERSONAL_INFO_PATTERNS)


def _msg_role_content(m, character_name: str):
    """Normalize both LangChain messages and plain dicts."""
    if hasattr(m, "type"):  # LangChain BaseMessage
        role = "User" if m.type == "human" else character_name
        content = getattr(m, "content", "") or ""
    elif isinstance(m, dict):
        role = "User" if m.get("role") == "user" else character_name
        content = m.get("content", "")
    else:
        role, content = character_name, str(m)
    return role, content


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def make_retrieve_node(ltm: LongTermMemory, character: dict):
    def retrieve_memories(state: CompanionState) -> dict:
        query = state.get("last_user_message") or ""
        if not query and state.get("messages"):
            last = state["messages"][-1]
            _, query = _msg_role_content(last, character["name"])

        memories = ltm.get_relevant(
            user_id=state["user_id"],
            character_id=state["character_id"],
            query=query,
            limit=6,
        )
        return {"retrieved_memories": memories}
    return retrieve_memories


def make_generate_node(llm: BaseLLM, character: dict, core: CoreMemory):
    def generate_response(state: CompanionState) -> dict:
        char = character
        traits = "\n".join(f"- {t}" for t in char["personality_traits"])
        values = "\n".join(f"{i+1}. {v}" for i, v in enumerate(char["values_ranked"]))
        flaws = "\n".join(f"- {f}" for f in char["flaws"])

        hist_lines = []
        for m in state.get("messages", [])[-8:]:
            role, content = _msg_role_content(m, char["name"])
            hist_lines.append(f"{role}: {content}")
        history_text = "\n".join(hist_lines) if hist_lines else "(no prior messages)"

        ltm_texts = state.get("retrieved_memories") or []
        ltm_block = "\n".join(f"- {t}" for t in ltm_texts) if ltm_texts else "- (none yet)"

        # Prefer live core summary if present
        core_block = core.to_prompt_block()
        if state.get("core_summary"):
            # Light override of the relationship line for this turn
            pass

        system_prompt = char["system_prompt_template"].format(
            name=char["name"],
            age=char["age"],
            role=char["role"],
            core_identity=char["core_identity"],
            traits=traits,
            speech_style=char["speech_style"],
            values=values,
            flaws=flaws,
            relationship_stance=char["relationship_stance"],
            core_memory=core_block,
            long_term_memories=ltm_block,
            history=history_text,
        )

        llm_messages = [{"role": "system", "content": system_prompt}]
        for m in state.get("messages", [])[-6:]:
            if hasattr(m, "type"):
                role = "user" if m.type == "human" else "assistant"
                content = getattr(m, "content", "") or ""
            elif isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            else:
                role, content = "assistant", str(m)
            llm_messages.append({"role": role, "content": content})

        reply_text = llm.chat(llm_messages)

        from langchain_core.messages import AIMessage
        turn_count = state.get("turn_count", 0) + 1
        return {
            "messages": [AIMessage(content=reply_text)],
            "turn_count": turn_count,
        }
    return generate_response


def _extract_with_llm(llm, user_text: str) -> list[dict]:
    """
    Ask the LLM to pull 0–3 short personal facts from the user message.
    Returns a list of {text, kind, importance}. Empty list on failure.
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "You extract personal facts about the USER from a single message. "
                "Return ONLY a JSON array of objects. Each object must have: "
                '"text" (short phrase, max 120 chars), '
                '"kind" (one of: fact, preference, relationship, emotion), '
                '"importance" (float 0.3–1.0). '
                "If nothing personal is shared, return []. "
                "Do not invent information. Do not include the character's thoughts."
            ),
        },
        {"role": "user", "content": user_text},
    ]
    try:
        raw = llm.chat(prompt, temperature=0.1)
        # Best-effort JSON extract
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            return []
        import json
        data = json.loads(raw[start : end + 1])
        out = []
        for item in data[:3]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text or len(text) > 140:
                continue
            kind = str(item.get("kind", "fact")).lower()
            if kind not in ("fact", "preference", "relationship", "emotion"):
                kind = "fact"
            try:
                imp = float(item.get("importance", 0.6))
            except Exception:
                imp = 0.6
            imp = max(0.3, min(1.0, imp))
            out.append({"text": text, "kind": kind, "importance": imp})
        return out
    except Exception:
        return []


def make_extract_node(ltm: LongTermMemory, core: CoreMemory, character: dict, llm=None):
    """
    Extraction node.
    - If a real LLM is available, ask it for structured facts.
    - Always fall back to the regex heuristic so the prototype works offline.
    """
    def extract_facts(state: CompanionState) -> dict:
        last_user = state.get("last_user_message") or ""
        if not last_user and state.get("messages"):
            for m in reversed(state["messages"]):
                role, content = _msg_role_content(m, character["name"])
                if role == "User":
                    last_user = content
                    break

        if not last_user:
            return {}

        facts_to_store = []

        # Try LLM extraction first when we have a non-mock LLM
        use_llm = llm is not None and type(llm).__name__ != "MockLLM"
        if use_llm:
            facts_to_store = _extract_with_llm(llm, last_user)

        # Fallback / supplement with regex when the message looks personal
        if not facts_to_store and looks_like_personal_info(last_user):
            snippet = last_user.strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            facts_to_store = [{"text": snippet, "kind": "fact", "importance": 0.75}]

        for f in facts_to_store:
            ltm.add_memory(
                state["user_id"],
                state["character_id"],
                f["text"],
                kind=f["kind"],
                importance=f["importance"],
            )
            core.add_user_fact(f["text"])
        return {}
    return extract_facts


def make_relationship_node(core: CoreMemory, character: dict):
    """
    Lightweight relationship-summary updater.
    In production this would call a small LLM; here we use simple heuristics
    so the prototype stays offline-friendly.
    """
    def update_relationship(state: CompanionState) -> dict:
        turn = state.get("turn_count", 0)
        facts = core.key_user_facts
        name = character["name"]

        if turn <= 1:
            summary = "Just started talking. No strong dynamic established yet."
        elif turn <= 4:
            summary = f"Early conversation with {name}. Still getting to know each other."
        elif len(facts) >= 3:
            summary = (
                f"Building rapport with {name}. "
                f"Several personal details shared ({len(facts)} key facts remembered)."
            )
        else:
            summary = f"Ongoing conversation with {name}. Comfortable, still light."

        # Only write when it actually changes
        if summary != core.relationship_summary:
            core.update_relationship_summary(summary)

        return {"core_summary": summary}
    return update_relationship


# ---------------------------------------------------------------------------
# Conditional router
# ---------------------------------------------------------------------------

def route_after_generate(state: CompanionState) -> str:
    """
    Decide whether to run the extraction node.
    Runs extraction when:
      - the latest user message looks like personal info, OR
      - every 4th turn (periodic light pass)
    """
    last_user = state.get("last_user_message") or ""
    turn = state.get("turn_count", 0)

    if looks_like_personal_info(last_user):
        return "extract"
    if turn > 0 and turn % 4 == 0:
        return "extract"
    return "update_relationship"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_companion_graph(
    character_name: str,
    user_id: str = "default_user",
    llm: Optional[BaseLLM] = None,
    storage_dir: str = "memory_store",
    checkpointer=None,
    backend: str = "sqlite",  # "chroma" | "sqlite" | "memory"
):
    """
    Build and compile a LangGraph StateGraph for the given character.
    """
    character = get_character(character_name)
    llm = llm or get_llm()
    core = CoreMemory(character, user_id)
    # Backend selection: "chroma" | "sqlite" | "memory"
    if backend == "chroma":
        raw = ChromaVectorStore(persist_dir=str(Path(storage_dir) / "chroma"))
    elif backend == "sqlite":
        raw = SQLiteVectorStore(db_path=str(Path(storage_dir) / "memories.db") if not str(storage_dir).endswith(".db") else storage_dir)
    else:
        raw = None

    if raw is not None:
        class _LTMAdapter:
            def __init__(self, store):
                self._s = store
            def add_memory(self, user_id, character_id, text, kind="fact", importance=0.5):
                return self._s.add(text, user_id, character_id, kind=kind, importance=importance)
            def get_relevant(self, user_id, character_id, query="", limit=6):
                hits = self._s.search(query or "", user_id, character_id, limit=limit)
                return [h["text"] for h in hits]
            def get_all(self, user_id, character_id, limit=50):
                return self._s.get_all(user_id, character_id, limit=limit)
        ltm = _LTMAdapter(raw)
    else:
        ltm = LongTermMemory(storage_dir)

    # Bootstrap core with a few existing memories
    existing = ltm.get_relevant(user_id, character["id"], limit=4)
    for fact in existing:
        core.add_user_fact(fact)

    retrieve = make_retrieve_node(ltm, character)
    generate = make_generate_node(llm, character, core)
    extract = make_extract_node(ltm, core, character, llm=llm)
    update_rel = make_relationship_node(core, character)

    builder = StateGraph(CompanionState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("extract", extract)
    builder.add_node("update_relationship", update_rel)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")

    # Conditional edge after generate
    builder.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "extract": "extract",
            "update_relationship": "update_relationship",
        },
    )
    builder.add_edge("extract", "update_relationship")
    builder.add_edge("update_relationship", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)

    # Attach helpers for the CLI / callers
    graph._character = character
    graph._core = core
    graph._ltm = ltm
    graph._user_id = user_id
    return graph


def chat_with_graph(graph, user_message: str, thread_id: str = "default") -> str:
    """
    Convenience helper: send one user message and get the assistant reply.
    """
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id}}

    # Read current turn count from checkpoint if possible
    turn_count = 0
    try:
        state = graph.get_state(config)
        if state and state.values:
            turn_count = state.values.get("turn_count", 0)
    except Exception:
        pass

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "user_id": graph._user_id,
            "character_id": graph._character["id"],
            "last_user_message": user_message,
            "retrieved_memories": [],
            "core_summary": graph._core.relationship_summary,
            "turn_count": turn_count,
            "should_extract": False,
        },
        config=config,
    )

    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return last.content if hasattr(last, "content") else last.get("content", "")
    return ""
