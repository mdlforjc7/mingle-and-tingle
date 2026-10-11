"""
Robust FastAPI web UI for the AI Companion.

- In-memory session store with automatic cleanup
- Character selection
- Session continuity
- Memory inspection
- Works with chroma / sqlite / memory backends
"""

from __future__ import annotations

import uuid
import time
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from characters import list_characters, get_character
from agents.langgraph_agent import build_companion_graph, chat_with_graph


# ---------------------------------------------------------------------------
# Session store with TTL
# ---------------------------------------------------------------------------

_sessions: Dict[str, dict] = {}
SESSION_TTL = 60 * 60 * 4  # 4 hours


def _cleanup_sessions():
    now = time.time()
    dead = [k for k, v in _sessions.items() if now - v.get("created", 0) > SESSION_TTL]
    for k in dead:
        _sessions.pop(k, None)


def get_or_create_graph(session_id: str, character_id: str, backend: str = "sqlite"):
    _cleanup_sessions()
    key = f"{session_id}:{character_id}"
    if key not in _sessions:
        graph = build_companion_graph(
            character_name=character_id,
            user_id=f"web_{session_id[:8]}",
            storage_dir="memory_store",
            backend=backend,
        )
        _sessions[key] = {"graph": graph, "created": time.time()}
    return _sessions[key]["graph"]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI Companion API ready")
    print("Characters:", list_characters())
    yield
    print("Shutdown")


app = FastAPI(
    title="AI Companion",
    description="Entertainment AI companions with personality + memory",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    character_id: str = Field(..., description="elara | sofia | kai | marcus")
    session_id: Optional[str] = None
    backend: str = Field("sqlite", description="chroma | sqlite | memory")


class ChatResponse(BaseModel):
    reply: str
    character_id: str
    character_name: str
    session_id: str


@app.get("/api/characters")
def api_list_characters():
    out = []
    for cid in list_characters():
        c = get_character(cid)
        out.append({
            "id": c["id"],
            "name": c["name"],
            "gender": c.get("gender", ""),
            "age": c.get("age", 0),
            "role": c.get("role", ""),
        })
    return out


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    try:
        char = get_character(req.character_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session_id = req.session_id or str(uuid.uuid4())
    graph = get_or_create_graph(session_id, req.character_id, backend=req.backend)
    thread_id = f"{session_id}_{req.character_id}"

    try:
        reply = chat_with_graph(graph, req.message, thread_id=thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return ChatResponse(
        reply=reply,
        character_id=char["id"],
        character_name=char["name"],
        session_id=session_id,
    )


@app.get("/api/memory/{character_id}")
def api_memory(character_id: str, session_id: str = "default"):
    try:
        get_character(character_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    key = f"{session_id}:{character_id}"
    if key not in _sessions:
        return {"core_memory": "(no session)", "key_facts": [], "session_id": session_id}

    graph = _sessions[key]["graph"]
    return {
        "core_memory": graph._core.to_prompt_block(),
        "relationship_summary": graph._core.relationship_summary,
        "key_facts": graph._core.key_user_facts,
        "session_id": session_id,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "sessions": len(_sessions), "characters": list_characters()}


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AI Companion</title>
<style>
:root {
  --bg: #0f0f12; --panel: #1a1a22; --border: #2a2a35;
  --text: #e8e6e3; --muted: #9a9690; --accent: #c4a77d;
  --user: #2d3a4a; --bot: #252530;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--bg); color: var(--text);
  min-height: 100vh; display: flex; flex-direction: column; align-items: center;
}
header {
  width: 100%; max-width: 720px; padding: 1.25rem 1rem 0.5rem;
  display: flex; justify-content: space-between; align-items: center;
}
header h1 { font-size: 1.15rem; font-weight: 600; color: var(--accent); }
select, button {
  background: var(--panel); color: var(--text); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.4rem 0.75rem; font-size: 0.9rem;
}
button { background: var(--accent); color: #1a150f; font-weight: 600; cursor: pointer; border: none; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.hint { width: 100%; max-width: 720px; padding: 0 1rem 0.5rem; font-size: 0.8rem; color: var(--muted); }
#chat {
  width: 100%; max-width: 720px; flex: 1; overflow-y: auto; padding: 1rem;
  display: flex; flex-direction: column; gap: 0.75rem;
}
.msg { max-width: 85%; padding: 0.75rem 1rem; border-radius: 14px; line-height: 1.45; font-size: 0.95rem; white-space: pre-wrap; }
.msg.user { align-self: flex-end; background: var(--user); border-bottom-right-radius: 4px; }
.msg.bot { align-self: flex-start; background: var(--bot); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
.msg .name { font-size: 0.75rem; color: var(--accent); margin-bottom: 0.25rem; font-weight: 600; }
footer {
  width: 100%; max-width: 720px; padding: 0.75rem 1rem 1.25rem; display: flex; gap: 0.5rem;
}
input[type="text"] {
  flex: 1; background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 0.75rem 1rem; color: var(--text); font-size: 0.95rem; outline: none;
}
input:focus { border-color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>AI Companion</h1>
  <div style="display:flex;gap:0.5rem;align-items:center">
    <select id="backend" title="Memory backend">
      <option value="sqlite">SQLite</option>
      <option value="chroma">Chroma</option>
      <option value="memory">Memory</option>
    </select>
    <select id="character"></select>
  </div>
</header>
<p class="hint">Entertainment only · Memory persists within a session</p>
<div id="chat"></div>
<footer>
  <input type="text" id="input" placeholder="Say something…" autocomplete="off" />
  <button id="send">Send</button>
</footer>
<script>
let sessionId = null;
let characterId = null;
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const charSelect = document.getElementById("character");
const backendSelect = document.getElementById("backend");

const openings = {
  elara: "Elara glances up. \\"Didn't expect company. What's on your mind?\\"",
  sofia: "Sofia smiles. \\"Hey! Perfect timing.\\"",
  kai: "Kai gives a small nod. \\"You're here. Talk.\\"",
  marcus: "Marcus meets your eyes. \\"Good to see you. How are you actually doing?\\""
};

async function loadCharacters() {
  try {
    const res = await fetch("/api/characters");
    const chars = await res.json();
    chars.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = c.name;
      charSelect.appendChild(opt);
    });
    characterId = chars[0]?.id || "elara";
    showOpening(characterId);
  } catch (e) {
    addBot("Could not load characters. Check the server.");
  }
}

function showOpening(id) {
  chat.innerHTML = "";
  sessionId = null;
  addBot(openings[id] || "Hello.", id);
}

charSelect.addEventListener("change", () => {
  characterId = charSelect.value;
  showOpening(characterId);
});

function addBot(text, cid) {
  const div = document.createElement("div");
  div.className = "msg bot";
  const name = document.createElement("div");
  name.className = "name";
  name.textContent = (cid || characterId || "Companion").charAt(0).toUpperCase() + (cid || characterId || "Companion").slice(1);
  div.appendChild(name);
  div.appendChild(document.createTextNode(text));
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addUser(text) {
  const div = document.createElement("div");
  div.className = "msg user";
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  addUser(text);
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        character_id: characterId,
        session_id: sessionId,
        backend: backendSelect.value
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error");
    sessionId = data.session_id;
    addBot(data.reply, data.character_id);
  } catch (e) {
    addBot("(Something went wrong: " + e.message + ")");
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
loadCharacters();
input.focus();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return CHAT_HTML
