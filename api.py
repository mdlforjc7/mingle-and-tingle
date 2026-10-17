"""
Robust FastAPI web UI for the Mingle and Tingle.

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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

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
# Free-tier rate limit (cost protection)
# ---------------------------------------------------------------------------
# Every /api/chat message costs real xAI credits (~$0.01 once conversation
# history accumulates). Without a cap, one heavy user or a simple loop could
# burn $25+/hour of prepaid credits. Cap attempts per IP per UTC day; the paid
# tier (phase 1 paywall) will replace this with account-based limits.
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "60"))

_usage: Dict[str, dict] = {}  # ip -> {"day": "YYYY-MM-DD", "count": int}


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str) -> Optional[str]:
    """Returns a friendly notice if the free daily budget is spent, else None."""
    today = time.strftime("%Y-%m-%d", time.gmtime())
    entry = _usage.get(ip)
    if not entry or entry["day"] != today:
        entry = {"day": today, "count": 0}
        _usage[ip] = entry
    if len(_usage) > 5000:  # opportunistic cleanup of stale IPs
        for k in [k for k, v in _usage.items() if v["day"] != today]:
            _usage.pop(k, None)
    if entry["count"] >= RATE_LIMIT_PER_DAY:
        return (
            f"That's all my free chats for today ({RATE_LIMIT_PER_DAY}). "
            "Unlimited plans are coming soon \u2014 come back tomorrow!"
        )
    entry["count"] += 1
    return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Mingle and Tingle API ready")
    print("Characters:", list_characters())
    yield
    print("Shutdown")


app = FastAPI(
    title="Mingle and Tingle",
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


# ---------------------------------------------------------------------------
# Character art: portraits + backgrounds served from ./img (no new dependencies)
# ---------------------------------------------------------------------------
_IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
_IMG_CACHE: Dict[str, bytes] = {}
if os.path.isdir(_IMG_DIR):
    for _fname in os.listdir(_IMG_DIR):
        if _fname.endswith(".webp"):
            with open(os.path.join(_IMG_DIR, _fname), "rb") as _f:
                _IMG_CACHE[_fname] = _f.read()


@app.get("/img/{name}")
def serve_img(name: str):
    data = _IMG_CACHE.get(name)
    if data is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(
        content=data,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=86400"},
    )


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
            "portrait": f"/img/{c['id']}.webp",
            "background": f"/img/{c['id']}-bg.webp",
        })
    return out


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest, request: Request):
    # 18+ gate: the chat UI sets the mt_age=1 cookie when the visitor passes
    # the age gate. The API refuses to generate content without it.
    if request.cookies.get("mt_age") != "1":
        raise HTTPException(
            status_code=403,
            detail="Age verification required. Please confirm you are 18 or older on the homepage first.",
        )
    try:
        char = get_character(req.character_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    limit_msg = check_rate_limit(_client_ip(request))
    if limit_msg:
        raise HTTPException(status_code=429, detail=limit_msg)

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
    return {
        "status": "ok",
        "sessions": len(_sessions),
        "characters": list_characters(),
        "free_chats_per_day": RATE_LIMIT_PER_DAY,
    }


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Mingle and Tingle</title>
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
#bg { position: fixed; inset: 0; background-size: cover; background-position: center;
  opacity: 0.22; pointer-events: none; z-index: 0; }
header, .hint, #chat, footer, #charbar { position: relative; z-index: 1; }
#charbar { width: 100%; max-width: 720px; padding: 0.25rem 1rem 0.75rem;
  display: flex; align-items: center; gap: 0.75rem; }
#portrait { width: 60px; height: 60px; border-radius: 50%; object-fit: cover;
  border: 2px solid var(--accent); background: var(--panel); flex: none; }
#charbar .nm { font-weight: 600; color: var(--accent); font-size: 1rem; }
#charbar .rl { font-size: 0.8rem; color: var(--muted); }
#hero { width: 100%; max-width: 720px; padding: 1.5rem 1rem 0.5rem; text-align: center; }
#hero h2 { font-size: 1.4rem; color: var(--accent); margin-bottom: 0.5rem; font-weight: 700; }
#hero p { font-size: 0.95rem; color: var(--text); opacity: 0.85; line-height: 1.5; }
.fineprint { width: 100%; max-width: 720px; margin: 0 auto; padding: 0.5rem 1rem 2rem; text-align: center; font-size: 0.7rem; color: var(--muted); opacity: 0.7; }
.fineprint a { color: var(--muted); }
</style>
</head>
<body>
<div id="bg"></div>
<header>
  <h1>Mingle and Tingle</h1>
  <div style="display:flex;gap:0.5rem;align-items:center">
    <select id="backend" title="Memory backend">
      <option value="sqlite">SQLite</option>
      <option value="chroma">Chroma</option>
      <option value="memory">Memory</option>
    </select>
    <select id="character"></select>
  </div>
</header>
<div id="hero">
  <h2>Your next great conversation starts here.</h2>
  <p>Meet Elara, Sofia, Kai, and Marcus. Chat, flirt, connect &mdash; and soon, meet real members too.</p>
</div>
<div id="charbar">
  <img id="portrait" alt="Character portrait" />
  <div><div class="nm" id="charname"></div><div class="rl" id="charrole"></div></div>
</div>
<p class="hint">Memory persists within a session</p>
<div id="chat"></div>
<footer>
  <input type="text" id="input" placeholder="Say something…" autocomplete="off" />
  <button id="send">Send</button>
</footer>
<div class="fineprint">All characters are fictional and 18 years of age or older. &middot; <a href="/terms">Terms</a> &middot; <a href="/privacy">Privacy</a></div>
<script>
let sessionId = null;
let characterId = null;
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const charSelect = document.getElementById("character");
const backendSelect = document.getElementById("backend");
const charData = {};

function setCharacterArt(id) {
  const c = charData[id] || {};
  const p = document.getElementById("portrait");
  if (c.portrait) { p.src = c.portrait; p.alt = (c.name || "Character") + " portrait"; }
  document.getElementById("charname").textContent = c.name || "";
  document.getElementById("charrole").textContent = c.role || "";
  if (c.background) document.getElementById("bg").style.backgroundImage = "url(" + c.background + ")";
}

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
      charData[c.id] = c;
    });
    characterId = chars[0]?.id || "elara";
    setCharacterArt(characterId);
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
  setCharacterArt(characterId);
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
    if (!res.ok) {
      if (res.status === 429) { addBot(data.detail || "Daily free chat limit reached.", characterId); return; }
      throw new Error(data.detail || "Error");
    }
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


# ---------------------------------------------------------------------------
# 18+ age gate + legal pages
# ---------------------------------------------------------------------------
# Mingle and Tingle carries adult-oriented AI companion content. A click-through
# 18+ gate covers the UI, and /api/chat additionally requires the mt_age=1
# cookie so the API cannot be used without passing the gate. (Stronger,
# ID-checked verification arrives with the paid tier and its payment setup.)

AGE_GATE_SNIPPET = """<div id="age-gate">
  <div class="age-card">
    <h2>Mingle and Tingle</h2>
    <p class="age-sub">Adults only &middot; 18+</p>
    <p>This site contains adult-oriented chat intended strictly for adults.</p>
    <p>You must be 18 years old or older (or the age of majority in your jurisdiction) to enter.</p>
    <div class="age-btns">
      <button id="age-enter">I am 18 or older &mdash; Enter</button>
      <button id="age-exit" class="ghost">I am under 18 &mdash; Exit</button>
    </div>
    <p class="age-fine">By entering you agree to our <a href="/terms">Terms of Service</a> and <a href="/privacy">Privacy Policy</a>.</p>
  </div>
</div>
<div class="legal-bar"><a href="/terms">Terms</a><span>&middot;</span><a href="/privacy">Privacy</a><span>&middot;</span><span>18+ only</span></div>
<style>
#age-gate { position: fixed; inset: 0; z-index: 9999; display: flex; align-items: center; justify-content: center; background: rgba(12,12,15,0.98); padding: 1rem; }
.age-card { max-width: 440px; background: #1a1a22; border: 1px solid #2a2a35; border-radius: 16px; padding: 2rem; text-align: center; }
.age-card h2 { color: #c4a77d; margin-bottom: 0.25rem; }
.age-sub { color: #9a9690; font-size: 0.85rem; margin-bottom: 1rem; letter-spacing: 0.05em; }
.age-card p { font-size: 0.95rem; line-height: 1.5; margin-bottom: 1rem; color: #e8e6e3; }
.age-btns { display: flex; flex-direction: column; gap: 0.6rem; margin: 1.25rem 0; }
.age-btns button { padding: 0.8rem; font-size: 1rem; }
#age-exit.ghost { background: transparent; border: 1px solid #2a2a35; color: #9a9690; }
.age-fine { font-size: 0.75rem !important; color: #9a9690 !important; margin-bottom: 0 !important; }
.age-fine a, .legal-bar a { color: #c4a77d; }
.legal-bar { width: 100%; max-width: 720px; margin: 0 auto; padding: 0.25rem 1rem 1.5rem; text-align: center; font-size: 0.75rem; color: #9a9690; display: flex; gap: 0.6rem; justify-content: center; }
</style>
<script>
(function () {
  var KEY = "mt_age_ok";
  function verified() { try { return localStorage.getItem(KEY) === "1"; } catch (e) { return false; } }
  function setVerified() {
    try { localStorage.setItem(KEY, "1"); } catch (e) {}
    document.cookie = "mt_age=1; path=/; max-age=31536000; SameSite=Lax";
  }
  var gate = document.getElementById("age-gate");
  document.getElementById("age-enter").addEventListener("click", function () { setVerified(); gate.style.display = "none"; });
  document.getElementById("age-exit").addEventListener("click", function () {
    document.body.innerHTML = '<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;text-align:center;padding:2rem;color:#9a9690;font-family:system-ui,sans-serif"><div><h2 style="color:#c4a77d;margin-bottom:1rem">Mingle and Tingle</h2><p>You must be 18 or older to use this site.</p></div></div>';
  });
  if (verified()) { setVerified(); gate.style.display = "none"; }
})();
</script>
"""

LEGAL_STYLE = """
<style>
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #0f0f12; color: #e8e6e3; margin: 0; padding: 0; }
main { max-width: 720px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; line-height: 1.65; }
h1 { color: #c4a77d; font-size: 1.6rem; margin-bottom: 0.25rem; }
.updated { color: #9a9690; font-size: 0.8rem; margin-bottom: 2rem; }
h2 { color: #c4a77d; font-size: 1.1rem; margin: 2rem 0 0.5rem; }
p, li { font-size: 0.95rem; }
ul { padding-left: 1.25rem; }
a { color: #c4a77d; }
.back { display: inline-block; margin-top: 2.5rem; font-size: 0.9rem; }
</style>
"""

TERMS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Terms of Service &mdash; Mingle and Tingle</title>
""" + LEGAL_STYLE + """
</head>
<body>
<main>
<h1>Terms of Service</h1>
<p class="updated">Last updated: October 17, 2026</p>

<h2>1. What Mingle and Tingle is</h2>
<p>Mingle and Tingle ("the Service", "we") is a subscription-based social and chat platform for adults. The Service currently offers AI characters you can chat with, and we are building features for real members to create profiles and message each other. <strong>The four launch characters (Elara, Sofia, Kai, and Marcus) are fictional entertainment characters.</strong> As real members join, the Service will include both fictional characters and real people.</p>

<h2>2. Who may use the Service</h2>
<p>You must be <strong>18 years old or older</strong> (or the age of majority in your jurisdiction, if higher) to use the Service. By using the Service you confirm that you meet this requirement. If you are under 18, you must leave immediately. We do not knowingly allow anyone under 18 to use the Service.</p>

<h2>3. Subscriptions and billing</h2>
<p>Access to the Service requires a paid subscription. There is no free tier.</p>
<ul>
<li><strong>Price:</strong> $19.99 per month, billed monthly.</li>
<li><strong>Free trial:</strong> New subscribers receive a 30-day free trial. You must provide a valid payment method when you subscribe. If you do not cancel before the trial ends, your payment method will be charged $19.99 and billing will continue monthly until you cancel.</li>
<li><strong>Auto-renewal:</strong> Subscriptions renew automatically each month. You may cancel at any time from your account settings; cancellation takes effect at the end of the current billing period and you retain access until then.</li>
<li><strong>Price changes:</strong> We will notify you at least 30 days before any price change takes effect.</li>
</ul>

<h2>4. Refunds</h2>
<p>Refund requests must be made within 7 days of the charge by contacting <a href="mailto:support@mingleandtingle.com">support@mingleandtingle.com</a>, and are reviewed case by case. The 30-day free trial is provided so you can evaluate the Service before being charged; charges after the trial period are generally non-refundable except as required by law.</p>

<h2>5. Acceptable use</h2>
<p>You agree not to:</p>
<ul>
<li>Use the Service for anything illegal, or to request or share content involving minors in any form &mdash; attempts of this kind will result in an immediate ban and may be reported to the authorities;</li>
<li>Harass, threaten, stalk, or abuse other users or the Service;</li>
<li>Share sexually explicit images of real people without their consent;</li>
<li>Attempt to break, overload, reverse-engineer, or scrape the Service;</li>
<li>Use automated scripts or bots to access the Service without permission;</li>
<li>Create accounts with false age information or impersonate another person.</li>
</ul>

<h2>6. Nature of AI character content</h2>
<p>Character replies are generated by artificial intelligence. They may be inaccurate, inconsistent, or inappropriate, and are not professional, medical, legal, or relationship advice. Nothing a character says reflects the views of Mingle and Tingle. AI characters are entertainment; they are not real people and cannot form real relationships with you.</p>

<h2>7. Real-member features</h2>
<p>As real-member features (profiles, matching, member-to-member messaging) launch, additional rules will apply: you must provide accurate profile information, you must be 18 or older, and you consent to our safety practices including profile review and moderation. These terms will be updated before those features go live.</p>

<h2>8. Suspension and termination</h2>
<p>We may suspend or terminate your access for violations of these terms, abuse of the Service, chargebacks, or unlawful use, without prior notice. If we terminate for cause, you are not entitled to a refund of the current billing period.</p>

<h2>9. Intellectual property</h2>
<p>The Service, its characters, design, and content are owned by Mingle and Tingle. You keep the rights to the messages you send, and you grant us a license to process them as needed to operate the Service (for example, to generate replies and maintain conversation memory).</p>

<h2>10. Disclaimers and liability</h2>
<p>The Service is provided "as is" without warranties of any kind, express or implied. To the maximum extent permitted by law, our total liability for any claim arising from your use of the Service is limited to the amounts you paid us in the 12 months before the claim. We are not liable for the actions, conduct, or content of other users.</p>

<h2>11. Changes to these terms</h2>
<p>We may update these terms as the Service evolves. Material changes will be announced on the Service at least 14 days before they take effect. Continued use after changes take effect means you accept the updated terms.</p>

<h2>12. Contact</h2>
<p>Questions about these terms: <a href="mailto:support@mingleandtingle.com">support@mingleandtingle.com</a></p>

<a class="back" href="/">&larr; Back to Mingle and Tingle</a>
</main>
</body>
</html>
"""

PRIVACY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Privacy Policy &mdash; Mingle and Tingle</title>
""" + LEGAL_STYLE + """
</head>
<body>
<main>
<h1>Privacy Policy</h1>
<p class="updated">Last updated: October 17, 2026</p>

<h2>1. What we collect</h2>
<ul>
<li><strong>Account information</strong> &mdash; when you subscribe: email address, password (stored hashed), and date of birth or age confirmation;</li>
<li><strong>Payment information</strong> &mdash; processed securely by our payment provider; we do not store full card numbers on our servers;</li>
<li><strong>Chat messages</strong> &mdash; what you type is processed to generate character replies and to maintain conversation memory;</li>
<li><strong>Profile information</strong> &mdash; when member profiles launch: photos, bio, interests, and preferences you provide;</li>
<li><strong>Technical data</strong> &mdash; IP address (for abuse prevention and usage limits), device and browser information;</li>
<li><strong>Browser storage</strong> &mdash; your browser keeps your age-verification choice and session identifier locally (localStorage) plus one cookie for the age gate.</li>
</ul>

<h2>2. How we use it</h2>
<p>We use the information above to operate the Service, process subscriptions and payments, generate replies, enforce these terms, prevent abuse and fraud, and comply with legal obligations. We do not use your messages for advertising and we do not sell your personal information.</p>

<h2>3. Who we share it with</h2>
<ul>
<li><strong>AI provider</strong> &mdash; processes chat messages to generate character replies;</li>
<li><strong>Payment processor</strong> &mdash; processes subscription payments and handles billing;</li>
<li><strong>Service providers</strong> &mdash; hosting and infrastructure providers that help us operate the Service;</li>
<li><strong>Law enforcement or regulators</strong> &mdash; where required by law or to protect safety, including reports of child-safety concerns.</li>
</ul>

<h2>4. How long we keep it</h2>
<p>Chat sessions expire after about 4 hours of inactivity. Account and subscription records are kept while your account is active and for as long as needed for legal, tax, and accounting purposes after cancellation. Browser storage stays on your device until you clear it.</p>

<h2>5. Your rights</h2>
<p>You may request access to, correction of, or deletion of your personal information by contacting <a href="mailto:support@mingleandtingle.com">support@mingleandtingle.com</a>. We will respond within 30 days. Deletion requests are subject to our legal retention obligations (for example, payment records).</p>

<h2>6. Children</h2>
<p>The Service is for adults 18 and older only. We do not knowingly collect information from anyone under 18; if we learn that we have, we will delete it and terminate the account.</p>

<h2>7. Security</h2>
<p>We use reasonable technical and organizational measures to protect your information, including encrypted connections (HTTPS) and hashed passwords. No system is completely secure, and you are responsible for keeping your login credentials confidential.</p>

<h2>8. Changes to this policy</h2>
<p>We will update this policy as the Service evolves and post the new version here. Material changes will be announced on the Service at least 14 days before they take effect.</p>

<h2>9. Contact</h2>
<p>Privacy questions: <a href="mailto:support@mingleandtingle.com">support@mingleandtingle.com</a></p>

<a class="back" href="/">&larr; Back to Mingle and Tingle</a>
</main>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return CHAT_HTML.replace("</body>", AGE_GATE_SNIPPET + "</body>")


@app.get("/terms", response_class=HTMLResponse)
def terms():
    return TERMS_HTML


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return PRIVACY_HTML
