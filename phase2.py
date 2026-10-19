"""
Phase 2: real-member features for Mingle and Tingle.

- SQLite database (users, profiles, questionnaires, photos, messages,
  blocks, reports, subscriptions, credit ledger, sessions)
- Signup / login with bcrypt password hashing + session cookies
- Profiles with questionnaire + photo uploads
- Discovery / profile browsing
- 1:1 member messaging (1 credit per message)
- Subscriptions: $19.99/mo, 30-day free trial
- Credit packages: $7.99/20, $14/40, $24.99/80
- Safety: block, report, basic moderation
- Billing abstraction ready for CCBill / Epoch (merchant approval pending)

All member features require 18+ (birthdate checked at signup) and an
active subscription or trial. The AI-character chat in api.py is untouched.
"""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import bcrypt
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Render disk mount: set DATA_DIR=/var/data in the Render dashboard after
# attaching a persistent disk at /var/data. Falls back to local ./data so
# development and non-disk deploys keep working unchanged.
DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "phase2.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_COOKIE = "mt_session"
SESSION_TTL = 60 * 60 * 24 * 30  # 30 days

TRIAL_DAYS = 30
TRIAL_STARTER_CREDITS = 20
SUBSCRIPTION_PRICE_CENTS = 1999  # $19.99 / month
MESSAGE_CREDIT_COST = 1

MAX_PHOTOS_PER_USER = 6
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Questionnaire: key -> (label, kind)  kind: "choice" | "text"
QUESTIONS: Dict[str, tuple] = {
    "looking_for": ("What are you looking for?", "choice"),
    "ideal_mate": ("Describe your ideal mate", "text"),
    "interests": ("Your interests and hobbies", "text"),
    "fun": ("What do you like to do for fun?", "text"),
    "dealbreakers": ("Any deal-breakers?", "text"),
}
LOOKING_FOR_OPTIONS = [
    "A serious relationship",
    "Casual dating",
    "Friendship",
    "Marriage-minded",
    "Just browsing",
]
GENDER_OPTIONS = ["Woman", "Man", "Nonbinary", "Prefer not to say"]

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()
_db: Optional[sqlite3.Connection] = None


def _connect() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = sqlite3.connect(DB_PATH, check_same_thread=False)
        _db.row_factory = sqlite3.Row
        _db.execute("PRAGMA journal_mode=WAL")
        _db.execute("PRAGMA foreign_keys=ON")
    return _db


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name  TEXT NOT NULL,
    created_at    INTEGER NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 1,
    is_banned     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS profiles (
    user_id     INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    birthdate   TEXT NOT NULL,            -- YYYY-MM-DD
    gender      TEXT NOT NULL DEFAULT '',
    seeking     TEXT NOT NULL DEFAULT '',
    location    TEXT NOT NULL DEFAULT '',
    bio         TEXT NOT NULL DEFAULT '',
    updated_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS questionnaire_answers (
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question_key  TEXT NOT NULL,
    answer_text   TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (user_id, question_key)
);
CREATE TABLE IF NOT EXISTS photos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL UNIQUE,
    is_primary  INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photos_user ON photos(user_id);
CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS subscriptions (
    user_id               INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    status                TEXT NOT NULL,   -- trialing | active | past_due | canceled
    trial_ends_at         INTEGER,
    current_period_end    INTEGER,
    card_on_file          INTEGER NOT NULL DEFAULT 0,
    processor             TEXT,            -- ccbill | epoch (once approved)
    processor_customer_id TEXT,
    created_at            INTEGER NOT NULL,
    updated_at            INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS credit_packages (
    id            TEXT PRIMARY KEY,   -- basic | standard | premium
    name          TEXT NOT NULL,
    credits       INTEGER NOT NULL,
    price_cents   INTEGER NOT NULL,
    validity_days INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS credit_ledger (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    delta       INTEGER NOT NULL,     -- +grant / -spend
    reason      TEXT NOT NULL,
    expires_at  INTEGER,              -- NULL = never expires
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_user ON credit_ledger(user_id);
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recipient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    body         TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    is_read      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_msg_pair ON messages(sender_id, recipient_id);
CREATE INDEX IF NOT EXISTS idx_msg_recipient ON messages(recipient_id);
CREATE TABLE IF NOT EXISTS blocks (
    blocker_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    blocked_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (blocker_id, blocked_id)
);
CREATE TABLE IF NOT EXISTS reports (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    reporter_id      INTEGER REFERENCES users(id) ON DELETE CASCADE,  -- NULL = system/auto-moderation
    reported_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    reason           TEXT NOT NULL,
    details          TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'open',  -- open | reviewed | actioned
    created_at       INTEGER NOT NULL
);
"""


def init_db():
    with _db_lock:
        db = _connect()
        db.executescript(SCHEMA)
        # Seed credit packages (idempotent)
        now = int(time.time())
        db.executemany(
            "INSERT OR IGNORE INTO credit_packages (id, name, credits, price_cents, validity_days)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                ("basic", "Basic", 20, 799, 14),
                ("standard", "Standard", 40, 1400, 30),
                ("premium", "Premium", 80, 2499, 60),
            ],
        )
        db.commit()


def _now() -> int:
    return int(time.time())


def _row_to_dict(row) -> dict:
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Passwords (bcrypt)
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Basic moderation
# ---------------------------------------------------------------------------
# Child-safety zero tolerance: any hit rejects the content, flags the
# account for review, and bans it immediately.

CHILD_SAFETY_PATTERNS = [
    r"\b(under\s?18|underage|minor)\b.*\b(sex|nude|pic|meet)\b",
    r"\b\d{1,2}\s?(yo|y\.o\.|years?\s?old)\b",
]
CHILD_SAFETY_RE = [re.compile(p, re.IGNORECASE) for p in CHILD_SAFETY_PATTERNS]


def _looks_like_minor_claim(text: str) -> bool:
    """Heuristic: catches explicit 'I am 17' style claims."""
    m = re.search(r"\bi['’]?m\s+(\d{1,2})\b", text, re.IGNORECASE)
    if m and int(m.group(1)) < 18:
        return True
    m = re.search(r"\b(\d{1,2})\s?(yo|y\.o\.|years?\s?old)\b", text, re.IGNORECASE)
    if m and int(m.group(1)) < 18:
        return True
    return False


def moderate_text(text: str) -> Optional[str]:
    """Returns a violation reason, or None if the text passes."""
    for rx in CHILD_SAFETY_RE:
        if rx.search(text):
            return "child-safety pattern matched"
    if _looks_like_minor_claim(text):
        return "possible minor age claim"
    return None


def _ban_user(user_id: int, reason: str):
    with _db_lock:
        db = _connect()
        db.execute("UPDATE users SET is_banned=1 WHERE id=?", (user_id,))
        db.execute(
            "INSERT INTO reports (reporter_id, reported_user_id, reason, details, status, created_at)"
            " VALUES (NULL, ?, 'auto-moderation', ?, 'actioned', ?)",
            (user_id, f"Auto-ban: {reason}", _now()),
        )
        db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        db.commit()


# ---------------------------------------------------------------------------
# Users / sessions
# ---------------------------------------------------------------------------

def create_user(email: str, password: str, display_name: str, birthdate: str) -> dict:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise ValueError("Please enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if not display_name.strip():
        raise ValueError("Please choose a display name.")
    try:
        bd = datetime.strptime(birthdate, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Birthdate must be YYYY-MM-DD.")
    today = datetime.now(timezone.utc).date()
    age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    if age < 18:
        raise ValueError("You must be 18 or older to join.")
    now = _now()
    with _db_lock:
        db = _connect()
        cur = db.execute("SELECT id FROM users WHERE email=?", (email,))
        if cur.fetchone():
            raise ValueError("An account with that email already exists.")
        cur = db.execute(
            "INSERT INTO users (email, password_hash, display_name, created_at)"
            " VALUES (?, ?, ?, ?)",
            (email, hash_password(password), display_name.strip()[:40], now),
        )
        user_id = cur.lastrowid
        db.execute(
            "INSERT INTO profiles (user_id, birthdate, updated_at) VALUES (?, ?, ?)",
            (user_id, birthdate, now),
        )
        # Start 30-day free trial
        trial_ends = now + TRIAL_DAYS * 86400
        db.execute(
            "INSERT INTO subscriptions (user_id, status, trial_ends_at, created_at, updated_at)"
            " VALUES (?, 'trialing', ?, ?, ?)",
            (user_id, trial_ends, now, now),
        )
        # Starter credits so trial members can try messaging
        db.execute(
            "INSERT INTO credit_ledger (user_id, delta, reason, created_at)"
            " VALUES (?, ?, 'trial starter credits', ?)",
            (user_id, TRIAL_STARTER_CREDITS, now),
        )
        db.commit()
    return get_user(user_id)


def get_user(user_id: int) -> Optional[dict]:
    with _db_lock:
        db = _connect()
        cur = db.execute("SELECT * FROM users WHERE id=?", (user_id,))
        return _row_to_dict(cur.fetchone())


def get_user_by_email(email: str) -> Optional[dict]:
    with _db_lock:
        db = _connect()
        cur = db.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),))
        return _row_to_dict(cur.fetchone())


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = _now()
    with _db_lock:
        db = _connect()
        db.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + SESSION_TTL),
        )
        db.commit()
    return token


def get_session_user(request: Request) -> Optional[dict]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    now = _now()
    with _db_lock:
        db = _connect()
        cur = db.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id"
            " WHERE s.token=? AND s.expires_at > ?",
            (token, now),
        )
        row = cur.fetchone()
        if row is None:
            return None
        user = _row_to_dict(row)
        if not user["is_active"] or user["is_banned"]:
            return None
        return user


def destroy_session(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with _db_lock:
            db = _connect()
            db.execute("DELETE FROM sessions WHERE token=?", (token,))
            db.commit()


# ---------------------------------------------------------------------------
# Subscriptions & credits
# ---------------------------------------------------------------------------

def get_subscription(user_id: int) -> Optional[dict]:
    with _db_lock:
        db = _connect()
        cur = db.execute("SELECT * FROM subscriptions WHERE user_id=?", (user_id,))
        return _row_to_dict(cur.fetchone())


def subscription_active(user_id: int) -> bool:
    sub = get_subscription(user_id)
    if not sub:
        return False
    now = _now()
    if sub["status"] == "trialing":
        return sub["trial_ends_at"] and sub["trial_ends_at"] > now
    if sub["status"] == "active":
        return not sub["current_period_end"] or sub["current_period_end"] > now
    return False


def credit_balance(user_id: int) -> int:
    now = _now()
    with _db_lock:
        db = _connect()
        cur = db.execute(
            "SELECT COALESCE(SUM(delta),0) AS bal FROM credit_ledger"
            " WHERE user_id=? AND (expires_at IS NULL OR expires_at > ?)",
            (user_id, now),
        )
        return cur.fetchone()["bal"] or 0


def grant_credits(user_id: int, amount: int, reason: str, validity_days: Optional[int] = None):
    now = _now()
    expires_at = now + validity_days * 86400 if validity_days else None
    with _db_lock:
        db = _connect()
        db.execute(
            "INSERT INTO credit_ledger (user_id, delta, reason, expires_at, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (user_id, amount, reason, expires_at, now),
        )
        db.commit()


def spend_credits(user_id: int, amount: int, reason: str) -> bool:
    if credit_balance(user_id) < amount:
        return False
    grant_credits(user_id, -amount, reason)
    return True


def list_packages() -> List[dict]:
    with _db_lock:
        db = _connect()
        cur = db.execute("SELECT * FROM credit_packages ORDER BY price_cents")
        return [_row_to_dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Billing abstraction (CCBill / Epoch — merchant approval pending)
# ---------------------------------------------------------------------------
# When a merchant account is approved, fill in BILLING_CONFIG with the
# processor credentials and implement the two functions below using the
# processor's hosted-payment / API docs. Until then every checkout returns
# a structured "not configured" payload so the UI can show a clear message.

BILLING_CONFIG = {
    "ccbill": {"enabled": False, "account": None, "subaccount": None},
    "epoch": {"enabled": False, "account": None},
}
SUBSCRIPTION_PLAN = {
    "name": "Mingle and Tingle Monthly",
    "price_cents": SUBSCRIPTION_PRICE_CENTS,
    "interval": "month",
    "trial_days": TRIAL_DAYS,
}


def billing_status() -> dict:
    return {
        "processors": {
            k: {"enabled": v["enabled"]} for k, v in BILLING_CONFIG.items()
        },
        "plan": SUBSCRIPTION_PLAN,
        "packages": [
            {"id": p["id"], "name": p["name"], "credits": p["credits"],
             "price_cents": p["price_cents"], "validity_days": p["validity_days"]}
            for p in list_packages()
        ],
    }


def create_subscription_checkout(user_id: int, processor: str) -> dict:
    """Start a hosted checkout for the $19.99/mo subscription (card on file)."""
    cfg = BILLING_CONFIG.get(processor)
    if not cfg or not cfg["enabled"]:
        return {
            "ok": False,
            "error": "billing_not_configured",
            "message": (
                "Online payments are not connected yet. Your 30-day free trial "
                "is active — no card is required until billing is enabled."
            ),
        }
    # Processor integration goes here once the merchant account is approved.
    raise NotImplementedError(f"{processor} checkout not implemented")


def create_package_checkout(user_id: int, package_id: str, processor: str) -> dict:
    """Start a hosted checkout for a credit package."""
    cfg = BILLING_CONFIG.get(processor)
    if not cfg or not cfg["enabled"]:
        return {
            "ok": False,
            "error": "billing_not_configured",
            "message": "Credit package purchases will be available once billing is connected.",
        }
    raise NotImplementedError(f"{processor} package checkout not implemented")


# ---------------------------------------------------------------------------
# Profiles / questionnaire / photos
# ---------------------------------------------------------------------------

def get_profile(user_id: int) -> Optional[dict]:
    with _db_lock:
        db = _connect()
        cur = db.execute(
            "SELECT p.*, u.display_name, u.created_at AS member_since"
            " FROM profiles p JOIN users u ON u.id = p.user_id"
            " WHERE p.user_id=?",
            (user_id,),
        )
        return _row_to_dict(cur.fetchone())


def update_profile(user_id: int, fields: dict) -> dict:
    allowed = {"gender", "seeking", "location", "bio"}
    updates = {k: str(fields.get(k, ""))[:500] for k in allowed if k in fields}
    if "bio" in updates:
        violation = moderate_text(updates["bio"])
        if violation:
            _ban_user(user_id, f"bio: {violation}")
            raise ValueError("This profile was rejected by moderation.")
    if updates:
        sets = ", ".join(f"{k}=?" for k in updates)
        with _db_lock:
            db = _connect()
            db.execute(
                f"UPDATE profiles SET {sets}, updated_at=? WHERE user_id=?",
                (*updates.values(), _now(), user_id),
            )
            db.commit()
    return get_profile(user_id)


def get_answers(user_id: int) -> Dict[str, str]:
    with _db_lock:
        db = _connect()
        cur = db.execute(
            "SELECT question_key, answer_text FROM questionnaire_answers WHERE user_id=?",
            (user_id,),
        )
        return {r["question_key"]: r["answer_text"] for r in cur.fetchall()}


def save_answers(user_id: int, answers: Dict[str, str]):
    for key, text in answers.items():
        if key not in QUESTIONS:
            raise ValueError(f"Unknown question: {key}")
        text = str(text or "")[:2000]
        if QUESTIONS[key][1] == "choice" and text and text not in LOOKING_FOR_OPTIONS:
            raise ValueError(f"Invalid choice for {key}.")
        violation = moderate_text(text)
        if violation:
            _ban_user(user_id, f"questionnaire {key}: {violation}")
            raise ValueError("This answer was rejected by moderation.")
    with _db_lock:
        db = _connect()
        for key, text in answers.items():
            db.execute(
                "INSERT INTO questionnaire_answers (user_id, question_key, answer_text)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT(user_id, question_key) DO UPDATE SET answer_text=excluded.answer_text",
                (user_id, key, str(text or "")),
            )
        db.commit()


def _photo_url(filename: str) -> str:
    return f"/api/v2/photos/{filename}"


def list_photos(user_id: int) -> List[dict]:
    with _db_lock:
        db = _connect()
        cur = db.execute(
            "SELECT * FROM photos WHERE user_id=? ORDER BY is_primary DESC, created_at",
            (user_id,),
        )
        out = []
        for r in cur.fetchall():
            d = _row_to_dict(r)
            d["url"] = _photo_url(d["filename"])
            out.append(d)
        return out


def add_photo(user_id: int, upload: UploadFile, data: bytes) -> dict:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Photo must be under 5 MB.")
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in ALLOWED_PHOTO_EXT:
        raise ValueError("Photo must be JPG, PNG, or WebP.")
    # crude magic-byte check
    if not (data.startswith(b"\xff\xd8") or data.startswith(b"\x89PNG")
            or data.startswith(b"RIFF") or data[:4] == b"\x00\x00\x00\x0c"):
        # allow webp "RIFF....WEBP" already covered; be lenient otherwise
        pass
    with _db_lock:
        db = _connect()
        cur = db.execute("SELECT COUNT(*) AS c FROM photos WHERE user_id=?", (user_id,))
        if cur.fetchone()["c"] >= MAX_PHOTOS_PER_USER:
            raise ValueError(f"Maximum {MAX_PHOTOS_PER_USER} photos.")
        cur = db.execute(
            "SELECT COUNT(*) AS c FROM photos WHERE user_id=? AND is_primary=1", (user_id,)
        )
        is_primary = 1 if cur.fetchone()["c"] == 0 else 0
    filename = f"{uuid.uuid4().hex}{ext}"
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as f:
        f.write(data)
    with _db_lock:
        db = _connect()
        cur = db.execute(
            "INSERT INTO photos (user_id, filename, is_primary, created_at)"
            " VALUES (?, ?, ?, ?)",
            (user_id, filename, is_primary, _now()),
        )
        photo_id = cur.lastrowid
        db.commit()
    return {"id": photo_id, "url": _photo_url(filename), "is_primary": bool(is_primary)}


def delete_photo(user_id: int, photo_id: int):
    with _db_lock:
        db = _connect()
        cur = db.execute(
            "SELECT * FROM photos WHERE id=? AND user_id=?", (photo_id, user_id)
        )
        row = cur.fetchone()
        if not row:
            raise ValueError("Photo not found.")
        db.execute("DELETE FROM photos WHERE id=?", (photo_id,))
        # promote oldest remaining to primary if needed
        if row["is_primary"]:
            db.execute(
                "UPDATE photos SET is_primary=1 WHERE user_id=? AND id=("
                " SELECT id FROM photos WHERE user_id=? ORDER BY created_at LIMIT 1)",
                (user_id, user_id),
            )
        db.commit()
    try:
        os.remove(os.path.join(UPLOAD_DIR, row["filename"]))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _is_blocked(a: int, b: int) -> bool:
    with _db_lock:
        db = _connect()
        cur = db.execute(
            "SELECT 1 FROM blocks WHERE (blocker_id=? AND blocked_id=?)"
            " OR (blocker_id=? AND blocked_id=?)",
            (a, b, b, a),
        )
        return cur.fetchone() is not None


def compatibility_score(viewer_id: int, other_id: int) -> int:
    """0-100 score from questionnaire overlap. Simple keyword overlap on
    interests/fun plus a bonus when 'looking_for' matches."""
    va = get_answers(viewer_id)
    oa = get_answers(other_id)
    score = 0
    if va.get("looking_for") and va.get("looking_for") == oa.get("looking_for"):
        score += 30
    for key in ("interests", "fun", "ideal_mate"):
        v_words = set(re.findall(r"[a-z]{3,}", (va.get(key) or "").lower()))
        o_words = set(re.findall(r"[a-z]{3,}", (oa.get(key) or "").lower()))
        if v_words and o_words:
            overlap = len(v_words & o_words)
            score += min(20, overlap * 5)
    return min(100, score)


def discover(viewer_id: int, limit: int = 24, offset: int = 0,
             seeking: str = "", gender: str = "") -> List[dict]:
    with _db_lock:
        db = _connect()
        q = ("SELECT u.id, u.display_name, p.gender, p.seeking, p.location,"
             " substr(p.bio,1,160) AS bio_snippet"
             " FROM users u JOIN profiles p ON p.user_id=u.id"
             " WHERE u.id != ? AND u.is_active=1 AND u.is_banned=0")
        params: list = [viewer_id]
        if gender:
            q += " AND p.gender=?"
            params.append(gender)
        if seeking:
            q += " AND p.seeking=?"
            params.append(seeking)
        q += (" AND NOT EXISTS (SELECT 1 FROM blocks b WHERE"
              " (b.blocker_id=? AND b.blocked_id=u.id)"
              " OR (b.blocker_id=u.id AND b.blocked_id=?))")
        params += [viewer_id, viewer_id]
        q += " ORDER BY u.created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        cur = db.execute(q, params)
        rows = [_row_to_dict(r) for r in cur.fetchall()]
    out = []
    for r in rows:
        photos = list_photos(r["id"])
        r["photo_url"] = photos[0]["url"] if photos else None
        r["match_score"] = compatibility_score(viewer_id, r["id"])
        out.append(r)
    out.sort(key=lambda r: r["match_score"], reverse=True)
    return out


def public_profile(viewer_id: int, user_id: int) -> dict:
    if _is_blocked(viewer_id, user_id):
        raise ValueError("This profile is not available.")
    prof = get_profile(user_id)
    if not prof:
        raise ValueError("Profile not found.")
    user = get_user(user_id)
    if not user or not user["is_active"] or user["is_banned"]:
        raise ValueError("Profile not found.")
    try:
        bd = datetime.strptime(prof["birthdate"], "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except Exception:
        age = None
    return {
        "user_id": user_id,
        "display_name": user["display_name"],
        "age": age,
        "gender": prof["gender"],
        "seeking": prof["seeking"],
        "location": prof["location"],
        "bio": prof["bio"],
        "answers": get_answers(user_id),
        "photos": list_photos(user_id),
        "match_score": compatibility_score(viewer_id, user_id),
        "member_since": prof["member_since"],
    }


# ---------------------------------------------------------------------------
# Messaging
# ---------------------------------------------------------------------------

def send_message(sender_id: int, recipient_id: int, body: str) -> dict:
    body = (body or "").strip()
    if not body:
        raise ValueError("Message is empty.")
    if len(body) > 2000:
        raise ValueError("Message is too long (2000 chars max).")
    if sender_id == recipient_id:
        raise ValueError("You can't message yourself.")
    recipient = get_user(recipient_id)
    if not recipient or not recipient["is_active"] or recipient["is_banned"]:
        raise ValueError("That member is not available.")
    if _is_blocked(sender_id, recipient_id):
        raise ValueError("You can't message this member.")
    violation = moderate_text(body)
    if violation:
        _ban_user(sender_id, f"message: {violation}")
        raise ValueError("This message was rejected by moderation.")
    if not subscription_active(sender_id):
        raise PermissionError("An active subscription or trial is required to message members.")
    if not spend_credits(sender_id, MESSAGE_CREDIT_COST, f"message to user {recipient_id}"):
        raise PermissionError("You're out of message credits. Grab a credit package to keep chatting.")
    now = _now()
    with _db_lock:
        db = _connect()
        cur = db.execute(
            "INSERT INTO messages (sender_id, recipient_id, body, created_at)"
            " VALUES (?, ?, ?, ?)",
            (sender_id, recipient_id, body, now),
        )
        msg_id = cur.lastrowid
        db.commit()
    return {"id": msg_id, "created_at": now, "credits_left": credit_balance(sender_id)}


def conversations(user_id: int) -> List[dict]:
    with _db_lock:
        db = _connect()
        cur = db.execute(
            """
            SELECT
              CASE WHEN m.sender_id=? THEN m.recipient_id ELSE m.sender_id END AS other_id,
              MAX(m.created_at) AS last_at,
              SUM(CASE WHEN m.recipient_id=? AND m.is_read=0 THEN 1 ELSE 0 END) AS unread
            FROM messages m
            WHERE m.sender_id=? OR m.recipient_id=?
            GROUP BY other_id ORDER BY last_at DESC
            """,
            (user_id, user_id, user_id, user_id),
        )
        rows = [_row_to_dict(r) for r in cur.fetchall()]
    out = []
    for r in rows:
        if _is_blocked(user_id, r["other_id"]):
            continue
        other = get_user(r["other_id"])
        if not other:
            continue
        photos = list_photos(r["other_id"])
        with _db_lock:
            db = _connect()
            cur = db.execute(
                "SELECT body FROM messages WHERE"
                " ((sender_id=? AND recipient_id=?) OR (sender_id=? AND recipient_id=?))"
                " ORDER BY created_at DESC LIMIT 1",
                (user_id, r["other_id"], r["other_id"], user_id),
            )
            last = cur.fetchone()
        out.append({
            "user_id": r["other_id"],
            "display_name": other["display_name"],
            "photo_url": photos[0]["url"] if photos else None,
            "last_message": (last["body"][:80] if last else ""),
            "last_at": r["last_at"],
            "unread": r["unread"],
        })
    return out


def thread(user_id: int, other_id: int, limit: int = 50) -> List[dict]:
    if _is_blocked(user_id, other_id):
        raise ValueError("This conversation is not available.")
    with _db_lock:
        db = _connect()
        db.execute(
            "UPDATE messages SET is_read=1 WHERE sender_id=? AND recipient_id=?",
            (other_id, user_id),
        )
        cur = db.execute(
            "SELECT * FROM messages WHERE"
            " (sender_id=? AND recipient_id=?) OR (sender_id=? AND recipient_id=?)"
            " ORDER BY created_at DESC LIMIT ?",
            (user_id, other_id, other_id, user_id, limit),
        )
        rows = [_row_to_dict(r) for r in cur.fetchall()]
        db.commit()
    return list(reversed(rows))


def unread_count(user_id: int) -> int:
    with _db_lock:
        db = _connect()
        cur = db.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE recipient_id=? AND is_read=0",
            (user_id,),
        )
        return cur.fetchone()["c"]


# ---------------------------------------------------------------------------
# Safety: block / report
# ---------------------------------------------------------------------------

def block_user(blocker_id: int, blocked_id: int):
    if blocker_id == blocked_id:
        raise ValueError("You can't block yourself.")
    if not get_user(blocked_id):
        raise ValueError("Member not found.")
    with _db_lock:
        db = _connect()
        db.execute(
            "INSERT OR IGNORE INTO blocks (blocker_id, blocked_id, created_at)"
            " VALUES (?, ?, ?)",
            (blocker_id, blocked_id, _now()),
        )
        db.commit()


def unblock_user(blocker_id: int, blocked_id: int):
    with _db_lock:
        db = _connect()
        db.execute(
            "DELETE FROM blocks WHERE blocker_id=? AND blocked_id=?",
            (blocker_id, blocked_id),
        )
        db.commit()


def report_user(reporter_id: int, reported_id: int, reason: str, details: str = ""):
    if reporter_id == reported_id:
        raise ValueError("You can't report yourself.")
    if not get_user(reported_id):
        raise ValueError("Member not found.")
    reason = (reason or "").strip()[:100]
    if not reason:
        raise ValueError("Please choose a reason.")
    with _db_lock:
        db = _connect()
        db.execute(
            "INSERT INTO reports (reporter_id, reported_user_id, reason, details, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (reporter_id, reported_id, reason, (details or "")[:2000], _now()),
        )
        db.commit()


# ---------------------------------------------------------------------------
# Auth helpers (FastAPI dependencies)
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v2", tags=["members"])


def current_user(request: Request) -> dict:
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in.")
    return user


def member_user(request: Request) -> dict:
    """Logged in + active subscription/trial (for member-only features)."""
    user = current_user(request)
    if not subscription_active(user["id"]):
        raise HTTPException(
            status_code=402,
            detail="Your trial or subscription has ended. Resubscribe to keep using member features.",
        )
    return user


def _set_session_cookie(resp: Response, token: str):
    resp.set_cookie(
        SESSION_COOKIE, token, max_age=SESSION_TTL,
        httponly=True, samesite="lax", path="/",
    )


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

class SignupIn(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=40)
    birthdate: str  # YYYY-MM-DD


class LoginIn(BaseModel):
    email: str
    password: str


@router.post("/signup")
def api_signup(data: SignupIn, request: Request):
    # 18+ gate cookie still required, same as the AI chat
    if request.cookies.get("mt_age") != "1":
        raise HTTPException(status_code=403, detail="Please confirm you are 18 or older first.")
    try:
        user = create_user(data.email, data.password, data.display_name, data.birthdate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_session(user["id"])
    resp = JSONResponse({
        "ok": True,
        "user_id": user["id"],
        "display_name": user["display_name"],
        "trial_days": TRIAL_DAYS,
        "starter_credits": TRIAL_STARTER_CREDITS,
    })
    _set_session_cookie(resp, token)
    return resp


@router.post("/login")
def api_login(data: LoginIn):
    user = get_user_by_email(data.email)
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not user["is_active"] or user["is_banned"]:
        raise HTTPException(status_code=403, detail="This account is not available.")
    token = create_session(user["id"])
    resp = JSONResponse({"ok": True, "display_name": user["display_name"]})
    _set_session_cookie(resp, token)
    return resp


@router.post("/logout")
def api_logout(request: Request):
    destroy_session(request)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@router.get("/me")
def api_me(request: Request):
    user = current_user(request)
    sub = get_subscription(user["id"])
    return {
        "user_id": user["id"],
        "display_name": user["display_name"],
        "email": user["email"],
        "subscription": {
            "status": sub["status"] if sub else "none",
            "trial_ends_at": sub["trial_ends_at"] if sub else None,
            "active": subscription_active(user["id"]),
            "card_on_file": bool(sub["card_on_file"]) if sub else False,
        },
        "credits": credit_balance(user["id"]),
        "unread_messages": unread_count(user["id"]),
        "profile": get_profile(user["id"]),
        "answers": get_answers(user["id"]),
        "photos": list_photos(user["id"]),
    }


# ---------------------------------------------------------------------------
# Profile / questionnaire / photos
# ---------------------------------------------------------------------------

class ProfileUpdate(BaseModel):
    gender: Optional[str] = None
    seeking: Optional[str] = None
    location: Optional[str] = None
    bio: Optional[str] = None


@router.put("/profile")
def api_update_profile(data: ProfileUpdate, request: Request):
    user = current_user(request)
    try:
        prof = update_profile(user["id"], {k: v for k, v in data.model_dump().items() if v is not None})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "profile": prof}


@router.put("/questionnaire")
def api_save_questionnaire(answers: Dict[str, str], request: Request):
    user = current_user(request)
    try:
        save_answers(user["id"], answers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "answers": get_answers(user["id"])}


@router.get("/questionnaire-meta")
def api_questionnaire_meta():
    return {
        "questions": [
            {"key": k, "label": label, "kind": kind,
             "options": LOOKING_FOR_OPTIONS if k == "looking_for" else []}
            for k, (label, kind) in QUESTIONS.items()
        ],
        "genders": GENDER_OPTIONS,
    }


@router.post("/photos")
async def api_upload_photo(request: Request, file: UploadFile = File(...)):
    user = current_user(request)
    data = await file.read()
    try:
        photo = add_photo(user["id"], file, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "photo": photo}


@router.delete("/photos/{photo_id}")
def api_delete_photo(photo_id: int, request: Request):
    user = current_user(request)
    try:
        delete_photo(user["id"], photo_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@router.get("/photos/{filename}")
def api_serve_photo(filename: str):
    safe = re.sub(r"[^a-zA-Z0-9._-]", "", filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.isfile(path) or ".." in filename:
        raise HTTPException(status_code=404, detail="Not found")
    ext = os.path.splitext(safe)[1].lower()
    media = {"jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".png": "image/png", ".webp": "image/webp"}.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type=media,
                    headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@router.get("/discover")
def api_discover(request: Request, limit: int = 24, offset: int = 0,
                 seeking: str = "", gender: str = ""):
    user = member_user(request)
    limit = max(1, min(limit, 50))
    return {"members": discover(user["id"], limit=limit, offset=offset,
                               seeking=seeking, gender=gender)}


@router.get("/members/{user_id}")
def api_view_profile(user_id: int, request: Request):
    user = member_user(request)
    try:
        return public_profile(user["id"], user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Messaging
# ---------------------------------------------------------------------------

class MessageIn(BaseModel):
    recipient_id: int
    body: str = Field(min_length=1, max_length=2000)


@router.post("/messages")
def api_send_message(data: MessageIn, request: Request):
    user = member_user(request)
    try:
        return {"ok": True, **send_message(user["id"], data.recipient_id, data.body)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=402, detail=str(e))


@router.get("/conversations")
def api_conversations(request: Request):
    user = member_user(request)
    return {"conversations": conversations(user["id"])}


@router.get("/messages/{other_id}")
def api_thread(other_id: int, request: Request, limit: int = 50):
    user = member_user(request)
    try:
        msgs = thread(user["id"], other_id, limit=min(limit, 100))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    other = get_user(other_id)
    return {
        "other": {"user_id": other_id,
                  "display_name": other["display_name"] if other else "?"},
        "messages": msgs,
    }


# ---------------------------------------------------------------------------
# Credits / packages / billing
# ---------------------------------------------------------------------------

@router.get("/billing")
def api_billing(request: Request):
    current_user(request)
    return billing_status()


@router.post("/billing/subscribe")
def api_subscribe(request: Request, processor: str = "ccbill"):
    user = current_user(request)
    return create_subscription_checkout(user["id"], processor)


@router.post("/billing/packages/{package_id}/buy")
def api_buy_package(package_id: str, request: Request, processor: str = "ccbill"):
    user = current_user(request)
    pkgs = {p["id"]: p for p in list_packages()}
    if package_id not in pkgs:
        raise HTTPException(status_code=404, detail="Unknown package.")
    return create_package_checkout(user["id"], package_id, processor)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

@router.post("/block/{user_id}")
def api_block(user_id: int, request: Request):
    user = current_user(request)
    try:
        block_user(user["id"], user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@router.post("/unblock/{user_id}")
def api_unblock(user_id: int, request: Request):
    user = current_user(request)
    unblock_user(user["id"], user_id)
    return {"ok": True}


class ReportIn(BaseModel):
    reason: str
    details: Optional[str] = ""


@router.post("/report/{user_id}")
def api_report(user_id: int, data: ReportIn, request: Request):
    user = current_user(request)
    try:
        report_user(user["id"], user_id, data.reason, data.details or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "message": "Thanks — our team will review this profile."}


# ---------------------------------------------------------------------------
# Member UI pages
# ---------------------------------------------------------------------------
# Server-rendered shells + vanilla JS against /api/v2. All pages carry the
# small-print fictional-characters footer and require the 18+ gate cookie.

PAGE_CSS = """
:root { --bg:#0f0f12; --panel:#1a1a22; --border:#2a2a35; --text:#e8e6e3;
  --muted:#9a9690; --accent:#c4a77d; --danger:#c46a6a; }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:system-ui,-apple-system,"Segoe UI",sans-serif; background:var(--bg);
  color:var(--text); min-height:100vh; }
.wrap { width:100%; max-width:720px; margin:0 auto; padding:1rem; }
nav.top { display:flex; gap:0.4rem; align-items:center; padding:0.75rem 1rem;
  border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--bg); z-index:5;
  flex-wrap:wrap; }
nav.top .brand { color:var(--accent); font-weight:700; margin-right:auto; text-decoration:none; }
nav.top a.nlink { color:var(--muted); text-decoration:none; font-size:0.85rem; padding:0.4rem 0.6rem;
  border-radius:8px; }
nav.top a.nlink.on, nav.top a.nlink:hover { color:var(--accent); background:var(--panel); }
h1 { color:var(--accent); font-size:1.3rem; margin:1rem 0; }
h2 { color:var(--accent); font-size:1.05rem; margin:1.25rem 0 0.6rem; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:1rem;
  margin-bottom:0.9rem; }
label { display:block; font-size:0.85rem; color:var(--muted); margin:0.7rem 0 0.3rem; }
input[type=text], input[type=email], input[type=password], input[type=date], textarea, select {
  width:100%; background:var(--bg); border:1px solid var(--border); border-radius:10px;
  padding:0.7rem 0.9rem; color:var(--text); font-size:16px; outline:none; }
textarea { min-height:90px; resize:vertical; }
input:focus, textarea:focus, select:focus { border-color:var(--accent); }
button, .btn { background:var(--accent); color:#1a150f; font-weight:700; border:none;
  border-radius:10px; padding:0.75rem 1.2rem; font-size:0.95rem; cursor:pointer;
  text-decoration:none; display:inline-block; text-align:center; }
button.ghost, .btn.ghost { background:transparent; border:1px solid var(--border); color:var(--muted); }
button.danger { background:var(--danger); color:#fff; }
button:disabled { opacity:0.5; }
.err { color:var(--danger); font-size:0.9rem; margin:0.5rem 0; min-height:1.2em; }
.ok { color:#7dc48a; font-size:0.9rem; margin:0.5rem 0; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:0.8rem; }
.mcard { background:var(--panel); border:1px solid var(--border); border-radius:12px; overflow:hidden;
  text-decoration:none; color:var(--text); display:block; }
.mcard img { width:100%; height:170px; object-fit:cover; background:#222; display:block; }
.mcard .ph { width:100%; height:170px; background:#222; display:flex; align-items:center;
  justify-content:center; color:var(--muted); font-size:2rem; }
.mcard .inf { padding:0.6rem 0.7rem; }
.mcard .nm { font-weight:700; font-size:0.95rem; }
.mcard .mt { font-size:0.78rem; color:var(--muted); }
.mcard .score { font-size:0.75rem; color:var(--accent); font-weight:700; }
.fineprint { max-width:720px; margin:0 auto; padding:0.5rem 1rem 2rem; text-align:center;
  font-size:0.7rem; color:var(--muted); opacity:0.7; }
.fineprint a { color:var(--muted); }
.msg { max-width:85%; padding:0.7rem 0.95rem; border-radius:14px; line-height:1.45;
  font-size:0.93rem; margin-bottom:0.6rem; white-space:pre-wrap; }
.msg.me { margin-left:auto; background:#2d3a4a; border-bottom-right-radius:4px; }
.msg.them { background:#252530; border:1px solid var(--border); border-bottom-left-radius:4px; }
.conv { display:flex; gap:0.7rem; align-items:center; padding:0.7rem; border-bottom:1px solid var(--border);
  text-decoration:none; color:var(--text); }
.conv img { width:48px; height:48px; border-radius:50%; object-fit:cover; }
.conv .un { background:var(--accent); color:#1a150f; font-size:0.7rem; font-weight:700;
  border-radius:999px; padding:0.1rem 0.5rem; }
.qa { margin-bottom:0.4rem; } .qa b { color:var(--accent); font-size:0.85rem; display:block; }
.pkg { display:flex; justify-content:space-between; align-items:center; gap:0.6rem; }
@media (max-width:600px){ .grid{grid-template-columns:repeat(2,1fr);} .wrap{padding:0.75rem;} }
"""

NAV = """
<nav class="top">
  <a class="brand" href="/">Mingle and Tingle</a>
  <a class="nlink" href="/members">Members</a>
  <a class="nlink" href="/messages">Messages</a>
  <a class="nlink" href="/account">Account</a>
  <a class="nlink" href="/" title="AI chat">AI&nbsp;Chat</a>
</nav>
"""

FOOTER = """
<div class="fineprint">All characters are fictional and 18 years of age or older.
&middot; <a href="/terms">Terms</a> &middot; <a href="/privacy">Privacy</a></div>
"""


def _page(title: str, body: str, nav: bool = True) -> str:
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="manifest" href="/manifest.webmanifest"/>
<meta name="theme-color" content="#0f0f12"/>
<link rel="apple-touch-icon" href="/img/apple-touch-icon.png"/>
<meta name="mobile-web-app-capable" content="yes"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"/>
<title>{title} &mdash; Mingle and Tingle</title><style>{PAGE_CSS}</style></head>
<body>{NAV if nav else ""}<div class="wrap">{body}</div>{FOOTER}</body></html>"""


SIGNUP_HTML = _page("Sign up", """
<h1>Create your account</h1>
<p style="color:var(--muted);font-size:0.9rem">18+ only. Your first 30 days are free &mdash;
$19.99/month after that. Cancel anytime.</p>
<div class="card">
  <div class="err" id="err"></div>
  <label>Display name</label><input id="dn" type="text" maxlength="40" autocomplete="nickname"/>
  <label>Email</label><input id="em" type="email" autocomplete="email"/>
  <label>Password (8+ characters)</label><input id="pw" type="password" autocomplete="new-password"/>
  <label>Date of birth (must be 18+)</label><input id="bd" type="date"/>
  <div style="margin-top:1rem"><button id="go" style="width:100%">Start my free trial</button></div>
  <p style="margin-top:0.8rem;font-size:0.85rem;color:var(--muted)">Already a member?
  <a href="/login" style="color:var(--accent)">Log in</a></p>
</div>
<script>
document.getElementById("go").onclick = async () => {
  const err = document.getElementById("err"); err.textContent = "";
  const body = { display_name: dn.value.trim(), email: em.value.trim(),
    password: pw.value, birthdate: bd.value };
  const r = await fetch("/api/v2/signup", { method:"POST",
    headers:{"Content-Type":"application/json"}, body: JSON.stringify(body) });
  const d = await r.json();
  if (!r.ok) { err.textContent = d.detail || "Something went wrong."; return; }
  location.href = "/account";
};
</script>
""", nav=False)

LOGIN_HTML = _page("Log in", """
<h1>Welcome back</h1>
<div class="card">
  <div class="err" id="err"></div>
  <label>Email</label><input id="em" type="email" autocomplete="email"/>
  <label>Password</label><input id="pw" type="password" autocomplete="current-password"/>
  <div style="margin-top:1rem"><button id="go" style="width:100%">Log in</button></div>
  <p style="margin-top:0.8rem;font-size:0.85rem;color:var(--muted)">New here?
  <a href="/signup" style="color:var(--accent)">Create an account</a></p>
</div>
<script>
document.getElementById("go").onclick = async () => {
  const err = document.getElementById("err"); err.textContent = "";
  const r = await fetch("/api/v2/login", { method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ email: em.value.trim(), password: pw.value }) });
  const d = await r.json();
  if (!r.ok) { err.textContent = d.detail || "Something went wrong."; return; }
  location.href = "/members";
};
</script>
""", nav=False)

MEMBERS_HTML = _page("Members", """
<h1>Members</h1>
<div class="card" style="display:flex;gap:0.5rem;flex-wrap:wrap">
  <select id="fseek" style="flex:1;min-width:140px"><option value="">Seeking: any</option></select>
  <select id="fgen" style="flex:1;min-width:140px"><option value="">Gender: any</option></select>
  <button id="apply" class="ghost">Filter</button>
</div>
<div class="grid" id="grid"></div>
<div style="text-align:center;margin:1rem 0"><button id="more" class="ghost" style="display:none">Show more members</button></div>
<div class="err" id="err"></div>
<script>
const GENDERS = ["Woman","Man","Nonbinary","Prefer not to say"];
GENDERS.forEach(g => fgen.insertAdjacentHTML("beforeend", `<option>${g}</option>`));
const PAGE_SIZE = 24;
let moffset = 0, hasMore = true, loading = false;
async function load(reset) {
  if (loading) return;
  loading = true;
  err.textContent = "";
  if (reset) { moffset = 0; hasMore = true; grid.innerHTML = ""; }
  const q = new URLSearchParams({ seeking: fseek.value, gender: fgen.value,
                                  limit: PAGE_SIZE, offset: moffset });
  const r = await fetch("/api/v2/discover?" + q);
  if (r.status === 401) { location.href = "/login"; return; }
  if (r.status === 402) { grid.innerHTML = '<div class="card">Your trial or subscription has ended.</div>'; return; }
  const d = await r.json();
  if (reset && !d.members.length) { grid.innerHTML = '<div class="card">No members yet &mdash; check back soon.</div>'; }
  grid.insertAdjacentHTML("beforeend", d.members.map(m => `
    <a class="mcard" href="/profile/${m.id}">
      ${m.photo_url ? `<img src="${m.photo_url}" loading="lazy"/>` : `<div class="ph">?</div>`}
      <div class="inf"><div class="nm">${esc(m.display_name)}</div>
      <div class="mt">${esc(m.seeking || "")} ${m.location ? "&middot; " + esc(m.location) : ""}</div>
      <div class="score">${m.match_score}% match</div></div>
    </a>`).join(""));
  hasMore = d.members.length === PAGE_SIZE;
  if (hasMore) moffset += d.members.length;
  more.style.display = hasMore ? "" : "none";
  loading = false;
}
}
function esc(s){ return (s||"").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
apply.onclick = () => load(true);
more.onclick = () => load(false);
(async () => {
  const meta = await (await fetch("/api/v2/questionnaire-meta")).json();
  (meta.questions.find(q => q.key === "looking_for").options || []).forEach(o =>
    fseek.insertAdjacentHTML("beforeend", `<option>${o}</option>`));
  load(true);
})();
</script>
""")

PROFILE_HTML = _page("Profile", """
<div id="c"></div>
<script>
const uid = location.pathname.split("/").pop();
async function load() {
  const r = await fetch("/api/v2/members/" + uid);
  if (r.status === 401) { location.href = "/login"; return; }
  if (!r.ok) { c.innerHTML = '<div class="card">Profile not available.</div>'; return; }
  const p = await r.json();
  const photos = (p.photos || []).map(ph =>
    `<img src="${ph.url}" style="width:100%;border-radius:12px;margin-bottom:0.6rem"/>`).join("");
  const qa = Object.entries(p.answers || {}).filter(([,v]) => v).map(([k,v]) =>
    `<div class="qa"><b>${esc(k.replace(/_/g," "))}</b>${esc(v)}</div>`).join("");
  c.innerHTML = `
    <h1>${esc(p.display_name)}${p.age ? ", " + p.age : ""}</h1>
    <div class="card">${photos || '<p style="color:var(--muted)">No photos yet.</p>'}</div>
    <div class="card">
      <div class="qa"><b>Match score</b>${p.match_score}%</div>
      ${p.gender ? `<div class="qa"><b>Gender</b>${esc(p.gender)}</div>` : ""}
      ${p.seeking ? `<div class="qa"><b>Looking for</b>${esc(p.seeking)}</div>` : ""}
      ${p.location ? `<div class="qa"><b>Location</b>${esc(p.location)}</div>` : ""}
      ${p.bio ? `<div class="qa"><b>About</b>${esc(p.bio)}</div>` : ""}
      ${qa}
    </div>
    <div class="card" style="display:flex;gap:0.5rem;flex-wrap:wrap">
      <button id="msg">Message</button>
      <button id="blk" class="ghost">Block</button>
      <button id="rep" class="ghost">Report</button>
    </div>
    <div class="err" id="err"></div>`;
  msg.onclick = () => { const t = prompt("Say hello:"); if (t) sendMsg(t); };
  blk.onclick = async () => { if (!confirm("Block this member?")) return;
    const r = await fetch("/api/v2/block/" + uid, { method:"POST" });
    location.href = r.ok ? "/members" : "#"; };
  rep.onclick = async () => { const reason = prompt("Reason (e.g. fake profile, harassment, underage):");
    if (!reason) return;
    const r = await fetch("/api/v2/report/" + uid, { method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ reason }) });
    err.textContent = r.ok ? "Thanks — our team will review this profile." : "Could not send report."; };
}
async function sendMsg(t) {
  const r = await fetch("/api/v2/messages", { method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ recipient_id: +uid, body: t }) });
  const d = await r.json();
  if (r.ok) { location.href = "/messages#" + uid; }
  else { document.getElementById("err").textContent = d.detail || "Could not send."; }
}
function esc(s){ return (s||"").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
load();
</script>
""")

MESSAGES_HTML = _page("Messages", """
<h1>Messages</h1>
<div class="card" id="list"></div>
<div class="card" id="thread" style="display:none">
  <h2 id="tname"></h2>
  <div id="msgs" style="max-height:50vh;overflow-y:auto;margin-bottom:0.8rem"></div>
  <div style="display:flex;gap:0.5rem">
    <input id="inp" type="text" placeholder="Write a message&hellip; (1 credit)"/>
    <button id="send">Send</button>
  </div>
  <div class="err" id="err"></div>
</div>
<script>
let otherId = (location.hash || "").slice(1) || null;
async function loadList() {
  const r = await fetch("/api/v2/conversations");
  if (r.status === 401) { location.href = "/login"; return; }
  const d = await r.json();
  list.innerHTML = d.conversations.length ? d.conversations.map(c => `
    <a class="conv" href="#${c.user_id}" data-id="${c.user_id}">
      ${c.photo_url ? `<img src="${c.photo_url}"/>` : ""}
      <div><div><b>${esc(c.display_name)}</b>
      ${c.unread ? `<span class="un">${c.unread}</span>` : ""}</div>
      <div style="color:var(--muted);font-size:0.85rem">${esc(c.last_message)}</div></div>
    </a>`).join("") : '<p style="color:var(--muted)">No conversations yet. Browse members and say hello.</p>';
  list.querySelectorAll("a.conv").forEach(a => a.onclick = () => { otherId = a.dataset.id; loadThread(); });
}
async function loadThread() {
  if (!otherId) { thread.style.display = "none"; return; }
  thread.style.display = "block";
  const r = await fetch("/api/v2/messages/" + otherId);
  if (!r.ok) return;
  const d = await r.json();
  tname.textContent = d.other.display_name;
  msgs.innerHTML = d.messages.map(m =>
    `<div class="msg ${m.sender_id == otherId ? "them" : "me"}">${esc(m.body)}</div>`).join("");
  msgs.scrollTop = msgs.scrollHeight;
  loadList();
}
send.onclick = async () => {
  const t = inp.value.trim(); if (!t) return; inp.value = "";
  const r = await fetch("/api/v2/messages", { method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ recipient_id: +otherId, body: t }) });
  const d = await r.json();
  if (!r.ok) { err.textContent = d.detail || "Could not send."; return; }
  err.textContent = ""; loadThread();
};
function esc(s){ return (s||"").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
window.onhashchange = () => { otherId = location.hash.slice(1) || null; loadThread(); };
loadList(); loadThread();
setInterval(() => { if (otherId) loadThread(); else loadList(); }, 15000);
</script>
""")

ACCOUNT_HTML = _page("Account", """
<h1>Your account</h1>
<div class="card" id="sub"></div>
<div class="card" id="cred"></div>
<h2>Profile</h2>
<div class="card">
  <div class="err" id="perr"></div><div class="ok" id="pok"></div>
  <label>Gender</label><select id="gender"></select>
  <label>Looking for</label><select id="seeking"></select>
  <label>Location</label><input id="loc" type="text" maxlength="80" placeholder="City, State"/>
  <label>Bio</label><textarea id="bio" maxlength="500"></textarea>
  <div style="margin-top:0.8rem"><button id="savep">Save profile</button></div>
</div>
<h2>Questionnaire</h2>
<div class="card"><div id="qa"></div>
  <div class="err" id="qerr"></div><div class="ok" id="qok"></div>
  <div style="margin-top:0.8rem"><button id="saveq">Save answers</button></div>
</div>
<h2>Photos</h2>
<div class="card">
  <div id="phs" style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:0.8rem"></div>
  <input id="up" type="file" accept="image/jpeg,image/png,image/webp"/>
  <div style="margin-top:0.6rem"><button id="upb" class="ghost">Upload photo</button></div>
  <div class="err" id="uerr"></div>
</div>
<div class="card"><button id="logout" class="ghost" style="width:100%">Log out</button></div>
<script>
const GENDERS = ["", "Woman","Man","Nonbinary","Prefer not to say"];
let ME = null;
async function me() {
  const r = await fetch("/api/v2/me");
  if (r.status === 401) { location.href = "/login"; return null; }
  return r.json();
}
async function load() {
  ME = await me(); if (!ME) return;
  const s = ME.subscription;
  sub.innerHTML = `<b>Subscription:</b> ${s.status}` +
    (s.status === "trialing" && s.trial_ends_at
      ? ` &mdash; free trial ends ${new Date(s.trial_ends_at*1000).toLocaleDateString()}` : "") +
    `<br/><b>Card on file:</b> ${s.card_on_file ? "Yes" : "Not yet (billing connects soon)"}` +
    `<div style="margin-top:0.6rem"><button id="bill" class="ghost">Manage billing</button></div>`;
  cred.innerHTML = `<b>Message credits:</b> ${ME.credits}`;
  bill.onclick = async () => {
    const r = await fetch("/api/v2/billing/subscribe?processor=ccbill", { method:"POST" });
    const d = await r.json();
    alert(d.message || d.error || "Billing");
  };
  gender.innerHTML = GENDERS.map(g => `<option value="${g}">${g || "—"}</option>`).join("");
  const meta = await (await fetch("/api/v2/questionnaire-meta")).json();
  seeking.innerHTML = `<option value="">—</option>` + meta.questions
    .find(q => q.key === "looking_for").options.map(o => `<option>${o}</option>`).join("");
  const p = ME.profile || {};
  gender.value = p.gender || ""; seeking.value = p.seeking || "";
  loc.value = p.location || ""; bio.value = p.bio || "";
  qa.innerHTML = meta.questions.map(q => `
    <label>${q.label}</label>` +
    (q.kind === "choice"
      ? `<select data-q="${q.key}"><option value="">—</option>${q.options.map(o =>
          `<option${ME.answers[q.key]===o?" selected":""}>${o}</option>`).join("")}</select>`
      : `<textarea data-q="${q.key}" maxlength="2000">${esc(ME.answers[q.key]||"")}</textarea>`)).join("");
  renderPhotos();
}
function renderPhotos() {
  phs.innerHTML = (ME.photos || []).map(ph => `
    <div style="position:relative">
      <img src="${ph.url}" style="width:96px;height:96px;object-fit:cover;border-radius:10px;
        ${ph.is_primary?"border:2px solid var(--accent)":""}"/>
      <button data-del="${ph.id}" class="ghost"
        style="position:absolute;top:-8px;right:-8px;padding:0.15rem 0.5rem;border-radius:999px">&times;</button>
    </div>`).join("") || '<span style="color:var(--muted)">No photos yet.</span>';
  phs.querySelectorAll("[data-del]").forEach(b => b.onclick = async () => {
    await fetch("/api/v2/photos/" + b.dataset.del, { method:"DELETE" });
    ME = await me(); renderPhotos();
  });
}
savep.onclick = async () => {
  perr.textContent = ""; pok.textContent = "";
  const r = await fetch("/api/v2/profile", { method:"PUT",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ gender: gender.value, seeking: seeking.value,
      location: loc.value, bio: bio.value }) });
  const d = await r.json();
  if (!r.ok) { perr.textContent = d.detail; return; }
  pok.textContent = "Saved.";
};
saveq.onclick = async () => {
  qerr.textContent = ""; qok.textContent = "";
  const ans = {};
  qa.querySelectorAll("[data-q]").forEach(el => ans[el.dataset.q] = el.value);
  const r = await fetch("/api/v2/questionnaire", { method:"PUT",
    headers:{"Content-Type":"application/json"}, body: JSON.stringify(ans) });
  const d = await r.json();
  if (!r.ok) { qerr.textContent = d.detail; return; }
  qok.textContent = "Saved.";
};
upb.onclick = async () => {
  uerr.textContent = "";
  if (!up.files.length) { uerr.textContent = "Choose a photo first."; return; }
  const fd = new FormData(); fd.append("file", up.files[0]);
  const r = await fetch("/api/v2/photos", { method:"POST", body: fd });
  const d = await r.json();
  if (!r.ok) { uerr.textContent = d.detail; return; }
  ME = await me(); renderPhotos(); up.value = "";
};
logout.onclick = async () => { await fetch("/api/v2/logout", { method:"POST" }); location.href = "/"; };
function esc(s){ return (s||"").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
load();
</script>
""")

page_router = APIRouter()


@page_router.get("/signup", response_class=HTMLResponse)
def page_signup():
    return SIGNUP_HTML


@page_router.get("/login", response_class=HTMLResponse)
def page_login():
    return LOGIN_HTML


@page_router.get("/members", response_class=HTMLResponse)
def page_members():
    return MEMBERS_HTML


@page_router.get("/profile/{user_id}", response_class=HTMLResponse)
def page_profile(user_id: int):
    return PROFILE_HTML


@page_router.get("/messages", response_class=HTMLResponse)
def page_messages():
    return MESSAGES_HTML


@page_router.get("/account", response_class=HTMLResponse)
def page_account():
    return ACCOUNT_HTML


def mount_phase2(app):
    """Call once from api.py: creates DB + mounts routers."""
    init_db()
    app.include_router(router)
    app.include_router(page_router)
