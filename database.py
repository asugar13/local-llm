import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "conversations.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                title         TEXT    NOT NULL DEFAULT 'New conversation',
                model         TEXT    NOT NULL,
                system_prompt TEXT    NOT NULL,
                created_at    TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL
                                REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                created_at      TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mood_ratings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL
                                REFERENCES conversations(id) ON DELETE CASCADE,
                rating          INTEGER NOT NULL,
                created_at      TEXT    NOT NULL
            );
        """)
        conn.commit()
    finally:
        conn.close()


def create_conversation(model: str, system_prompt: str) -> int:
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO conversations (title, model, system_prompt, created_at) VALUES (?, ?, ?, ?)",
            ("New conversation", model, system_prompt, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def set_conversation_title(conversation_id: int, title: str) -> None:
    title = title[:40].strip()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_conversation_meta(conversation_id: int, model: str, system_prompt: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE conversations SET model = ?, system_prompt = ? WHERE id = ?",
            (model, system_prompt, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def save_message(conversation_id: int, role: str, content: str) -> int:
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_conversations() -> list:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT c.id, c.title, c.model, c.system_prompt, c.created_at "
            "FROM conversations c "
            "WHERE EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id) "
            "ORDER BY c.created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def load_conversation(conversation_id: int) -> dict | None:
    conn = _connect()
    try:
        conv = conn.execute(
            "SELECT id, title, model, system_prompt, created_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conv is None:
            return None
        msgs = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()
        result = dict(conv)
        result["messages"] = [dict(m) for m in msgs]
        return result
    finally:
        conn.close()


def replace_messages(conversation_id: int, messages: list) -> None:
    """Delete all messages for a conversation and re-insert the given list."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        for msg in messages:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, msg["role"], msg["content"], datetime.utcnow().isoformat()),
            )
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conversation_id: int) -> None:
    conn = _connect()
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()


def save_mood_rating(conversation_id: int, rating: int) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO mood_ratings (conversation_id, rating, created_at) VALUES (?, ?, ?)",
            (conversation_id, rating, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_mood_history() -> list:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT c.title, c.created_at, mr.rating "
            "FROM mood_ratings mr "
            "JOIN conversations c ON c.id = mr.conversation_id "
            "ORDER BY mr.created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
