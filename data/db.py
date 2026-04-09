import sqlite3
import json
import os

DB_PATH = "results/chat_history.db"

def init_db():
    os.makedirs("results", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            refs TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        );

        CREATE TABLE IF NOT EXISTS saved_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER,
            case_id TEXT,
            case_title TEXT,
            notes TEXT,
            saved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        );

        CREATE TABLE IF NOT EXISTS legal_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER,
            source_doc TEXT,
            self_edit_json TEXT NOT NULL,
            keywords TEXT,
            query_count INTEGER DEFAULT 0,
            save_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


# ── Workspaces ──────────────────────────────────────────────────────────────
def create_workspace(name: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO workspaces (name) VALUES (?)", (name,))
    conn.commit()
    ws_id = c.lastrowid
    c.execute("SELECT id, name, created_at FROM workspaces WHERE id=?", (ws_id,))
    row = c.fetchone()
    conn.close()
    return {"id": row[0], "name": row[1], "created_at": row[2]}

def list_workspaces() -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, created_at FROM workspaces ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]

# ── Messages ─────────────────────────────────────────────────────────────────
def save_message(role: str, content: str, refs=None, workspace_id=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (workspace_id, role, content, refs) VALUES (?, ?, ?, ?)",
        (workspace_id, role, content, json.dumps(refs) if refs else None)
    )
    conn.commit()
    conn.close()

def get_history(workspace_id=None, limit=50) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if workspace_id:
        c.execute(
            "SELECT role, content, refs, timestamp FROM messages WHERE workspace_id=? ORDER BY timestamp DESC LIMIT ?",
            (workspace_id, limit)
        )
    else:
        c.execute(
            "SELECT role, content, refs, timestamp FROM messages ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "refs": json.loads(r[2]) if r[2] else [], "timestamp": r[3]} for r in rows][::-1]

# ── Saved Cases ───────────────────────────────────────────────────────────────
def save_case(workspace_id: int, case_id: str, case_title: str, notes: str = "") -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO saved_cases (workspace_id, case_id, case_title, notes) VALUES (?, ?, ?, ?)",
        (workspace_id, case_id, case_title, notes)
    )
    conn.commit()
    sc_id = c.lastrowid
    conn.close()
    return {"id": sc_id, "case_id": case_id, "case_title": case_title}

def get_saved_cases(workspace_id: int) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, case_id, case_title, notes, saved_at FROM saved_cases WHERE workspace_id=? ORDER BY saved_at DESC",
        (workspace_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "case_id": r[1], "case_title": r[2], "notes": r[3], "saved_at": r[4]} for r in rows]
