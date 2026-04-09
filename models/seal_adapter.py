"""
SEAL-Inspired Legal Knowledge Adapter
======================================
Inspired by MIT CSAIL's "Self-Adapting Language Models" (SEAL, arXiv:2506.10943).

Full SEAL: LLM generates "self-edits" → lightweight SFT + RL reward loop → weight updates.

Our implementation (practical adaptation without GPU fine-tuning infra):
  ┌─ SEAL Core Principle ──────────────────────────────────────────────────────┐
  │  The model should generate its own structured representation of new        │
  │  knowledge, store it persistently, and use it to improve future responses. │
  └────────────────────────────────────────────────────────────────────────────┘

When a PDF is uploaded:
  1. Gemini "studies" the document → produces a structured "Legal Self-Edit"
     (implications, analogies, doctrinal fingerprint, key statutes)
  2. Self-Edit is stored persistently in SQLite (legal_knowledge table)
  3. Future queries: SEAL Adapter retrieves relevant self-edits via keyword match
     → prepended to Gemini prompt as "learned prior knowledge"
  4. Adaptation signal: cases saved by user after SEAL-augmented responses
     are tracked → boosts that self-edit's priority for future use

This is SEAL's "knowledge incorporation" mode, mapped to legal domain.
"""
import json
import re
import sqlite3
from typing import Optional


DB_PATH = "results/chat_history.db"


# ── Schema ────────────────────────────────────────────────────────────────────

def init_knowledge_table():
    """Add legal_knowledge table if it doesn't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS legal_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER,
            source_doc TEXT,
            self_edit_json TEXT NOT NULL,
            keywords TEXT,
            query_count INTEGER DEFAULT 0,
            save_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# ── Self-Edit Generation ───────────────────────────────────────────────────────

SELF_EDIT_PROMPT = """You are EtaNexus Legal Memory Engine (inspired by MIT SEAL).
Analyze this legal document and produce a structured "Legal Self-Edit" — a machine-readable
knowledge record that can help answer future questions about similar legal situations.

DOCUMENT (excerpt):
\"\"\"
{text}
\"\"\"

TASK: Generate a JSON object with these fields:
{{
  "doctrinal_fingerprint": "One-sentence core legal principle this case establishes",
  "key_articles": ["Article 21", "Article 14", ...],
  "key_statutes": ["IPC §302", "CrPC §161", ...],
  "logical_implications": [
    "IF <condition> AND <legal principle> THEN <outcome>",
    ...  (generate 3-5 implications)
  ],
  "landmark_analogies": [
    "Similar to <landmark case> because <reason>",
    ...  (generate 2-3 analogies)
  ],
  "doctrinal_cluster": "Basic Structure | Privacy | Reservation | Environment | Criminal | Other",
  "precedent_strength": "Binding | Persuasive | Historical",
  "one_line_summary": "Plain English summary of the case's legal impact"
}}

Return ONLY valid JSON. No markdown, no explanation."""


def generate_self_edit(text: str, gemini_model) -> Optional[dict]:
    """
    Uses Gemini to generate a structured Legal Self-Edit from document text.
    This is the SEAL "self-edit generation" step.
    
    Returns:
        Parsed self-edit dict, or None if generation fails.
    """
    prompt = SELF_EDIT_PROMPT.format(text=text[:3000])
    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[SEAL] Self-edit generation failed: {e}")
        # Fallback: minimal structured edit from text alone
        return _fallback_self_edit(text)


def _fallback_self_edit(text: str) -> dict:
    """Regex-based fallback when Gemini is unavailable."""
    articles = list(set(re.findall(r"Article\s+\d+[A-Z]?", text)))
    ipc = list(set(re.findall(r"(?:IPC|Section)\s*[§]?\s*(\d+[A-Z]?)", text)))
    return {
        "doctrinal_fingerprint": "Extracted via pattern analysis",
        "key_articles": articles[:5],
        "key_statutes": [f"IPC §{s}" for s in ipc[:3]],
        "logical_implications": [],
        "landmark_analogies": [],
        "doctrinal_cluster": "Other",
        "precedent_strength": "Persuasive",
        "one_line_summary": text[:200].replace("\n", " ")
    }


# ── Persistent Storage ────────────────────────────────────────────────────────

def store_knowledge(workspace_id: int, source_doc: str, self_edit: dict) -> int:
    """
    Persist a self-edit to the knowledge store.
    Returns the inserted row ID.
    """
    # Build keyword index from self-edit for retrieval
    keywords = " ".join([
        self_edit.get("doctrinal_cluster", ""),
        self_edit.get("doctrinal_fingerprint", ""),
        " ".join(self_edit.get("key_articles", [])),
        " ".join(self_edit.get("key_statutes", [])),
        self_edit.get("one_line_summary", "")[:200],
    ]).lower()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO legal_knowledge
           (workspace_id, source_doc, self_edit_json, keywords)
           VALUES (?, ?, ?, ?)""",
        (workspace_id, source_doc, json.dumps(self_edit), keywords)
    )
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    print(f"[SEAL] Self-edit stored → knowledge_id={row_id}, doc={source_doc}")
    return row_id


def get_relevant_knowledge(query: str, workspace_id: Optional[int] = None,
                            top_k: int = 3) -> list:
    """
    Retrieve relevant self-edits for a given query using keyword matching.
    Priority: save_count DESC (cases saved after this knowledge was used = quality signal)
    
    Returns list of self-edit dicts.
    """
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 3]

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if workspace_id:
        c.execute(
            """SELECT id, source_doc, self_edit_json, save_count FROM legal_knowledge
               WHERE workspace_id = ?
               ORDER BY save_count DESC, created_at DESC LIMIT 20""",
            (workspace_id,)
        )
    else:
        c.execute(
            """SELECT id, source_doc, self_edit_json, save_count FROM legal_knowledge
               ORDER BY save_count DESC, created_at DESC LIMIT 20"""
        )
    rows = c.fetchall()
    conn.close()

    # Score by keyword overlap
    scored = []
    for row_id, source_doc, self_edit_json, save_count in rows:
        try:
            self_edit = json.loads(self_edit_json)
        except Exception:
            continue
        keywords = " ".join([
            self_edit.get("doctrinal_fingerprint", ""),
            " ".join(self_edit.get("key_articles", [])),
            self_edit.get("one_line_summary", ""),
            self_edit.get("doctrinal_cluster", ""),
        ]).lower()
        overlap = sum(1 for w in query_words if w in keywords)
        if overlap > 0:
            scored.append((overlap + save_count * 0.5, row_id, source_doc, self_edit))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"knowledge_id": r[1], "source_doc": r[2], "self_edit": r[3]}
        for r in scored[:top_k]
    ]


def record_usage(knowledge_id: int, saved: bool = False):
    """
    Track usage of a self-edit (query_count) and quality signal (save_count).
    This is the lightweight analog of SEAL's RL reward loop.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if saved:
        c.execute(
            "UPDATE legal_knowledge SET query_count=query_count+1, save_count=save_count+1 WHERE id=?",
            (knowledge_id,)
        )
    else:
        c.execute(
            "UPDATE legal_knowledge SET query_count=query_count+1 WHERE id=?",
            (knowledge_id,)
        )
    conn.commit()
    conn.close()


def list_knowledge(workspace_id: int) -> list:
    """List all self-edits stored for a workspace."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """SELECT id, source_doc, self_edit_json, query_count, save_count, created_at
           FROM legal_knowledge WHERE workspace_id=?
           ORDER BY created_at DESC""",
        (workspace_id,)
    )
    rows = c.fetchall()
    conn.close()
    result = []
    for row in rows:
        try:
            self_edit = json.loads(row[2])
        except Exception:
            self_edit = {}
        result.append({
            "id": row[0],
            "source_doc": row[1],
            "fingerprint": self_edit.get("doctrinal_fingerprint", "N/A"),
            "cluster": self_edit.get("doctrinal_cluster", "Other"),
            "query_count": row[3],
            "save_count": row[4],
            "created_at": row[5],
        })
    return result


# ── Prompt Augmentation ───────────────────────────────────────────────────────

def augment_prompt_with_knowledge(base_prompt: str, query: str,
                                   workspace_id: Optional[int] = None) -> tuple[str, list]:
    """
    Prepend relevant self-edits to the Gemini prompt.
    Returns (augmented_prompt, list_of_used_knowledge_ids).
    
    This is the SEAL "knowledge-augmented inference" step.
    """
    relevant = get_relevant_knowledge(query, workspace_id=workspace_id, top_k=2)
    if not relevant:
        return base_prompt, []

    knowledge_blocks = []
    used_ids = []
    for item in relevant:
        se = item["self_edit"]
        block = (
            f"[PRIOR KNOWLEDGE from '{item['source_doc']}']\n"
            f"  Core Principle: {se.get('doctrinal_fingerprint', 'N/A')}\n"
            f"  Key Articles: {', '.join(se.get('key_articles', []))}\n"
            f"  Logical Implications:\n"
            + "\n".join(f"    • {imp}" for imp in se.get("logical_implications", [])[:3])
            + f"\n  Analogies: {'; '.join(se.get('landmark_analogies', [])[:2])}"
        )
        knowledge_blocks.append(block)
        used_ids.append(item["knowledge_id"])
        record_usage(item["knowledge_id"])

    seal_context = (
        "=== SEAL LEGAL MEMORY (Prior Cases Analyzed) ===\n"
        + "\n\n".join(knowledge_blocks)
        + "\n=== END SEAL CONTEXT ===\n\n"
    )
    return seal_context + base_prompt, used_ids
