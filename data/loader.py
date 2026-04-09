"""
Data Loader — EtaNexus v6.0
===========================
Loads cases from JSON and generates dual embeddings.
v6 adds:
  - extract_statutes()  : IPC/CrPC/bare-act section parser
  - statute enrichment  : stamped onto each case dict at load time
"""
import json
import re
import torch
from sentence_transformers import SentenceTransformer

# ── Statute Whitelist ──────────────────────────────────────────────────────────
# Curated top-30 most-litigated IPC/CrPC sections in Indian constitutional law.
# Avoids noisy matches on every number in the text.
IPC_WHITELIST = {
    "302", "304", "304B", "307", "354", "376", "377", "420", "498A",
    "499", "500", "120B", "34", "149", "186", "336", "337", "338",
}
CRPC_WHITELIST = {
    "41", "154", "161", "162", "164", "167", "173", "197", "227",
    "313", "320", "357", "438", "439", "482",
}

IPC_PATTERN  = re.compile(r"(?:IPC|Indian Penal Code)[^0-9]{0,15}[Ss](?:ection)?\s*[§]?\s*(\d+[A-Z]?)")
CRPC_PATTERN = re.compile(r"(?:Cr\.?P\.?C\.?|Code of Criminal Procedure)[^0-9]{0,15}[Ss](?:ection)?\s*[§]?\s*(\d+[A-Z]?)")
BARE_SEC_PATTERN = re.compile(r"[Ss]ection\s+(\d+[A-Z]?)\s+of\s+(?:the\s+)?(?:IPC|Indian Penal Code)")


def extract_statutes(text: str) -> list[str]:
    """
    Extract IPC and CrPC section references from legal text.
    Uses a curated whitelist to avoid false positives on every numeric reference.

    Returns deduplicated list like ["IPC §302", "CrPC §161"].
    """
    statutes = set()
    for m in IPC_PATTERN.findall(text):
        sec = m.strip().upper()
        if sec in IPC_WHITELIST:
            statutes.add(f"IPC §{sec}")
    for m in CRPC_PATTERN.findall(text):
        sec = m.strip().upper()
        if sec in CRPC_WHITELIST:
            statutes.add(f"CrPC §{sec}")
    for m in BARE_SEC_PATTERN.findall(text):
        sec = m.strip().upper()
        if sec in IPC_WHITELIST:
            statutes.add(f"IPC §{sec}")
    return sorted(statutes)


def load_cases(file_path: str) -> list:
    """Load case data from JSON file, enriching with statute tags."""
    with open(file_path, 'r') as f:
        cases = json.load(f)
    # Stamp statute extraction onto each case (in-place, non-destructive)
    for case in cases:
        if 'statutes' not in case:
            text = case.get('summary', '') + ' ' + case.get('title', '')
            case['statutes'] = extract_statutes(text)
    print(f"Loaded {len(cases)} cases from {file_path}")
    return cases


def generate_text_embeddings(cases: list, model_name: str = 'all-MiniLM-L6-v2') -> torch.Tensor:
    """Generate semantic embeddings for case summaries."""
    print(f"Generating embeddings using {model_name}...")
    model = SentenceTransformer(model_name)
    summaries = [c['summary'] for c in cases]
    embeddings = model.encode(summaries)
    return torch.tensor(embeddings, dtype=torch.float)

