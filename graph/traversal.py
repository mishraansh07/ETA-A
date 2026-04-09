"""
Citation Graph Traversal — GraphRAG path finding for legal reasoning.

Used by the GraphRAG Narrative Engine to inject structured citation paths
into Gemini's prompt, enabling it to reason about doctrinal evolution:
  "Kesavananda Bharati (1973) → Minerva Mills (1980) → I.R. Coelho (2007)"
"""
from collections import deque
from typing import Optional


def find_citation_path(
    source_id: str,
    target_id: str,
    cases: list,
    max_hops: int = 3,
) -> Optional[list[str]]:
    """
    BFS shortest citation path from source to target within max_hops.
    
    Returns:
        List of case IDs representing the path, or None if not reachable.
        e.g. ["case_001", "case_004", "case_019"]
    """
    if source_id == target_id:
        return [source_id]

    # Build adjacency map: case_id → list of cited case_ids
    adj: dict[str, list[str]] = {}
    for case in cases:
        adj[case["id"]] = case.get("citations", [])

    # BFS
    queue = deque([[source_id]])
    visited = {source_id}

    while queue:
        path = queue.popleft()
        current = path[-1]

        if len(path) > max_hops + 1:
            break

        for neighbor in adj.get(current, []):
            if neighbor == target_id:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    return None  # No path within max_hops


def get_ego_graph(case_id: str, cases: list, depth: int = 2) -> dict:
    """
    Returns the ego-graph (neighbourhood) around a given case.
    Useful for the Citation Graph UI to highlight connected subgraphs.

    Returns:
        {"nodes": [{"id", "title", "year"}], "edges": [{"source", "target"}]}
    """
    id_to_case = {c["id"]: c for c in cases}
    visited: dict[str, int] = {}  # case_id → depth level

    queue = deque([(case_id, 0)])
    nodes, edges = [], []

    while queue:
        cid, d = queue.popleft()
        if cid in visited or d > depth:
            continue
        visited[cid] = d
        c = id_to_case.get(cid)
        if not c:
            continue
        nodes.append({"id": cid, "title": c["title"], "year": c.get("year")})

        for cited_id in c.get("citations", []):
            edges.append({"source": cid, "target": cited_id})
            if cited_id not in visited and d + 1 <= depth:
                queue.append((cited_id, d + 1))

    return {"nodes": nodes, "edges": edges}


def format_path_narrative(path: list[str], cases: list) -> str:
    """
    Converts a citation path list of IDs into a human-readable narrative string.
    Used for injecting into Gemini's prompt context.

    Example output:
      "Kesavananda Bharati v. State of Kerala (1973)
       → CITED BY → Minerva Mills v. Union of India (1980)
       → CITED BY → I.R. Coelho v. State of Tamil Nadu (2007)"
    """
    id_to_case = {c["id"]: c for c in cases}
    parts = []
    for cid in path:
        c = id_to_case.get(cid, {})
        title = c.get("title", cid)
        year = c.get("year", "")
        label = f"{title} ({year})" if year else title
        parts.append(label)
    return "\n  → CITED BY → ".join(parts)


def find_all_paths_from(source_id: str, cases: list, max_hops: int = 2) -> list:
    """
    Find all reachable cases from source within max_hops via citation edges.
    Used for GraphRAG context building — returns the 'doctrinal neighbourhood'.

    Returns:
        List of {"id", "title", "year", "hops"} dicts, ordered by hop distance.
    """
    id_to_case = {c["id"]: c for c in cases}
    adj: dict[str, list[str]] = {c["id"]: c.get("citations", []) for c in cases}

    visited: dict[str, int] = {}
    queue = deque([(source_id, 0)])
    reachable = []

    while queue:
        cid, hops = queue.popleft()
        if cid in visited or hops > max_hops:
            continue
        visited[cid] = hops
        if cid != source_id:
            c = id_to_case.get(cid, {})
            reachable.append({
                "id": cid,
                "title": c.get("title", cid),
                "year": c.get("year"),
                "hops": hops,
            })
        for neighbor in adj.get(cid, []):
            if neighbor not in visited:
                queue.append((neighbor, hops + 1))

    return sorted(reachable, key=lambda x: x["hops"])
