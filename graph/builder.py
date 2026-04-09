import torch
import numpy as np
from torch_geometric.data import Data

def build_pyg_graph(cases, text_embeddings):
    """
    Build a Professional Directed Graph from case data.
    Nodes are initialized with features: [Text_Embedding (384) + Year (1) + Court_Level (1) + Article_Count (1)]
    """
    id_to_idx = {case['id']: i for i, case in enumerate(cases)}
    
    # 1. Feature Engineering (Professional Metadata)
    num_nodes = len(cases)
    text_feat = text_embeddings
    if isinstance(text_feat, np.ndarray):
        text_feat = torch.from_numpy(text_feat).float()
    
    meta_feats = []
    for case in cases:
        # Normalize Year (1950-2025)
        year = case.get('year', 2000)
        norm_year = (year - 1950) / (2025 - 1950)
        
        # Normalize Court Level (0-2)
        court = case.get('court_level', 2)
        norm_court = court / 2.0
        
        # Normalize Article Count (assume max 10 for normalization)
        num_articles = len(case.get('articles', []))
        norm_articles = min(num_articles / 10.0, 1.0)
        
        meta_feats.append([norm_year, norm_court, norm_articles])
    
    meta_feats = torch.tensor(meta_feats, dtype=torch.float)
    x = torch.cat([text_feat, meta_feats], dim=1)
    
    # 2. Directed Edges (Citations)
    edge_index = []
    for i, case in enumerate(cases):
        for citation_id in case.get('citations', []):
            if citation_id in id_to_idx:
                target_idx = id_to_idx[citation_id]
                # Directed edge: Case i (newer) cites target_idx (older precedent)
                edge_index.append([i, target_idx])
    
    if len(edge_index) == 0:
        # Fallback to self-loops if no citations found
        edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
    data = Data(x=x, edge_index=edge_index)
    return data, id_to_idx
