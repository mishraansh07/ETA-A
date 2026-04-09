"""
LegalGAT — Graph Attention Network for Legal Citation Analysis.
Drop-in replacement for GraphSAGE; outputs 32-dim node embeddings.

Architecture:
  Layer 1: GATConv(in → hidden, heads=4, concat=True)  → hidden*4
  Layer 2: GATConv(hidden*4 → out, heads=1, concat=False) → out
  
Why GAT over GraphSAGE for legal graphs:
- Legal cases cite each other with UNEQUAL importance (landmark case ≠ minor reference)
- Multi-head attention learns WHY a precedent is cited, not just THAT it is cited
- Produces interpretable attention weights — we can visualize which citations the model
  considers most doctrinally significant
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class LegalGAT(torch.nn.Module):
    """
    Graph Attention Network tuned for legal citation graphs.
    Uses 4-head attention in layer 1 to capture diverse citation reasons
    (procedural, doctrinal, constitutional, evidentiary).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32,
                 out_channels: int = 32, heads: int = 4, dropout: float = 0.3):
        super(LegalGAT, self).__init__()
        self.dropout = dropout

        # Layer 1: multi-head attention (concat=True → output = hidden * heads)
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        # Layer 2: single-head, averaging (concat=False → output = out_channels)
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # ELU preferred over ReLU for attention-based models
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_attention_weights(self, x: torch.Tensor,
                              edge_index: torch.Tensor):
        """
        Returns (embeddings, attention_weights) for visualization.
        Useful for explaining WHY the model relates two cases.
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index_out, alpha1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (_, alpha2) = self.conv2(x, edge_index, return_attention_weights=True)
        return x, alpha1.mean(dim=-1)  # Return layer-1 attention (most interpretable)
