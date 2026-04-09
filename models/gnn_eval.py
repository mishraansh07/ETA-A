"""
GNN Stress Test & Performance Harness — EtaNexus v6.0
==================================================
Runs industrial-grade benchmarking of:
  • GraphSAGE
  • LegalGAT
  • GCN (baseline)

Metric Scope: MRR Stability (5 trials), Throughput, and Latency.
"""
import argparse
import sys
import time
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# ── Paths setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.loader import load_cases, generate_text_embeddings
from src.graph.builder import build_pyg_graph
from src.models.gnn_sage import GraphSAGE
from src.models.gnn_gat import LegalGAT
from src.models.trainer import train_link_prediction

# ── GCN baseline ───────────────────────────────────────────────────────────────
try:
    from torch_geometric.nn import GCNConv
    class LegalGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels=64, out_channels=32):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index)
            return x
    GCN_AVAILABLE = True
except ImportError:
    GCN_AVAILABLE = False


# ── Non-Graph Baseline: Simple MLP ─────────────────────────────────────────────

class SimpleMLP(torch.nn.Module):
    """
    Baseline that ignores graph structure. 
    Predicts links purely based on semantic and metadata feature similarity.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=32):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None):
        # edge_index is ignored to simulate a purely feature-based model
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        return x

# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_mrr(embeddings: np.ndarray, pos_edges: np.ndarray,
                all_edges_set: set, num_neg: int = 50) -> float:
    n = len(embeddings)
    rr_list = []
    for src, tgt in pos_edges:
        neg_targets = []
        attempts = 0
        while len(neg_targets) < num_neg and attempts < num_neg * 10:
            rand_t = np.random.randint(0, n)
            if (src, rand_t) not in all_edges_set and rand_t != src:
                neg_targets.append(rand_t)
            attempts += 1
        if not neg_targets: continue
        pos_score = float((embeddings[src] * embeddings[tgt]).sum())
        neg_scores = [(embeddings[src] * embeddings[nt]).sum() for nt in neg_targets]
        rank = 1 + sum(1 for ns in neg_scores if ns >= pos_score)
        rr_list.append(1.0 / rank)
    return float(np.mean(rr_list)) if rr_list else 0.0

def compute_precision_at_k(embeddings: np.ndarray, pos_edges: np.ndarray,
                            all_edge_set: set, k: int = 5) -> float:
    hits = 0
    total = 0
    sims_all = cos_sim(embeddings, embeddings)
    for src, tgt in pos_edges:
        scores = sims_all[src].copy()
        scores[src] = -1
        top_k_ids = np.argsort(scores)[::-1][:k]
        if tgt in top_k_ids: hits += 1
        total += 1
    return hits / total if total > 0 else 0.0

def compute_hit_at_k(embeddings: np.ndarray, pos_edges: np.ndarray,
                      all_edge_set: set, k: int = 10) -> float:
    return compute_precision_at_k(embeddings, pos_edges, all_edge_set, k=k)

def compute_auc(embeddings: np.ndarray, pos_edges: np.ndarray,
                num_neg: int = 500) -> float:
    n = len(embeddings)
    all_pos_set = set(map(tuple, pos_edges.tolist()))
    neg_edges = []
    attempts = 0
    while len(neg_edges) < num_neg and attempts < num_neg * 10:
        src = np.random.randint(0, n)
        tgt = np.random.randint(0, n)
        if (src, tgt) not in all_pos_set and src != tgt:
            neg_edges.append((src, tgt))
        attempts += 1
    if not neg_edges: return 0.5
    src_emb = embeddings[pos_edges[:num_neg, 0]]
    tgt_emb = embeddings[pos_edges[:num_neg, 1]]
    pos_scores = (src_emb * tgt_emb).sum(axis=1)
    neg_edges_arr = np.array(neg_edges)
    src_neg_emb = embeddings[neg_edges_arr[:, 0]]
    tgt_neg_emb = embeddings[neg_edges_arr[:, 1]]
    neg_scores = (src_neg_emb * tgt_neg_emb).sum(axis=1)
    y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
    y_score = np.concatenate([pos_scores, neg_scores])
    try: return float(roc_auc_score(y_true, y_score))
    except: return 0.5

def split_edges(data, train_ratio: float = 0.8):
    edge_index = data.edge_index.t().numpy()
    n = len(edge_index)
    idx = np.random.permutation(n)
    train_n = int(n * train_ratio)
    train_edges = edge_index[idx[:train_n]]
    test_edges = edge_index[idx[train_n:]]
    train_edge_index = torch.tensor(train_edges.T, dtype=torch.long)
    return train_edge_index, train_edges, test_edges

# ── Stress Test Runners ────────────────────────────────────────────────────────

def run_model_trials(name: str, model_class, data, train_edge_index, test_edges, all_edges_set, epochs, trials=5) -> dict:
    print(f"\nStress Testing: {name} ({trials} Trials)")
    trial_metrics = {"MRR": [], "P@5": [], "Hit@10": [], "AUC": [], "train_time": []}
    loss_histories = []
    latencies = []
    total_nodes = data.num_nodes

    for t in range(trials):
        print(f"  > Trial {t+1}...")
        torch.manual_seed(42 + t)
        np.random.seed(42 + t)
        
        if name == "GraphSAGE": model = model_class(in_channels=data.num_node_features, hidden_channels=64, out_channels=32)
        elif name == "LegalGAT": model = model_class(in_channels=data.num_node_features, hidden_channels=32, out_channels=32, heads=4)
        elif name == "SimpleMLP": model = model_class(in_channels=data.num_node_features, hidden_channels=64, out_channels=32)
        else: model = model_class(in_channels=data.num_node_features, hidden_channels=64, out_channels=32)

        start_t = time.time()
        from torch_geometric.data import Data
        train_data = Data(x=data.x, edge_index=train_edge_index)
        embeddings_tensor, loss_hist = train_link_prediction(train_data, model, epochs=epochs)
        train_duration = time.time() - start_t
        trial_metrics["train_time"].append(train_duration)
        loss_histories.append(loss_hist)
        
        inf_start = time.time()
        with torch.no_grad():
            embeddings = model(data.x, train_edge_index).cpu().numpy()
        inf_duration = (time.time() - inf_start) * 1000
        latencies.append(inf_duration / total_nodes)

        trial_metrics["MRR"].append(compute_mrr(embeddings, test_edges, all_edges_set))
        trial_metrics["P@5"].append(compute_precision_at_k(embeddings, test_edges, all_edges_set))
        trial_metrics["Hit@10"].append(compute_hit_at_k(embeddings, test_edges, all_edges_set))
        trial_metrics["AUC"].append(compute_auc(embeddings, test_edges))

    summary = {
        "model": name,
        "MRR_mean": float(np.mean(trial_metrics["MRR"])),
        "MRR_std": float(np.std(trial_metrics["MRR"])),
        "AUC_mean": float(np.mean(trial_metrics["AUC"])),
        "AUC_std": float(np.std(trial_metrics["AUC"])),
        "Latency_ms_per_node": float(np.mean(latencies)),
        "Throughput_nodes_per_sec": float(total_nodes / np.mean(trial_metrics["train_time"])),
        "loss_histories": loss_histories,
        "final_embeddings": embeddings
    }
    print(f"  [RESULT] MRR: {summary['MRR_mean']:.4f} (±{summary['MRR_std']:.4f}) | Latency: {summary['Latency_ms_per_node']:.4f} ms/node")
    return summary

def generate_visuals(results):
    os.makedirs("results", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Loss Convergence
    plt.figure(figsize=(10, 6))
    for r in results:
        h = np.array(r["loss_histories"])
        m, s = h.mean(axis=0), h.std(axis=0)
        plt.plot(range(len(m)), m, label=r["model"])
        plt.fill_between(range(len(m)), m-s, m+s, alpha=0.1)
    plt.title("GNN Training Convergence (5-Trial Shaded Variance)")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.savefig("results/gnn_loss_convergence.png", dpi=300)
    plt.close()

    # MRR Comparison
    plt.figure(figsize=(8, 6))
    models = [r["model"] for r in results]
    means = [r["MRR_mean"] for r in results]
    stds = [r["MRR_std"] for r in results]
    plt.bar(models, means, yerr=stds, capsize=10, color=['#94A3B8', '#4F46E5', '#10B981', '#F59E0B'], alpha=0.8)
    plt.title("MRR Stability Comparison")
    plt.savefig("results/gnn_mrr_stability.png", dpi=300)
    plt.close()

    # Stress Test Scatter
    plt.figure(figsize=(8, 6))
    for r in results:
        plt.scatter(r["Throughput_nodes_per_sec"], r["Latency_ms_per_node"], s=200, label=r["model"])
    plt.title("Industrial Stress Test: Throughput vs Latency")
    plt.xlabel("Training Throughput (Nodes/Sec)")
    plt.ylabel("Inference Latency (ms/Node)")
    plt.legend()
    plt.savefig("results/gnn_stress_test.png", dpi=300)
    plt.close()

def run_benchmark(data_path="data/dataset_v4.json", epochs=50, trials=5):
    print("\nEtaNexus v6.0 — INDUSTRIAL STRESS TEST")
    cases = load_cases(data_path)
    text_embeddings = generate_text_embeddings(cases)
    data, id_to_idx = build_pyg_graph(cases, text_embeddings)
    all_edges = data.edge_index.t().numpy()
    all_edges_set = set(map(tuple, all_edges.tolist()))
    train_edge_index, _, test_edges = split_edges(data)

    results = []
    # Non-Graph Baseline
    results.append(run_model_trials("SimpleMLP", SimpleMLP, data, train_edge_index, test_edges, all_edges_set, epochs, trials))
    
    # Graph-Based Models
    results.append(run_model_trials("GraphSAGE", GraphSAGE, data, train_edge_index, test_edges, all_edges_set, epochs, trials))
    results.append(run_model_trials("LegalGAT", LegalGAT, data, train_edge_index, test_edges, all_edges_set, epochs, trials))
    if GCN_AVAILABLE:
        results.append(run_model_trials("GCN", LegalGCN, data, train_edge_index, test_edges, all_edges_set, epochs, trials))

    generate_visuals(results)
    winner = sorted(results, key=lambda x: x["MRR_mean"], reverse=True)[0]
    torch.save(torch.tensor(winner["final_embeddings"]), "results/gnn_embeddings.pt")
    
    with open("results/performance_report.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k not in ["loss_histories", "final_embeddings"]} for r in results], f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    run_benchmark(epochs=args.epochs, trials=args.trials)
