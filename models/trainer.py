import torch
import torch.nn.functional as F

def train_link_prediction(data, model, epochs=50, lr=0.005):
    """
    Train the GNN model using a Link Prediction objective.
    Works for GraphSAGE, GAT, and GCN architectures.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    num_nodes = data.num_nodes
    pos_edge_index = data.edge_index
    
    if pos_edge_index.shape[1] == 0:
        print("[WARN] No edges found for training. Returning initial state.")
        return model(data.x, data.edge_index)

    model.train()
    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward pass: get node embeddings
        z = model(data.x, pos_edge_index)
        
        # 2. Positive edges
        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        
        # 3. Negative sampling
        neg_edge_index = torch.randint(0, num_nodes, (2, pos_edge_index.size(1)), device=z.device)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        
        # 4. Binary Cross Entropy Loss
        pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(float(loss.item()))
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    model.eval()
    with torch.no_grad():
        final_z = model(data.x, pos_edge_index)
        return final_z, loss_history

