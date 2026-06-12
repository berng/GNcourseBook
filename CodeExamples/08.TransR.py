import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling, train_test_split_edges
from torch_geometric.nn import MessagePassing
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt

class TransR(nn.Module):
    """TransR implementation using PyTorch Geometric"""
    def __init__(self, num_entities, num_relations, ent_dim, rel_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        # Entity embeddings
        self.ent_emb = nn.Embedding(num_entities, ent_dim)
        # Relation embeddings
        self.rel_emb = nn.Embedding(num_relations, rel_dim)
        # Relation-specific projection matrices
        self.rel_proj = nn.Embedding(num_relations, ent_dim * rel_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.rel_proj.weight)
    
    def _project(self, ent_emb, rel_id):
        """Project entity embeddings to relation space"""
        # Get projection matrix for relation
        proj_mat = self.rel_proj(rel_id).view(-1, self.ent_dim, self.rel_dim)
        # Project entity embeddings: (batch, ent_dim) @ (batch, ent_dim, rel_dim)
        projected = torch.bmm(ent_emb.unsqueeze(1), proj_mat).squeeze(1)
        return projected
    def forward(self, head_idx, rel_idx, tail_idx):
        """
        Compute scores for triplets
        Returns: scores (higher = more likely)
        """
        # Get embeddings
        h_emb = self.ent_emb(head_idx)  # (batch, ent_dim)
        r_emb = self.rel_emb(rel_idx)   # (batch, rel_dim)
        t_emb = self.ent_emb(tail_idx)  # (batch, ent_dim)
        # Project entities to relation space
        h_proj = self._project(h_emb, rel_idx)  # (batch, rel_dim)
        t_proj = self._project(t_emb, rel_idx)  # (batch, rel_dim)
        # TransR scoring function: -||h_proj + r - t_proj||_2
        score = torch.norm(h_proj + r_emb - t_proj, p=2, dim=1)
        return -score  # Negative distance (higher = better)

def train(model, optimizer, data, device):
    """Training loop with negative sampling"""
    model.train()
    optimizer.zero_grad()
    # Positive samples
    pos_edge_index = data.train_pos_edge_index
    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1),
        method='sparse'
    )

    # Create relation indices (all 0 since Cora has single relation type)
    rel_idx = torch.zeros(pos_edge_index.size(1), dtype=torch.long, device=device)
    # Positive scores
    pos_scores = model(pos_edge_index[0], rel_idx, pos_edge_index[1])
    # Negative scores
    neg_scores = model(neg_edge_index[0], rel_idx, neg_edge_index[1])

    # Margin ranking loss
    margin = 1.0
    loss = torch.mean(F.relu(margin - pos_scores + neg_scores))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, device, pos_edge_index, neg_edge_index):
    """Evaluate model on given edge sets"""
    model.eval()

    rel_idx = torch.zeros(pos_edge_index.size(1), dtype=torch.long, device=device)
    # Positive scores
    pos_scores = model(pos_edge_index[0], rel_idx, pos_edge_index[1])
    # Negative scores
    neg_scores = model(neg_edge_index[0], rel_idx, neg_edge_index[1])
    # Combine scores and labels
    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]).cpu().numpy()
    # Compute metrics
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dataset = Planetoid(root='./', name='Cora')
    data = dataset[0]
    data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)
    num_entities = data.num_nodes
    num_relations = 1  # Cora has single relation type (citation)
    ent_dim = 100
    rel_dim = 100
    model = TransR(num_entities, num_relations, ent_dim, rel_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
#    print(f"\nModel parameters:")
#    print(f"- Entity embeddings: {num_entities} x {ent_dim}")
#    print(f"- Relation embeddings: {num_relations} x {rel_dim}")
#    print(f"- Projection matrices: {num_relations} x ({ent_dim} x {rel_dim})")    
    # Training loop
    best_val_auc = 0
    patience = 50
    patience_counter = 0
    
    print(f"\nStarting training...")
    for epoch in range(1, 501):
        loss = train(model, optimizer, data.to(device), device)
        # Validation
        if epoch % 10 == 0:
            val_auc, val_ap = evaluate(
                model, data, device,
                data.val_pos_edge_index,
                data.val_neg_edge_index
            )
            print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}')
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_transr_cora.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Load best model and test
    model.load_state_dict(torch.load('best_transr_cora.pth'))
    test_auc, test_ap = evaluate(
        model, data, device,
        data.test_pos_edge_index,
        data.test_neg_edge_index
    )
    print(f"\nFinal Results:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP: {test_ap:.4f}")
    # Additional link prediction example
    print(f"\nLink Prediction Example:")

    quit()

    with torch.no_grad():
        # Predict score for first test positive edge
        score_pos,score_neg,score_pn=[],[],[]
        print('pos idx: ',data.test_pos_edge_index.shape[1])
        print('neg idx: ',data.test_neg_edge_index.shape[1])

        for idx4test in range(data.test_pos_edge_index.shape[1]):
         h, t = data.test_pos_edge_index[:, idx4test]
         rel = torch.tensor([0], device=device) # link type
         sp=model(h.unsqueeze(0), rel, t.unsqueeze(0)).to('cpu')
         h, t = data.test_neg_edge_index[:, idx4test]
         sn=model(h.unsqueeze(0), rel, t.unsqueeze(0)).to('cpu')

         score_pos.append(sp)
         score_neg.append(sn)
         score_pn.append(sp-sn)
        score_neg=np.array(score_neg)
        score_pos=np.array(score_pos)
        score_pn=np.array(score_pn)
        print(f"Score for edge: ",score_pos.mean()-2*score_pos.std(),score_pos.mean()+2*score_pos.std())
        print(f"Score for non-edge: ",score_neg.mean()-2*score_neg.std(),score_neg.mean()+2*score_neg.std())
        print(f"Score for pn: ",score_pn.mean(),score_pn.std(),score_pn.mean()-2*score_pn.std(),score_pn.mean()+2*score_pn.std())
        plt.hist(score_pn,bins=int(np.sqrt(score_pn.shape[0])))
        plt.show()

main()
