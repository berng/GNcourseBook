#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().system('pip install python-louvain')


# In[5]:


import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
import numpy as np

G = nx.karate_club_graph()
partition = community_louvain.best_partition(G, random_state=42)
num_communities = len(set(partition.values()))
print(f"Number of communities found: {num_communities}")
# Print the community assignment for each node
print("\nNode -> Community")
for node, community_id in sorted(partition.items()):
    print(f"{node:2d} -> {community_id}")
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducible results
colors = [partition[node] for node in G.nodes()]
cmap = plt.cm.Set3  # Choose a qualitative colormap
vmin, vmax = min(colors), max(colors)
nx.draw_networkx_edges(G, pos, alpha=0.7)
nx.draw_networkx_nodes(G, pos,
                       node_color=colors,
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       node_size=500)
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cmap(i / vmax), label=f'Community {i}')
                   for i in range(num_communities)]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[1]:


get_ipython().system('pip install torch_geometric')


# In[35]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np


# In[36]:


# Load CORA dataset
dataset = Planetoid(root='./', name='Cora')
data = dataset[0]

# Parameters
num_classes = dataset.num_classes  # 7 for CORA
num_features = dataset.num_features
num_nodes = data.num_nodes

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NOCD Model
class NOCD(nn.Module):
    def __init__(self, nfeat=1, ncomm=7):
        super(NOCD, self).__init__()
        self.ncomm = ncomm
        # GNN encoder
        self.conv1 = GCNConv(1, 4)
        self.conv2 = GCNConv(4, ncomm)
        # Community affinity matrix
        self.B = nn.Parameter(torch.randn(ncomm, ncomm) * 0.1)

    def forward(self, x, edge_index):
        # Encode
        x=x[:,0:1]
        x[:,0:1]=1
        x = F.relu(self.conv1(x, edge_index))
        logits = self.conv2(x, edge_index)  # [N, K]
        # Soft assignment
        Q = F.softmax(logits, dim=1)  # [N, K]
        return Q

# Reconstruction loss
def reconstruction_loss(Q, edge_index, B):
    # Compute expected adjacency matrix
    print('Q:',Q.shape,'B:',B.shape)
#    A_pred = torch.mm(Q, torch.mm(Q,B.t))  # [N, N]
    s1=torch.mm(Q,B.T)
    print(s1.shape)
    A_pred = torch.mm(Q,s1)  # [N, N]
    A_pred = torch.clamp(A_pred, 0, 1)

    # Positive loss (reconstruct existing edges)
    row, col = edge_index
    pos_loss = -torch.log(A_pred[row, col] + 1e-10).mean()

    # Negative loss (avoid reconstructing non-edges)
    # Sample negative edges (same number as positive edges)
    neg_row = torch.randint(0, num_nodes, (edge_index.size(1),), device=device)
    neg_col = torch.randint(0, num_nodes, (edge_index.size(1),), device=device)
    # Avoid self-loops and existing edges
    mask = (neg_row != neg_col) & (A_pred[neg_row, neg_col] < 1e-5)
    neg_row = neg_row[mask]
    neg_col = neg_col[mask]
    if len(neg_row) > 0:
        neg_loss = -torch.log(1 - A_pred[neg_row, neg_col] + 1e-10).mean()
    else:
        neg_loss = torch.tensor(0.0, device=device)

    return pos_loss + neg_loss

# Entropy regularization
def entropy_loss(Q):
    return -(Q * torch.log(Q + 1e-10)).sum(1).mean()

# Initialize model
model = NOCD(num_features, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Move data to device
data = data.to(device)

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()

    # Forward pass
    Q = model(data.x, data.edge_index)

    # Compute losses
    B=F.one_hot(data.y, num_classes=7).to(torch.float)
    print(B.shape)
    recon_loss = reconstruction_loss(Q, data.edge_index, B)
    ent_loss = entropy_loss(Q)
    loss = recon_loss + 0.1 * ent_loss  # Weight for entropy

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, Entropy: {ent_loss.item():.4f}')

# Get final assignments
model.eval()
with torch.no_grad():
    Q_final = model(data.x, data.edge_index)
    pred_labels = Q_final.argmax(dim=1).cpu().numpy()

# Evaluate against ground truth
true_labels = data.y.cpu().numpy()

# Calculate metrics
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)

print(f"\nNOCD Results on CORA:")
print(f"Predicted Communities: {len(np.unique(pred_labels))}")
print(f"Ground Truth Classes: {len(np.unique(true_labels))}")
print(f"ARI (Adjusted Rand Index): {ari:.4f}")
print(f"NMI (Normalized Mutual Info): {nmi:.4f}")

# Show community distribution
comm_counts = np.bincount(pred_labels)
print(f"Community sizes: {comm_counts}")


# In[38]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

dataset = Planetoid(root='./', name='Cora')
data = dataset[0]
num_classes = dataset.num_classes  # 7 for CORA
num_nodes = data.num_nodes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
node_features = torch.eye(num_nodes).to(device)


# In[37]:


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(1, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x=x[:,0:1]
        x[:,0:1]=1
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model
model = GCN(1, 64, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()
data = data.to(device)
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(node_features, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    out = model(node_features, data.edge_index)
    pred = out.argmax(dim=1)

    # Calculate metrics for train, val, and test sets
    for mask_name, mask in [('Train', data.train_mask),
                            ('Validation', data.val_mask),
                            ('Test', data.test_mask)]:
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        print(f'{mask_name} Accuracy: {acc:.4f}, Micro-F1: {micro_f1:.4f}, Macro-F1: {macro_f1:.4f}')

# Final test set results
test_acc = accuracy_score(data.y[data.test_mask].cpu().numpy(),
                         pred[data.test_mask].cpu().numpy())
print(f'\nFinal Test Accuracy: {test_acc:.4f}')


# In[16]:


yp=model(data.x,data.edge_index)
yp=np.argmax(yp.detach().to('cpu').numpy(),axis=-1)
np.unique(yp)

