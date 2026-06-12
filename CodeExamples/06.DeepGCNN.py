from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np



dataset = Planetoid(root=".", name="Cora")
data = dataset[0]
print('x:',data.x.shape)
print('y:',np.unique(data.y))
degrees = degree(data.edge_index[0]).numpy()
print('deg:',degrees)
numbers = Counter(degrees)


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def accuracy(pred_y, y):
  return ((pred_y == y).sum() / len(y)).item()

class GCN(torch.nn.Module):
  """Graph Convolutional Network"""
  def __init__(self, dim_in, dim_h, dim_out, layers=3):
    super().__init__()
    self.layers_num=layers
#    self.gcn=[1]*self.layers_num
#    for i in range(self.layers_num):
#     if i==0:
    self.gcn0 = GCNConv(dim_in, dim_h)
#     else:
    self.gcn1 = GCNConv(dim_h, dim_h)
    self.gcn2 = GCNConv(dim_h, dim_h)
    self.gcn3 = GCNConv(dim_h, dim_h)
    self.gcn4 = GCNConv(dim_h, dim_h)
    self.gcn_out = GCNConv(dim_h, dim_out)
    
  def forward(self, x, edge_index):
    h = self.gcn0(x, edge_index)
    h = torch.selu(h)
#    print('xsh:',x.shape,'hshape:',h.shape)
    h_old =h
    h = self.gcn1(h, edge_index)
    h = torch.selu(h)
    h +=h_old
    h_old =h
    h = self.gcn2(h, edge_index)
    h = torch.selu(h)
    h +=h_old
    h = self.gcn_out(h, edge_index)
    return F.log_softmax(h, dim=1)
  
  
  def fit(self, data, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.parameters(),lr=0.01,weight_decay=5e-4)
    self.train()
    old_val_loss=1e100
    old_best_epoch=-1
    patience=100
    val_arr=[]
    for epoch in range(epochs+1):
      optimizer.zero_grad()
      out = self(data.x, data.edge_index)
      loss = criterion(out[data.train_mask],data.y[data.train_mask])
      acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      if(epoch % 1 == 0):
        val_loss = criterion(out[data.val_mask],data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
        val_arr.append([epoch,loss.detach().numpy(),acc,val_loss.detach().numpy(),val_acc])
        if val_loss<old_val_loss:
         old_val_loss=val_loss
         old_best_epoch=epoch
        print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
        if epoch>old_best_epoch+patience:
         break
    val_arr=np.array(val_arr)
    plt.plot(val_arr[:,0],val_arr[:,2],label='acc@train')
    plt.plot(val_arr[:,0],val_arr[:,4],label='acc@val')
    plt.legend()
    plt.ylim(0.6,1)
    plt.savefig('5layers.png')
  @torch.no_grad()
  def test(self, data):
    self.eval()
    out = self(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask],
    data.y[data.test_mask])
    return acc


gcn = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn)
gcn.fit(data, epochs=1000)
