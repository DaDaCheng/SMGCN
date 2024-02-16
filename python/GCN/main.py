import torch 
import os
import numpy as np
from load_utils import load_data
import torch
from torch.nn import Linear
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GCNConv
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.') #Default seed same as GCNII
parser.add_argument('--name', type=str, default='ncora')
parser.add_argument('--net', type=int, default=0)
parser.add_argument('--rate', type=float, default=0.01)
args = parser.parse_args()
print(args)
net=args.net
name=args.name
seed=args.seed
rate=args.rate

dataset, data=load_data(args.name)
data=data.to(device)


if name=='chameleon':
    if net==0:
        class GCN(torch.nn.Module):
            def __init__(self, hidden_channels,dropout=False):
                super().__init__()
                torch.manual_seed(seed)
                self.conv1 = GCNConv(dataset.num_features, dataset.num_classes,flow='target_to_source')
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                return x
    else:
        class GCN(torch.nn.Module):
            def __init__(self, hidden_channels, dropout=False):
                super().__init__()
                torch.manual_seed(seed)
                self.conv1 = GCNConv(dataset.num_features, hidden_channels,flow='target_to_source')
                self.conv2 = GCNConv(hidden_channels, dataset.num_classes,flow='target_to_source')
                self.dropout = dropout

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = x.relu()
                if self.dropout:
                    x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
                return x
    
else:
    if net==0:
        class GCN(torch.nn.Module):
            def __init__(self, hidden_channels,dropout=False):
                super().__init__()
                torch.manual_seed(seed)
                self.conv1 = GCNConv(dataset.num_features, dataset.num_classes)
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                return x
    else:
        class GCN(torch.nn.Module):
            def __init__(self, hidden_channels, dropout=False):
                super().__init__()
                torch.manual_seed(seed)
                self.conv1 = GCNConv(dataset.num_features, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
                self.dropout = dropout

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = x.relu()
                if self.dropout:
                    x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
                return x
if net==0:
    MSE=True
    dropout=False
if net==1:
    MSE=True
    dropout=False
if net==2:
    MSE=True
    dropout=True
if net==3:
    MSE=False
    dropout=True
    
if MSE:
    criterion=torch.nn.MSELoss()
else:
    criterion = torch.nn.CrossEntropyLoss()
uniform_sample=False


torch.manual_seed(seed+int(rate*100000)+100)
N=data.num_nodes
if uniform_sample:

    y=data.y.detach().clone().cpu()

    train_mask=torch.zeros(N,dtype=torch.bool)
    test_mask=torch.zeros(N,dtype=torch.bool)
    torch.manual_seed(seed+int(rate*100000)+100)
    nn=int(rate*N)
    train_mask[:nn]=True
    test_mask[nn:]=True
else:
    y=data.y.detach().clone().cpu()
    num_label=torch.max(y)+1
    hash=torch.ones(num_label,dtype=torch.int)
    for i in range(num_label):
        hash[i]=torch.sum(y==i)
    hash=(hash*rate).int()
    train_mask=torch.zeros(N,dtype=torch.bool)
    test_mask=torch.ones(N,dtype=torch.bool)
    rangemask=torch.randperm(N)
    for i in range(N):
        ii=rangemask[i]
        yi=y[ii]
        if hash[yi]>0:
            train_mask[ii]=True
            test_mask[ii]=False
            hash[yi]=hash[yi]-1
        if torch.sum(hash)==0:
            break
model = GCN(hidden_channels=16,dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)


lowest_loss=10
epoch_report=0
for epoch in range(1, 10000):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    if MSE:
        T=F.one_hot(data.y, num_classes=dataset.num_classes).to(device).float()
    else:
        T=data.y
    loss = criterion(out[train_mask], T[train_mask])
    #loss = criterion(out[train_mask], data.y[train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()
    t=loss.item()

    if t<lowest_loss:
        epoch_report=epoch
        lowest_loss=t
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[test_mask] == data.y[test_mask]  # Check against ground-truth labels.
        loss = criterion(out[test_mask], T[test_mask])
        loss_test =loss.detach().clone().cpu().numpy().item()
        test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
        #print(epoch_report,t,test_acc)
print(rate,t,test_acc,loss_test,epoch_report)

import sys
log_file = open('re/'+str(name)+'_'+str(net)+'.txt', 'a')
sys.stdout = log_file
print('rate:{:.8f}, acc:{:.8f}, loss:{:.8f}, epoch:{:.5f}'.format(rate,test_acc,loss_test,epoch_report))
sys.stdout = sys.__stdout__
log_file.close()
