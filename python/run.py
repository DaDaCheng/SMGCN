from utils import full_load_data,sparse_mx_to_torch_sparse_tensor
import torch.nn.functional as F
import torch
from GCNlayer import GCNcConv



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#load data
datastr = 'chameleon'
split_i=0
splitstr = 'splits/'+datastr+'_split_0.6_0.2_'+str(split_i)+'.npz'
g, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr, splitstr)


#model
class twolayerGCNc(torch.nn.Module):
    def __init__(self, num_features,hidden_channels,num_labels,c=1):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNcConv(num_features, hidden_channels,flow='target_to_source',add_self_loops=True,selfc=c)
        self.conv2 = GCNcConv(hidden_channels, num_labels,flow='target_to_source',add_self_loops=True,selfc=c)

    def forward(self, x, edge_index):
        x= self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
model=twolayerGCNc(num_features,128,num_labels,c=-0.5).to(device)



gg=sparse_mx_to_torch_sparse_tensor(g)
link_indices=gg.coalesce().indices() 
edge_index=link_indices.to(device)
edge_index=edge_index.to(device)
labels=labels.to(device)
x=features.to(device)




optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(x, edge_index)  # Perform a single forward pass.
    loss = criterion(out[idx_train], labels[idx_train])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss
def test():
    model.eval()
    out = model(x, edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    loss = criterion(out[idx_test], labels[idx_test])
    test_loss=loss.item()
    test_correct = pred[idx_test] == labels[idx_test]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(idx_test.sum())  # Derive ratio of correct predictions.
    return test_acc,test_loss

def val():
    model.eval()
    out = model(x, edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    val_correct = pred[idx_val] == labels[idx_val]  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / int(idx_val.sum())  # Derive ratio of correct predictions.
    return val_acc

pick_test_acc=0
pick_test_loss=100
pick_val_acc=0
for epoch in range(1, 2000):
    loss = train()
    val_acc = val()
    if val_acc>pick_val_acc:
        test_acc,test_loss = test()
        pick_val_acc=val_acc
        pick_test_acc=test_acc
        pick_test_loss=test_loss
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test loss: {pick_test_loss:.4f}, Test Accuracy: {pick_test_acc:.4f}')
