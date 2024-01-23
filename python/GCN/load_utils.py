import argparse
from tqdm import tqdm
import copy
import numpy as np

import networkx as nx
import numpy as np
from utils import *
from torch_geometric.data import Data
import os
from torch_geometric.utils import dense_to_sparse,coalesce,remove_self_loops,to_undirected
import torch
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device ='cpu'
print(device)
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
class EmptyObject:
    pass

def load_data(name):
    if name=='cora':
        dataset = Planetoid(root='data/Planetoid', name='Cora')
        print()
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

        data = dataset[0]  # Get the first graph object.

        print()
        print(data)
        print('===========================================================================================================')

        # Gather some statistics about the graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Number of training nodes: {data.train_mask.sum()}')
        print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        data=data.to(device)
        return dataset, data

    if name=='citeseer': 
        dataset = Planetoid(root='data/Planetoid', name='citeseer')
        print()
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

        data = dataset[0]  # Get the first graph object.

        print()
        print(data)
        print('===========================================================================================================')

        # Gather some statistics about the graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Number of training nodes: {data.train_mask.sum()}')
        print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        data=data.to(device)
        return dataset, data
    if name=='ncora':
        dataset = Planetoid(root='data/Planetoid', name='Cora')
        print()
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

        data = torch.load('ncora.pt')
        return dataset, data

    if name=='ncoranew':
        dataset = Planetoid(root='data/Planetoid', name='Cora')
        print()
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

        data = dataset[0]  # Get the first graph object.

        print()
        print(data)
        print('===========================================================================================================')

        # Gather some statistics about the graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Number of training nodes: {data.train_mask.sum()}')
        print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        data=data.to(device)


        
        noise_rate=0.3
        old_index=data.edge_index.detach().clone().cpu()
        N=data.num_nodes
        row,col=old_index
        mask=row<col
        row,col=row[mask],col[mask]
        rangemask=torch.randperm(len(row))
        cut_index=int(noise_rate*len(row))
        old_index=torch.vstack((row[cut_index:],col[cut_index:]))
        p=(2*cut_index)/(N*N)

        A_p=torch.ones((N,N))*p
        A=torch.bernoulli(torch.tensor(A_p))
        noise_edge_index,_=dense_to_sparse(A)
        row,col=noise_edge_index
        mask=row<col
        row,col=row[mask],col[mask]
        noise_edge_index=torch.vstack((row,col))

        edge_index=torch.cat((old_index,noise_edge_index),dim=1)

        edge_index,_=remove_self_loops(edge_index)
        edge_index=to_undirected(edge_index)
        data.edge_index=edge_index.to(device)


        old_x=data.x.detach().clone().cpu()
        shapea,shapeb=old_x.shape

        row,col=torch.where(old_x==1)
        rangemask=torch.randperm(len(row))
        cut=int(noise_rate*len(row))
        cut_index=rangemask[:cut]
        old_x[row[cut_index],col[cut_index]]=0
        add_indexa=torch.randint(shapea,(cut,))
        add_indexb=torch.randint(shapeb,(cut,))
        old_x[add_indexa,add_indexb]=1
        data.x=old_x.to(device)
        torch.save(data,'ncora.pt')
        return dataset, data
        
    if name=='texas':
        #dataset_name='chameleon'
        dataset_name='texas'



        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name=='film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])

        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                                label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                                label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        print(G.number_of_edges())

        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
        g = adj

        gt=sparse_mx_to_torch_sparse_tensor(g)
        row,col=gt.coalesce().indices()
        edge_index=torch.cat((row.reshape(-1,1),col.reshape(-1,1)),dim=1).T

        data=Data(edge_index=edge_index,x=torch.tensor(features,dtype=torch.float),y=torch.tensor(labels))
        dataset=Data()
        dataset.num_features=features.shape[1]
        dataset.num_classes=max(labels)+1
        
        return dataset, data
        
    if name=='chameleon':
        dataset_name='chameleon'
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name=='film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])

        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                                label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                                label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        print(G.number_of_edges())

        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
        g = adj

        gt=sparse_mx_to_torch_sparse_tensor(g)
        row,col=gt.coalesce().indices()
        edge_index=torch.cat((row.reshape(-1,1),col.reshape(-1,1)),dim=1).T

        data=Data(edge_index=edge_index,x=torch.tensor(features,dtype=torch.float),y=torch.tensor(labels))
        dataset=Data()
        dataset.num_features=features.shape[1]
        dataset.num_classes=max(labels)+1
        
        return dataset, data
    
    if name=='csbmnew':
        N=1000
        alpha=0.5
        l=1
        mu=1
        d=30
        hN=int(N/2)
        F=int(alpha*N)
        y=np.ones(hN)
        y=np.concatenate([y,-y],axis=0).reshape(N,1)
        
        u=np.random.randn(F,1)/np.sqrt(F)
        Omega=np.random.randn(N,F)
        X=np.sqrt(mu/N)*y@u.T+Omega/np.sqrt(F)
        cin=(d+l*np.sqrt(d))/N
        cout=(d-l*np.sqrt(d))/N
        Aones=np.ones([hN,hN])
        A_p=np.concatenate([np.concatenate([cin*Aones,cout*Aones],axis=1),np.concatenate([cout*Aones,cin*Aones],axis=1)],axis=0)
        A_p=np.triu(A_p, k=0)
        A=torch.bernoulli(torch.tensor(A_p))
        A=(A+A.T).to(torch.long).numpy()
        A[A>1]=1
        A=A/np.sqrt(d)
        edge_index,_=dense_to_sparse(torch.tensor(A))
        y=np.round((y+1)/2)
        data = Data(x=torch.tensor(X).float(), edge_index=edge_index, y=torch.tensor(y.reshape(-1), dtype=torch.long))
        dataset=EmptyObject()
        dataset.num_classes=2
        dataset.num_features=F
        torch.save(data,'csbm.pt')
        return dataset,data
    
    if name=='csbm':
        N=1000
        alpha=0.5
        l=1
        mu=1
        d=30
        hN=int(N/2)
        F=int(alpha*N)
        dataset=EmptyObject()
        dataset.num_classes=2
        dataset.num_features=F
        data=torch.load('csbm.pt')
        print(data)
        return dataset,data