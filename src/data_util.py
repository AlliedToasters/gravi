import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.utils.undirected import to_undirected, is_undirected#TORCH_GEOMETRIC.UTILS.UNDIRECTED
from build_graph import build_graph
import pickle
from sklearn.preprocessing import StandardScaler
import os

def load_state(path='./data/states/12340689.txt'):
    with open(path, 'r') as f:
        d = f.read().split("\n")[:-1]
        
    positions, forces, ticker = [], [], 0
    pos, forc = [], []
    for item in d:
        if ticker < 3:
            pos.append(float(item))
        else:
            forc.append(float(item))
        ticker += 1
        if ticker > 5:
            ticker = 0
            positions.append(pos), forces.append(forc)
            pos, forc = [], []
            
    return np.array(positions), np.array(forces)

def train_test_split(states_path="data/states/"):
    train_size = 0.75
    states = os.listdir(states_path)
    train_states = set(np.random.choice(states, size=int(train_size*len(states)), replace=False).tolist())
    valid_states = set(states) - set(train_states)
    print(len(train_states)/len(states))
    print(len(valid_states)/len(states))
    assert len(train_states) + len(valid_states) == len(states)
    return train_states, valid_states

def scale_3vec(p, scl):
    scl_p = scl.transform(p.flatten().reshape(-1, 1)).reshape(p.shape)
    return scl_p

def graph_to_edge_index(g):
    edges = list(g.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    return edge_index

def preprocess_data():
    try:
        with open('model/input_scaler.pkl', 'rb') as f:
            input_scl = pickle.load(f)
        with open('model/output_scaler.pkl', 'rb') as f:
            output_scl = pickle.load(f)
        output_train_graphs = torch.load('data/train_graphs.pt')
        output_valid_graphs = torch.load('data/valid_graphs.pt')
        # input_scl, output_scl, train_graphs, valid_graphs
    except FileNotFoundError:
        tr, va = train_test_split()
        tr_p, tr_f, va_p, va_f, all_p, all_f = [], [], [], [], [], []

        print('parsing train data...')
        for item in tqdm(tr):
            p, f = load_state("data/states/"+item)
            tr_p.append(p)
            tr_f.append(f)
            all_p.extend(p.flatten().tolist())
            all_f.extend(f.flatten().tolist())

        print('parsing validation data...')
        for item in tqdm(va):
            p, f = load_state("data/states/"+item)
            va_p.append(p)
            va_f.append(f)

        print('fitting scalers...')
        input_scl = StandardScaler(with_mean=False)
        output_scl = StandardScaler(with_mean=False)
        input_scl = input_scl.fit(np.array(all_p).reshape(-1, 1))
        output_scl = output_scl.fit(np.array(all_f).reshape(-1, 1))

        with open('model/input_scaler.pkl', 'wb') as f:
            pickle.dump(input_scl, f)

        with open('model/output_scaler.pkl', 'wb') as f:
            pickle.dump(output_scl, f)


        output_train_graphs, output_valid_graphs = [], []
        train_graphs, valid_graphs = [], []

        print('computing train graphs...')
        for i in tqdm(range(len(tr_p))):
            p = tr_p[i]
            g = build_graph(p)
            f = tr_f[i]
            p = scale_3vec(p, input_scl)
            f = scale_3vec(f, output_scl)
            x = torch.tensor(p).float()
            y = torch.tensor(f).float()
            edge_index = graph_to_edge_index(g)
            d = Data(x=x, y=y, edge_index=edge_index)
            train_graphs.append(d)

        for d in train_graphs:
            new_ei = to_undirected(d.edge_index)
            new_d = Data(x=d.x, y=d.y, edge_index=new_ei)
            assert is_undirected(new_d.edge_index)
            output_train_graphs.append(new_d)

        torch.save(output_train_graphs, 'data/train_graphs.pt')

        print('computing valid graphs...')
        for i in tqdm(range(len(va_p))):
            p = va_p[i]
            g = build_graph(p)
            f = va_f[i]
            p = scale_3vec(p, input_scl)
            f = scale_3vec(f, output_scl)
            x = torch.tensor(p).float()
            y = torch.tensor(f).float()
            edge_index = graph_to_edge_index(g)
            d = Data(x=x, y=y, edge_index=edge_index)
            valid_graphs.append(d)

        for d in valid_graphs:
            new_ei = to_undirected(d.edge_index)
            new_d = Data(x=d.x, y=d.y, edge_index=new_ei)
            assert is_undirected(new_d.edge_index)
            output_valid_graphs.append(new_d)

        torch.save(output_valid_graphs, 'data/valid_graphs.pt')

    return input_scl, output_scl, output_train_graphs, output_valid_graphs