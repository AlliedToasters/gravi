import torch
from torch_geometric.data import Data
import os
import numpy as np
from data_util import load_state
from build_graph import build_graph
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

def train_test_split(states_path="data/states/"):
    train_size = 0.75
    states = os.listdir(states_path)
    train_states = set(np.random.choice(states, size=int(train_size*len(states))).tolist())
    valid_states = set(states) - set(train_states)
    return train_states, valid_states

def scale_3vec(p, scl):
    scl_p = scl.transform(p.flatten().reshape(-1, 1)).reshape(p.shape)
    return scl_p


def main():
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