from flask import Flask, request
import flask
import torch
from model import GNN
from build_graph import build_graph
from data_util import graph_to_edge_index
import numpy as np
import json
import pickle

model = torch.load('model/v0.pt')
model.eval()
with open('model/input_scaler.pkl', 'rb') as f:
    input_scl = pickle.load(f)
with open('model/output_scaler.pkl', 'rb') as f:
    output_scl = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def health_check():
    resp = flask.Response("{}", status=200)
    resp.headers['Access-Control-Allow-Methods'] = 'GET,OPTIONS'
    return resp

@app.route('/get_forces')
def get_forces():
    coords = request.args.get('coords')
    coords = np.array([float(x) for x in coords.split(",")])
    graph = build_graph(coords.reshape(-1, 3))
    edge_index = graph_to_edge_index(graph)
    print(edge_index)
    coords = input_scl.transform(coords.reshape(-1, 1))
    coords = coords.reshape(-1, 3)
    with torch.no_grad():
        out = model.forward(torch.tensor(coords).float(), edge_index)
    forces = out.cpu().numpy()
    forces = output_scl.inverse_transform(forces.flatten().reshape(-1, 1))
    body = {
        "forces":forces.flatten().tolist()
    }
    r = json.dumps(body)
    resp = flask.Response(r, status=200)
    resp.headers['Access-Control-Allow-Methods'] = 'GET,OPTIONS'
    return resp