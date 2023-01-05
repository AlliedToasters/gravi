import faiss
import networkx as nx

try:
    res = faiss.StandardGpuResources()
except AttributeError:
    res = None

def build_nn_graph(p, k=6, g=None, use_gpu=True):
    if g is None:
        g = nx.Graph()
#     t1 = time.time()
    p = p.astype("float32")
    index_flat = faiss.IndexFlatL2(3)
    if use_gpu:
        index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index_flat.add(p)
    _, I = index_flat.search(p, k+1)
    I = I[:, 1:]
#     print(time.time() - t1)
#     edges = set()
    for i in range(len(p)):
        row = I[i, :]
        for item in row:
            edge = (i, item)
            edge = (min(edge), max(edge))
            if edge[0] != edge[1]:
                g.add_edge(edge[0], edge[1])
    return g

def build_random_graph(p, n_edges_per_node=3, g=None):
    if g is None:
        g = nx.Graph()
    _p = n_edges_per_node/len(p)
    _g = nx.erdos_renyi_graph(len(p), _p)
    g = nx.compose(g, _g)
    return g

def build_throughline_graph(p, g=None):
    if g is None:
        g = nx.Graph()
    last_node = len(p)
    for i in range(last_node):
        other = i+1
        if other == last_node:
            print('reached last node...')
            other = 0
        g.add_edge(i, other)
    return g

def build_graph(p, n_nn_edges=6, n_random_edges=3, do_build_throughline_graph=True, use_gpu=False):
    g = nx.Graph()
    if n_nn_edges > 0:
        g = build_nn_graph(p, k=n_nn_edges, g=g, use_gpu=use_gpu)
    if n_random_edges > 0:
        g = build_random_graph(p, n_edges_per_node=n_random_edges, g=g)
    if do_build_throughline_graph:
        g = build_throughline_graph(p, g)
    return g