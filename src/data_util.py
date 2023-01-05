import numpy as np

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