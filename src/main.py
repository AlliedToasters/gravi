from data_util import preprocess_data
from model import GNN
import torch
import time
from tqdm import tqdm

def main():
    input_scl, output_scl, train_graphs, valid_graphs = preprocess_data()
    model = GNN(hidden_channels=128, out_channels=3, dropout=0.0)
    lossr = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
    for item in tqdm(train_graphs):
        out = model.forward(item.x, item.edge_index)
        loss = lossr(out, item.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model, "model/v0.pt")

    # g = train_graphs[0]
    # print(g.x.shape)
    # t1 = time.time()
    # with torch.no_grad():
    #     out = model.forward(g.x, g.edge_index)
    # print(time.time() - t1)
    # print(out.shape)

if __name__ in "__main__":
    main()