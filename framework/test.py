# test.py
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "pyproject.toml").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import yaml
import torch
from torch_geometric.loader import DataLoader
import pickle
import time

from framework import utils as utils
from framework import model as md

def inverse(pred_channel, scaler, G, N, T):
    """ Un-scale one channel back to physical units. """

    scaled_2d = pred_channel.reshape(G, N, T).transpose(0, 2, 1).reshape(-1, N)
    unscaled_2d = scaler.inverse_transform(scaled_2d)
    return unscaled_2d.reshape(G, T, N).transpose(0, 2, 1).reshape(G * N, T)

def main(cfg_path: str, seed: int = 42):

    # load config
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # paths
    GRAPH_DIR = PROJECT_ROOT / config["data"]["graph_dir"]
    out_dir = PROJECT_ROOT / config["run"]["out_dir"]
    if seed != 42:
        out_dir = out_dir.with_name(f"{out_dir.name}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_name = config['data']['train_file']
    train_name = train_name[1]
    test_name = config['data']['test_file']
    test_name = test_name[1]
    batch_size = config["dataloader"]["batch_size"]
    channel = config["data"]["channel"]
    scaler_path = out_dir / config["scaling"]["scaler_path"]
    model_path = out_dir / "model.pt"

    # loading data
    # If train and test share a file, reuse the held-out split saved at train
    # time  otherwise load the test file and scale it with the fitted scaler.
    if train_name == test_name:
        testset = torch.load(out_dir / "testset.pt", weights_only=False)
    else:
        test_set = torch.load(GRAPH_DIR / test_name, weights_only=False)
        testset = utils.scale(test_set, path=scaler_path, channel=channel, isTest=True)

    # optional noise
    if config["data"]["noise_level"] != 0:
        testset = utils.add_noise(testset,config["data"]["noise_level"])
        torch.save(testset, out_dir / f"testset_noisy_{config['data']['noise_level']}.pt")

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # rebuild model
    mp_type = config["model"]["mp_type"]
    hidden_gnn = config["model"]["hidden_gnn"]
    hidden_lstm = config["model"]["hidden_lstm"]
    n_layers_gnn = config["model"]["n_layers_gnn"]
    n_layers_lstm = config["model"]["n_layers_lstm"]
    num_heads = config["model"]["num_heads"]
    dropout = config["model"]["dropout"]

    x_dim = testset[0].x.shape[1]
    edge_dim = testset[0].edge_attr.shape[1]
    in_dim = x_dim + 1
    out_dim =1

    Arch = getattr(md, config['model']['arch'])

    model = Arch(
        mp_type=mp_type,
        num_heads=num_heads,
        dropout=dropout,
        in_dim=in_dim,
        edge_in=edge_dim,
        hidden_gnn=hidden_gnn,
        n_layers_gnn=n_layers_gnn,
        hidden_lstm=hidden_lstm,
        n_layers_lstm=n_layers_lstm,
        out_dim=out_dim,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # load scalers
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    G = len(testset)
    N = testset[0].target.shape[0]
    T = testset[0].target.shape[1]

    # predict
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    y_pred_scaled = md.test(model, test_loader, device).numpy()
    y_true_scaled = torch.cat([g.target  for g in testset], dim=0).cpu().numpy()
    target_scaler  = scalers["y"]
    y_pred  = inverse(y_pred_scaled, target_scaler,  G, N, T)
    y_true  = inverse(y_true_scaled,  target_scaler,  G, N, T)


    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    inference_time = t1 - t0
    print(f"\nInference time: {inference_time:.4f} s")
    print(f"Graphs: {len(testset)}")
    print(f"Time per graph: {inference_time / len(testset):.6f} s")


    # compute errors per node and per graph, saved as parquet

    df_node   = utils.error_per_node(y_true,  y_pred,  testset)
    df_graph  = utils.error_per_graph(y_true,  y_pred,  testset)
    df_node.to_parquet(out_dir  / f"df_node_{channel}.parquet")
    df_graph.to_parquet(out_dir  / f"df_graph_{channel}.parquet")

    print(f"\nSaved: {out_dir / f'df_node_{channel}.parquet'}")
    print(f"Saved: {out_dir / f'df_graph_{channel}.parquet'}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml or config.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.config, args.seed)
