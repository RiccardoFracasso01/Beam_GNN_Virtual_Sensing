import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "pyproject.toml").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import yaml
import torch
import numpy as np
import pickle
import time

from framework import utils as utils


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def inverse(pred_channel, scaler, G, N, T):
    """ Un-scale one channel back to physical units. """
    
    scaled_2d = pred_channel.reshape(G, N, T).transpose(0, 2, 1).reshape(-1, N)
    unscaled_2d = scaler.inverse_transform(scaled_2d)
    return unscaled_2d.reshape(G, T, N).transpose(0, 2, 1).reshape(G * N, T)


def mde_reconstruct(y_meas, Phi_full, sensor_idx):
    """ Reconstruct the full field from the sensor signals in `y_meas` using modal expansion. 

        args:
        y_meas: (N, T) np.array, the measured signals, zero at the unsensed nodes
        Phi_full: (N, n_modes) np.array, the mode shapes at every node
        sensor_idx: list of int, the indices of the sensed nodes
    """
    N = Phi_full.shape[0]
    unmeasured_idx = [k for k in range(N) if k not in sensor_idx]

    Phi_m = Phi_full[sensor_idx, :]      # (n_sensors, n_modes) mode shape at sensors
    Phi_u = Phi_full[unmeasured_idx, :]  # (n_unmeasured, n_modes)
    u_m   = y_meas[sensor_idx, :]        # (n_sensors, T) measured responses (noisy if noise added)

    #  modal expansion: q = pinv(Phi_m) @ u_m,  u_hat = Phi_u @ q
    T_mat = Phi_u @ np.linalg.pinv(Phi_m)
    u_hat = T_mat @ u_m

    u_reconstructed = np.zeros_like(y_meas)
    u_reconstructed[sensor_idx, :]     = u_m
    u_reconstructed[unmeasured_idx, :] = u_hat
    return u_reconstructed


def main(cfg_path: str, meta_path: str, seed: int = 42):

    # load config
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    GRAPH_DIR = PROJECT_ROOT / config["data"]["graph_dir"]
    out_dir = PROJECT_ROOT / config["run"]["out_dir"]
    if seed != 42:
        out_dir = out_dir.with_name(f"{out_dir.name}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_name = config["data"]["train_file"][1]
    test_name  = config["data"]["test_file"][1]
    channel = config["data"]["channel"]
    scaler_path = out_dir / config["scaling"]["scaler_path"]

    sensor_idx = list(config["data"]["selected_nodes"])

    # load data to test
    if train_name == test_name:
        testset = torch.load(out_dir / "testset.pt", weights_only=False)
    else:
        test_set = torch.load(GRAPH_DIR / test_name, weights_only=False)
        testset = utils.scale(test_set, path=scaler_path, channel=channel, isTest=True)

    noise_level = config["data"]["noise_level"]
    if noise_level != 0:
        testset = utils.add_noise(testset, noise_level)



    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    acc_scaler = scalers["y"]

    # load mode shapes
    with open(meta_path, "rb") as f:
        df_meta = pickle.load(f)

    Phi_full = to_numpy(df_meta.mode_shapes[0][:2, :].T)   # using only system 0's mode shapes for every system is intentional, simulates modeling errors

    if Phi_full.shape[1] > len(sensor_idx):
        raise ValueError( f"MDE needs at least as many sensors as modes: got {len(sensor_idx)} sensors and {Phi_full.shape[1]} modes." )

    G = len(testset)
    N = testset[0].target.shape[0]
    T = testset[0].target.shape[1]


    y_true_scaled = torch.cat([g.target for g in testset], dim=0).cpu().numpy()
    y_true  = inverse(y_true_scaled, acc_scaler, G, N, T)   


    y_meas_scaled = torch.cat([g.y for g in testset], dim=0).cpu().numpy()
    y_meas  = inverse(y_meas_scaled, acc_scaler, G, N, T)   # (G*N, T)

    t0 = time.perf_counter()
    blocks = []
    for i in range(G):
        y_meas_i = y_meas[i * N:(i + 1) * N, :]            # (N, T)
        blocks.append(mde_reconstruct(y_meas_i, Phi_full, sensor_idx))
    y_pred_acc = np.concatenate(blocks, axis=0)                # (G*N, T)
    t1 = time.perf_counter()

    inference_time = t1 - t0
    print(f"\nMDE inference time: {inference_time:.4f} s")
    print(f"Graphs: {G}")
    print(f"Time per graph: {inference_time / G:.6f} s")

    # per-node and per-graph errors, saved as parquet

    df_node  = utils.error_per_node(y_true,  y_pred_acc,  testset)
    df_graph = utils.error_per_graph(y_true, y_pred_acc, testset)

    df_node.to_parquet(out_dir  / f"df_node_{channel}_MDE.parquet")
    df_graph.to_parquet(out_dir / f"df_graph_{channel}_MDE.parquet")

    print(f"\nSaved: {out_dir / f'df_node_{channel}_MDE.parquet'}")
    print(f"Saved: {out_dir / f'df_graph_{channel}_MDE.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml (same one used for test.py)")
    ap.add_argument("--meta",   required=True, help="Path to a pickle holding df_meta (must expose .mode_shapes)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.config, args.meta, args.seed)
