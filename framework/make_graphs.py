import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import degree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()

def build_graphs(data, config):
    """ Convert a dataframe of simulated structures into a list of PyG graphs.
        args:
        data: pd.DataFrame, the simulation results, one row per simulated structure
        config: dict, the loaded yaml config, sets the target channel, the direction and the sensed nodes
    """
    # which simulated field is used as the target 
    channel = config["data"]["channel"]
    direction = config["data"]["direction"]
    acc_name  = 'acc_'  + direction
    vel_name  = 'vel_'  + direction
    disp_name = 'disp_' + direction


    n_nodes = len(data.loc[0, 'n_properties'])
    sensed_nodes = torch.tensor(config["data"]["selected_nodes"])
    sensed_mask = torch.zeros(n_nodes, dtype=torch.bool)
    sensed_mask[sensed_nodes] = True

    # 1D chain edges, made undirected by appending the reversed pairs
    edge_index_chain = torch.tensor([list(range(n_nodes - 1)), list(range(1, n_nodes))], dtype=torch.long)
    edge_index_chain = torch.cat([edge_index_chain, edge_index_chain.flip(0)], dim=1)

    # shortcut edges: connect every sensed node to all other nodes
    source_idx = sensed_nodes.repeat_interleave(n_nodes)
    destination_idx = torch.arange(n_nodes).repeat(len(sensed_nodes))
    not_cyclic = source_idx != destination_idx
    sensed_nodes_idx = torch.stack([source_idx[not_cyclic], destination_idx[not_cyclic]], dim=0)
    sensed_nodes_idx = torch.cat([sensed_nodes_idx, sensed_nodes_idx.flip(0)], dim=1)

    # remove shortcut edges that already exist in the chain.
    existing_keys = edge_index_chain[0] * n_nodes + edge_index_chain[1]
    sensor_keys = sensed_nodes_idx[0] * n_nodes + sensed_nodes_idx[1]
    to_add_mask = ~torch.isin(sensor_keys, existing_keys)
    sensed_nodes_idx = sensed_nodes_idx[:, to_add_mask]

    lapl_posEnc = AddLaplacianEigenvectorPE(k=3, attr_name=None, is_undirected=True)
    all_graphs = []

    for i in range(len(data)):
        # node features: raw properties + normalised first column (L_norm) + sensed flag
        x = torch.tensor(data.loc[i, 'n_properties'], dtype=torch.float)
        L_norm = (x[:, 0] / x[:, 0].max()).unsqueeze(-1)
        x = torch.cat([x, L_norm, sensed_mask.unsqueeze(-1).float()], dim=1)

        # edge features, duplicated to match the undirected chain
        edge_attr = torch.tensor(data.loc[i, 'el_properties'], dtype=torch.float)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # chain edges
        chain_indicator = torch.tensor([[1.0, 0.0]]).repeat(edge_attr.size(0), 1)
        edge_attr = torch.cat([edge_attr, chain_indicator], dim=1)

        # shortcut edges with no physical features  just binary indicators
        n_shortcut = sensed_nodes_idx.size(1)
        n_edge_feats = edge_attr.size(1) - 2          # everything except the chain indicator
        sensed_edge_attr = torch.zeros((n_shortcut, n_edge_feats))
        shortcut_onehot = torch.tensor([[0.0, 1.0]]).repeat(n_shortcut, 1)
        sensed_edge_attr = torch.cat([sensed_edge_attr, shortcut_onehot], dim=1)

        # combine chain + shortcut into the full connectivity
        edge_index_fc = torch.cat([edge_index_chain, sensed_nodes_idx], dim=1)
        edge_attr_fc = torch.cat([edge_attr, sensed_edge_attr], dim=0)

        graph = Data(x=x, edge_attr=edge_attr_fc, edge_index=edge_index_fc, num_nodes=n_nodes)

        #laplacian eigs encoding
        graph = lapl_posEnc(graph)

        # degree centrality
        deg = degree(edge_index_fc[0], num_nodes=n_nodes, dtype=torch.float)
        deg_cent = (deg / (n_nodes - 1)).unsqueeze(1)
        graph.x = torch.cat([graph.x, deg_cent], dim=1)

        graph.acc  = torch.from_numpy(data.loc[i, acc_name]).float()   
        graph.vel  = torch.from_numpy(data.loc[i, vel_name]).float()   
        graph.disp = torch.from_numpy(data.loc[i, disp_name]).float()  

        target = {"acc": graph.acc, "vel": graph.vel, "disp": graph.disp}[channel]
        graph.mask = sensed_mask
        graph.y = torch.zeros_like(target, dtype=torch.float)
        graph.y[sensed_mask.bool()] = target[sensed_mask.bool()]
        graph.target = target   # unmasked, full field version of whatever channel is selected


        graph.sys_id = data.index[i]
        graph.M =  data.loc[i, "M"]
        graph.C =  data.loc[i, "C"]
        graph.K =  data.loc[i, "K"]
        graph.F =  data.loc[i, "F"]


        all_graphs.append(graph)

    return all_graphs


def main():
    args = parse_args()

    with args.config.open() as f:
        config = yaml.safe_load(f)

    PROJECT_ROOT = Path.cwd().resolve()
    while not (PROJECT_ROOT / "pyproject.toml").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
        PROJECT_ROOT = PROJECT_ROOT.parent

    OUT_DIR = PROJECT_ROOT / Path(config["data"]["graph_dir"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_src, train_out = config["data"]["train_file"]
    test_src,  test_out  = config["data"]["test_file"]

    simulation_dir = PROJECT_ROOT / 'simulationResults'
    if train_src == test_src:
        out_path = OUT_DIR / train_out
        if out_path.exists():
            print(f"Skipping — file already exists: {out_path}")
        else:
            data = pd.read_pickle(simulation_dir / train_src)
            graphs = build_graphs(data, config)
            torch.save(graphs, out_path)
            print(f"Saved {len(graphs)} graphs to: {out_path}")
    else:
        train_out_path = OUT_DIR / train_out
        if train_out_path.exists():
            print(f"Skipping train — file already exists: {train_out_path}")
        else:
            train_data = pd.read_pickle(simulation_dir / train_src)
            train_graphs = build_graphs(train_data, config)
            torch.save(train_graphs, train_out_path)
            print(f"Saved {len(train_graphs)} graphs to: {train_out_path}")

        test_out_path = OUT_DIR / test_out
        if test_out_path.exists():
            print(f"Skipping test — file already exists: {test_out_path}")
        else:
            test_data = pd.read_pickle(simulation_dir / test_src)
            test_graphs = build_graphs(test_data, config)
            torch.save(test_graphs, test_out_path)
            print(f"Saved {len(test_graphs)} graphs to: {test_out_path}")


if __name__ == "__main__":
    main()
