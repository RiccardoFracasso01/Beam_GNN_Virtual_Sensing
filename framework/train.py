# imports
import sys
from pathlib import Path
import random

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "pyproject.toml").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm


from framework import utils as utils
from framework import model as md

def set_seed(seed: int):
    """ Seed every random number generator (python, numpy, torch cpu and cuda) for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(cfg_path: str, run_type: str, seed: int = 42):
    # load config
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)


    # paths
    GRAPH_DIR = PROJECT_ROOT / config['data']['graph_dir']
    out_dir = PROJECT_ROOT / config['run']['out_dir']
    if seed != 42:
        out_dir = out_dir.with_name(f"{out_dir.name}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = out_dir / config['scaling']['scaler_path']
    # parameters
    train_name = config['data']['train_file']
    train_name = train_name[1]
    test_name = config['data']['test_file']
    test_name = test_name[1]
    batch_size = config['dataloader']['batch_size']

    mp_type=config['model']['mp_type']
    hidden_gnn  = config['model']['hidden_gnn']
    hidden_lstm = config['model']['hidden_lstm']
    n_layers_gnn  = config['model']['n_layers_gnn']
    n_layers_lstm = config['model']['n_layers_lstm']
    num_heads= config['model']['num_heads']
    channel = config['data']['channel']
    dropout= config['model']['dropout']

    lr = config['train']['lr']
    weight_decay = config['train']['weight_decay']
    epochs = config['train']['epochs']
    eta_min = config['train']['eta_min']

    # loading data
    train_set = torch.load(GRAPH_DIR / train_name, weights_only = False )
    if train_name == test_name:
        trainset,testset = utils.scale_same(train_set, path=scaler_path, test_size=config['data']["test_size"],channel=channel)
        torch.save(testset,  out_dir / "testset.pt")
    else:
        trainset = utils.scale(train_set, path=scaler_path,channel=channel, isTest = False)

    if config["data"]["noise_level"] != 0:
        trainset = utils.add_noise(trainset,config["data"]["noise_level"])


    # validation split
    val_frac = 0.1
    trainset, valset = train_test_split(trainset, test_size=val_frac, random_state=42)
    set_seed(seed)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model initialisation
    x_dim    = trainset[0].x.shape[1]
    edge_dim = trainset[0].edge_attr.shape[1]
    y_step_dim = 1
    in_dim = x_dim + y_step_dim
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
        out_dim=out_dim
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)


   # training
    if run_type == "data":
        train_fn = md.train
    elif  run_type == "phys":
        train_fn = md.train_phys
    else:
        raise ValueError(f"Unknown run_type: '{run_type}'")

    # data loss
    if run_type == "data":
        train_losses, val_losses = [], []
        best_val = float('inf')
        best_epoch = 0
        t0 = time.perf_counter()
        pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch", dynamic_ncols=True)
        for epoch in pbar:
            e0 = time.perf_counter()
            train_loss = train_fn(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                grad_clip=1,
            )
            val_loss = md.validate(model, val_loader, criterion, device)
            scheduler.step()

            e1 = time.perf_counter()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), out_dir / "model.pt")

            pbar.set_postfix(
                train=f"{train_loss:.4e}", val=f"{val_loss:.4e}",
                best=f"{best_val:.4e}@{best_epoch}",
                epoch_s=f"{(e1-e0):.1f}", refresh=True,
            )

        t1 = time.perf_counter()
        tqdm.write(f"Done. Total: {(t1-t0)/60:.2f} min | Avg/epoch: {(t1-t0)/max(epochs,1):.2f} s\n")


        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
        ax.plot(range(1, len(val_losses) + 1),   val_losses,   label="Val loss")
        ax.axvline(best_epoch, color='k', ls=':', alpha=0.5, label=f"Best val @ {best_epoch}")
        ax.set_title(f"Training Progress [{run_type}]")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / "loss_curve.png", dpi=150)
        plt.close(fig)

    # physics loss
    elif run_type == "phys":
        phys_dt = config['phys_loss']['dt']
        phys_highpass = config['phys_loss']['highpass_hz']
        history = {"total": [], "data": [], "physics": []}
        val_losses = []
        best_val = float('inf')
        best_epoch = 0
        t0 = time.perf_counter()
        pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch", dynamic_ncols=True)
        for epoch in pbar:
            losses = train_fn(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch =epoch,
                scaler_path = scaler_path,
                channel = channel,
                grad_clip=1,
                dt = phys_dt,
                highpass_hz = phys_highpass,
            )
            val_loss = md.validate(model, val_loader, criterion, device)
            scheduler.step()

            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), out_dir / "model.pt")

            for k, v in losses.items():
                history[k].append(v)

            pbar.set_postfix(
                total=f"{losses['total']:.4f}", data=f"{losses['data']:.4f}",
                phys=f"{losses['physics']:.4f}",
                val=f"{val_loss:.4e}", best=f"{best_val:.4e}@{best_epoch}",
                refresh=True,
            )

        t1 = time.perf_counter()
        tqdm.write(f"Done. Total: {(t1-t0)/60:.2f} min | Avg/epoch: {(t1-t0)/max(epochs,1):.2f} s\n")


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        epochs_range = range(1, epochs + 1)

        ax1.plot(epochs_range, history["total"],   label="Total loss")
        ax1.plot(epochs_range, history["data"],    label="Data loss")
        ax1.plot(epochs_range, history["physics"], label="Physics loss")
        ax1.plot(epochs_range, val_losses, label="Val loss (data only)")
        ax1.axvline(best_epoch, color='k', ls=':', alpha=0.5, label=f"Best val @ {best_epoch}")
        ax1.set_title(f"Training Progress [ {run_type}]")
        ax1.set_ylabel("Loss")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(True, which="both", ls="--", alpha=0.4)

        ratio = np.array(history["physics"]) / (np.array(history["data"]) + 1e-12)
        ax2.plot(epochs_range, ratio, label="Physics / Data ratio")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Ratio")
        ax2.legend()
        ax2.grid(True, which="both", ls="--", alpha=0.4)

        fig.tight_layout()
        fig.savefig(out_dir / "loss_curve.png", dpi=150)
        plt.close(fig)




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml or config.json")
    ap.add_argument("--type",   required=True, help="data or phys")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()
    main(args.config, args.type, args.seed)
