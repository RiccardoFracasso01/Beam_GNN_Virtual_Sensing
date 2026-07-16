import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "pyproject.toml").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import math
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from framework import utils
from framework import model as md

# settings
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.15
BATCH_SIZE   = 10
GRAD_CLIP    = 1.0
MP_TYPE      = "GAT"
ARCH         = "GNN_LSTM"
OUT_DIM      = 1
CHANNEL      = "acc"      # 'acc' | 'vel' | 'disp' which signal is predicted by the model
PHYS_DT      = 0.002      # timestep of the signal
PHYS_HIGHPASS = 2.0       # high-pass cutoff (Hz)
HPO_DIR      = PROJECT_ROOT / "hpo_results"
GRAPH_DIR    = PROJECT_ROOT / "graphs"
GRAPHS_FILE  = "example.pt"
SCALER_PATH  = HPO_DIR / "scalers.pkl"

# optuna search
N_TRIALS     = 50
TUNE_EPOCHS  = 200      # epochs per trial during the search
FINAL_EPOCHS = 200        # epochs written into the final config for full training

# lambda_phys search
PHYS_EPOCHS      = 200
LAMBDA_DATA      = 1.0
LAMBDA_PHYS_GRID = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]

def build_search_space(trial: optuna.Trial) -> dict:
    """ Sample one set of hyperparameters from the Optuna search space """
    hp = {
        "lr":            trial.suggest_float("lr",           5e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "hidden_gnn":    trial.suggest_categorical("hidden_gnn",  [32, 64, 128]),
        "n_layers_gnn":  trial.suggest_int("n_layers_gnn", 1, 2),
        "hidden_lstm":   trial.suggest_categorical("hidden_lstm", [64, 128, 256]),
        "n_layers_lstm": trial.suggest_int("n_layers_lstm", 1, 2),
        "dropout":       trial.suggest_float("dropout", 0.0, 0.2),
        "eta_min":       trial.suggest_float("eta_min", 1e-7, 1e-5, log=True),
    }
    # GAT needs an attention head other message passing types use 1
    hp["num_heads"] = trial.suggest_categorical("num_heads", [2, 4]) if MP_TYPE == "GAT" else 1
    return hp


def build_model(hp: dict, in_dim: int, edge_dim: int, device: torch.device) -> nn.Module:
    Arch = getattr(md, ARCH)
    return Arch(
        mp_type       = MP_TYPE,
        num_heads     = hp.get("num_heads", 1) if MP_TYPE == "GAT" else 1,
        dropout       = hp["dropout"],
        in_dim        = in_dim,
        edge_in       = edge_dim,
        hidden_gnn    = hp["hidden_gnn"],
        n_layers_gnn  = hp["n_layers_gnn"],
        hidden_lstm   = hp["hidden_lstm"],
        n_layers_lstm = hp["n_layers_lstm"],
        out_dim       = OUT_DIM,
    ).to(device)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """ Compute the RMSE and MAE of the model over a set """
    model.eval()

    sum_sq  = 0
    sum_abs = 0
    n_elem  = 0

    for batch in loader:
        batch  = batch.to(device)
        y_pred = model(batch).squeeze(-1)
        y_true = batch.target
        diff = y_pred - y_true
        sum_sq  += diff.pow(2).sum().item()
        sum_abs += diff.abs().sum().item()
        n_elem  += diff.numel()

    model.train()
    n = max(n_elem, 1)
    return { "rmse": math.sqrt(sum_sq / n), "mae":  sum_abs / n,}


# optuna search
def make_objective(trainset, valset, device: torch.device):
    """ Builds the Optuna objective closure over the given data splits.

        The returned objective trains a model for TUNE_EPOCHS, reports validation
        RMSE to the trial for pruning, and returns the best achieved RMSE.

        args:
        trainset: list of Data, the training graphs
        valset: list of Data, the validation graphs
        device: torch.device, the device to train on
    """
    in_dim    = trainset[0].x.shape[1] + 1
    edge_dim  = trainset[0].edge_attr.shape[1]
    criterion = nn.MSELoss()

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False)

    def objective(trial: optuna.Trial) -> float:
        """ Train one model and return its best validation RMSE. """
        hp = build_search_space(trial)

        model     = build_model(hp, in_dim, edge_dim, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TUNE_EPOCHS, eta_min=hp["eta_min"])

        best_rmse    = float("inf")
        best_metrics = {}

        for epoch in range(1, TUNE_EPOCHS + 1):
            md.train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                grad_clip=GRAD_CLIP,
            )
            metrics = evaluate(model, val_loader, device)
            scheduler.step()

            trial.report(metrics["rmse"], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if metrics["rmse"] < best_rmse:
                best_rmse    = metrics["rmse"]
                best_metrics = metrics

        for k, v in best_metrics.items():
            trial.set_user_attr(k, round(v, 6))

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

        return best_rmse

    return objective


# lambda physics grid search
def train_phys_config(lambda_phys: float, hp: dict, trainset, valset, device: torch.device) -> dict:
    """ Trains a model with physics-informed loss for a single lambda_phys weight.
        Used for the grid search to find optimal weighting.
    """
    in_dim    = trainset[0].x.shape[1] + 1
    edge_dim  = trainset[0].edge_attr.shape[1]
    criterion = nn.MSELoss()

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False)

    model     = build_model(hp, in_dim, edge_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHYS_EPOCHS, eta_min=hp["eta_min"])

    best_rmse    = float("inf")
    best_metrics = {}

    for epoch in range(1, PHYS_EPOCHS + 1):
        md.train_phys(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler_path=SCALER_PATH,
            channel=CHANNEL,
            device=device,
            lambda_data=LAMBDA_DATA,
            lambda_phy=lambda_phys,
            dt = PHYS_DT,
            highpass_hz = PHYS_HIGHPASS,
        )
        metrics = evaluate(model, val_loader, device)
        scheduler.step()

        if metrics["rmse"] < best_rmse:
            best_rmse    = metrics["rmse"]
            best_metrics = metrics

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return best_metrics


def main():
    HPO_DIR.mkdir(parents=True, exist_ok=True)

    graphs           = torch.load(GRAPH_DIR / GRAPHS_FILE, weights_only=False)
    trainval, _      = train_test_split(graphs,   test_size=1 - TRAIN_RATIO, random_state=42)
    trainval = utils.scale(trainval, path=SCALER_PATH, channel=CHANNEL, isTest=False)
    trainset, valset = train_test_split(trainval, test_size=VAL_RATIO,        random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Total  : {len(graphs)} graphs  →  train {len(trainset)} | val {len(valset)} | held-out {len(graphs) - len(trainval)}")
    print(f"Stage 1: {N_TRIALS} trials × {TUNE_EPOCHS} epochs")
    print(f"Stage 2: {len(LAMBDA_PHYS_GRID)} lambda values × {PHYS_EPOCHS} epochs\n")


    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=15),
    )
    study.optimize(
        make_objective(trainset, valset, device),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n[Stage 1] Best trial #{best.number}")
    print(f"  RMSE : {best.user_attrs['rmse']}")
    print(f"  MAE  : {best.user_attrs['mae']}")

    # collect hyperparameters
    best_hp = {
        "lr":            best.params["lr"],
        "weight_decay":  best.params["weight_decay"],
        "hidden_gnn":    best.params["hidden_gnn"],
        "n_layers_gnn":  best.params["n_layers_gnn"],
        "hidden_lstm":   best.params["hidden_lstm"],
        "n_layers_lstm": best.params["n_layers_lstm"],
        "dropout":       best.params["dropout"],
        "eta_min":       best.params["eta_min"],
        "num_heads":     best.params.get("num_heads", 1),
    }

    out_data = {
        "model": {
            "mp_type":       MP_TYPE,
            "arch":          ARCH,
            "hidden_gnn":    best_hp["hidden_gnn"],
            "hidden_lstm":   best_hp["hidden_lstm"],
            "n_layers_gnn":  best_hp["n_layers_gnn"],
            "n_layers_lstm": best_hp["n_layers_lstm"],
            "num_heads":     best_hp["num_heads"],
            "dropout":       round(best_hp["dropout"], 6),
        },
        "train": {
            "lr":           best_hp["lr"],
            "weight_decay": best_hp["weight_decay"],
            "eta_min":      best_hp["eta_min"],
        },
        "val": {
            "rmse": best.user_attrs["rmse"],
            "mae":  best.user_attrs["mae"],
        },
    }
    data_path = HPO_DIR / "best_params.json"
    with open(data_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved → {data_path}")

    # lambda physics grid search
    print(f"\n Grid search over lambda_phys using best  hyperparameters\n")
    results      = []
    best_overall = {"rmse": float("inf")}

    for i, lp in enumerate(LAMBDA_PHYS_GRID, 1):
        print(f"[{i}/{len(LAMBDA_PHYS_GRID)}] lambda_phys = {lp:.2e}")
        metrics = train_phys_config(lp, best_hp, trainset, valset, device)
        print(f"   rmse = {metrics['rmse']:.6f} | mae = {metrics['mae']:.6f}\n")
        results.append({"lambda_phys": lp, **metrics})
        if metrics["rmse"] < best_overall["rmse"]:
            best_overall = {"lambda_phys": lp, **metrics}

    print(f"[Stage 2] Best: lambda_phys = {best_overall['lambda_phys']:.2e}")
    print(f"  RMSE : {best_overall['rmse']:.6f}")
    print(f"  MAE  : {best_overall['mae']:.6f}")

    out_pi = {
        "model": {
            "mp_type":       MP_TYPE,
            "arch":          ARCH,
            "hidden_gnn":    best_hp["hidden_gnn"],
            "hidden_lstm":   best_hp["hidden_lstm"],
            "n_layers_gnn":  best_hp["n_layers_gnn"],
            "n_layers_lstm": best_hp["n_layers_lstm"],
            "num_heads":     best_hp["num_heads"],
            "dropout":       round(best_hp["dropout"], 6),
        },
        "train": {
            "lr":           best_hp["lr"],
            "weight_decay": best_hp["weight_decay"],
            "eta_min":      best_hp["eta_min"],
            "lambda_data":  LAMBDA_DATA,
            "lambda_phys":  best_overall["lambda_phys"],
        },
        "val": {
            "rmse": round(best_overall["rmse"], 6),
            "mae":  round(best_overall["mae"], 6),
        },
        # full grid results kept for inspection
        "grid": [
            {"lambda_phys": r["lambda_phys"],
             "rmse": round(r["rmse"], 6),
             "mae":  round(r["mae"], 6)}
            for r in results
        ],
    }
    pi_path = HPO_DIR / "best_params_PI.json"
    with open(pi_path, "w") as f:
        json.dump(out_pi, f, indent=2)
    print(f"\nSaved → {pi_path}")


if __name__ == "__main__":
    main()
