import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


def iter_block(y_true, y_pred, testset):
    """ Yields individual graphs and their unstacked predictions one at a time.

        args:
        y_true: (sum(n_nodes), T) np.array, the stacked ground-truth signals
        y_pred: (sum(n_nodes), T) np.array, the stacked predicted signals
        testset: list of Data, the graphs the arrays were stacked from, in order
    """
    cursor = 0
    for g in testset:
        n = g.num_nodes
        yield g, y_true[cursor:cursor + n], y_pred[cursor:cursor + n]
        cursor += n


def node_metrics(yt, yp, tol=1e-9):
    """ Compute the per-node error metrics along the time axis. Metrics with a zero denominator are set to NaN
        args:
        yt: (n_nodes, T) np.array, the ground-truth signals
        yp: (n_nodes, T) np.array, the predicted signals
        tol: float, the threshold below which a denominator is treated as zero
    """
    diff = yt - yp
    mae = np.mean(np.abs(diff), axis=1)
    rmse = np.sqrt(np.mean(diff**2, axis=1))

    abs_true = np.sum(np.abs(yt), axis=1)
    ape = np.where(abs_true > tol, np.sum(np.abs(diff), axis=1) / abs_true * 100, np.nan)

    num = np.sum(yt * yp, axis=1)**2
    den = np.sum(yt * yt, axis=1) * np.sum(yp * yp, axis=1)
    trac = np.where(den > tol, num / den, np.nan)

    rng = yt.max(axis=1) - yt.min(axis=1)
    nrmse = np.where(rng > tol, rmse / rng, np.nan)

    ss_res = np.sum(diff**2, axis=1)
    ss_tot = np.sum((yt - yt.mean(axis=1, keepdims=True))**2, axis=1)
    r2 = np.where(ss_tot > tol, 1 - ss_res / ss_tot, np.nan)

    return {"mae": mae, "ape": ape, "trac": trac, "rmse": rmse, "nrmse": nrmse, "r2": r2}


def safe_nanmean(a):
    """ Mean of an array, returning NaN when none are valid. Unlike np.nanmean this stays quiet on an all-NaN input instead of warning. """
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def error_per_node(y_true, y_pred, testset, tol=1e-9):
    """ Build the per-node error  for the whole test set. One row per node  per graph.

        args:
        y_true: (sum(n_nodes), T) np.array, the stacked ground-truth signals
        y_pred: (sum(n_nodes), T) np.array, the stacked predicted signals
        testset: list of Data, the graphs the arrays were stacked from
        tol: float, the threshold below which a metric denominator is treated as zero
    """
    rows = []
    for g, yt, yp in iter_block(y_true, y_pred, testset):
        m = node_metrics(yt, yp, tol)
        sys_id = int(g.sys_id)
        for i in range(len(yt)):
            rows.append({
                "sys_id": sys_id, "n_nodes": g.num_nodes, "node_pos": i + 1,
                "y_true": yt[i], "y_pred": yp[i],
                **{k: v[i] for k, v in m.items()},
            })
    return pd.DataFrame(rows)


def error_per_graph(y_true, y_pred, testset, tol=1e-9):
    """ Build the per-graph error. One row per graph

        args:
        y_true: (sum(n_nodes), T) np.array, the stacked ground-truth signals
        y_pred: (sum(n_nodes), T) np.array, the stacked predicted signals
        testset: list of Data, the graphs the arrays were stacked from
        tol: float, the threshold below which a metric denominator is treated as zero
    """
    rows = []
    for gi, (g, yt, yp) in enumerate(iter_block(y_true, y_pred, testset)):
        if np.max(np.abs(yt)) <= tol:
            continue
        m = node_metrics(yt, yp, tol)
        sys_id = int(g.sys_id) if hasattr(g, "sys_id") else gi
        rows.append({
            "sys_id": sys_id, "n_nodes": g.num_nodes,
            **{k: safe_nanmean(v) for k, v in m.items()},
        })
    return pd.DataFrame(rows)


def fit_scaler(graphs, attr_name, transpose=False, row_limit=None, col_limit=None):
    """ Fits a StandardScaler on a pooled attribute across all graphs.
        Standardization occurs along columns. Signals stored as (n_nodes, T) 
        must be transposed to (T, n_nodes) so each node is standardized individually.

        args:
        graphs: list of Data, the graphs to pool the statistics over
        attr_name: str, the attribute to fit on, e.g. 'x', 'acc', 'edge_attr'
        transpose: bool, whether to transpose the attribute before fitting (True for the signals)
        row_limit: int, fit on the first row_limit rows only, None for all
        col_limit: int, fit on the first col_limit columns only, None for all

    """
    def to_np(g):
        a = g[attr_name].detach().cpu().numpy()
        a = a.T if transpose else a
        return a[:row_limit, :col_limit]

    data = np.concatenate([to_np(g) for g in graphs], axis=0)
    return StandardScaler().fit(data)


def transform(graphs, attr_name, scaler, transpose=False, row_limit=None, col_limit=None):
    """ Apply a fitted scaler to one attribute of every graph, in place.

        args:
        graphs: list of Data, the graphs to scale
        attr_name: str, the attribute to scale
        scaler: StandardScaler, the fitted scaler to apply
        transpose: bool, whether to transpose before scaling, must match the fit_scaler call
        row_limit: int, scale the first row_limit rows only, None for all
        col_limit: int, scale the first col_limit columns only, None for all
    """
    for g in graphs:
        data = g[attr_name].detach().cpu().numpy()
        data = data.T if transpose else data
        to_scale = data[:row_limit, :col_limit]
        scaled = scaler.transform(to_scale)
        data[:row_limit, :col_limit] = scaled
        data = data.T if transpose else data
        g[attr_name] = torch.from_numpy(data).float()


# the attributes that get scaled, as (attribute name, scaler key, transpose before scaling).
# the three signals are transposed so that each node is standardized independently, the node and
# edge features are not, since their columns are already the features
FIELDS = [("x", "x", False), ("acc", "acc", True), ("vel", "vel", True), ("disp", "disp", True), ("edge_attr", "edge_attr", False)]


def fit_all(graphs, channel):
    """ Fit one scaler per entry of FIELDS on the given graphs.

        args:
        graphs: list of Data, the graphs to fit on (the train set, never the test set)
        channel: str, the target signal, one of ['acc', 'vel', 'disp']
    """
    scalers = {key: fit_scaler(graphs, attr, transpose=t) for attr, key, t in FIELDS}
    scalers["y"] = scalers[channel]
    return scalers


def apply_all(graphs, scalers, channel):
    """ Scale all graph attributes in place and construct the model input/target pair.
        Adds two attributes to every graph:
        - target: the full-field signal of the selected channel
        - y: the target masked to the sensed nodes, zero elsewhere

        args:
        graphs: list of Data, the graphs to scale, modified in place
        scalers: dict, the fitted scalers keyed by attribute
        channel: str, the target signal, one of ['acc', 'vel', 'disp']
    """
    for attr, key, t in FIELDS:
        transform(graphs, attr, scalers[key], transpose=t)
    for g in graphs:
        g.target = g[channel].clone()
        g.y = torch.zeros_like(g.target)
        g.y[g.mask.bool()] = g.target[g.mask.bool()]


def scale(graphs, path, channel, isTest=False):
    """ Fit or load the scalers, then apply them to the graphs.
        Used when the train and test sets come from different simulation files: the train call fits
        and writes the scalers, the test call loads the very same ones back.

        args:
        graphs: list of Data, the graphs to scale, modified in place
        path: str or Path, where the scalers are written to (train) or read from (test)
        channel: str, the target signal, one of ['acc', 'vel', 'disp']
        isTest: bool, False fits the scalers on these graphs and saves them, True loads them instead
    """
    path = str(path)
    if not isTest:
        scalers = fit_all(graphs, channel)
        with open(path, "wb") as f: pickle.dump(scalers, f)
    else:
        with open(path, "rb") as f: scalers = pickle.load(f)

    apply_all(graphs, scalers, channel)
    return graphs


def scale_same(graphs, path, test_size, channel, seed=42):
    """ Split the graphs into train and test, fit the scalers on the train part only, then apply them to both parts in place.
        Used when the train and test sets come from the same simulation file. 
    """
    if path is not None:
        path = str(path)
    trainset, testset = train_test_split(graphs, test_size=test_size, shuffle=True, random_state=seed)

    # fit on the train half only, so that the test statistics do not leak into the scaling
    scalers = fit_all(trainset, channel)
    if path is not None:
        with open(path, "wb") as f: pickle.dump(scalers, f)

    apply_all(trainset, scalers, channel)
    apply_all(testset, scalers, channel)

    if path is None:
        return trainset, testset, scalers
    else:
        return trainset, testset


def add_noise(graphs, noise_level, seed=42):
    """ Adds peak-scaled Gaussian noise to the sensed nodes of each graph in place.
        The noise is scaled per signal by its own peak value to maintain a constant signal-to-noise
        ratio across all nodes. The clean ground truth (`target`) remains unmodified.

        args:
        graphs: list of Data, the graphs to modify in place
        noise_level: float, the noise standard deviation as a fraction of each signal's peak
        seed: int, the seed of the noise generator, for reproducibility
    """
    gen = torch.Generator(device=graphs[0].y.device).manual_seed(seed)
    for g in graphs:
        signals = g.y[g.mask.bool(), :]
        max_values = signals.abs().max(dim=1, keepdim=True).values
        noise = torch.randn(signals.shape, generator=gen, dtype=signals.dtype, device=signals.device)
        g.y[g.mask.bool(), :] = signals + noise * (noise_level * max_values)
    return graphs


def integrate(y, dt, channel, highpass_hz=2.0):
    """ Recover the kinematic triplet (acc, vel, disp) from a single channel prediction.
        Integration and differentiation are performed in the frequency domain. To prevent
        low-frequency drift during integration, frequencies below `highpass_hz` are zeroed
        out (high-pass filtered).

        args:
        y: (batch, n_nodes, T) torch.Tensor, the predicted signal
        dt: float, the timestep of the signal
        channel: str, the predicted signal type, one of ['acc', 'vel', 'disp']
        highpass_hz: float, the cutoff frequency [Hz] below which content is discarded
    """
    n = y.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=dt).to(device=y.device, dtype=y.dtype)
    omega = 2 * np.pi * freqs

    iw = 1j * omega
    iw2 = -(omega ** 2)

    # dividing by inf below the cutoff zeroes those bins out, which is the high-pass filter
    iw_safe = torch.where(freqs < highpass_hz, torch.full_like(iw, float('inf')), iw)
    iw2_safe = torch.where(freqs < highpass_hz, torch.full_like(iw2, float('inf')), iw2)

    Y = torch.fft.rfft(y, dim=-1)

    # integrate (divide by i*omega) or differentiate (multiply by i*omega) to reach the other two
    if channel == "acc":
        A, V, X = Y, Y / iw_safe[None, None, :], Y / iw2_safe[None, None, :]
    elif channel == "vel":
        A, V, X = Y * iw[None, None, :], Y, Y / iw_safe[None, None, :]
    elif channel == "disp":
        A, V, X = Y * iw2[None, None, :], Y * iw[None, None, :], Y
    else:
        raise ValueError(f"unknown channel: {channel}")

    a = torch.fft.irfft(A, n=n, dim=-1)
    v = torch.fft.irfft(V, n=n, dim=-1)
    x = torch.fft.irfft(X, n=n, dim=-1)
    return a, v, x


def eom_loss(M, C, K, F, a, v, x):
    """ Computes the mean squared residual of the equation of motion: M*a + C*v + K*x - F.
        The system matrices are condensed to the last n_dof translational degrees of freedom,
        so the kinematic states and the forcing are sliced to match.

        args:
        M, C, K: (batch, n_dof, n_dof) torch.Tensor, the system matrices
        F: (batch, n_nodes, T) torch.Tensor, the applied forcing
        a, v, x: (batch, n_nodes, T) torch.Tensor, the acceleration, velocity and displacement
    """
    n_dof = M.shape[1]

    # keep only the DOFs the system matrices were condensed down to
    a = a[:, -n_dof:, :]
    v = v[:, -n_dof:, :]
    x = x[:, -n_dof:, :]
    F = F[:, -n_dof:, :]

    Ma = torch.bmm(M, a)
    Cv = torch.bmm(C, v)
    Kx = torch.bmm(K, x)

    r = Ma + Cv + Kx - F
    return torch.mean(r**2)
