import openseespy.opensees as ops
import numpy as np
import pandas as pd
import yaml
import math
import argparse
from scipy.signal import decimate, butter, sosfiltfilt
from scipy.stats import qmc
from pathlib import Path
from joblib import Parallel, delayed
from dataclasses import dataclass
from scipy.linalg import solve
from scipy import signal


@dataclass
class SimResult:
    """ The full output of a single simulated beam.

        The response is stored for both bending directions 
        fields:
        acc_z, acc_y: (n_nodes, T) np.array, the nodal accelerations, decimated and trimmed
        vel_z, vel_y: (n_nodes, T) np.array, the nodal velocities
        disp_z, disp_y: (n_nodes, T) np.array, the nodal displacements
        el_properties: (n_nodes - 1, 5) np.array, one row per element: [A, Iz, Iy, J, mu]
        n_properties: (n_nodes, 2) np.array, one row per node: [x, spring stiffness] (0 if cantilever)
        f0: (5,) np.array, the natural frequencies [Hz] of the 5 retained bending modes
        mode_shapes: (5, n_nodes) np.array, the mass-normalized mode shapes, sign-fixed to be positive at the tip
        M, C, K: (n_dof, n_dof) np.array, the system matrices, Guyan-condensed to the translational DOFs
        F: (n_nodes, T) np.array, the applied forcing on the same time grid as the response
    """
    acc_z:          np.ndarray   # (n_nodes, T)
    acc_y:          np.ndarray
    vel_z:          np.ndarray
    vel_y:          np.ndarray
    disp_z:         np.ndarray
    disp_y:         np.ndarray
    el_properties:  np.ndarray   # element properties
    n_properties:   np.ndarray   # node properties
    f0:             np.ndarray   # (5,)
    mode_shapes:    np.ndarray   # (5, n_nodes)
    M: np.array
    C: np.array
    K: np.array
    F: np.array


def get_args():
    parser = argparse.ArgumentParser(prog='makeDataset', description='Generate and simulate response of a population of beams')
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel workers (-1 = all cores)")
    return parser.parse_args()


def excitation_number(n_samples, n_exc_pos):
    """ Draws the number of simultaneous excitations for each system in the population.
        If n_exc_pos is a scalar, all systems receive that number of excitations. If it is a
        sequence, the population is evenly divided among the choices and shuffled to ensure a balanced mix of loading cases.

        args:
        n_samples: int, the total size of the population
        n_exc_pos: int or list of int, the allowed number(s) of simultaneous excitations
    """
    if np.isscalar(n_exc_pos):
        allowed = [int(n_exc_pos)]
    else:
        allowed = [int(x) for x in n_exc_pos]
    allowed_arr = np.array(allowed)
    reps = math.ceil(n_samples / len(allowed_arr))
    load_vector = np.tile(allowed_arr, reps)[:n_samples]
    np.random.shuffle(load_vector)
    return load_vector


def excitation_position(n_excitation, target_nodes):
    """ Draws unique node indices where excitations will be applied.
        Supports two targeting modes:
        - a single flat range [low, high]: the nodes are drawn without replacement
        - a list of ranges [[l1, h1], [l2, h2], ...]: one node is drawn per selected sub-range,
          to keep the excitations physically distributed

        args:
        n_excitation: int, the number of excitations to place
        target_nodes: list, a flat [low, high] range or a list of [low, high] ranges
    """
    is_flat_range = (
        len(target_nodes) == 2
        and np.isscalar(target_nodes[0])
        and np.isscalar(target_nodes[1])
    )

    if is_flat_range:
        low, high = int(target_nodes[0]), int(target_nodes[1])
        available = np.arange(low, high + 1)
        nodes = np.random.choice(available, size=n_excitation, replace=False)
        return sorted(nodes.tolist())

    chosen_idx = np.random.choice(len(target_nodes), size=n_excitation, replace=False)
    nodes = []
    for idx in chosen_idx:
        low, high = target_nodes[idx]
        nodes.append(np.random.randint(low, high + 1))
    return sorted(nodes)


# sobol sampling
def generate_sobol_population(param_cfg, load_cfg, n_samples):
    """ Samples a population of systems using a Sobol sequence.

        args:
        param_cfg: dict, the parameter names and their sampling ranges
        load_cfg: dict, the loading config, sets the excitation counts and the target nodes
        n_samples: int, the size of the population to generate
    """
    keys = list(param_cfg.keys())
    dim = len(keys)
    sobol_sampler = qmc.Sobol(d=dim, scramble=True)
    points = sobol_sampler.random(n_samples)

    n_exc_points = load_cfg['n_exc_points']
    target_ranges = load_cfg['target_nodes']
    n_loads_array = excitation_number(n_samples, n_exc_points)

    population = []
    for j in range(n_samples):
        p = {}
        selected_range = {}

        for i, k in enumerate(keys):
            # split the Sobol coordinate into a range choice and a position inside that range
            u = points[j, i]
            ranges = param_cfg[k]["ranges"]
            n_ranges = len(ranges)
            idx = min(int(u * n_ranges), n_ranges - 1)
            low, high = map(float, ranges[idx])
            u_local = (u * n_ranges) - idx
            p[k] = low + u_local * (high - low)
            selected_range[k] = (low, high)

        p["n_forces"] = int(n_loads_array[j])
        p["target_nodes"] = excitation_position(p["n_forces"], target_ranges)

        # one amplitude per excitation, drawn from the range the Sobol sample landed in
        F_low, F_high = selected_range["force_amplitude"]
        p["force_amplitude"] = np.random.uniform(F_low, F_high, size=p["n_forces"]).tolist()

        population.append(p)

    return population


def guyan(M, C, K, targetdof):
    """ Guyan (static) condensation of the system matrices onto the translational DOFs.

        args:
        M: (n, n) np.array, the full mass matrix
        C: (n, n) np.array, the full damping matrix
        K: (n, n) np.array, the full stiffness matrix
        targetdof: int, 1 for y-direction, 2 for z-direction
    """
    all_indices = np.arange(M.shape[0])
    master_indices = np.arange(targetdof, M.shape[0], 6)
    slave_indices = np.array([i for i in all_indices if i not in master_indices])

    # K partition
    Kmm = K[np.ix_(master_indices, master_indices)]
    Kms = K[np.ix_(master_indices, slave_indices)]
    Ksm = K[np.ix_(slave_indices, master_indices)]
    Kss = K[np.ix_(slave_indices, slave_indices)]

    # the slaves follow the masters statically, which gives the transformation L
    T_static = solve(Kss, -Ksm)
    identity = np.eye(len(master_indices))
    L_mat = np.zeros((M.shape[0], len(master_indices)))
    L_mat[master_indices, :] = identity
    L_mat[slave_indices, :] = T_static

    # project the full matrices onto the retained DOFs
    K_red = L_mat.T @ K @ L_mat
    M_red = L_mat.T @ M @ L_mat
    C_red = L_mat.T @ C @ L_mat

    return M_red, C_red, K_red


def run_simulation(p, cfg):
    """ Build one beam and simulate its transient response in OpenSeesPy.
        args:
        p: dict, the parameter set of this system, as produced by generate_sobol_population
        cfg: dict, the parsed simulation config

    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # beam properties
    E = float(p["E"])
    L = float(p["L"])
    b0 = float(p["b0"]) * L       # the cross-section is sampled as a fraction of the length,
    h0 = float(p["h0"]) * L       # so that the slenderness stays in a sensible range
    if cfg["taper"]["isTapered"]:
        db = cfg["taper"]["db"]
        dh = cfg["taper"]["dh"]
        l_taper_start = cfg["taper"]["l_start"]
    else:
        db = 0
        dh = 0
        l_taper_start = 0

    rho = float(p["rho"])
    zeta = float(p["zeta"])
    nu = 0.3
    G = E / (2 * (1 + nu))

    # if an embedded length was sampled, the beam is supported by springs instead of being clamped
    L_embed = p.get("L_embedded")
    if L_embed is not None:
        L_embed = float(L_embed) * L
        k_x_max = float(p["k_x"])
        k_r_max = float(p["k_r"])

    # load properties
    load_dir = cfg["load"]["force_dir"]

    # nodes, equally spaced along the beam axis
    n_nodes = cfg["simulation"]["num_nodes"]
    xs = np.linspace(0, L, n_nodes)

    n_properties = np.zeros((n_nodes, 1))
    el_properties = np.zeros((n_nodes - 1, 5))

    for i, x in enumerate(xs):
        ops.node(i + 1, x, 0.0, 0.0)
        n_properties[i] = x

    ops.geomTransf("Linear", 1, 0, 1, 0)

    # elements, with the cross-section shrinking linearly past the start of the taper
    L_taper = L - l_taper_start
    for i in range(1, n_nodes):
        if L_taper > 0 and xs[i-1] > l_taper_start:
            t = (xs[i] - l_taper_start) / L_taper
            b = b0 * (1 - db * t)
            h = h0 * (1 - dh * t)
        else:
            b = b0
            h = h0
        A = b * h
        mu = rho * A
        Iz = b * h ** 3 / 12.0
        Iy = h * b ** 3 / 12.0

        # torsional constant of a rectangular section
        side1, side2 = max(b, h), min(b, h)
        J = (side1 * side2**3) * (1/3 - 0.21 * (side2/side1) * (1 - (side2**4)/(12 * side1**4)))

        ops.element("ElasticTimoshenkoBeam", i, i, i+1, E, G, A, J, Iy, Iz, 5/6*A, 5/6*A, 1, "-mass", mu, "-cMass")
        el_properties[i-1] = [A, Iz, Iy, J, mu]

    # boundary conditions
    springs = np.zeros((n_nodes, 1))
    if L_embed is None:
        ops.fix(1, 1, 1, 1, 1, 1, 1)

    elif L_embed is not None:
        # embedded: every node within L_embed is tied to a fixed anchor by a zeroLength spring
        # element, which models a foundation rather than an ideal clamp
        embedded = [(i + 1, x) for i, x in enumerate(xs) if x <= L_embed]

        # material tags
        mat_tag_offset = 200   # material tags start at 200 + 3*i
        elem_tag_offset = 2000  # element tags start at 2000 + i

        for idx, (node_tag, x) in enumerate(embedded):
            anchor_tag = 1000 + idx
            ops.node(anchor_tag, x, 0.0, 0.0)
            ops.fix(anchor_tag, 1, 1, 1, 1, 1, 1)

            # the springs stiffen linearly with depth, so the support is soft at the surface
            # and approaches a clamp at the bottom of the embedded length
            depth = L_embed - x
            factor = depth / L_embed

            k_x = k_x_max * factor
            k_r = k_r_max * factor
            k_x = max(k_x, 1.0)
            k_r = max(k_r, 1.0)

            mt = mat_tag_offset + 4 * idx     # base material tag for this node
            ops.uniaxialMaterial('Elastic', mt + 0, k_x)
            ops.uniaxialMaterial('Elastic', mt + 3, k_r)             # rotational spring

            et = elem_tag_offset + idx
            ops.element('zeroLength', et, anchor_tag, node_tag, '-mat', mt+0, mt+0, mt+0, mt+3, mt+3, mt+3, '-dir', 1, 2, 3, 4, 5, 6)
            springs[node_tag-1] = k_x

    # the node feature vector the graphs are built from: position along the beam and support stiffness
    n_properties = np.hstack((n_properties.reshape(-1, 1), springs))

    # additional lumped mass at the free end of the cantilever beam
    m_tip = float(p.get("m_tip", 0.0))
    if m_tip > 0.0:
        ops.mass(n_nodes, m_tip, m_tip, m_tip, 0.0, 0.0, 0.0)

    # eigenproblem
    n_modes = 30
    eigs = ops.eigen(n_modes)
    omegas = np.sqrt(eigs)
    f0 = omegas / (2 * np.pi)

    beam_nodes = [n for n in ops.getNodeTags() if n < 1000]

    valid_shapes = []
    valid_f0 = []

    # keep the first 5 modes that actually deflect in the loaded direction, discarding the axial,
    # torsional and out-of-plane ones. 
    for mode in range(1, n_modes + 1):
        if load_dir == "z":
            nodes_eigen = np.array([ops.nodeEigenvector(node, mode, 3) for node in beam_nodes])
        elif load_dir == "y":
            nodes_eigen = np.array([ops.nodeEigenvector(node, mode, 2) for node in beam_nodes])
        if nodes_eigen[-1] < 0:
            nodes_eigen = -nodes_eigen
        if np.max(np.abs(nodes_eigen)) > 1e-6 and len(valid_shapes) < 5:
            valid_shapes.append(nodes_eigen)
            valid_f0.append(f0[mode - 1])

    mode_shapes = np.array(valid_shapes)
    f0 = np.array(valid_f0)

    # Rayleigh damping, calibrated so that the damping ratio is  zeta at the first two bending modes
    w1 = f0[0]*2*np.pi
    w2 = f0[1]*2*np.pi

    alpha_m = zeta * (2 * w1 * w2) / (w1 + w2)
    beta_k = zeta * 2 / (w1 + w2)

    ops.rayleigh(alpha_m, 0.0, beta_k, 0.0)

    # loads
    A_force = p['force_amplitude']
    load_type = cfg['load']['load_type']
    dt = float(cfg["simulation"]["dt"])
    duration = float(cfg["load"]["duration"])
    target_nodes = p["target_nodes"]

    signals = []

    if load_type == 'sine':
        # a sum of sines at the configured frequencies, each with a random phase
        f_hz = cfg["load"]["frequency"]
        if np.isscalar(f_hz):
            f_list = [float(f_hz)]
        else:
            f_list = [float(f) for f in f_hz]
        t = np.arange(0, duration, dt)
        for A in A_force:
                sig = np.zeros_like(t)
                for f in f_list:
                    phi = np.random.uniform(0, 2 * np.pi)
                    sig += np.sin(2 * np.pi * f * t + phi)
                sig *= A
                signals.append(sig)

    elif load_type == 'white_noise':
        # band-limited white noise: gaussian noise low-passed at the configured cutoff, then
        # rescaled to the sampled amplitude
        nSteps = int(duration / dt)
        f_cut = cfg['load']['frequency']
        f_cut = float(f_cut) if np.isscalar(f_cut) else float(f_cut[0])
        sos = butter(cfg['load']['order'], f_cut, 'low', fs=1/dt, output='sos')
        for A in A_force:
            wn = np.random.normal(0.0, 1.0, nSteps)
            filtered = sosfiltfilt(sos, wn)
            filtered *= A / np.max(np.abs(filtered))
            signals.append(filtered)

    elif load_type == 'colored_noise':
        # noise filtered by a SDOF response with the specified resonant frequency and damping
        f_hz = cfg["load"]["frequency"]
        zeta = cfg["load"]["zeta"]
        if np.isscalar(f_hz):
            f_list = [float(f_hz)]
            z_list = [float(zeta)]
        else:
            f_list = [float(f) for f in f_hz]
            z_list = [float(z) for z in zeta]
        t = np.arange(0, duration, dt)
        nSteps = len(t)
        for A in A_force:
                sig = np.zeros_like(t)
                for f, z in zip(f_list, z_list):
                    noise = np.random.normal(0.0, 1.0, nSteps)
                    wn = 2*np.pi*f
                    sys = signal.lti([1.0], [1.0, 2*z*wn, wn**2])
                    _, colored, _ = signal.lsim(sys, U=noise, T=t)
                    sig += colored
                sig *= A / np.max(np.abs(sig))
                signals.append(sig)

    else:
        raise ValueError(f"Unknown load_type: {load_type}")

    # apply load to the nodes
    for k, (node, sig) in enumerate(zip(target_nodes, signals), start=1):
        ops.timeSeries("Path", k, "-dt", dt, "-values", *sig.tolist())
        ops.pattern("Plain", k, k)
        if load_dir == "z":
            ops.load(node, 0.0, 0.0, 1.0, 0, 0, 0)
        elif load_dir == "y":
            ops.load(node, 0.0, 1.0, 0.0, 0, 0, 0)

    # extract forces
    q = cfg['simulation']['decimation']
    s_to_keep = cfg['simulation']['seconds_to_keep']
    fs_dec = 1/dt // q
    n_keep = int(s_to_keep * fs_dec)
    steps = int(cfg["simulation"]["tEnd"] / dt)          # same length as the response

    # pad the force history if it last less than the simulation time
    force_full = np.zeros((n_nodes, steps))
    for node, sig in zip(target_nodes, signals):
        n_valid = min(len(sig), steps)                    # truncate if duration > tEnd
        force_full[node - 1, :n_valid] = sig[:n_valid]

    # decimate and keep the same tail window as the response
    force_history = decimate(force_full, q, axis=1, ftype='iir', zero_phase=True)
    force_history = force_history[:, -n_keep:]

    # analysis: linear Newmark (unconditionally stable)
    ops.wipeAnalysis()
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.algorithm("Linear")
    ops.analysis("Transient")

    dt = float(cfg["simulation"]["dt"])
    steps = int(cfg["simulation"]["tEnd"] / dt)
    q = cfg['simulation']['decimation']
    s_to_keep = cfg['simulation']['seconds_to_keep']
    fs_dec = 1/dt // q

    node_tags = ops.getNodeTags()
    n_nodes = len([n for n in node_tags if n < 1000])

    disp_z_history = np.zeros((n_nodes, steps))
    disp_y_history = np.zeros((n_nodes, steps))
    vel_z_history = np.zeros((n_nodes, steps))
    vel_y_history = np.zeros((n_nodes, steps))
    acc_z_history = np.zeros((n_nodes, steps))
    acc_y_history = np.zeros((n_nodes, steps))

    # step through time, recording the full kinematic state of every beam node
    for k in range(steps):
        ok = ops.analyze(1, dt)
        if ok != 0:
            raise RuntimeError("Analysis failed")

        for j, n in enumerate([n for n in node_tags if n < 1000]):  # remove anchor nodes that have tags from 1000 onwards
            acc_z_history[j, k] = ops.nodeAccel(n, 3)  # direction Z
            acc_y_history[j, k] = ops.nodeAccel(n, 2)  # direction Y
            vel_z_history[j, k] = ops.nodeVel(n, 3)
            vel_y_history[j, k] = ops.nodeVel(n, 2)
            disp_z_history[j, k] = ops.nodeDisp(n, 3)
            disp_y_history[j, k] = ops.nodeDisp(n, 2)

    # the solver runs at a small dt for stability,then the response is decimated to reduce data volume
    acc_z_history = decimate(acc_z_history, q, axis=1, ftype='iir', zero_phase=True)
    acc_y_history = decimate(acc_y_history, q, axis=1, ftype='iir', zero_phase=True)
    vel_z_history = decimate(vel_z_history, q, axis=1, ftype='iir', zero_phase=True)
    vel_y_history = decimate(vel_y_history, q, axis=1, ftype='iir', zero_phase=True)
    disp_z_history = decimate(disp_z_history, q, axis=1, ftype='iir', zero_phase=True)
    disp_y_history = decimate(disp_y_history, q, axis=1, ftype='iir', zero_phase=True)

    # keep only the tail (steady-state response)
    acc_z_history = acc_z_history[:, -int(s_to_keep * fs_dec):]
    acc_y_history = acc_y_history[:, -int(s_to_keep * fs_dec):]
    vel_z_history = vel_z_history[:, -int(s_to_keep * fs_dec):]
    vel_y_history = vel_y_history[:, -int(s_to_keep * fs_dec):]
    disp_z_history = disp_z_history[:, -int(s_to_keep * fs_dec):]
    disp_y_history = disp_y_history[:, -int(s_to_keep * fs_dec):]

    # extracting the MCK matrices
    ops.wipeAnalysis()
    ops.system('FullGeneral')
    ops.numberer('Plain')
    ops.analysis('Transient')
    ops.integrator('GimmeMCK', 1.0, 0.0, 0.0)
    ops.algorithm("Linear")
    ops.analyze(1, 0.0)
    N = ops.systemSize()
    M = np.array(ops.printA('-ret')).reshape((N, N))

    ops.integrator('GimmeMCK', 0.0, 0.0, 1.0)
    ops.algorithm("Linear")
    ops.analyze(1, 0.0)
    K = np.array(ops.printA('-ret')).reshape((N, N))

    ops.integrator('GimmeMCK', 0.0, 1.0, 0.0)
    ops.algorithm("Linear")
    ops.analyze(1, 0.0)
    C = np.array(ops.printA('-ret')).reshape((N, N))

    # condense the 6-DOF-per-node system onto the translation of the loaded direction
    if load_dir == "z":
        M, C, K = guyan(M, C, K, 2)
    elif load_dir == "y":
        M, C, K = guyan(M, C, K, 1)

    return SimResult(acc_z=acc_z_history, acc_y=acc_y_history, vel_z=vel_z_history, vel_y=vel_y_history, disp_z=disp_z_history, disp_y=disp_y_history, el_properties=el_properties, n_properties=n_properties, f0=f0, mode_shapes=mode_shapes, M=M, C=C, K=K, F=force_history)


def run_simulation_safe(i, p, cfg):
    try:
        result = run_simulation(p, cfg)
        return i, result
    except Exception as e:
        print(f"  [WARNING] Sample {i} failed: {e}")
        return i, None


def main():
    """ Sample the population, simulate every system in parallel and save the dataset. """
    PROJECT_ROOT = Path.cwd().resolve()
    while not (PROJECT_ROOT / "pyproject.toml").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
        PROJECT_ROOT = PROJECT_ROOT.parent

    results_dir = PROJECT_ROOT / "simulationResults"
    results_dir.mkdir(parents=True, exist_ok=True)

    args = get_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    n_samples = cfg["sampling"]["samples"]

    parameter_list = generate_sobol_population(cfg["population_param"], cfg["load"], n_samples)

    print(f"Generating {n_samples} systems")

    # simulate the population in parallel, then restore the sampling order
    raw_results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(run_simulation_safe)(i, parameter_list[i], cfg)
        for i in range(n_samples))

    raw_results.sort(key=lambda x: x[0])

    data_dict = {
        "acc_z": [],
        "acc_y": [],
        "vel_z": [],
        "vel_y": [],
        "disp_z": [],
        "disp_y": [],
        "n_properties": [],
        "el_properties": [],
        "M": [],
        "C": [],
        "K": [],
        "F": [],
    }

    f0_list = [None] * n_samples
    mode_shapes_list = [None] * n_samples

    for i, result in raw_results:

        f0_list[i] = result.f0
        mode_shapes_list[i] = result.mode_shapes

        for field in ["acc_z", "acc_y", "vel_z", "vel_y", "disp_z", "disp_y", "el_properties", "n_properties", "M", "C", "K", "F"]:
            data_dict[field].append(getattr(result, field))

    # the metadata table: the sampled parameters of every system, alongside its modal properties
    taper_cfg = cfg.get("taper", {})
    meta_rows = []
    for i, p in enumerate(parameter_list):
        row = {
            "system_id":     i,
            "E":             p.get("E"),
            "b0":            p.get("b0") * p.get("L"),   # stored in absolute units, not as a fraction of L
            "h0":            p.get("h0") * p.get("L"),
            "L":             p.get("L"),
            "rho":           p.get("rho"),
            "zeta":          p.get("zeta"),
            "db":            taper_cfg.get("db", 0),
            "dh":            taper_cfg.get("dh", 0),
            "l_taper_start": taper_cfg.get("l_start", 0) if taper_cfg.get("isTapered") else 0,
            "L_embedded":    p.get("L_embedded"),
            "k_x":           p.get("k_x"),
            "k_r":           p.get("k_r"),
            "m_tip":         p.get("m_tip", 0.0),
            "F":             p.get("force_amplitude"),
            "f0":            f0_list[i],
            "n_loads":       p.get("n_forces"),
            "target_nodes":  p.get("target_nodes"),
            "mode_shapes":   mode_shapes_list[i],
        }
        meta_rows.append(row)

    meta = pd.DataFrame(meta_rows).set_index("system_id")
    data = pd.DataFrame(data_dict)
    data.index.name = "system_id"

    out = cfg["output"]["dataset_name"]
    meta_path = results_dir / f"{out}_meta.pkl"
    data_path = results_dir / f"{out}_data.pkl"

    meta.to_pickle(meta_path)
    data.to_pickle(data_path)

    print(f"Saved meta → {meta_path}  ({len(meta)} rows)")
    print(f"Saved data → {data_path}  ({len(data)} rows)")


if __name__ == "__main__":
    main()
