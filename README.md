# GNN-based Virtual Sensing on Beams

This repository is dedicated to a graph-based virtual sensing (VS) approach on beam structures, wherein the strucutral response is predicted at unmeasured nodes from a handful of sensed locations. We simulate structures as cantilever beams with randomized geometry, material and loading conditions using [OpenSeesPy](https://openseespydoc.readthedocs.io/), and a GNN + LSTM model is trained to reconstruct the full-field response from a sparse sensor layout. The framework also benchmarks the learned models against Modal Decomposition and Expansion (MDE). This code is associated with the following paper:
- "Towards Generalisable Virtual Sensing: A Physics-Informed Graph Neural Network Approach On Cantilever Beams", submitted to the special collection 'Focus on Machine Learning for Structural and Infrastructure Systems' of Machine Learning: Engineering.

## Methodology overview

[vsFramework.pdf](https://github.com/user-attachments/files/30092508/vsFramework.pdf)


## Project structure
```
├── simulate.py               # OpenSeesPy FEM simulation of the beam population
├── simulationConfigs/        # config templates for simulate.py
│
├── framework/
│   ├── make_graphs.py        # builds PyG graphs from simulation output
│   ├── train.py              # trains the GNN_LSTM model (data-driven or physics-informed)
│   ├── test.py               # evaluates a trained model on a test set
│   ├── test_MDE.py           # Modal Decomposition and Expansion baseline
│   ├── optimize.py           # hyperparameter search
│   ├── utils.py              # scaling, noise, error metrics, physics-loss helpers
│   └── model.py              # GNN_LSTM architecture + train/validate/test loops
│
├── VSconfigs/                # config templates for make_graphs.py / train.py / test.py / test_MDE.py
└── notebooks/                # result-exploration notebooks
```

The scripts create four more directories as they run: `simulationResults/` (simulation pickles), `graphs/`, the directory named by `run.out_dir` (model, scalers, loss curve, metrics) and `hpo_results/` (hyperparameter search output).

## Usage

Scripts can be run with `uv run` from anywhere inside the repo.

### Simulating a beam population

To generate a population of beams with the configured  geometry and material properties, first set up a config yaml file in `simulationConfigs/` (see `example_simConfig.yaml` as a template). Then launch the simulation script:

`uv run simulate.py --config simulationConfigs/example_simConfig.yaml --n_jobs -1`

This runs the FEM simulation for every sampled system in parallel and saves the time histories and mode shapes to `simulationResults/`.

### Building graph datasets

Once a simulation result exists, turn it into a graph dataset with a `VSconfigs/` config (see  `example_VSConfig.yaml` as a template).

`uv run framework/make_graphs.py --config VSconfigs/example_VSConfig.yaml`

### Training a model

To train the GNN_LSTM model, use the same kind of `VSconfigs/` config and choose a training mode with `--type`:

`uv run framework/train.py --config VSconfigs/example_VSConfig.yaml --type data`  (data-driven)

`uv run framework/train.py --config VSconfigs/example_VSConfig.yaml --type phys`  (physics-informed)

`--seed` (default `42`) sets the seed so a run is reproducible. See [Seed sweeps](#seed-sweeps) below.

### Testing a model

To evaluate a trained model on a dataset variant:

`uv run framework/test.py --config VSconfigs/example_VSConfig.yaml`

### Running the MDE baseline

MDE requires a config and a metadata pickle holding the mode shapes to reconstruct with. They are automatically taken from the files of the specified simulation result:

`uv run framework/test_MDE.py --config VSconfigs/example_VSConfig.yaml --meta simulationResults/example_meta.pkl`

Every test script writes two parquet files per run: a per-node metrics file and a per-system metrics file (averaged over nodes). The filenames are suffixed with the configured `channel` (`acc`, `vel`, or `disp`): `test.py` writes `df_node_<channel>.parquet` / `df_graph_<channel>.parquet`, while the MDE baseline writes `df_node_<channel>_MDE.parquet` / `df_graph_<channel>_MDE.parquet`.  For example, with `channel: acc` these become `df_node_acc.parquet` / `df_graph_acc.parquet` and `df_node_acc_MDE.parquet` / `df_graph_acc_MDE.parquet`. See `framework/utils.py:node_metrics` for the exact metric definitions (MAE, APE, TRAC, RMSE, NRMSE, R²).

### Seed sweeps

`train.py`, `test.py` and `test_MDE.py` all accept `--seed` (default `42`). Any seed other than the default reads and writes `out_dir` suffixed with `_seed<N>`, so the runs of a sweep do not overwrite each other. The same `--seed` must be passed to every script in the sweep, otherwise the test scripts look in the unsuffixed `out_dir` and will not find the model, scalers and test set that training wrote:

```
uv run framework/train.py    --config VSconfigs/example_VSConfig.yaml --type data --seed 7
uv run framework/test.py     --config VSconfigs/example_VSConfig.yaml --seed 7
uv run framework/test_MDE.py --config VSconfigs/example_VSConfig.yaml --meta simulationResults/example_meta.pkl --seed 7
```

`--seed` only varies the model initialisation and the batch shuffling. The train/test split, the train/val split and the sensor noise are all drawn with a fixed seed and do not follow `--seed`, so a sweep can be used to  measure the variance of the model on a fixed dataset rather than the variance of the data. The MDE baseline holds no trained weights, so its metrics are identical in every seeded directory; it takes `--seed` only so it can be pointed at the matching `out_dir`.

### Hyperparameter search

`optimize.py` runs a two-stage search: an Optuna search to find the best hyperparameters, followed by a grid search over `lambda_phys` for the physics-informed loss. Results are written to `hpo_results/` as `best_params.json` and `best_params_PI.json`, in the same layout as the `model:` and `train:` blocks of a `VSconfigs/` config, so they can be pasted straight in.

The graph file, channel, timestep, trial and epoch counts and the `lambda_phys` grid are all constants at the top of `framework/optimize.py` and must be edited there before running.

`uv run framework/optimize.py`

## Notebooks

- `notebooks/explore_simRes.ipynb` - inspects raw simulation output   (parameter distributions, mode shapes, PSDs).
- `notebooks/explore_graphs.ipynb` -  inspects the built graph dataset.
- `notebooks/explore_results.ipynb` - inspects the model predictions and the error metrics.

## License

MIT — see [LICENSE](LICENSE).
