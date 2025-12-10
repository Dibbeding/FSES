# Community-based Sampling in Hypergraphs: Frontier Scoring Expansion Sampling

This repository accompanies the thesis _“Community-based Sampling in Hypergraphs: Frontier Scoring Expansion Sampling (FSES)”_.

It contains:

- the implementation of FSES,
- baseline samplers for comparison,
- scripts to run experiments, and
- utilities to generate community-retention and parameter-sweep plots.

The main focus is **community retention**: how many distinct communities from the original hypergraph remain represented in a sampled sub-hypergraph.

## Repository layout

- `run_fses.py`  
  Command-line interface (CLI) to run FSES, the random frontier baseline, JURW, or NB-HO-RW on a hypergraph and save the sampled hyperedges.

- `plot_retention.py`  
  CLI to compute and plot community retention from previously saved samples.

- `plot_community_sizes.py`  
  CLI to plot the community size distribution from a `<name>_assign.txt` file.

- `run_param_analysis.py`  
  CLI to perform one-at-a-time parameter sweeps for FSES (over `k`, unknown score, `alpha`, or `F_lim`) and plot retention curves.

- `fses/`  
  Python package with:
  - FSES and the random frontier baseline,
  - an in-repo implementation of JURW, and
  - an in-repo implementation of NB-HO-RW,
  as well as I/O helpers.

- `data/input/`  
  Place input hypergraphs and community files here (files named `<dataset>_he.txt` and `<dataset>_assign.txt`).

- `data/sampled/`  
  Directory where sampled hypergraphs are written, in subfolders per algorithm.

- `data/plots/`  
  Directory for generated plots (community retention, community size distributions, parameter sweeps).

## Baseline implementations and references

The thesis compares FSES against several baseline sampling methods. Some are used via the public MiDaS repository, as they were already implemented there. JURW and NB-HO-RW are included here because they were not available as public implementations at the time of writing.

- **Node Sampling (NS), Forest Fire (FF), Random Walk (RW)**  
  Implemented in the MiDaS repository:  
  <https://github.com/young917/MiDaS>

- **MiDaS hyperedge sampler**  
  The main MiDaS algorithm, also from:  
  <https://github.com/young917/MiDaS>

- **Joint Unbiased Random Walk (JURW)**  
  In-repo implementation based on the paper _“Sampling hypergraphs via joint unbiased random walk”_.  
  Code: `fses/algorithms/jurw.py`.

- **Non-Backtracking Higher-Order Random Walk (NB-HO-RW)**  
  In-repo implementation based on _“Sampling nodes and hyperedges via random walks on large hypergraphs”_.  
  Code: `fses/algorithms/nbho_rw.py`.

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Preparing data
Place two files in `data/input/`:
- `<name>_he.txt` — one hyperedge per line, nodes separated by comma or whitespace.
- `<name>_assign.txt` — one community id per line (1-indexed node ids).

You can use any hypergraph as long as it matches this format. If you have other generators or real data, convert them into these two files and keep the shared `<name>` prefix; FSES will pick them up via `--dataset <name>`. The `_assign.txt` file is required for community retention evaluation (plots); without it you can still sample, but you cannot compute retention.

### Generating synthetic hypergraphs (h-ABCD)
Generator: https://github.com/bkamins/ABCDHypergraphGenerator.jl

Template command (fill in your own paths and parameter values):
```bash
julia --project abcdh.jl \
  -n <NODES> -d <deg_min>,<deg_mean>,<deg_max> -c <cs_alpha>,<cs_min>,<cs_max> \
  -x <mixing> -q <q_vector> -w :linear -s <seed> \
  -o <output_prefix>
```
Each run produces `<output_prefix>_he.txt` and `<output_prefix>_assign.txt`, which you can place in `data/input/`.

## Running FSES
Example: sample 5% and 10% of the nodes from `balanced`, two runs each (pass a list of fractions):
```bash
python run_fses.py --dataset balanced --sample-frac 0.05 0.10 --runs 2
```
Key options:
- Input: `--dataset NAME` expects `data/input/NAME_he.txt` and `NAME_assign.txt`; or use `--hyperedges <path>` plus optional `--communities <path>`.
- Sampling sizes: use fractions only via `--sample-frac` (default `[0.25]`; e.g., `0.05` for 5%).
- Algorithm: `--algo fses` (default), `--algo fses-random` (random frontier baseline), `--algo jurw`, or `--algo nbho-rw`.
- Parameters (defaults: unknown_score=1.0, k=0.10, alpha=0.0, candidate_limit=100, refresh on): `--unknown-score`, `--k`, `--alpha`, `--candidate-limit`, `--no-refresh`.
- Output: `data/sampled/<algo>/<name>_n<N>_<pct>pct_<params>_<timestamp>.txt` (one hyperedge per line, space-separated). Timestamps in filenames make it easy to pick the latest samples per variant.

## Plotting community retention
Compute retention for saved samples at thresholds like `1n` (at least 1 node from the community) or `5p` (5% of the community), and generate a plot:
```bash
python plot_retention.py --dataset balanced --thresholds 1n 5p 10p --algos FSES FSES_random
```
Options:
- `--assign` to point to a specific `_assign.txt` (defaults to `data/input/<dataset>_assign.txt`).
- `--samples-dir` (default `data/sampled`), `--algos` selects the subfolders to include (e.g., `FSES`, `FSES_random`).
- `--plots-dir` (default `data/plots`).
- `--max-samples N` to only use the newest N samples per percentage (per algo); defaults to all.
Output: `data/plots/retention_<dataset>_<algo-tag>.png`.

## Plotting community size distribution
Use `_assign.txt` directly to inspect community sizes:
```bash
python plot_community_sizes.py --dataset balanced
```
Options:
- `--assign` to pass a direct path to `<name>_assign.txt` (overrides `--dataset`).
- `--plots-dir` (default `data/plots`), `--no-show` to skip opening a window.
Output: `data/plots/community_sizes_<dataset>.png`.

## Parameter analysis (one-at-a-time sweeps)
Sweep exactly one parameter list at a time; keep others as single values. Example (sweep k, fix others):
```bash
python run_param_analysis.py --datasets balanced --k-values 0.05 0.10 0.20 --unknown-values 1 --alpha-values 0 --flim-values 100 --sample-pct 5 10 15 20 25 --runs 3
```
Rules:
- Only one of `--k-values`, `--unknown-values`, `--alpha-values`, `--flim-values` may have multiple entries; the rest must be single values.
- Sample sizes: pass a list via `--sample-pct ...` or use presets with `--sample-preset fine` (5..50 step 5, default) or `coarse` (10..50 step 10).
- Thresholds: `--thresholds` (default `1n`; first threshold used for plotting).
- Outputs: CSV `data/plots/param_sweep_<dataset>.csv` and plot `data/plots/param_sweep_<dataset>_<threshold>.png`. Defaults: k=0.10, unknown=1.0, alpha=0.0, flim=100, refresh on.
Quick preset example (sweep unknown score, coarse sampling):
```bash
python run_param_analysis.py --datasets highmix --k-values 0.10 --unknown-values 1 5 --alpha-values 0 --flim-values 100 --sample-preset coarse --runs 1
```

## File formats (expected)
- Hyperedges: one hyperedge per line, nodes separated by comma or space, e.g., `1 4 9 12`.
- Communities: `_assign.txt` with one community id per node (line number = node id, starting at 1).

