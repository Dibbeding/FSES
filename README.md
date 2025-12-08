# Community-based Sampling in Hypergraphs: Frontier Scoring Expansion Sampling

This repo accompanies the thesis "_Community-based Sampling in Hypergraphs: Frontier Scoring Expansion Sampling (FSES)_". It lets you:
- run FSES on your own hypergraph files,
- compare to a simple random frontier baseline,
- save sampled hypergraphs, and
- generate community-retention plots and parameter-sweep plots to reproduce the experiments.

## Repository layout
- `run_fses.py` — CLI to run FSES (or the random baseline) on a hypergraph and save sampled hyperedges.
- `plot_retention.py` — CLI to plot community retention from saved samples.
- `plot_community_sizes.py` — CLI to plot the community size distribution from an `_assign.txt`.
- `run_param_analysis.py` — CLI to sweep FSES parameters (k, unknown score, alpha, F_lim) and plot retention curves.
- `fses/` — Python package with the sampling algorithms and I/O helpers.
- `data/input/` — put `<dataset>_he.txt` and `<dataset>_assign.txt` here.
- `data/sampled/` — samples are written here in subfolders per algorithm.
- `data/plots/` — generated retention plots are stored here.

## Installation
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
Example: sample 5% and 10% of the nodes from `balanced`, two runs each:
```bash
python run_fses.py --dataset balanced --sample-frac 0.05 0.10 --runs 2
```
Key options:
- Input: `--dataset NAME` expects `data/input/NAME_he.txt` and `NAME_assign.txt`; or use `--hyperedges <path>` plus optional `--communities <path>`.
- Sampling sizes: use fractions only via `--sample-frac` (default `[0.25]`; e.g., `0.05` for 5%).
- Algorithm: `--algo fses` (default) or `--algo fses-random` (random frontier baseline).
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

