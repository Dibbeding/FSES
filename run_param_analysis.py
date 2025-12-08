"""Run FSES parameter sweeps and plot community retention curves.

Allows sweeping k, unknown node score, alpha, and candidate limit over one or
more datasets. Outputs CSV summaries and plots with mean/std shading.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fses.algorithms import FSESParams, expansion_sampling_F_score_with_params
from fses.graph_io import EdgeList, NodeCommunities, NodeList, load_hypergraph


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def parse_threshold(token: str) -> Tuple[str, float]:
    token = token.strip()
    if token.endswith("n"):
        return "abs", float(token[:-1])
    if token.endswith("p"):
        return "pct", float(token[:-1])
    raise ValueError(f"Invalid threshold token: {token} (use 5n or 10p)")


def build_comm_lookup(assignments: NodeCommunities) -> Dict[int, set]:
    lookup: Dict[int, set] = {}
    for node, comm in assignments.items():
        lookup.setdefault(comm, set()).add(node)
    return lookup


def compute_retention(comm_lookup: Dict[int, set], edges: Iterable[Iterable[int]], threshold: Tuple[str, float]) -> float:
    if not comm_lookup:
        return 0.0
    sampled_nodes = {node for edge in edges for node in edge}
    retained = 0
    mode, value = threshold
    for members in comm_lookup.values():
        present = sampled_nodes & members
        if mode == "pct":
            coverage = (len(present) / max(1, len(members))) * 100
            if coverage >= value:
                retained += 1
        else:
            if len(present) >= value or len(present) == len(members):
                retained += 1
    return retained / len(comm_lookup) * 100


def _params_label(p: FSESParams) -> str:
    parts = []
    parts.append(f"k{p.k:.2f}")
    parts.append(f"u{int(p.unknown_node_score)}")
    parts.append(f"a{p.size_penalty_alpha:.2f}")
    parts.append(f"fl{p.candidate_limit}")
    if p.zero_known:
        parts.append("zeroknown")
    if not p.use_refresh:
        parts.append("norefresh")
    return "_".join(parts)


@dataclass
class VariantResult:
    label: str
    sample_pcts: List[int]
    means: List[float]
    stds: List[float]


# --------------------------------------------------------------------------- #
# Core sweep                                                                  #
# --------------------------------------------------------------------------- #


def run_single(
    edge_list: EdgeList,
    node_list: NodeList,
    comm_lookup: Dict[int, set],
    total_nodes: int,
    sample_pcts: List[int],
    runs: int,
    threshold: Tuple[str, float],
    params: FSESParams,
    base_seed: int,
) -> VariantResult:
    means: List[float] = []
    stds: List[float] = []

    for idx, pct in enumerate(sample_pcts):
        target_nodes = max(1, int(total_nodes * (pct / 100.0)))
        scores: List[float] = []
        for run_idx in range(runs):
            seed = base_seed + idx * 1000 + run_idx
            random.seed(seed)
            result = expansion_sampling_F_score_with_params(edge_list, node_list, target_nodes, params)
            sampled_edges = result["sampled_edges"]
            sampled = [list(edge_list[eid]) for eid in sampled_edges]
            scores.append(compute_retention(comm_lookup, sampled, threshold))
        means.append(float(np.mean(scores)))
        stds.append(float(np.std(scores)))

    return VariantResult(label=_params_label(params), sample_pcts=sample_pcts, means=means, stds=stds)


def plot_results(
    dataset: str,
    threshold_label: str,
    variants: List[VariantResult],
    save_path: Path,
) -> None:
    plt.figure(figsize=(6, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "D", "^", "v", "X", "P", "*"]

    for idx, vr in enumerate(variants):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.plot(vr.sample_pcts, vr.means, marker=marker, color=color, label=vr.label)
        if vr.stds:
            m = np.array(vr.means)
            s = np.array(vr.stds)
            plt.fill_between(vr.sample_pcts, m - s, m + s, color=color, alpha=0.15)

    plt.xlabel("Sampled nodes (%)")
    plt.ylabel(f"Community retention ({threshold_label})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(ticks=[i for i in range(5, 55, 5)])
    plt.ylim(0, 100)
    plt.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[saved] {save_path}")
    plt.close()


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter sweeps for FSES and plot community retention.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset base names (expects data/input/<name>_he.txt and _assign.txt)")
    parser.add_argument("--data-root", type=str, default="data/input", help="Input folder (default: data/input)")
    parser.add_argument("--sample-pct", nargs="+", type=int, default=None, help="Custom sample percentages (e.g., 5 10 15)")
    parser.add_argument("--sample-preset", choices=["fine", "coarse"], default="fine", help="Preset sample schedule: fine=5..50 step 5, coarse=10..50 step 10")
    parser.add_argument("--runs", type=int, default=3, help="Runs per parameter setting")
    parser.add_argument("--thresholds", nargs="+", default=["1n"], help="Thresholds like 1n or 5p")
    parser.add_argument("--k-values", nargs="+", type=float, default=[0.10], help="k decay values to sweep")
    parser.add_argument("--unknown-values", nargs="+", type=float, default=[1.0], help="unknown node scores to sweep")
    parser.add_argument("--alpha-values", nargs="+", type=float, default=[0.0], help="size penalty alpha values to sweep")
    parser.add_argument("--flim-values", nargs="+", type=int, default=[100], help="candidate_limit (F_lim) values to sweep")
    parser.add_argument("--zero-known", action="store_true", help="Include zero_known=True variant")
    parser.add_argument("--no-refresh", action="store_true", help="Disable refresh budgeting")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (auto if omitted)")
    parser.add_argument("--plots-dir", type=str, default="data/plots", help="Where to store plots")
    parser.add_argument("--csv-dir", type=str, default="data/plots", help="Where to store CSV summaries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_seed = args.seed if args.seed is not None else int(time.time()) % 10000000
    thresholds = [parse_threshold(t) for t in args.thresholds]

    # Resolve sample percentages from preset or custom list
    if args.sample_pct is not None:
        sample_pcts = args.sample_pct
    else:
        sample_pcts = list(range(5, 55, 5)) if args.sample_preset == "fine" else list(range(10, 55, 10))

    # One-at-a-time sweep: only one parameter list may have multiple values
    sweep_fields: List[Tuple[str, List[float]]] = []
    if len(args.k_values) > 1:
        sweep_fields.append(("k", args.k_values))
    if len(args.unknown_values) > 1:
        sweep_fields.append(("unknown", args.unknown_values))
    if len(args.alpha_values) > 1:
        sweep_fields.append(("alpha", args.alpha_values))
    if len(args.flim_values) > 1:
        sweep_fields.append(("flim", args.flim_values))
    if len(sweep_fields) > 1:
        raise SystemExit("Only one parameter can be swept at a time; pass a list for a single parameter and single values for the rest.")

    base_k = args.k_values[0]
    base_unknown = args.unknown_values[0]
    base_alpha = args.alpha_values[0]
    base_flim = args.flim_values[0]

    for dataset in args.datasets:
        hyperedges = Path(args.data_root) / f"{dataset}_he.txt"
        assigns = Path(args.data_root) / f"{dataset}_assign.txt"
        if not hyperedges.exists() or not assigns.exists():
            print(f"[skip] missing files for {dataset}")
            continue

        edge_list, node_list, assignments = load_hypergraph(hyperedges, assigns)
        comm_lookup = build_comm_lookup(assignments)
        total_nodes = max(len(assignments), len(node_list))

        variants: List[VariantResult] = []
        # Build variants based on sweep (or single baseline)
        sweep_name = sweep_fields[0][0] if sweep_fields else None
        sweep_values = sweep_fields[0][1] if sweep_fields else [None]

        for val in sweep_values:
            k = base_k if sweep_name != "k" else val
            unk = base_unknown if sweep_name != "unknown" else val
            alpha = base_alpha if sweep_name != "alpha" else val
            flim = base_flim if sweep_name != "flim" else int(val) if val is not None else base_flim

            params = FSESParams(
                unknown_node_score=unk,
                k=k,
                size_penalty_alpha=alpha,
                candidate_limit=flim,
                zero_known=args.zero_known,
                use_refresh=not args.no_refresh,
            )
            vr = run_single(
                edge_list=edge_list,
                node_list=node_list,
                comm_lookup=comm_lookup,
                total_nodes=total_nodes,
                sample_pcts=sample_pcts,
                runs=args.runs,
                threshold=thresholds[0],
                params=params,
                base_seed=base_seed,
            )
            variants.append(vr)

        # Write CSV summary
        csv_dir = Path(args.csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"param_sweep_{dataset}.csv"
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("dataset,variant,sample_pct,mean_retention,std_retention,runs\n")
            for vr in variants:
                for pct, mean, std in zip(vr.sample_pcts, vr.means, vr.stds):
                    fh.write(f"{dataset},{vr.label},{pct},{mean:.4f},{std:.4f},{args.runs}\n")
        print(f"[saved] {csv_path}")

        # Plot per threshold (using first/only threshold label)
        thresh_label = args.thresholds[0]
        plot_path = Path(args.plots_dir) / f"param_sweep_{dataset}_{thresh_label}.png"
        plot_results(dataset, thresh_label, variants, plot_path)


if __name__ == "__main__":  # pragma: no cover
    main()
