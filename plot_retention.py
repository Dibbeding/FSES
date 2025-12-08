"""Compute and plot community retention for sampled hypergraphs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fses.graph_io import load_assignments

FILENAME_PATTERN = re.compile(
    r"(?P<dataset>.+?)_n(?P<nodes>\d+?)_(?P<sample>\d+)pct_(?P<params>.+)_(?P<timestamp>\d{8}_\d{6})\.txt"
)


def parse_threshold(token: str) -> Tuple[str, float]:
    """Parse threshold tokens like '1n' (absolute) or '5p' (percent)."""
    token = token.strip()
    if token.endswith("n"):
        return "abs", float(token[:-1])
    if token.endswith("p"):
        return "pct", float(token[:-1])
    raise ValueError(f"Invalid threshold token: {token} (use 5n or 10p)")


def load_sample_edges(path: Path) -> List[List[int]]:
    """Load sampled hyperedges from disk."""
    edges: List[List[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",") if "," in line else line.split()
            cleaned = [int(p) for p in parts if p.strip()]
            if cleaned:
                edges.append(cleaned)
    return edges


def build_comm_lookup(assignments: Dict[int, int]) -> Dict[int, set]:
    """Map community id -> set of member nodes."""
    comms: Dict[int, set] = defaultdict(set)
    for node, comm in assignments.items():
        comms[comm].add(node)
    return comms


def compute_retention(
    comm_lookup: Dict[int, set],
    edges: Iterable[Iterable[int]],
    threshold: Tuple[str, float],
) -> float:
    """Compute community retention percentage for a given threshold."""
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


def collect_samples(samples_root: Path, algo_dir: str, dataset: str, max_samples: int = 0) -> Dict[int, List[Path]]:
    """Group sample files by percentage for one algorithm.

    If max_samples > 0, keep only the newest N (by filename) per percentage.
    """
    folder = samples_root / algo_dir
    if not folder.exists():
        raise FileNotFoundError(f"Samples folder not found: {folder}")
    grouped: Dict[int, List[Path]] = defaultdict(list)
    for path in folder.glob("*.txt"):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        info = match.groupdict()
        if info["dataset"] != dataset:
            continue
        pct = int(info["sample"])
        grouped[pct].append(path)
    if max_samples > 0:
        for pct, paths in list(grouped.items()):
            grouped[pct] = sorted(paths, key=lambda p: p.name, reverse=True)[:max_samples]
    return grouped


def _compute_series_stats(
    samples: Dict[int, List[Path]],
    comm_lookup: Dict[int, set],
    threshold: Tuple[str, float],
) -> Tuple[List[int], List[float], List[float]]:
    """Return xs, mean retention, std retention per sample percentage."""
    xs: List[int] = []
    means: List[float] = []
    stds: List[float] = []
    for pct in sorted(samples.keys()):
        scores: List[float] = []
        for sample_path in samples[pct]:
            edges = load_sample_edges(sample_path)
            scores.append(compute_retention(comm_lookup, edges, threshold))
        if not scores:
            continue
        xs.append(pct)
        means.append(float(np.mean(scores)))
        stds.append(float(np.std(scores)))
    return xs, means, stds


def generate_plot(
    all_samples: Dict[str, Dict[int, List[Path]]],
    comm_lookup: Dict[int, set],
    thresholds: List[Tuple[str, float]],
    title: str,
    save_path: Path,
    show_errorband: bool = True,
) -> None:
    """Plot retention curves for one dataset across algorithms."""
    plt.figure(figsize=(6, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "D", "^", "v", "X", "P", "*"]

    for algo_idx, (algo_name, samples) in enumerate(all_samples.items()):
        color = colors[algo_idx % len(colors)]
        marker = markers[algo_idx % len(markers)]
        for thresh_idx, threshold in enumerate(thresholds):
            xs, means, stds = _compute_series_stats(samples, comm_lookup, threshold)
            label = f"{algo_name} ({int(threshold[1])}{'%' if threshold[0] == 'pct' else 'n'})" if len(thresholds) > 1 else algo_name
            plt.plot(xs, means, marker=marker, color=color, label=label)
            if show_errorband and stds:
                means_arr = np.array(means)
                std_arr = np.array(stds)
                plt.fill_between(xs, means_arr - std_arr, means_arr + std_arr, color=color, alpha=0.15)

    plt.xlabel("Sampled nodes (%)")
    plt.ylabel("Avg. community retention (min. 1 nodes present)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(ticks=[i for i in range(5, 55, 5)])
    plt.ylim(0, 100)
    plt.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[saved] {save_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot community retention for sampled hypergraphs.")
    parser.add_argument("--dataset", required=True, help="Dataset base name used in the sampled filenames")
    parser.add_argument("--assign", type=str, default="", help="Path to *_assign.txt (defaults to data/input/<dataset>_assign.txt)")
    parser.add_argument("--samples-dir", type=str, default="data/sampled", help="Root folder that contains algo subfolders")
    parser.add_argument("--algos", nargs="+", default=["FSES"], help="Algorithm subfolder names inside samples-dir (e.g., FSES FSES_random)")
    parser.add_argument("--thresholds", nargs="*", default=["1n"], help="Thresholds like 1n (absolute) or 10p (percent)")
    parser.add_argument("--plots-dir", type=str, default="data/plots", help="Where to store generated plots")
    parser.add_argument("--no-errorband", action="store_true", help="Disable stddev shaded band")
    parser.add_argument("--max-samples", type=int, default=0, help="Use only the newest N samples per percentage (per algo); 0 means use all")
    return parser.parse_args()


def main() -> None:
    """Entry point: load samples, compute retention, write plot."""
    args = parse_args()
    assign_path = Path(args.assign) if args.assign else Path("data/input") / f"{args.dataset}_assign.txt"
    if not assign_path.exists():
        raise FileNotFoundError(f"Assignment file not found: {assign_path}")

    thresholds = [parse_threshold(tok) for tok in args.thresholds]
    assignments = load_assignments(assign_path)
    comm_lookup = build_comm_lookup(assignments)
    all_samples: Dict[str, Dict[int, List[Path]]] = {}
    for algo in args.algos:
        grouped = collect_samples(Path(args.samples_dir), algo, args.dataset, max_samples=args.max_samples)
        if grouped:
            all_samples[algo] = grouped
    if not all_samples:
        raise RuntimeError(f"No samples found for dataset '{args.dataset}' in {args.samples_dir} for algos {args.algos}")

    algo_tag = "_".join(args.algos)
    save_path = Path(args.plots_dir) / f"retention_{args.dataset}_{algo_tag}.png"
    generate_plot(
        all_samples=all_samples,
        comm_lookup=comm_lookup,
        thresholds=thresholds,
        title=args.dataset,
        save_path=save_path,
        show_errorband=not args.no_errorband,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
