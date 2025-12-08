"""Plot community size distribution for a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from fses.graph_io import load_assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot community size distribution.")
    parser.add_argument("--dataset", type=str, help="Dataset base name (expects data/input/<name>_assign.txt)")
    parser.add_argument("--assign", type=str, help="Path to *_assign.txt (overrides --dataset)")
    parser.add_argument("--plots-dir", type=str, default="data/plots", help="Where to store the plot")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window; just save")
    return parser.parse_args()


def load_sizes(assign_path: Path) -> list[int]:
    assignments: Dict[int, int] = load_assignments(assign_path)
    counts: Dict[int, int] = {}
    for comm in assignments.values():
        counts[comm] = counts.get(comm, 0) + 1
    return sorted(counts.values(), reverse=True)


def plot_sizes(sizes: list[int], title: str, save_path: Path, no_show: bool) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(range(1, len(sizes) + 1), sizes)
    plt.xlabel("Community rank by size")
    plt.ylabel("Community size (|C|)")
    plt.grid(True, linestyle="--", alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[saved] {save_path}")
    if not no_show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    args = parse_args()
    if args.assign:
        assign_path = Path(args.assign)
        dataset = assign_path.stem.replace("_assign", "")
    elif args.dataset:
        dataset = args.dataset
        assign_path = Path("data/input") / f"{dataset}_assign.txt"
    else:
        raise SystemExit("Provide --dataset or --assign")

    if not assign_path.exists():
        raise FileNotFoundError(f"Assignment file not found: {assign_path}")

    sizes = load_sizes(assign_path)
    save_path = Path(args.plots_dir) / f"community_sizes_{dataset}.png"
    plot_sizes(sizes, title=dataset, save_path=save_path, no_show=args.no_show)


if __name__ == "__main__":  # pragma: no cover
    main()
