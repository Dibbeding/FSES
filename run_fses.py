"""CLI to run the Frontier Scoring Expansion Sampling (FSES) algorithm."""

from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

from fses.algorithms import (
    FSESParams,
    expansion_sampling_F_random,
    expansion_sampling_F_score_with_params,
    expansion_sampling_JURW_MHNorm,
    expansion_sampling_nbho_rw,
)
from fses.graph_io import EdgeList, NodeList, load_hypergraph


def _resolve_paths(args: argparse.Namespace) -> Tuple[str, Path, Path]:
    """Resolve dataset/hyperedge paths based on CLI flags."""
    if args.dataset:
        base = args.dataset
        data_root = Path(args.data_root)
        hyperedges = data_root / f"{base}_he.txt"
        assignments = data_root / f"{base}_assign.txt"
    else:
        hyperedges = Path(args.hyperedges).resolve()
        base = args.dataset or hyperedges.stem.replace("_he", "")
        if args.communities:
            assignments = Path(args.communities).resolve()
        else:
            assignments = hyperedges.with_name(hyperedges.name.replace("_he", "_assign"))
    if not hyperedges.exists():
        raise FileNotFoundError(f"Hyperedge file not found: {hyperedges}")
    if not assignments.exists():
        raise FileNotFoundError(f"Community assignment file not found: {assignments}")
    return base, hyperedges, assignments


def _compute_targets(
    total_nodes: int,
    sample_fracs: List[float],
) -> List[int]:
    """Compute target node counts from fractional inputs."""
    targets: List[int] = []
    for frac in sample_fracs:
        if frac <= 0:
            continue
        targets.append(max(1, int(total_nodes * frac)))
    unique_targets = sorted({t for t in targets if t <= total_nodes})
    if not unique_targets:
        raise ValueError("No valid sample sizes provided (use --sample-frac values in (0,1]).")
    return unique_targets


def _save_sample(
    output_dir: Path,
    dataset: str,
    total_nodes: int,
    target_nodes: int,
    params_str: str,
    edge_list: EdgeList,
    sampled_edges: Iterable[int],
) -> Path:
    """Persist sampled hyperedges to disk and return the path."""
    pct = int(round((target_nodes / max(1, total_nodes)) * 100))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset}_n{total_nodes}_{pct}pct_{params_str}_{timestamp}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as handle:
        for edge_id in sampled_edges:
            members = edge_list.get(edge_id, [])
            handle.write(" ".join(map(str, members)) + "\n")
    return output_path


def _run_once(
    expansion_func: Callable[[EdgeList, NodeList, int], Dict[str, object]],
    edge_list: EdgeList,
    node_list: NodeList,
    target_nodes: int,
    seed: int,
) -> Dict[str, object]:
    """Run a single sampling call with a given seed."""
    random.seed(seed)
    result = expansion_func(edge_list, node_list, target_nodes)
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run FSES sampling on a hypergraph.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=str, help="Dataset base name (expects <data_root>/<name>_he.txt and _assign.txt)")
    src.add_argument("--hyperedges", type=str, help="Path to a hyperedge file (comma or space separated)")
    parser.add_argument("--communities", type=str, help="Path to the community assignment file (defaults to *_assign.txt next to hyperedges)")
    parser.add_argument("--data-root", type=str, default="data/input", help="Base folder when using --dataset (default: data/input)")
    parser.add_argument(
        "--algo",
        choices=["fses", "fses-random", "jurw", "nbho-rw"],
        default="fses",
        help="Sampling algorithm variant",
    )
    parser.add_argument("--sample-frac", type=float, nargs="+", default=[0.25], help="Sample sizes as fractions (0.05 for 5%%)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per sample size")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (auto if omitted)")
    parser.add_argument("--output-dir", type=str, default="data/sampled", help="Directory to store sampled graphs")
    parser.add_argument("--tag", type=str, default="", help="Optional string appended to the params token in filenames")

    # Parameter overrides
    parser.add_argument("--unknown-score", type=float, default=1.0, help="Score bonus for unseen nodes")
    parser.add_argument("--k", type=float, default=0.10, help="Frequency decay for known nodes")
    parser.add_argument("--alpha", type=float, default=0.0, help="Hyperedge size penalty alpha")
    parser.add_argument("--candidate-limit", type=int, default=100, help="Frontier size limit (F_lim)")
    parser.add_argument("--no-refresh", action="store_true", help="Disable refresh budget recomputation")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, hyperedge_path, assign_path = _resolve_paths(args)
    edge_list, node_list, assignments = load_hypergraph(hyperedge_path, assign_path)
    total_nodes = max(len(assignments), len(node_list))

    targets = _compute_targets(total_nodes, args.sample_frac)
    print(f"[load] dataset={dataset} edges={len(edge_list)} nodes={total_nodes} targets={targets}")

    params = FSESParams(
        unknown_node_score=args.unknown_score,
        k=args.k,
        size_penalty_alpha=args.alpha,
        candidate_limit=args.candidate_limit,
        zero_known=False,
        use_refresh=not args.no_refresh,
    )

    algo = args.algo
    use_jurw_seed = False

    if algo == "fses":
        expansion_func = lambda el, nl, n: expansion_sampling_F_score_with_params(el, nl, n, params)
        variant_folder = "FSES"
        param_suffix = "fses"
    elif algo == "fses-random":
        expansion_func = lambda el, nl, n: expansion_sampling_F_random(el, nl, n)
        variant_folder = "FSES_random"
        param_suffix = "random"
    elif algo == "jurw":
        expansion_func = lambda el, nl, n: expansion_sampling_JURW_MHNorm(el, nl, n)
        variant_folder = "JURW"
        param_suffix = "jurw"
        use_jurw_seed = True
    else:  # nbho-rw
        expansion_func = lambda el, nl, n: expansion_sampling_nbho_rw(el, nl, n)
        variant_folder = "NBHO_RW"
        param_suffix = "nbho"
    if args.tag:
        param_suffix = f"{param_suffix}_{args.tag}"

    output_dir = Path(args.output_dir) / variant_folder
    base_seed = args.seed if args.seed is not None else int(time.time()) % 10000000

    for target_nodes in targets:
        for run_idx in range(args.runs):
            seed = base_seed + run_idx * 9973 + target_nodes
            print(f"[run] {dataset} | target_nodes={target_nodes} | run={run_idx + 1}/{args.runs} | seed={seed}")
            if use_jurw_seed:
                os.environ["JURW_SEED"] = str(seed)
            result = _run_once(expansion_func, edge_list, node_list, target_nodes, seed)
            sampled_edges = result["sampled_edges"]
            params_token = f"{result.get('params', param_suffix)}"
            if args.tag and args.tag not in params_token:
                params_token = f"{params_token}_{args.tag}"
            path = _save_sample(
                output_dir=output_dir,
                dataset=dataset,
                total_nodes=total_nodes,
                target_nodes=target_nodes,
                params_str=params_token,
                edge_list=edge_list,
                sampled_edges=sampled_edges,
            )
            print(f"[save] {path}")

    print("[done] Sampling finished.")


if __name__ == "__main__":  # pragma: no cover
    main()
