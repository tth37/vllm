#!/usr/bin/env python3
"""
Exp2 Analysis: Compare NCCL performance across isolation configurations.

Reads benchmark JSON results and NCCL debug logs from results/ directory.
Produces a comparison table showing transport selection and bandwidth per config.

Usage:
    python3 analyze_exp2.py [--results-dir=results/]
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


# Config metadata: (name, description, isolation_level)
CONFIGS = [
    # Phase 1
    ("baseline", "All GPUs + host IPC + host net", "none"),
    ("private_ipc", "All GPUs + private IPC + host net", "IPC"),
    ("private_net", "All GPUs + host IPC + bridge net", "NET"),
    ("private_both", "All GPUs + private IPC + bridge net", "IPC+NET"),
    ("single_gpu", "1 GPU + private IPC + bridge net", "full"),
    ("shared_shm", "1 GPU + private IPC + bridge net + shared SHM", "full+SHM"),
    # Phase 2
    ("shm_forced", "1 GPU + P2P disabled + shared SHM", "full+SHM"),
    ("shm_p2p_combo", "All GPUs + shared SHM + shared /tmp", "IPC+NET+SHM"),
    # Phase 3
    ("patched_nccl", "Patched NCCL + per-GPU isolation", "full+patch"),
]


def load_results(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load benchmark results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def extract_transport(log_filepath: Path) -> str:
    """Parse NCCL debug log to determine transport type.

    Looks for lines like:
        NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
        NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM
        NCCL INFO Channel 00/0 : 0[0] -> 1[1] via NET/Socket/2
    """
    try:
        with open(log_filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return "NO_LOG"

    # Look for transport lines
    p2p_match = re.search(r'via P2P(/\w+)*', content)
    shm_match = re.search(r'via SHM', content)
    net_match = re.search(r'via NET(/\w+)*', content)

    if p2p_match:
        return p2p_match.group(0)  # e.g., "via P2P/CUMEM/read"
    elif shm_match:
        return "via SHM"
    elif net_match:
        return net_match.group(0)  # e.g., "via NET/Socket/2"
    else:
        return "UNKNOWN"


def extract_nccl_topology(log_filepath: Path) -> Dict[str, str]:
    """Extract NCCL topology info from debug logs."""
    info = {}
    try:
        with open(log_filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return info

    # nNodes and localRanks
    topo_match = re.search(
        r'nRanks (\d+) nNodes (\d+) localRanks (\d+)', content
    )
    if topo_match:
        info['nRanks'] = topo_match.group(1)
        info['nNodes'] = topo_match.group(2)
        info['localRanks'] = topo_match.group(3)

    return info


def get_bandwidth_at_size(results: Dict[str, Any], operation: str,
                          target_size_mb: float = 100.0) -> Optional[float]:
    """Get bandwidth for a specific operation at a target size.

    Finds the closest available size to target_size_mb.
    """
    op_data = results.get('results', {}).get(operation, {})
    if not op_data:
        return None

    # Find closest size
    best_key = None
    best_diff = float('inf')
    for key in op_data:
        size_mb = float(key.replace('MB', ''))
        diff = abs(size_mb - target_size_mb)
        if diff < best_diff:
            best_diff = diff
            best_key = key

    if best_key:
        return op_data[best_key].get('bandwidth_gbps')
    return None


def generate_report(results_dir: Path):
    """Generate multi-config comparison report."""
    print("\n" + "=" * 110)
    print("  EXP2: NCCL ISOLATION MATRIX — PERFORMANCE COMPARISON")
    print("=" * 110)

    # Collect data for each config
    rows: List[Tuple[str, str, str, str, Optional[float], Optional[float],
                      Dict[str, str]]] = []

    for config_name, description, isolation in CONFIGS:
        result_file = results_dir / f"{config_name}_results.json"
        log_file = results_dir / f"{config_name}_log.txt"

        results = load_results(result_file)
        transport = extract_transport(log_file)
        topo = extract_nccl_topology(log_file)

        ar_bw = get_bandwidth_at_size(results, 'all_reduce') if results else None
        ag_bw = get_bandwidth_at_size(results, 'all_gather') if results else None

        rows.append((config_name, description, isolation, transport,
                      ar_bw, ag_bw, topo))

    # Print summary table
    print("\n  Summary (bandwidth at ~100 MB transfer size):\n")
    header = (f"  {'Config':<18} {'Transport':<22} {'AllReduce':>12} "
              f"{'AllGather':>12} {'nNodes':>7} {'localRanks':>11} {'Isolation':<14}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    baseline_ar = None
    for (name, desc, isolation, transport, ar_bw, ag_bw, topo) in rows:
        if name == "baseline" and ar_bw is not None:
            baseline_ar = ar_bw

        ar_str = f"{ar_bw:.2f} GB/s" if ar_bw is not None else "N/A"
        ag_str = f"{ag_bw:.2f} GB/s" if ag_bw is not None else "N/A"
        n_nodes = topo.get('nNodes', '-')
        local_ranks = topo.get('localRanks', '-')

        print(f"  {name:<18} {transport:<22} {ar_str:>12} "
              f"{ag_str:>12} {n_nodes:>7} {local_ranks:>11} {isolation:<14}")

    # Relative performance vs baseline
    if baseline_ar is not None:
        print(f"\n  Relative to baseline (all-reduce ~100MB):\n")
        for (name, desc, isolation, transport, ar_bw, ag_bw, topo) in rows:
            if ar_bw is not None and name != "baseline":
                pct = (ar_bw / baseline_ar) * 100
                bar_len = int(pct / 2)
                bar = "#" * min(bar_len, 50)
                print(f"  {name:<18} {pct:6.1f}%  {bar}")

    # Detailed per-operation breakdown for available configs
    print("\n" + "=" * 110)
    print("  DETAILED BREAKDOWN BY OPERATION")
    print("=" * 110)

    operations = ['all_reduce', 'all_gather', 'reduce_scatter', 'broadcast']

    for op in operations:
        print(f"\n  {op.upper().replace('_', '-')}:")
        print(f"  {'Config':<18} ", end="")

        # Get size keys from first available result
        size_keys = None
        for (name, _, _, _, _, _, _) in rows:
            result_file = results_dir / f"{name}_results.json"
            results = load_results(result_file)
            if results:
                op_data = results.get('results', {}).get(op, {})
                if op_data:
                    size_keys = sorted(
                        op_data.keys(),
                        key=lambda x: op_data[x]['size_elements']
                    )
                    break

        if not size_keys:
            print("  No data available")
            continue

        for sk in size_keys:
            print(f"{sk:>12}", end=" ")
        print()
        print("  " + "-" * (18 + len(size_keys) * 13))

        for (name, _, _, _, _, _, _) in rows:
            result_file = results_dir / f"{name}_results.json"
            results = load_results(result_file)
            if not results:
                continue

            op_data = results.get('results', {}).get(op, {})
            if not op_data:
                continue

            print(f"  {name:<18} ", end="")
            for sk in size_keys:
                if sk in op_data:
                    bw = op_data[sk]['bandwidth_gbps']
                    print(f"{bw:>9.2f} GB ", end="")
                else:
                    print(f"{'N/A':>12} ", end="")
            print()

    # Conclusions
    print("\n" + "=" * 110)
    print("  KEY FINDINGS")
    print("=" * 110)

    fast_configs = []
    slow_configs = []
    for (name, desc, isolation, transport, ar_bw, ag_bw, topo) in rows:
        if ar_bw is None:
            continue
        if baseline_ar and ar_bw >= baseline_ar * 0.7:
            fast_configs.append((name, transport, ar_bw, isolation))
        else:
            slow_configs.append((name, transport, ar_bw, isolation))

    if fast_configs:
        print("\n  Configs achieving >= 70% of baseline bandwidth:")
        for name, transport, bw, iso in fast_configs:
            print(f"    {name:<18} {transport:<22} {bw:.2f} GB/s  (isolation: {iso})")

    if slow_configs:
        print("\n  Configs with degraded performance (<70% baseline):")
        for name, transport, bw, iso in slow_configs:
            print(f"    {name:<18} {transport:<22} {bw:.2f} GB/s  (isolation: {iso})")

    print("\n" + "=" * 110 + "\n")


if __name__ == '__main__':
    results_dir = Path("results")

    for arg in sys.argv[1:]:
        if arg.startswith("--results-dir="):
            results_dir = Path(arg.split("=", 1)[1])

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    generate_report(results_dir)
