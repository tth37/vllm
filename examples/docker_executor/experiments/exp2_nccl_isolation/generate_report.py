#!/usr/bin/env python3
"""Generate plots and an HTML report for exp2: NCCL transport under per-GPU container isolation.

Reads benchmark JSON results from results/ and produces:
  - Matplotlib figures in analysis_assets/
  - Formal HTML report (analysis_report.html)
"""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
ASSET_DIR = ROOT / "analysis_assets"
HTML_REPORT = ROOT / "analysis_report.html"

CONFIGS = [
    {
        "key": "baseline",
        "file": "node192_baseline.json",
        "label": "Baseline (P2P/NVLink)",
        "short_label": "Baseline",
        "color": "#2c2c2c",
        "transport": "P2P/CUMEM/read",
        "gpu_visibility": "All GPUs per container",
        "ipc": "Host",
        "network": "Host",
        "shared_volumes": "None",
        "nccl_env": "Default",
    },
    {
        "key": "shm_isolation",
        "file": "node192_isolated_shm.json",
        "label": "SHM Isolation",
        "short_label": "SHM Isolation",
        "color": "#4c72b0",
        "transport": "SHM/direct/direct",
        "gpu_visibility": "1 GPU per container (CUDA), all (NVML)",
        "ipc": "Private",
        "network": "Bridge",
        "shared_volumes": "/dev/shm (4 GB), /tmp (200 MB)",
        "nccl_env": "NCCL_P2P_DISABLE=1, NCCL_HOSTID, NCCL_NET_DISABLE_INTRA=1",
    },
    {
        "key": "naive_isolation",
        "file": "node192_isolated_p2p.json",
        "label": "Naive Isolation (TCP)",
        "short_label": "Naive (TCP)",
        "color": "#c44e52",
        "transport": "NET/Socket",
        "gpu_visibility": "1 GPU per container (CUDA), all (NVML)",
        "ipc": "Private",
        "network": "Bridge",
        "shared_volumes": "/tmp (200 MB)",
        "nccl_env": "NCCL_HOSTID only",
    },
]

OPERATIONS = ["all_reduce", "all_gather", "reduce_scatter", "broadcast", "p2p"]
OP_LABELS = {
    "all_reduce": "All-Reduce",
    "all_gather": "All-Gather",
    "reduce_scatter": "Reduce-Scatter",
    "broadcast": "Broadcast",
    "p2p": "P2P Send/Recv",
}

# Size keys in the JSON, ordered small to large
SIZE_KEYS = ["0.00MB", "0.01MB", "0.10MB", "0.98MB", "9.77MB", "97.66MB"]
SIZE_LABELS = ["1 KB", "10 KB", "100 KB", "1 MB", "10 MB", "100 MB"]


def load_results() -> dict[str, dict]:
    """Load all config result JSONs."""
    data = {}
    for cfg in CONFIGS:
        path = RESULTS_DIR / cfg["file"]
        if path.exists():
            with open(path) as f:
                data[cfg["key"]] = json.load(f)
        else:
            print(f"  WARNING: {path} not found")
    return data


def get_bandwidth(data: dict, config_key: str, op: str, size_key: str) -> float:
    """Extract bandwidth_gbps for a given config/operation/size."""
    try:
        return data[config_key]["results"][op][size_key]["bandwidth_gbps"]
    except (KeyError, TypeError):
        return 0.0


def get_time(data: dict, config_key: str, op: str, size_key: str) -> float:
    """Extract avg_time_ms for a given config/operation/size."""
    try:
        return data[config_key]["results"][op][size_key]["avg_time_ms"]
    except (KeyError, TypeError):
        return 0.0


def _setup_plot_style():
    """Configure matplotlib for formal academic style matching exp1."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#999",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_bandwidth_by_size(data: dict, filename: str) -> Path:
    """Plot bandwidth vs message size for all 3 configs, one subplot per operation."""
    _setup_plot_style()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=False)
    fig.suptitle("NCCL Bandwidth vs Message Size — node192 (2× A100-SXM4, NVLink NV12)",
                 fontsize=14, fontweight="bold", y=1.02)

    x = np.arange(len(SIZE_KEYS))

    for idx, op in enumerate(OPERATIONS):
        ax = axes[idx]
        for cfg in CONFIGS:
            bw = [get_bandwidth(data, cfg["key"], op, sk) for sk in SIZE_KEYS]
            ax.plot(x, bw, marker="o", markersize=5, linewidth=2,
                    color=cfg["color"], label=cfg["short_label"])

        ax.set_title(OP_LABELS[op], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(SIZE_LABELS, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Bandwidth (GB/s)" if idx == 0 else "")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.01)
        if idx == 0:
            ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out = ASSET_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bandwidth_bars_100mb(data: dict, filename: str) -> Path:
    """Bar chart comparing bandwidth at 100 MB across operations and configs."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(OPERATIONS))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cfg in enumerate(CONFIGS):
        bw = [get_bandwidth(data, cfg["key"], op, "97.66MB") for op in OPERATIONS]
        bars = ax.bar(x + offsets[i], bw, width, label=cfg["short_label"],
                      color=cfg["color"], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, bw):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("NCCL Bandwidth at 100 MB — node192 (2× A100-SXM4, NVLink NV12)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([OP_LABELS[op] for op in OPERATIONS])
    ax.legend()
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = ASSET_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_degradation(data: dict, filename: str) -> Path:
    """Bar chart showing % of baseline bandwidth retained at 100 MB."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(OPERATIONS))
    width = 0.3
    offsets = [-width / 2, width / 2]

    # Only SHM and Naive (relative to baseline)
    for i, cfg in enumerate(CONFIGS[1:]):  # skip baseline
        pcts = []
        for op in OPERATIONS:
            base_bw = get_bandwidth(data, "baseline", op, "97.66MB")
            cfg_bw = get_bandwidth(data, cfg["key"], op, "97.66MB")
            pcts.append(cfg_bw / base_bw * 100 if base_bw > 0 else 0)

        bars = ax.bar(x + offsets[i], pcts, width, label=cfg["short_label"],
                      color=cfg["color"], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("% of Baseline Bandwidth")
    ax.set_title("Bandwidth Retained Under Isolation (100 MB) — vs P2P/NVLink Baseline",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([OP_LABELS[op] for op in OPERATIONS])
    ax.legend()
    ax.axhline(y=100, color="#2c2c2c", linestyle="--", linewidth=1, alpha=0.5, label="_nolegend_")
    ax.set_ylim(0, max(20, 15))

    fig.tight_layout()
    out = ASSET_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_latency_bars_100mb(data: dict, filename: str) -> Path:
    """Bar chart comparing latency (avg_time_ms) at 100 MB."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(OPERATIONS))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cfg in enumerate(CONFIGS):
        times = [get_time(data, cfg["key"], op, "97.66MB") for op in OPERATIONS]
        bars = ax.bar(x + offsets[i], times, width, label=cfg["short_label"],
                      color=cfg["color"], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, times):
            if val > 0.1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("NCCL Operation Latency at 100 MB — node192 (2× A100-SXM4, NVLink NV12)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([OP_LABELS[op] for op in OPERATIONS])
    ax.legend()
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = ASSET_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _swatch(color: str) -> str:
    """Colored square swatch for HTML tables."""
    return (f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{color};border:1px solid #999;margin-right:5px;'
            f'vertical-align:middle"></span>')


def generate_html(data: dict, figures: dict[str, Path]) -> Path:
    """Generate the formal HTML report."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        commit = "unknown"

    from datetime import datetime, timezone
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build the 100 MB comparison table for results section
    def bw_table_100mb() -> str:
        rows = ""
        for op in OPERATIONS:
            base_bw = get_bandwidth(data, "baseline", op, "97.66MB")
            shm_bw = get_bandwidth(data, "shm_isolation", op, "97.66MB")
            naive_bw = get_bandwidth(data, "naive_isolation", op, "97.66MB")
            shm_pct = (shm_bw / base_bw * 100) if base_bw > 0 else 0
            naive_pct = (naive_bw / base_bw * 100) if base_bw > 0 else 0
            rows += (f"<tr><td>{OP_LABELS[op]}</td>"
                     f"<td>{base_bw:.2f}</td>"
                     f"<td>{shm_bw:.2f}</td>"
                     f"<td>{naive_bw:.2f}</td>"
                     f"<td><b>{shm_pct:.1f}%</b></td>"
                     f"<td><b>{naive_pct:.1f}%</b></td></tr>\n")
        return rows

    # Build full size-sweep tables for appendix
    def size_sweep_table(config_key: str) -> str:
        rows = ""
        for si, (sk, sl) in enumerate(zip(SIZE_KEYS, SIZE_LABELS)):
            cols = f"<td>{sl}</td>"
            for op in OPERATIONS:
                bw = get_bandwidth(data, config_key, op, sk)
                cols += f"<td>{bw:.2f}</td>"
            rows += f"<tr>{cols}</tr>\n"
        return rows

    # Figure HTML helper
    def fig_html(key: str, alt: str) -> str:
        if key not in figures:
            return ""
        rel = figures[key].relative_to(ROOT)
        return f'''
      <div class="figure">
        <img src="{html.escape(str(rel))}" alt="{html.escape(alt)}">
      </div>'''

    # Config comparison table
    config_rows = ""
    for cfg in CONFIGS:
        config_rows += (f"<tr>"
                        f"<td>{_swatch(cfg['color'])}<b>{cfg['label']}</b></td>"
                        f"<td>{cfg['gpu_visibility']}</td>"
                        f"<td>{cfg['ipc']}</td>"
                        f"<td>{cfg['network']}</td>"
                        f"<td>{cfg['shared_volumes']}</td>"
                        f"<td>{cfg['transport']}</td>"
                        f"<td><code>{html.escape(cfg['nccl_env'])}</code></td>"
                        f"</tr>\n")

    # Appendix: per-config size sweep tables
    appendix_tables = ""
    for cfg in CONFIGS:
        appendix_tables += f"""
      <h3>{_swatch(cfg['color'])}{html.escape(cfg['label'])} &mdash; {html.escape(cfg['transport'])}</h3>
      <div style="overflow-x:auto">
      <table>
        <thead>
          <tr><th>Size</th><th>All-Reduce</th><th>All-Gather</th><th>Reduce-Scatter</th><th>Broadcast</th><th>P2P</th></tr>
        </thead>
        <tbody>
          {size_sweep_table(cfg['key'])}
        </tbody>
      </table>
      </div>
      <p style="font-size:12px;color:#666">All values in GB/s.</p>
"""

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Experiment 2: NCCL Transport Under Per-GPU Container Isolation</title>
  <style>
    body {{
      margin: 0;
      font-family: "Times New Roman", "Nimbus Roman", Times, serif;
      background: #fff;
      color: #111;
      line-height: 1.6;
    }}
    main {{
      max-width: 900px;
      margin: 0 auto;
      padding: 40px 32px 60px;
    }}
    h1 {{
      font-size: 22px;
      text-align: center;
      margin: 0 0 4px;
      line-height: 1.3;
    }}
    .subtitle {{
      text-align: center;
      font-size: 14px;
      color: #444;
      margin: 0 0 24px;
    }}
    h2 {{
      font-size: 17px;
      border-bottom: 1px solid #999;
      padding-bottom: 4px;
      margin: 28px 0 12px;
    }}
    h3 {{
      font-size: 15px;
      margin: 20px 0 8px;
    }}
    p, li {{
      font-size: 14px;
      margin: 0 0 10px;
    }}
    ul, ol {{
      padding-left: 24px;
    }}
    .figure {{
      margin: 16px 0;
      text-align: center;
    }}
    .figure img {{
      max-width: 100%;
      border: 1px solid #ccc;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin: 12px 0;
    }}
    th, td {{
      border: 1px solid #bbb;
      padding: 5px 7px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f0f0f0;
      font-weight: bold;
    }}
    td {{
      font-variant-numeric: tabular-nums;
    }}
    code {{
      font-family: "Courier New", Courier, monospace;
      font-size: 13px;
      background: #f5f5f5;
      padding: 1px 4px;
    }}
    .config-table {{
      width: auto;
      margin: 12px 0;
    }}
    .config-table td:first-child {{
      font-weight: bold;
      white-space: nowrap;
      padding-right: 16px;
    }}
    hr.appendix {{
      border: none;
      border-top: 2px solid #333;
      margin: 36px 0 24px;
    }}
    .diagram {{
      font-family: "Courier New", Courier, monospace;
      font-size: 13px;
      background: #fafafa;
      border: 1px solid #ddd;
      padding: 12px 16px;
      margin: 12px 0;
      white-space: pre;
      overflow-x: auto;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Experiment 2: NCCL Transport Under Per-GPU Container Isolation</h1>
    <p class="subtitle">
      node192 &middot; 2&times; A100-SXM4-40GB &middot; NVLink NV12<br>
      Generated {generated} &middot; commit {html.escape(commit)}
    </p>

    <h2>1. Objective</h2>
    <p>Quantify the NCCL communication bandwidth loss when GPU workers run in
       per-GPU isolated containers (one physical GPU per container, private IPC
       and network namespaces) on NVLink-connected hardware. The experiment
       tests whether environment-variable-only configurations can recover
       NVLink P2P performance without code changes to NCCL.</p>

    <h2>2. Background</h2>
    <p>NCCL selects transport based on what the runtime environment makes available.
       On NVLink-connected GPUs, the optimal path is <b>P2P/CUMEM/read</b>, which
       performs direct GPU-to-GPU DMA over NVLink via CUDA IPC. Three barriers
       prevent this under container isolation:</p>
    <ol>
      <li><b>Hostname mismatch</b> &mdash; containers have different hostnames, causing
          NCCL to treat them as separate nodes. Solved with <code>NCCL_HOSTID</code>.</li>
      <li><b>GPU topology invisible</b> &mdash; per-GPU <code>CUDA_VISIBLE_DEVICES</code>
          hides the interconnect topology. Partially solved with
          <code>NVIDIA_VISIBLE_DEVICES=all</code> (NVML can see topology).</li>
      <li><b>CUDA peer access impossible</b> &mdash; each container sees only one GPU as
          <code>cuda:0</code>. <code>cudaDeviceCanAccessPeer()</code> fails because
          there is no peer device. <b>This cannot be solved with environment variables.</b></li>
    </ol>

    <h2>3. Configurations Under Test</h2>
    <p>Three Docker Compose configurations test the isolation spectrum on node192
       (GPU 0 and GPU 1, connected via 12&times; NVLink).</p>
    <table>
      <thead>
        <tr><th>Configuration</th><th>GPU Visibility</th><th>IPC</th>
            <th>Network</th><th>Shared Volumes</th>
            <th>NCCL Transport</th><th>NCCL Env</th></tr>
      </thead>
      <tbody>
        {config_rows}
      </tbody>
    </table>

    <div class="diagram">+-----------------+     +-----------------+
|  Container 0    |     |  Container 1    |
|  cuda:0 = GPU0  |     |  cuda:0 = GPU1  |
|  (no peer GPU)  |     |  (no peer GPU)  |
+--------+--------+     +--------+--------+
         |                       |
         |   NVLink (unused!)    |
    GPU0 ===================== GPU1
         |                       |
    +----+---------------------------+----+
    |         Host Memory (DRAM)          |
    |     Shared /dev/shm (tmpfs)         |
    |   &lt;- SHM transport goes here -&gt;    |
    +-------------------------------------+</div>
    <p style="font-size:12px;color:#666;text-align:center">
      Figure: Per-GPU container isolation forces data through host DRAM (SHM)
      instead of the direct NVLink path.</p>

    <h2>4. Experimental Setup</h2>
    <table class="config-table">
      <tr><td>Machine</td><td>node192 (10.0.2.192)</td></tr>
      <tr><td>GPUs</td><td>2&times; NVIDIA A100-SXM4-40GB (GPU 0, GPU 1)</td></tr>
      <tr><td>GPU Interconnect</td><td>NV12 (12&times; NVLink, ~600 GB/s bidirectional)</td></tr>
      <tr><td>NCCL</td><td>2.21.5 (bundled with PyTorch 2.5.1+cu124)</td></tr>
      <tr><td>CUDA</td><td>12.4</td></tr>
      <tr><td>Docker</td><td>27.3.1 with nvidia runtime</td></tr>
      <tr><td>Benchmark</td><td>Custom PyTorch NCCL benchmark (all-reduce, all-gather, reduce-scatter, broadcast, P2P)</td></tr>
      <tr><td>Message sizes</td><td>1 KB &ndash; 100 MB (6 sizes, log-spaced)</td></tr>
      <tr><td>Timing</td><td>CUDA events, trimmed mean (drop top/bottom 5%)</td></tr>
    </table>

    <h2>5. Results</h2>

    <h3>5.1 Bandwidth at 100 MB</h3>
    <table>
      <thead>
        <tr><th>Operation</th>
            <th>{_swatch(CONFIGS[0]['color'])}Baseline (GB/s)</th>
            <th>{_swatch(CONFIGS[1]['color'])}SHM Isolation (GB/s)</th>
            <th>{_swatch(CONFIGS[2]['color'])}Naive TCP (GB/s)</th>
            <th>SHM / Baseline</th>
            <th>TCP / Baseline</th></tr>
      </thead>
      <tbody>
        {bw_table_100mb()}
      </tbody>
    </table>

    {fig_html('bars_100mb', 'Bandwidth at 100 MB')}

    <h3>5.2 Bandwidth vs Message Size</h3>
    {fig_html('bw_by_size', 'Bandwidth vs message size')}

    <h3>5.3 Degradation Under Isolation</h3>
    {fig_html('degradation', 'Bandwidth retained under isolation')}

    <h3>5.4 Latency at 100 MB</h3>
    {fig_html('latency', 'Latency at 100 MB')}

    <h2>6. Analysis</h2>

    <h3>6.1 Environment variables alone cannot restore NVLink P2P</h3>
    <p>The fundamental barrier is <code>cudaDeviceCanAccessPeer()</code>: with per-GPU
       <code>CUDA_VISIBLE_DEVICES</code>, each container sees only <code>cuda:0</code>.
       There is no peer device to check, so NCCL&rsquo;s P2P transport selection
       (<code>p2pCanConnect</code>) fails unconditionally. No combination of
       <code>NCCL_HOSTID</code>, <code>NVIDIA_VISIBLE_DEVICES</code>, or other
       environment variables can change this &mdash; it is a CUDA runtime constraint.</p>

    <h3>6.2 SHM is the best achievable fallback without code changes</h3>
    <p>By sharing <code>/dev/shm</code> and <code>/tmp</code> as Docker volumes and
       setting <code>NCCL_P2P_DISABLE=1</code> + <code>NCCL_HOSTID</code>, NCCL selects
       SHM/direct/direct transport. This routes data through host DRAM (CPU memory),
       achieving 13&ndash;23 GB/s &mdash; about <b>9&ndash;12% of the NVLink P2P baseline</b>
       (157&ndash;261 GB/s). While substantially better than the naive TCP fallback
       (4&ndash;9 GB/s), it still represents a <b>~7&ndash;12&times; degradation</b>.</p>

    <h3>6.3 Without SHM workaround, NCCL falls to TCP sockets</h3>
    <p>The naive isolation config (no shared <code>/dev/shm</code>, no
       <code>NCCL_P2P_DISABLE</code>) causes NCCL to detect <code>nNodes=2</code>
       and fall back to NET/Socket transport over the Docker bridge network. This
       achieves only 2&ndash;5% of baseline bandwidth. The SHM workaround provides
       a <b>3&ndash;4&times; improvement</b> over this TCP baseline.</p>

    <h3>6.4 The penalty is architectural, not configurational</h3>
    <p>NVLink P2P transport requires CUDA IPC (<code>cudaIpcGetMemHandle</code> /
       <code>cudaIpcOpenMemHandle</code>), which requires both GPUs to be visible in
       the same CUDA context. Per-GPU container isolation makes this impossible by
       design. Recovering NVLink performance would require one of:</p>
    <ol>
      <li>Patching NCCL to use <code>pidfd_getfd()</code> (kernel 5.6+) for
          cross-namespace CUDA IPC file descriptor passing</li>
      <li>Using the CUDA VMM API (<code>cuMemExportToShareableHandle</code> with
          <code>CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR</code>) to export GPU memory
          as POSIX FDs shareable via Unix domain sockets</li>
      <li>Giving each container visibility to all GPUs (defeats isolation)</li>
    </ol>

    <h2>7. Conclusions</h2>
    <ol>
      <li>Per-GPU container isolation on NVLink hardware causes an <b>88&ndash;91%
          NCCL bandwidth loss</b> compared to the P2P/NVLink baseline. This is a
          fundamental CUDA runtime constraint, not a configuration problem.</li>
      <li>The best environment-variable-only workaround (shared <code>/dev/shm</code>
          + NCCL SHM transport) recovers to ~10% of baseline, achieving 13&ndash;23 GB/s
          through host DRAM.</li>
      <li>Without the SHM workaround, NCCL falls back to TCP sockets at 2&ndash;5%
          of baseline (4&ndash;9 GB/s).</li>
      <li>Restoring NVLink P2P performance under per-GPU isolation requires NCCL
          source-level modifications to enable cross-container CUDA IPC.</li>
    </ol>

    <hr class="appendix">

    <h2>Appendix A: Full Bandwidth Tables</h2>
    <p>Complete bandwidth measurements (GB/s) across all message sizes and operations.</p>

    {appendix_tables}

  </main>
</body>
</html>
"""

    HTML_REPORT.write_text(html_text, encoding="utf-8")
    return HTML_REPORT


def main() -> None:
    ASSET_DIR.mkdir(exist_ok=True)

    print("Loading results...")
    data = load_results()
    if len(data) < 3:
        print(f"  WARNING: Only {len(data)}/3 result files found")
        if not data:
            print("  No results to process!")
            return

    print("Generating plots...")
    figures: dict[str, Path] = {}

    figures["bw_by_size"] = plot_bandwidth_by_size(data, "bandwidth_vs_size.png")
    print(f"  {figures['bw_by_size']}")

    figures["bars_100mb"] = plot_bandwidth_bars_100mb(data, "bandwidth_100mb_bars.png")
    print(f"  {figures['bars_100mb']}")

    figures["degradation"] = plot_degradation(data, "degradation_100mb.png")
    print(f"  {figures['degradation']}")

    figures["latency"] = plot_latency_bars_100mb(data, "latency_100mb_bars.png")
    print(f"  {figures['latency']}")

    print("Generating HTML report...")
    report = generate_html(data, figures)
    print(f"Report written to {report}")


if __name__ == "__main__":
    main()
