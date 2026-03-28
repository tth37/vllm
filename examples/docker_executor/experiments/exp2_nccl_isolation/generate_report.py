#!/usr/bin/env python3
"""Generate plots and an HTML report for exp2: NCCL transport under per-GPU container isolation.

Reads benchmark JSON results from results/ and produces:
  - Matplotlib figures in analysis_assets/
  - Formal HTML report (analysis_report.html)

Report objective: Demonstrate that NCCL cuMem VMM API recovers full NVLink P2P
bandwidth under per-GPU Docker container isolation.
"""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
        "key": "cumem_isolation",
        "file": "node192_cumem_isolation.json",
        "label": "CUMEM Isolation (P2P Recovery)",
        "short_label": "CUMEM (P2P)",
        "color": "#55a868",
        "transport": "P2P/CUMEM/read",
        "gpu_visibility": "1 GPU per container (CUDA), all (NVML)",
        "ipc": "Private",
        "network": "Host",
        "shared_volumes": "/dev/shm (4 GB)",
        "nccl_env": "NCCL_CUMEM_ENABLE=1",
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

SIZE_KEYS = ["0.00MB", "0.01MB", "0.10MB", "0.98MB", "9.77MB", "97.66MB"]
SIZE_LABELS = ["1 KB", "10 KB", "100 KB", "1 MB", "10 MB", "100 MB"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_results() -> dict[str, dict]:
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
    try:
        return data[config_key]["results"][op][size_key]["bandwidth_gbps"]
    except (KeyError, TypeError):
        return 0.0


def get_time(data: dict, config_key: str, op: str, size_key: str) -> float:
    try:
        return data[config_key]["results"][op][size_key]["avg_time_ms"]
    except (KeyError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def _setup_plot_style():
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


# ---------------------------------------------------------------------------
# Figure 1: Architecture Diagram (replaces ASCII art)
# ---------------------------------------------------------------------------

def _draw_box(ax, x, y, w, h, label, color="#e8e8e8", border="#555",
              fontsize=9, fontweight="normal", text_color="#111", alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=border, linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color)
    return box


def plot_architecture_diagram(filename: str) -> Path:
    """Two-panel architecture diagram: SHM fallback vs CUMEM P2P recovery."""
    _setup_plot_style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7.5))

    for ax in (ax_left, ax_right):
        ax.set_xlim(-0.5, 6.0)
        ax.set_ylim(-0.8, 7.0)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── Layout constants (centered in 0..5.5 range) ──
    C0_X, C1_X = 0.2, 3.0          # container left edges
    C_W, C_H = 2.3, 2.4            # container size
    G_PAD = 0.25                     # GPU inset from container
    G_W, G_H = 1.8, 1.2            # GPU box size
    GPU_Y = 4.4                      # GPU bottom y
    CONT_Y = 4.0                     # Container bottom y
    MID_X = 2.75                     # horizontal center

    # ===================================================================
    # LEFT PANEL: Without CUMEM (SHM fallback)
    # ===================================================================
    ax_left.set_title("Without CUMEM: SHM Fallback\n(9\u201312% of NVLink bandwidth)",
                      fontsize=14, fontweight="bold", color="#c44e52", pad=14)

    # Container outlines
    _draw_box(ax_left, C0_X, CONT_Y, C_W, C_H, "", color="#fff3e0", border="#e65100")
    ax_left.text(C0_X + C_W / 2, CONT_Y + C_H + 0.15, "Container 0",
                 ha="center", fontsize=11, fontweight="bold", color="#e65100")
    _draw_box(ax_left, C1_X, CONT_Y, C_W, C_H, "", color="#e3f2fd", border="#1565c0")
    ax_left.text(C1_X + C_W / 2, CONT_Y + C_H + 0.15, "Container 1",
                 ha="center", fontsize=11, fontweight="bold", color="#1565c0")

    # GPU boxes
    g0_cx = C0_X + C_W / 2
    g1_cx = C1_X + C_W / 2
    _draw_box(ax_left, C0_X + G_PAD, GPU_Y, G_W, G_H, "GPU 0\ncuda:0",
              color="#fff9c4", border="#f57f17", fontsize=10, fontweight="bold")
    ax_left.text(g0_cx, GPU_Y - 0.18, "No peer visible",
                 ha="center", fontsize=8, color="#999", style="italic")
    _draw_box(ax_left, C1_X + G_PAD, GPU_Y, G_W, G_H, "GPU 1\ncuda:0",
              color="#fff9c4", border="#f57f17", fontsize=10, fontweight="bold")
    ax_left.text(g1_cx, GPU_Y - 0.18, "No peer visible",
                 ha="center", fontsize=8, color="#999", style="italic")

    # NVLink (crossed out)
    nvl_y = GPU_Y + G_H / 2
    ax_left.annotate("", xy=(C1_X + G_PAD, nvl_y), xytext=(C0_X + G_PAD + G_W, nvl_y),
                     arrowprops=dict(arrowstyle="-", color="#ccc", lw=3, linestyle="--"))
    ax_left.text(MID_X, nvl_y + 0.2, "NVLink", ha="center", fontsize=9, color="#bbb")
    # Red X
    xc = MID_X
    ax_left.plot([xc - 0.35, xc + 0.35], [nvl_y - 0.3, nvl_y + 0.3], color="#c44e52", lw=3, zorder=10)
    ax_left.plot([xc - 0.35, xc + 0.35], [nvl_y + 0.3, nvl_y - 0.3], color="#c44e52", lw=3, zorder=10)

    # Host DRAM box
    dram_y, dram_h = 0.6, 2.0
    _draw_box(ax_left, 0.4, dram_y, 4.7, dram_h, "", color="#e8eaf6", border="#3949ab")
    ax_left.text(MID_X, dram_y + dram_h - 0.4, "Host Memory (DRAM)",
                 ha="center", fontsize=11, fontweight="bold", color="#3949ab")
    ax_left.text(MID_X, dram_y + dram_h / 2, "Shared /dev/shm (tmpfs)",
                 ha="center", fontsize=10, color="#5c6bc0")
    ax_left.text(MID_X, dram_y + 0.35, "SHM/direct/direct transport",
                 ha="center", fontsize=10, color="#7986cb", style="italic")

    # Data path arrows GPU0 -> DRAM -> GPU1
    ax_left.annotate("", xy=(g0_cx, dram_y + dram_h), xytext=(g0_cx, CONT_Y),
                     arrowprops=dict(arrowstyle="-|>", color="#4c72b0", lw=2.5, mutation_scale=16))
    ax_left.annotate("", xy=(g1_cx, CONT_Y), xytext=(g1_cx, dram_y + dram_h),
                     arrowprops=dict(arrowstyle="-|>", color="#4c72b0", lw=2.5, mutation_scale=16))
    ax_left.text(g0_cx - 0.55, (CONT_Y + dram_y + dram_h) / 2, "copy to\nhost mem",
                 ha="center", fontsize=8, color="#4c72b0", style="italic")
    ax_left.text(g1_cx + 0.55, (CONT_Y + dram_y + dram_h) / 2, "copy from\nhost mem",
                 ha="center", fontsize=8, color="#4c72b0", style="italic")

    # Bandwidth result
    _draw_box(ax_left, 1.35, -0.55, 2.8, 0.65, "13\u201323 GB/s",
              color="#ffcdd2", border="#c62828", fontsize=12, fontweight="bold", text_color="#b71c1c")

    # ===================================================================
    # RIGHT PANEL: With CUMEM (P2P Recovery)
    # ===================================================================
    ax_right.set_title("With CUMEM: P2P/NVLink Recovery\n(99.5\u2013100.1% of baseline bandwidth)",
                       fontsize=14, fontweight="bold", color="#2e7d32", pad=14)

    # Container outlines
    _draw_box(ax_right, C0_X, CONT_Y, C_W, C_H, "", color="#fff3e0", border="#e65100")
    ax_right.text(C0_X + C_W / 2, CONT_Y + C_H + 0.15, "Container 0",
                  ha="center", fontsize=11, fontweight="bold", color="#e65100")
    _draw_box(ax_right, C1_X, CONT_Y, C_W, C_H, "", color="#e3f2fd", border="#1565c0")
    ax_right.text(C1_X + C_W / 2, CONT_Y + C_H + 0.15, "Container 1",
                  ha="center", fontsize=11, fontweight="bold", color="#1565c0")

    # GPU boxes
    _draw_box(ax_right, C0_X + G_PAD, GPU_Y, G_W, G_H, "GPU 0\ncuda:0",
              color="#fff9c4", border="#f57f17", fontsize=10, fontweight="bold")
    ax_right.text(g0_cx, GPU_Y - 0.18, "cuMem export",
                  ha="center", fontsize=8, color="#2e7d32", fontweight="bold")
    _draw_box(ax_right, C1_X + G_PAD, GPU_Y, G_W, G_H, "GPU 1\ncuda:0",
              color="#fff9c4", border="#f57f17", fontsize=10, fontweight="bold")
    ax_right.text(g1_cx, GPU_Y - 0.18, "cuMem import",
                  ha="center", fontsize=8, color="#2e7d32", fontweight="bold")

    # NVLink (active, bold green double-arrow)
    ax_right.annotate("", xy=(C1_X + G_PAD + 0.05, nvl_y),
                      xytext=(C0_X + G_PAD + G_W - 0.05, nvl_y),
                      arrowprops=dict(arrowstyle="<|-|>", color="#2e7d32", lw=4, mutation_scale=20))
    ax_right.text(MID_X, nvl_y + 0.25, "NVLink DMA",
                  ha="center", fontsize=10, fontweight="bold", color="#2e7d32")

    # Shared Network Namespace zone
    ns_y, ns_h = 1.6, 1.9
    _draw_box(ax_right, 0.4, ns_y, 4.7, ns_h, "", color="#e8f5e9", border="#2e7d32")
    ax_right.text(MID_X, ns_y + ns_h - 0.35, "Shared Network Namespace (host)",
                  ha="center", fontsize=11, fontweight="bold", color="#2e7d32")
    ax_right.text(MID_X, ns_y + ns_h / 2, "Abstract Unix Domain Socket",
                  ha="center", fontsize=10, color="#388e3c")
    ax_right.text(MID_X, ns_y + 0.4, "SCM_RIGHTS: POSIX FD passing",
                  ha="center", fontsize=10, color="#43a047", style="italic")

    # FD passing arrows (dashed green, from GPUs down into NS zone)
    fd_target_y = ns_y + ns_h - 0.6
    ax_right.annotate("", xy=(MID_X - 0.3, fd_target_y), xytext=(g0_cx, CONT_Y),
                      arrowprops=dict(arrowstyle="-|>", color="#66bb6a", lw=2.2, linestyle="--", mutation_scale=14))
    ax_right.annotate("", xy=(g1_cx, CONT_Y), xytext=(MID_X + 0.3, fd_target_y),
                      arrowprops=dict(arrowstyle="-|>", color="#66bb6a", lw=2.2, linestyle="--", mutation_scale=14))
    ax_right.text(g0_cx - 0.6, (CONT_Y + fd_target_y) / 2, "export\nFD",
                  ha="center", fontsize=8, color="#388e3c", style="italic")
    ax_right.text(g1_cx + 0.55, (CONT_Y + fd_target_y) / 2, "import\nFD",
                  ha="center", fontsize=8, color="#388e3c", style="italic")

    # Isolation badges in a row
    badge_y = 0.7
    badges = ["Private PID NS", "Private IPC NS", "Private Mount NS", "Per-GPU CUDA"]
    badge_w = 1.05
    total_w = len(badges) * badge_w + (len(badges) - 1) * 0.12
    start_x = MID_X - total_w / 2
    for i, label_text in enumerate(badges):
        bx = start_x + i * (badge_w + 0.12)
        _draw_box(ax_right, bx, badge_y, badge_w, 0.35, label_text,
                  color="#e1f5fe", border="#0288d1", fontsize=7.5, fontweight="bold", text_color="#01579b")

    ax_right.text(MID_X, badge_y - 0.3, "Isolation preserved",
                  ha="center", fontsize=10, fontweight="bold", color="#0277bd", style="italic")

    # Bandwidth result
    _draw_box(ax_right, 1.35, -0.55, 2.8, 0.65, "157\u2013261 GB/s",
              color="#c8e6c9", border="#2e7d32", fontsize=12, fontweight="bold", text_color="#1b5e20")

    fig.tight_layout(w_pad=4)
    out = ASSET_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: CUMEM Data Path Flow Diagram
# ---------------------------------------------------------------------------

def plot_cumem_datapath(filename: str) -> Path:
    """Horizontal flow diagram showing the cuMem FD export/import mechanism."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_xlim(-0.8, 15.5)
    ax.set_ylim(-1.0, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title("NCCL cuMem VMM Data Path: Cross-Container GPU Memory Sharing",
                 fontsize=15, fontweight="bold", pad=18)

    # ── Layout constants ──
    BOX_W, BOX_H = 3.2, 1.4
    c0_x, c1_x = 0.0, 10.0          # container column x
    mid_x = 5.0                       # middle column x
    top_y = 4.2                       # top row y
    bot_y = 1.8                       # bottom row y

    # ── Container header bars ──
    _draw_box(ax, c0_x, 5.8, BOX_W, 0.5, "Container 0  (GPU 0)",
              color="#fff3e0", border="#e65100", fontsize=10, fontweight="bold", text_color="#bf360c")
    _draw_box(ax, c1_x, 5.8, BOX_W, 0.5, "Container 1  (GPU 1)",
              color="#e3f2fd", border="#1565c0", fontsize=10, fontweight="bold", text_color="#0d47a1")

    # ── Step boxes ──
    steps = [
        # Container 0 steps (orange)
        (c0_x, top_y, "Step 1: cuMemCreate\nAllocate GPU memory\non GPU 0", "#fff3e0", "#e65100"),
        (c0_x, bot_y, "Step 2: cuMemExport\nToShareableHandle\n\u2192 POSIX file descriptor", "#fff3e0", "#e65100"),
        # Middle (green)
        (mid_x, (top_y + bot_y) / 2, "Step 3: sendmsg()\nSCM_RIGHTS\nAbstract Unix domain socket\n(shared network namespace)", "#e8f5e9", "#2e7d32"),
        # Container 1 steps (blue)
        (c1_x, top_y, "Step 4: cuMemImport\nFromShareableHandle\nReceive POSIX FD", "#e3f2fd", "#1565c0"),
        (c1_x, bot_y, "Step 5: cuMemMap +\ncuMemSetAccess\nMap remote GPU memory", "#e3f2fd", "#1565c0"),
    ]

    for sx, sy, label, fc, ec in steps:
        h = 1.7 if "Step 3" in label else BOX_H
        _draw_box(ax, sx, sy, BOX_W, h, label, color=fc, border=ec, fontsize=9, fontweight="bold")

    mid_step_h = 1.7
    mid_step_cy = (top_y + bot_y) / 2 + mid_step_h / 2

    # ── Arrows ──
    arrow_kw = dict(arrowstyle="-|>", color="#555", lw=2.2, mutation_scale=16)

    # Step 1 -> Step 2 (down)
    ax.annotate("", xy=(c0_x + BOX_W / 2, bot_y + BOX_H),
                xytext=(c0_x + BOX_W / 2, top_y),
                arrowprops=arrow_kw)

    # Step 2 -> Step 3 (right-up)
    ax.annotate("", xy=(mid_x, mid_step_cy - 0.3),
                xytext=(c0_x + BOX_W, bot_y + BOX_H / 2),
                arrowprops=arrow_kw)

    # Step 3 -> Step 4 (right-up)
    ax.annotate("", xy=(c1_x, top_y + BOX_H / 2),
                xytext=(mid_x + BOX_W, mid_step_cy - 0.3),
                arrowprops=arrow_kw)

    # Step 4 -> Step 5 (down)
    ax.annotate("", xy=(c1_x + BOX_W / 2, bot_y + BOX_H),
                xytext=(c1_x + BOX_W / 2, top_y),
                arrowprops=arrow_kw)

    # ── NVLink DMA bar at bottom ──
    nvl_y, nvl_h = -0.4, 0.8
    _draw_box(ax, 1.5, nvl_y, 10.2, nvl_h, "", color="#c8e6c9", border="#2e7d32")
    ax.annotate("", xy=(10.5, nvl_y + nvl_h / 2),
                xytext=(2.7, nvl_y + nvl_h / 2),
                arrowprops=dict(arrowstyle="<|-|>", color="#2e7d32", lw=4.5, mutation_scale=22))
    ax.text(6.6, nvl_y + nvl_h / 2, "Step 6: NVLink DMA  \u2014  GPU driver resolves physical topology",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1b5e20")

    # Arrow from step 5 down to NVLink
    ax.annotate("", xy=(c1_x + BOX_W / 2, nvl_y + nvl_h),
                xytext=(c1_x + BOX_W / 2, bot_y),
                arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=2, mutation_scale=15, linestyle="--"))

    # ── Key insight callout ──
    ax.text(14.0, 3.2,
            "Key insight:\n\nThe GPU driver\nknows the real\nNVLink topology\neven though each\ncontainer\u2019s CUDA\nruntime sees only\none GPU.",
            ha="center", va="center", fontsize=8.5, color="#444", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fafafa", edgecolor="#bbb", linewidth=1.2))

    fig.tight_layout()
    out = ASSET_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figures 3-6: Data plots (same as before)
# ---------------------------------------------------------------------------

def plot_bandwidth_by_size(data: dict, filename: str) -> Path:
    _setup_plot_style()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=False)
    fig.suptitle("NCCL Bandwidth vs Message Size \u2014 node192 (2\u00d7 A100-SXM4, NVLink NV12)",
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
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(OPERATIONS))
    n = len(CONFIGS)
    width = 0.8 / n
    offsets = [(i - (n - 1) / 2) * width for i in range(n)]

    for i, cfg in enumerate(CONFIGS):
        bw = [get_bandwidth(data, cfg["key"], op, "97.66MB") for op in OPERATIONS]
        bars = ax.bar(x + offsets[i], bw, width, label=cfg["short_label"],
                      color=cfg["color"], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, bw):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("NCCL Bandwidth at 100 MB \u2014 node192 (2\u00d7 A100-SXM4, NVLink NV12)",
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
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    non_baseline = [c for c in CONFIGS if c["key"] != "baseline"]
    x = np.arange(len(OPERATIONS))
    n = len(non_baseline)
    width = 0.8 / n
    offsets = [(i - (n - 1) / 2) * width for i in range(n)]

    for i, cfg in enumerate(non_baseline):
        pcts = []
        for op in OPERATIONS:
            base_bw = get_bandwidth(data, "baseline", op, "97.66MB")
            cfg_bw = get_bandwidth(data, cfg["key"], op, "97.66MB")
            pcts.append(cfg_bw / base_bw * 100 if base_bw > 0 else 0)
        bars = ax.bar(x + offsets[i], pcts, width, label=cfg["short_label"],
                      color=cfg["color"], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("% of Baseline Bandwidth")
    ax.set_title("Bandwidth Retained Under Isolation (100 MB) \u2014 vs P2P/NVLink Baseline",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([OP_LABELS[op] for op in OPERATIONS])
    ax.legend()
    ax.axhline(y=100, color="#2c2c2c", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylim(0, 115)

    fig.tight_layout()
    out = ASSET_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_latency_bars_100mb(data: dict, filename: str) -> Path:
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(OPERATIONS))
    n = len(CONFIGS)
    width = 0.8 / n
    offsets = [(i - (n - 1) / 2) * width for i in range(n)]

    for i, cfg in enumerate(CONFIGS):
        times = [get_time(data, cfg["key"], op, "97.66MB") for op in OPERATIONS]
        bars = ax.bar(x + offsets[i], times, width, label=cfg["short_label"],
                      color=cfg["color"], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, times):
            if val > 0.1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("NCCL Operation Latency at 100 MB \u2014 node192 (2\u00d7 A100-SXM4, NVLink NV12)",
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


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _swatch(color: str) -> str:
    return (f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{color};border:1px solid #999;margin-right:5px;'
            f'vertical-align:middle"></span>')


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html(data: dict, figures: dict[str, Path]) -> Path:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        commit = "unknown"

    from datetime import datetime, timezone
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def fig_html(key: str, alt: str, caption: str = "") -> str:
        if key not in figures:
            return ""
        rel = figures[key].relative_to(ROOT)
        cap = f'<p style="font-size:12px;color:#666;text-align:center;margin-top:4px">{caption}</p>' if caption else ""
        return f'''
      <div class="figure">
        <img src="{html.escape(str(rel))}" alt="{html.escape(alt)}">
        {cap}
      </div>'''

    # Results table
    def bw_table_100mb() -> str:
        rows = ""
        for op in OPERATIONS:
            base_bw = get_bandwidth(data, "baseline", op, "97.66MB")
            cumem_bw = get_bandwidth(data, "cumem_isolation", op, "97.66MB")
            shm_bw = get_bandwidth(data, "shm_isolation", op, "97.66MB")
            naive_bw = get_bandwidth(data, "naive_isolation", op, "97.66MB")
            cumem_pct = (cumem_bw / base_bw * 100) if base_bw > 0 else 0
            shm_pct = (shm_bw / base_bw * 100) if base_bw > 0 else 0
            naive_pct = (naive_bw / base_bw * 100) if base_bw > 0 else 0
            rows += (f"<tr><td>{OP_LABELS[op]}</td>"
                     f"<td>{base_bw:.2f}</td>"
                     f"<td style=\"background:#e8f5e9\"><b>{cumem_bw:.2f}</b></td>"
                     f"<td>{shm_bw:.2f}</td>"
                     f"<td>{naive_bw:.2f}</td>"
                     f"<td style=\"background:#e8f5e9\"><b>{cumem_pct:.1f}%</b></td>"
                     f"<td>{shm_pct:.1f}%</td>"
                     f"<td>{naive_pct:.1f}%</td></tr>\n")
        return rows

    def size_sweep_table(config_key: str) -> str:
        rows = ""
        for sk, sl in zip(SIZE_KEYS, SIZE_LABELS):
            cols = f"<td>{sl}</td>"
            for op in OPERATIONS:
                bw = get_bandwidth(data, config_key, op, sk)
                cols += f"<td>{bw:.2f}</td>"
            rows += f"<tr>{cols}</tr>\n"
        return rows

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
  <title>Experiment 2: Recovering NVLink Performance Under Per-GPU Container Isolation</title>
  <style>
    body {{
      margin: 0;
      font-family: "Times New Roman", "Nimbus Roman", Times, serif;
      background: #fff;
      color: #111;
      line-height: 1.6;
    }}
    main {{
      max-width: 920px;
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
    pre {{
      font-family: "Courier New", Courier, monospace;
      font-size: 12px;
      background: #f8f8f8;
      border: 1px solid #ddd;
      padding: 12px 16px;
      margin: 12px 0;
      overflow-x: auto;
      line-height: 1.5;
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
    .highlight-box {{
      background: #e8f5e9;
      border: 2px solid #2e7d32;
      border-radius: 6px;
      padding: 12px 16px;
      margin: 16px 0;
    }}
    .highlight-box p {{
      margin: 0;
    }}
    .warning-box {{
      background: #fff3e0;
      border: 2px solid #e65100;
      border-radius: 6px;
      padding: 12px 16px;
      margin: 16px 0;
    }}
    hr.appendix {{
      border: none;
      border-top: 2px solid #333;
      margin: 36px 0 24px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Experiment 2: Recovering NVLink Performance<br>Under Per-GPU Container Isolation</h1>
    <p class="subtitle">
      node192 &middot; 2&times; A100-SXM4-40GB &middot; NVLink NV12<br>
      Generated {generated} &middot; commit {html.escape(commit)}
    </p>

    <!-- ============================================================ -->
    <!-- SECTION 1: OBJECTIVE                                         -->
    <!-- ============================================================ -->

    <h2>1. Objective</h2>
    <p>Achieve <b>bare-metal NCCL NVLink bandwidth</b> while maintaining
       <b>per-GPU Docker container isolation</b> (one physical GPU per container,
       private PID/IPC/mount namespaces). The goal is to enable vLLM&rsquo;s
       DockerDistributedExecutor to run GPU workers in isolated containers without
       sacrificing the NVLink interconnect performance that tensor-parallel inference
       depends on.</p>

    <div class="highlight-box">
      <p><b>Result:</b> NCCL&rsquo;s cuMem VMM API (<code>NCCL_CUMEM_ENABLE=1</code>)
         with host networking recovers <b>99.5&ndash;100.1%</b> of the uncontainerized
         NVLink P2P baseline across all collective operations. No NCCL source code
         changes required.</p>
    </div>

    <!-- ============================================================ -->
    <!-- SECTION 2: THE PROBLEM                                       -->
    <!-- ============================================================ -->

    <h2>2. The Problem: Container Isolation Breaks NVLink P2P</h2>

    <p>When GPU workers run in separate containers with per-GPU
       <code>CUDA_VISIBLE_DEVICES</code>, each container sees only one GPU as
       <code>cuda:0</code>. NCCL&rsquo;s standard P2P transport relies on
       <code>cudaDeviceCanAccessPeer()</code> to verify GPU-to-GPU access, but
       this call fails because there is no visible peer device. Without P2P,
       NCCL falls back to slower transports:</p>

    <table>
      <thead>
        <tr><th>Fallback Transport</th><th>Data Path</th><th>100 MB Bandwidth</th><th>vs NVLink P2P</th></tr>
      </thead>
      <tbody>
        <tr><td><b>SHM/direct/direct</b></td><td>GPU &rarr; Host DRAM &rarr; GPU</td>
            <td>13&ndash;23 GB/s</td><td style="color:#c44e52"><b>9&ndash;12%</b></td></tr>
        <tr><td><b>NET/Socket (TCP)</b></td><td>GPU &rarr; CPU &rarr; TCP &rarr; CPU &rarr; GPU</td>
            <td>4&ndash;9 GB/s</td><td style="color:#c44e52"><b>2&ndash;5%</b></td></tr>
      </tbody>
    </table>

    <p>This represents a <b>7&ndash;12&times; bandwidth degradation</b> on NVLink
       hardware &mdash; making per-GPU container isolation impractical for
       tensor-parallel workloads without a recovery mechanism.</p>

    {fig_html('architecture', 'Architecture: SHM fallback vs CUMEM P2P recovery',
              'Figure 1: Per-GPU container isolation forces data through host DRAM (left). '
              'The cuMem VMM API recovers NVLink DMA by exporting GPU memory as POSIX FDs '
              'passed via Unix domain sockets (right).')}

    <!-- ============================================================ -->
    <!-- SECTION 3: THE SOLUTION                                      -->
    <!-- ============================================================ -->

    <h2>3. The Solution: NCCL cuMem VMM API</h2>

    <p>NCCL 2.x includes an alternative GPU memory sharing path based on the
       <b>CUDA Virtual Memory Management (VMM) API</b>, activated by setting
       <code>NCCL_CUMEM_ENABLE=1</code>. Unlike traditional CUDA IPC (which
       requires both GPUs visible in one CUDA context), the cuMem path exports
       GPU memory as <b>POSIX file descriptors</b> that can be passed between
       processes via Unix domain sockets. The NVIDIA driver resolves the physical
       NVLink topology for DMA, regardless of CUDA runtime GPU visibility.</p>

    <h3>3.1 How P2P Transport Selection Succeeds</h3>

    <p>Despite each container seeing only one GPU, NCCL&rsquo;s
       <code>p2pCanConnect()</code> in <code>p2p.cc</code> does not reject the
       P2P transport. Tracing the decision path:</p>

    <ol>
      <li><b><code>ncclTopoCheckP2p()</code></b> (paths.cc) &mdash; checks two
          conditions: (a) <code>hostHash</code> matches (same hostname via host
          networking), and (b) <code>shmDev</code> matches (same <code>/dev/shm</code>
          device number via shared volume). Both pass &rarr; P2P is <b>not rejected</b>
          at the topology level.</li>
      <li><b><code>busIdToCudaDev()</code></b> (p2p.cc:156) &mdash; tries to convert the
          peer GPU&rsquo;s PCI bus ID to a CUDA device index. Returns <code>-1</code>
          because the peer GPU is not visible. However, on <b>CUDA &ge;10.1</b>
          (we have 12.4), the code simply <b>returns without disabling P2P</b>
          (lines 159&ndash;161). This is the critical code path.</li>
      <li><b><code>cudaDeviceCanAccessPeer()</code></b> (p2p.cc:170) &mdash;
          <b>never reached</b> because step 2 returned early when
          <code>busIdToCudaDev</code> failed.</li>
      <li><b>P2P transport is selected.</b> At setup time,
          <code>ncclCuMemEnable()</code> detects that the cuMem VMM API is available
          and selects the <code>P2P_CUMEM</code> subtype, which uses POSIX FD-based
          memory sharing instead of CUDA IPC handles.</li>
    </ol>

    <h3>3.2 The cuMem Data Path</h3>

    <p>Once P2P/CUMEM transport is selected, NCCL establishes GPU memory sharing
       through a six-step process:</p>

    {fig_html('datapath', 'cuMem VMM data path flow diagram',
              'Figure 2: The cuMem VMM data path. GPU memory is exported as a POSIX file '
              'descriptor, passed to the peer container via Unix domain socket SCM_RIGHTS, '
              'then imported and mapped. The GPU driver handles NVLink DMA.')}

    <ol>
      <li><b><code>cuMemCreate()</code></b> &mdash; allocates GPU memory using the
          VMM API on the local GPU (e.g., GPU 0 in Container 0).</li>
      <li><b><code>cuMemExportToShareableHandle()</code></b> with
          <code>CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR</code> &mdash; exports the
          GPU memory allocation as a <b>POSIX file descriptor</b>. This FD represents
          the physical GPU memory, not a CUDA runtime handle.</li>
      <li><b><code>sendmsg()</code> with <code>SCM_RIGHTS</code></b> &mdash; NCCL&rsquo;s
          proxy thread sends the FD to the peer container&rsquo;s proxy thread over an
          <b>abstract Unix domain socket</b> (SOCK_DGRAM). The <code>SCM_RIGHTS</code>
          ancillary message passes the file descriptor across process boundaries.</li>
      <li><b><code>cuMemImportFromShareableHandle()</code></b> &mdash; the peer
          container receives the FD and imports it as a cuMem allocation handle.</li>
      <li><b><code>cuMemMap()</code> + <code>cuMemSetAccess()</code></b> &mdash;
          maps the imported GPU memory into the local virtual address space and grants
          the local GPU access permissions.</li>
      <li><b>NVLink DMA</b> &mdash; the NVIDIA driver knows the real physical topology.
          When the local GPU accesses the mapped memory, the driver routes the DMA
          transfer over the NVLink interconnect, achieving full hardware bandwidth.</li>
    </ol>

    <h3>3.3 Why Shared Network Namespace Is Required</h3>

    <p>NCCL&rsquo;s IPC socket implementation (<code>ipcsocket.cc</code>) uses
       <b>abstract Unix domain sockets</b> by default
       (<code>NCCL_PARAM(IpcUseAbstractSocket, "IPC_USE_ABSTRACT_SOCKET", 1)</code>).
       Abstract sockets are created by setting <code>sun_path[0] = '\\0'</code> in the
       <code>sockaddr_un</code> structure. In Linux, abstract sockets are bound to the
       <b>network namespace</b>, not the filesystem &mdash; meaning an abstract socket
       created in one network namespace is invisible to processes in another.</p>

    <p>With Docker bridge networking, each container has a <b>separate network
       namespace</b>. When Container 1&rsquo;s NCCL proxy tries to <code>sendmsg()</code>
       to Container 0&rsquo;s abstract socket, it gets <code>ECONNREFUSED</code> (errno 111)
       because the socket literally does not exist in its namespace.</p>

    <p>The solution is <code>network_mode: host</code>, which places both containers
       in the <b>host&rsquo;s network namespace</b>. Abstract UDS sockets are then
       visible across containers. This does not share IPC, PID, or mount namespaces
       &mdash; those remain private.</p>

    <h3>3.4 Required Docker Configuration</h3>

    <pre>services:
  rank0:
    image: gpu-comm-benchmark:latest
    runtime: nvidia
    network_mode: host                  # Shared network NS for abstract UDS
    environment:
      - NVIDIA_VISIBLE_DEVICES=all      # NVML topology discovery
      - CUDA_VISIBLE_DEVICES=0          # Per-GPU compute isolation
      - NCCL_CUMEM_ENABLE=1             # Enable cuMem VMM API
    volumes:
      - shared-shm:/dev/shm            # shmDev match for same-node detection
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']    # Both GPUs to NVML
              capabilities: [gpu]</pre>

    <h3>3.5 Isolation Properties</h3>

    <table>
      <thead>
        <tr><th>Namespace / Resource</th>
            <th>Baseline<br>(No Isolation)</th>
            <th>CUMEM Isolation<br>(This Solution)</th>
            <th>SHM Isolation<br>(Fallback)</th></tr>
      </thead>
      <tbody>
        <tr><td><b>PID namespace</b></td>
            <td>Host (shared)</td>
            <td style="background:#e8f5e9"><b>Private</b></td>
            <td>Private</td></tr>
        <tr><td><b>IPC namespace</b></td>
            <td>Host (shared)</td>
            <td style="background:#e8f5e9"><b>Private</b></td>
            <td>Private</td></tr>
        <tr><td><b>Mount namespace</b></td>
            <td>Host (shared)</td>
            <td style="background:#e8f5e9"><b>Private</b></td>
            <td>Private</td></tr>
        <tr><td><b>Network namespace</b></td>
            <td>Host</td>
            <td>Host</td>
            <td style="background:#e8f5e9">Bridge (private)</td></tr>
        <tr><td><b>GPU (CUDA runtime)</b></td>
            <td>All visible</td>
            <td style="background:#e8f5e9"><b>1 per container</b></td>
            <td>1 per container</td></tr>
        <tr><td><b>Process visibility</b></td>
            <td>All processes visible</td>
            <td style="background:#e8f5e9"><b>Container-only</b></td>
            <td>Container-only</td></tr>
        <tr><td><b>NCCL bandwidth</b></td>
            <td>157&ndash;261 GB/s</td>
            <td style="background:#e8f5e9"><b>157&ndash;260 GB/s</b></td>
            <td>13&ndash;23 GB/s</td></tr>
      </tbody>
    </table>

    <p>The CUMEM isolation configuration provides <b>stronger isolation than the
       baseline</b> (private PID, IPC, mount namespaces; per-GPU CUDA visibility)
       while matching baseline NVLink bandwidth. The network namespace is shared,
       but this is the same as the baseline configuration.</p>

    <!-- ============================================================ -->
    <!-- SECTION 4: EXPERIMENTAL SETUP                                -->
    <!-- ============================================================ -->

    <h2>4. Experimental Setup</h2>
    <table class="config-table">
      <tr><td>Machine</td><td>node192 (10.0.2.192)</td></tr>
      <tr><td>GPUs</td><td>2&times; NVIDIA A100-SXM4-40GB (GPU 0, GPU 1)</td></tr>
      <tr><td>GPU Interconnect</td><td>NV12 (12&times; NVLink, ~600 GB/s bidirectional theoretical)</td></tr>
      <tr><td>NCCL</td><td>2.21.5 (bundled with PyTorch 2.5.1+cu124)</td></tr>
      <tr><td>CUDA</td><td>12.4 (driver 12.6)</td></tr>
      <tr><td>Docker</td><td>27.3.1 with nvidia runtime</td></tr>
      <tr><td>Benchmark</td><td>Custom PyTorch NCCL benchmark: all-reduce, all-gather, reduce-scatter, broadcast, P2P send/recv</td></tr>
      <tr><td>Message sizes</td><td>1 KB &ndash; 100 MB (6 sizes, log-spaced)</td></tr>
      <tr><td>Timing</td><td>CUDA events, 100 iterations, trimmed mean (drop top/bottom 5%)</td></tr>
    </table>

    <!-- ============================================================ -->
    <!-- SECTION 5: CONFIGURATIONS                                    -->
    <!-- ============================================================ -->

    <h2>5. Configurations Under Test</h2>
    <p>Four Docker Compose configurations test the isolation spectrum on node192:</p>
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

    <!-- ============================================================ -->
    <!-- SECTION 6: RESULTS                                           -->
    <!-- ============================================================ -->

    <h2>6. Results</h2>

    <h3>6.1 Bandwidth at 100 MB</h3>
    <table>
      <thead>
        <tr><th>Operation</th>
            <th>{_swatch(CONFIGS[0]['color'])}Baseline</th>
            <th>{_swatch(CONFIGS[1]['color'])}CUMEM</th>
            <th>{_swatch(CONFIGS[2]['color'])}SHM</th>
            <th>{_swatch(CONFIGS[3]['color'])}TCP</th>
            <th style="background:#e8f5e9">CUMEM/Base</th>
            <th>SHM/Base</th>
            <th>TCP/Base</th></tr>
      </thead>
      <tbody>
        {bw_table_100mb()}
      </tbody>
    </table>
    <p style="font-size:12px;color:#666">All bandwidth values in GB/s. Percentages show fraction of baseline retained.</p>

    {fig_html('bars_100mb', 'Bandwidth at 100 MB',
              'Figure 3: NCCL bandwidth at 100 MB message size. CUMEM isolation (green) matches '
              'the baseline, while SHM and TCP fall far behind.')}

    <h3>6.2 Bandwidth vs Message Size</h3>
    {fig_html('bw_by_size', 'Bandwidth vs message size',
              'Figure 4: Bandwidth scaling across message sizes. Baseline and CUMEM lines overlap '
              'at all sizes, confirming full NVLink recovery.')}

    <h3>6.3 Bandwidth Recovery Relative to Baseline</h3>
    {fig_html('degradation', 'Bandwidth retained under isolation',
              'Figure 5: Percentage of baseline bandwidth retained. CUMEM achieves 99.5\u2013100.1% '
              'across all operations.')}

    <h3>6.4 Latency at 100 MB</h3>
    {fig_html('latency', 'Latency at 100 MB',
              'Figure 6: Operation latency. CUMEM latency matches baseline; SHM and TCP show '
              '7\u201312\u00d7 and 15\u201340\u00d7 higher latency respectively.')}

    <!-- ============================================================ -->
    <!-- SECTION 7: CONCLUSIONS                                       -->
    <!-- ============================================================ -->

    <h2>7. Conclusions</h2>

    <ol>
      <li><b>Full NVLink bandwidth recovery:</b> NCCL&rsquo;s cuMem VMM API
          (<code>NCCL_CUMEM_ENABLE=1</code>) achieves <b>99.5&ndash;100.1%</b> of the
          uncontainerized P2P/NVLink baseline across all NCCL collective operations,
          with per-GPU container isolation. No NCCL source code modifications are
          needed.</li>

      <li><b>Mechanism:</b> The cuMem path exports GPU memory as POSIX file
          descriptors via <code>cuMemExportToShareableHandle</code>, passes them
          between containers via abstract Unix domain sockets (<code>SCM_RIGHTS</code>),
          and the peer imports and maps the memory with <code>cuMemMap</code> +
          <code>cuMemSetAccess</code>. The NVIDIA driver routes DMA over NVLink
          regardless of CUDA runtime GPU visibility.</li>

      <li><b>Required configuration:</b>
          <code>NCCL_CUMEM_ENABLE=1</code>,
          <code>NVIDIA_VISIBLE_DEVICES=all</code>,
          <code>network_mode: host</code>,
          shared <code>/dev/shm</code> volume.
          Per-GPU <code>CUDA_VISIBLE_DEVICES</code> provides compute isolation.</li>

      <li><b>Stronger isolation than baseline:</b> The CUMEM configuration provides
          private PID, IPC, and mount namespaces (which the baseline does not), while
          maintaining identical NVLink bandwidth.</li>

      <li><b>Without CUMEM:</b> Per-GPU isolation causes 88&ndash;91% bandwidth loss.
          The best environment-variable-only fallback (SHM) recovers to ~10% of
          baseline; naive TCP achieves only 2&ndash;5%.</li>
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ASSET_DIR.mkdir(exist_ok=True)

    print("Loading results...")
    data = load_results()
    if len(data) < 4:
        print(f"  WARNING: Only {len(data)}/4 result files found")
        if not data:
            print("  No results to process!")
            return

    print("Generating figures...")
    figures: dict[str, Path] = {}

    figures["architecture"] = plot_architecture_diagram("architecture_diagram.png")
    print(f"  {figures['architecture']}")

    figures["datapath"] = plot_cumem_datapath("cumem_datapath.png")
    print(f"  {figures['datapath']}")

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
