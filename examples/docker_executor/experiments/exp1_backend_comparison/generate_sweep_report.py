#!/usr/bin/env python3
"""Generate plots and an HTML report for the exp1 multi-model sweep.

Supports multiple nodes with different directory layouts:
  - node192/<model>/<variant>/<bench_file>   (new layout)
  - node196/<model>/<variant>/<bench_file>   (reorganized)
"""

from __future__ import annotations

import csv
import html
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "analysis_assets"
HTML_REPORT = ROOT / "analysis_report.html"
CSV_REPORT = ROOT / "analysis_metrics.csv"

MODELS = [
    ("qwen3_4b", "Qwen3-4B"),
    ("qwen3_8b", "Qwen3-8B"),
    ("qwen3_14b", "Qwen3-14B"),
]

NODES = [
    {
        "key": "node192",
        "label": "node192 (A100-SXM4, NVLink NV12)",
        "gpu_info": "2x A100-SXM4-40GB, GPU 0,1, NVLink NV12",
        "dir": "node192",
        "tp_sizes": [1, 2],
        "variants": [
            {
                "key": "baseline",
                "label": "Baseline Docker+MP",
                "dir": "baseline",
                "color": "#2c2c2c",
                "tp_files": {
                    1: "3_docker_mp_tp1_bench.txt",
                    2: "4_docker_mp_tp2_bench.txt",
                },
            },
            {
                "key": "dockerbe_sync_output",
                "label": "DockerBE Sync Output",
                "dir": "dockerbe_sync_output",
                "color": "#c44e52",
                "tp_files": {
                    1: "5_host_dockerbe_tp1_bench.txt",
                    2: "6_host_dockerbe_tp2_bench.txt",
                },
            },
            {
                "key": "dockerbe_hybrid_shm",
                "label": "DockerBE Hybrid SHM",
                "dir": "dockerbe_hybrid_shm",
                "color": "#8da0cb",
                "tp_files": {
                    1: "5_host_dockerbe_tp1_bench.txt",
                    2: "6_host_dockerbe_tp2_bench.txt",
                },
            },
            {
                "key": "dockerbe_single_visible_gpu",
                "label": "DockerBE Single GPU/Node",
                "dir": "dockerbe_single_visible_gpu",
                "color": "#e5ae38",
                "tp_files": {
                    1: "8_host_dockerbe_single_visible_gpu_tp1_bench.txt",
                    2: "9_host_dockerbe_single_visible_gpu_tp2_bench.txt",
                },
            },
            {
                "key": "dockerbe_full_shm",
                "label": "DockerBE Full SHM",
                "dir": "dockerbe_full_shm",
                "color": "#66c2a5",
                "tp_files": {
                    1: "5_host_dockerbe_tp1_bench.txt",
                    2: "6_host_dockerbe_tp2_bench.txt",
                },
            },
        ],
    },
    {
        "key": "node196",
        "label": "node196 (A100-PCIE, PCIe only)",
        "gpu_info": "2x A100-PCIE-40GB, GPU 1,2, PCIe 4.0",
        "dir": "node196",
        "tp_sizes": [1, 2, 4],
        "variants": [
            {
                "key": "baseline",
                "label": "Baseline Docker+MP",
                "dir": "baseline",
                "color": "#2c2c2c",
                "tp_files": {
                    1: "3_docker_mp_tp1_bench.txt",
                    2: "4_docker_mp_tp2_bench.txt",
                    4: "5_docker_mp_tp4_bench.txt",
                },
            },
            {
                "key": "dockerbe_sync_output",
                "label": "DockerBE Sync Output",
                "dir": "dockerbe_sync_output",
                "color": "#c44e52",
                "tp_files": {
                    1: "5_host_dockerbe_tp1_bench.txt",
                    2: "6_host_dockerbe_tp2_bench.txt",
                    4: "7_host_dockerbe_tp4_bench.txt",
                },
            },
            {
                "key": "dockerbe_hybrid_shm",
                "label": "DockerBE Hybrid SHM",
                "dir": "dockerbe_hybrid_shm",
                "color": "#8da0cb",
                "tp_files": {
                    1: "5_host_dockerbe_tp1_bench.txt",
                    2: "6_host_dockerbe_tp2_bench.txt",
                    4: "7_host_dockerbe_tp4_bench.txt",
                },
            },
            {
                "key": "dockerbe_single_visible_gpu",
                "label": "DockerBE Single GPU/Node",
                "dir": "dockerbe_single_visible_gpu",
                "color": "#e5ae38",
                "tp_files": {
                    1: "8_host_dockerbe_single_visible_gpu_tp1_bench.txt",
                    2: "9_host_dockerbe_single_visible_gpu_tp2_bench.txt",
                    4: "10_host_dockerbe_single_visible_gpu_tp4_bench.txt",
                },
            },
            {
                "key": "dockerbe_full_shm",
                "label": "DockerBE Full SHM",
                "dir": "dockerbe_full_shm",
                "color": "#66c2a5",
                "tp_files": {
                    1: "5_host_dockerbe_tp1_bench.txt",
                    2: "6_host_dockerbe_tp2_bench.txt",
                    4: "7_host_dockerbe_tp4_bench.txt",
                },
            },
        ],
    },
]

METRIC_KEYS = {
    "Successful requests": "successful_requests",
    "Failed requests": "failed_requests",
    "Benchmark duration (s)": "benchmark_duration_s",
    "Request throughput (req/s)": "request_throughput_rps",
    "Output token throughput (tok/s)": "output_token_throughput_tps",
    "Peak concurrent requests": "peak_concurrent_requests",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "Median TTFT (ms)": "median_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "Median TPOT (ms)": "median_tpot_ms",
    "P99 TPOT (ms)": "p99_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "Median ITL (ms)": "median_itl_ms",
    "P99 ITL (ms)": "p99_itl_ms",
}


@dataclass
class Record:
    node_key: str
    model_slug: str
    model_label: str
    variant_key: str
    variant_label: str
    color: str
    tp_size: int
    bench_path: Path
    metrics: dict[str, float]


def parse_bench_file(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8").strip()

    # JSON key -> internal key mapping (vllm bench JSON uses different names)
    json_key_map = {
        "completed": "successful_requests",
        "failed": "failed_requests",
        "duration": "benchmark_duration_s",
        "request_throughput": "request_throughput_rps",
        "output_throughput": "output_token_throughput_tps",
        "max_concurrent_requests": "peak_concurrent_requests",
    }

    # Try JSON format first (vllm bench serve --save-result output)
    if text.startswith("{"):
        try:
            data = json.loads(text)
            metrics: dict[str, float] = {}
            for internal_key in METRIC_KEYS.values():
                if internal_key in data:
                    metrics[internal_key] = float(data[internal_key])
            # Also check mapped keys
            for json_key, internal_key in json_key_map.items():
                if json_key in data and internal_key not in metrics:
                    metrics[internal_key] = float(data[json_key])
            if metrics:
                return metrics
        except (json.JSONDecodeError, ValueError):
            pass

    # Fall back to text format (stdout capture)
    metrics = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key not in METRIC_KEYS:
            continue
        try:
            metrics[METRIC_KEYS[key]] = float(raw.strip())
        except ValueError:
            continue
    return metrics


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT.parents[3]), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_node_records(node: dict) -> tuple[list[Record], dict[int, list[tuple[str, str]]]]:
    """Load records for a single node, auto-detecting available models."""
    node_dir = ROOT / node["dir"]
    records: list[Record] = []
    available: dict[int, list[tuple[str, str]]] = {}

    for tp_size in node["tp_sizes"]:
        model_rows: list[tuple[str, str]] = []
        for model_slug, model_label in MODELS:
            has_all = all(
                (node_dir / model_slug / v["dir"] / v["tp_files"][tp_size]).exists()
                for v in node["variants"]
            )
            if has_all:
                model_rows.append((model_slug, model_label))
        if model_rows:
            available[tp_size] = model_rows

    for tp_size in sorted(available):
        for model_slug, model_label in available[tp_size]:
            for variant in node["variants"]:
                bench_path = node_dir / model_slug / variant["dir"] / variant["tp_files"][tp_size]
                records.append(Record(
                    node_key=node["key"],
                    model_slug=model_slug,
                    model_label=model_label,
                    variant_key=variant["key"],
                    variant_label=variant["label"],
                    color=variant["color"],
                    tp_size=tp_size,
                    bench_path=bench_path,
                    metrics=parse_bench_file(bench_path),
                ))

    return records, available


def records_by_key(records: list[Record]) -> dict[tuple[str, int, str], Record]:
    return {
        (r.model_slug, r.tp_size, r.variant_key): r
        for r in records
    }


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def variant_offsets(count: int) -> list[float]:
    center = (count - 1) / 2
    return [i - center for i in range(count)]


def _setup_plot_style() -> None:
    """Configure matplotlib for formal report style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#ddd",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
    })


def annotate_bars(ax: Any, bars: list[Any], precision: int) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{precision}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7, rotation=90, color="#333",
        )


def plot_absolute_metric(
    records: list[Record],
    available: dict[int, list[tuple[str, str]]],
    variants: list[dict],
    metric_name: str,
    ylabel: str,
    title: str,
    filename: str,
    precision: int,
) -> Path:
    _setup_plot_style()
    tp_sizes = sorted(available)
    fig, axes = plt.subplots(1, len(tp_sizes), figsize=(5.5 * len(tp_sizes), 4.2),
                             sharey=False, squeeze=False)
    width = 0.72 / len(variants)
    offsets = variant_offsets(len(variants))
    record_map = records_by_key(records)

    for ax, tp_size in zip(axes[0], tp_sizes):
        models = available[tp_size]
        model_labels = [label for _, label in models]
        centers = list(range(len(models)))
        ax.set_axisbelow(True)

        for offset, variant in zip(offsets, variants):
            values = [
                record_map[(ms, tp_size, variant["key"])].metrics[metric_name]
                for ms, _ in models
            ]
            positions = [c + offset * width for c in centers]
            bars = ax.bar(positions, values, width=width, label=variant["label"],
                          color=variant["color"], edgecolor="#333", linewidth=0.5)
            annotate_bars(ax, list(bars), precision)

        ax.set_title(f"TP={tp_size}", fontsize=12)
        ax.set_xticks(centers)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel(ylabel)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=min(len(variants), 5), frameon=True, edgecolor="#ccc",
               fancybox=False, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    output_path = ASSET_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_deltas(
    records: list[Record],
    available: dict[int, list[tuple[str, str]]],
    variants: list[dict],
    filename: str,
) -> Path:
    _setup_plot_style()
    tp_sizes = sorted(available)
    delta_variants = [v for v in variants if v["key"] != "baseline"]
    fig, axes = plt.subplots(2, len(tp_sizes), figsize=(5.5 * len(tp_sizes), 7),
                             sharex=True, squeeze=False)
    width = 0.72 / len(delta_variants)
    offsets = variant_offsets(len(delta_variants))
    record_map = records_by_key(records)

    plot_specs = [
        ("median_tpot_ms", "TPOT delta vs baseline (ms)", 0, 2),
        ("output_token_throughput_tps", "Throughput delta vs baseline (tok/s)", 1, 1),
    ]

    for col, tp_size in enumerate(tp_sizes):
        models = available[tp_size]
        model_labels = [label for _, label in models]
        centers = list(range(len(models)))
        for metric_name, ylabel, row, precision in plot_specs:
            ax = axes[row][col]
            ax.set_axisbelow(True)
            baselines = [
                record_map[(ms, tp_size, "baseline")].metrics[metric_name]
                for ms, _ in models
            ]

            for offset, variant in zip(offsets, delta_variants):
                values = [
                    record_map[(ms, tp_size, variant["key"])].metrics[metric_name] - baselines[i]
                    for i, (ms, _) in enumerate(models)
                ]
                positions = [c + offset * width for c in centers]
                bars = ax.bar(positions, values, width=width, label=variant["label"],
                              color=variant["color"], edgecolor="#333", linewidth=0.5)
                annotate_bars(ax, list(bars), precision)

            ax.axhline(0.0, color="#333", linewidth=0.8)
            ax.set_title(f"TP={tp_size}", fontsize=12)
            ax.set_xticks(centers)
            ax.set_xticklabels(model_labels)
            ax.set_ylabel(ylabel)
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin
            ax.set_ylim(ymin - span * 0.06, ymax + span * 0.12)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Delta Relative to Baseline", fontsize=13, y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=min(len(delta_variants), 4), frameon=True, edgecolor="#ccc",
               fancybox=False, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    output_path = ASSET_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_cross_node(
    node_data: list[tuple[dict, list[Record], dict[int, list[tuple[str, str]]]]],
    filename: str,
) -> Path | None:
    """Bar chart comparing baseline and full_shm TPOT across nodes for shared models."""
    _setup_plot_style()
    shared_model = "qwen3_8b"
    shared_label = "Qwen3-8B"

    node_records = {}
    for node, records, available in node_data:
        rmap = records_by_key(records)
        node_records[node["key"]] = rmap

    # Check both nodes have Qwen3-8B TP=2
    for nk in node_records:
        if (shared_model, 2, "baseline") not in node_records[nk]:
            return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_axisbelow(True)

    groups = []
    values = []
    colors = []

    for tp in [1, 2]:
        for node, _, _ in node_data:
            nk = node["key"]
            rmap = node_records[nk]
            if (shared_model, tp, "baseline") not in rmap:
                continue
            bl = rmap[(shared_model, tp, "baseline")].metrics["median_tpot_ms"]
            fs = rmap[(shared_model, tp, "dockerbe_full_shm")].metrics["median_tpot_ms"]
            short = node["key"].replace("node", "n")
            groups.extend([f"{short} BL TP{tp}", f"{short} FS TP{tp}"])
            values.extend([bl, fs])
            colors.extend(["#555", "#999"])

    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="#333", linewidth=0.5)
    annotate_bars(ax, list(bars), 2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Median TPOT (ms)")
    ax.set_title(f"Cross-Node Comparison: {shared_label}", fontsize=13)

    # Manual legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#555", edgecolor="#333", label="Baseline"),
                       Patch(facecolor="#999", edgecolor="#333", label="Full SHM")],
              frameon=True, edgecolor="#ccc", fancybox=False, framealpha=0.9)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)
    fig.tight_layout()

    output_path = ASSET_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_csv(all_records: list[Record]) -> Path:
    fieldnames = [
        "node", "model_slug", "model_label", "variant_key", "variant_label",
        "tp_size", "successful_requests", "failed_requests",
        "benchmark_duration_s", "request_throughput_rps",
        "output_token_throughput_tps", "peak_concurrent_requests",
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "p99_itl_ms", "bench_path",
    ]
    with CSV_REPORT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            row = {
                "node": r.node_key, "model_slug": r.model_slug,
                "model_label": r.model_label, "variant_key": r.variant_key,
                "variant_label": r.variant_label, "tp_size": r.tp_size,
                "bench_path": str(r.bench_path),
            }
            row.update(r.metrics)
            writer.writerow(row)
    return CSV_REPORT


def table_rows_html(records: list[Record], available: dict[int, list[tuple[str, str]]],
                    variants: list[dict]) -> str:
    record_map = records_by_key(records)
    rows: list[str] = []
    for model_slug, model_label in MODELS:
        for tp_size in sorted(available):
            if model_slug not in {s for s, _ in available[tp_size]}:
                continue
            baseline = record_map[(model_slug, tp_size, "baseline")]
            for variant in variants:
                key = (model_slug, tp_size, variant["key"])
                if key not in record_map:
                    continue
                r = record_map[key]
                bl_tpot = baseline.metrics["median_tpot_ms"]
                bl_tps = baseline.metrics["output_token_throughput_tps"]
                delta_tpot = r.metrics["median_tpot_ms"] - bl_tpot
                delta_tps = r.metrics["output_token_throughput_tps"] - bl_tps
                pct = (delta_tpot / bl_tpot * 100) if bl_tpot else 0

                if variant["key"] == "baseline":
                    delta_cell = "&mdash;"
                elif abs(pct) < 1:
                    delta_cell = f'{delta_tpot:+.2f} ({pct:+.1f}%)'
                elif pct < 5:
                    delta_cell = f'{delta_tpot:+.2f} ({pct:+.1f}%)'
                else:
                    delta_cell = f'<b>{delta_tpot:+.2f} ({pct:+.1f}%)</b>'

                rows.append(
                    "<tr>"
                    f"<td>{html.escape(model_label)}</td>"
                    f"<td>{tp_size}</td>"
                    f"<td><span style=\"display:inline-block;width:10px;height:10px;background:{variant['color']};border:1px solid #999;margin-right:5px;vertical-align:middle\"></span>{html.escape(variant['label'])}</td>"
                    f"<td>{r.metrics['median_tpot_ms']:.2f}</td>"
                    f"<td>{delta_cell}</td>"
                    f"<td>{r.metrics['output_token_throughput_tps']:.1f}</td>"
                    f"<td>{delta_tps:+.1f}</td>"
                    f"<td>{r.metrics['median_ttft_ms']:.1f}</td>"
                    f"<td>{r.metrics['median_itl_ms']:.2f}</td>"
                    f"<td>{r.metrics['successful_requests']:.0f}/{r.metrics['failed_requests']:.0f}</td>"
                    "</tr>"
                )
    return "\n".join(rows)


def generate_html(
    node_data: list[tuple[dict, list[Record], dict[int, list[tuple[str, str]]]]],
    figures: dict[str, dict[str, Path]],
    cross_node_fig: Path | None,
) -> Path:
    commit = git_commit()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build results sections (figures only, no tables)
    results_sections = []
    appendix_sections = []
    for node, records, available in node_data:
        if not records:
            continue
        node_key = node["key"]
        figs = figures.get(node_key, {})

        tp_sizes = sorted(available)
        seen: set[str] = set()
        unique_models: list[str] = []
        for tp in tp_sizes:
            for _, label in available[tp]:
                if label not in seen:
                    seen.add(label)
                    unique_models.append(label)
        models_str = ", ".join(unique_models)

        table = table_rows_html(records, available, node["variants"])

        fig_html = ""
        for fig_key, fig_title in [("tpot", "Median TPOT"), ("throughput", "Output Throughput"), ("delta", "Delta vs Baseline")]:
            if fig_key in figs:
                rel = figs[fig_key].relative_to(ROOT)
                fig_html += f"""
      <div class="figure">
        <img src="{html.escape(str(rel))}" alt="{fig_title}">
      </div>"""

        results_sections.append(f"""
      <h3 id="{node_key}">{html.escape(node['label'])}</h3>
      <p><b>Hardware:</b> {html.escape(node['gpu_info'])}<br>
         <b>Models:</b> {html.escape(models_str)}<br>
         <b>TP sizes:</b> {', '.join(str(t) for t in tp_sizes)}</p>
      {fig_html}""")

        appendix_sections.append(f"""
      <h3>{html.escape(node['label'])} &mdash; Per-Run Metrics</h3>
      <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Model</th><th>TP</th><th>Variant</th>
            <th>Med TPOT (ms)</th><th>TPOT Delta</th>
            <th>Output tok/s</th><th>tok/s Delta</th>
            <th>Med TTFT (ms)</th><th>Med ITL (ms)</th><th>Succ/Fail</th>
          </tr>
        </thead>
        <tbody>
          {table}
        </tbody>
      </table>
      </div>""")

    cross_node_html = ""
    if cross_node_fig:
        rel = cross_node_fig.relative_to(ROOT)
        cross_node_html = f"""
      <h3>Cross-Node Comparison</h3>
      <p>Qwen3-8B baseline and full_shm TPOT compared across nodes.
         At TP=1 (no inter-GPU communication) both nodes perform identically,
         confirming the TP=2 difference is purely from interconnect speed (NVLink vs PCIe).</p>
      <div class="figure">
        <img src="{html.escape(str(rel))}" alt="Cross-node comparison">
      </div>"""

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Experiment 1: DockerDistributedExecutor Backend Comparison</title>
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
  </style>
</head>
<body>
  <main>
    <h1>Experiment 1: DockerDistributedExecutor Backend Comparison</h1>
    <p class="subtitle">
      Generated {generated} &middot; commit {html.escape(commit)}
    </p>

    <h2>1. Objective</h2>
    <p>Quantify the serving-latency overhead introduced by <code>DockerDistributedExecutor</code>
       compared to the baseline configuration (vLLM running inside a single Docker container
       with the multiprocess backend). The experiment isolates three potential overhead sources:
       synchronous output copying, SHM-vs-TCP worker response transport, and per-GPU container
       isolation (single visible GPU per container).</p>

    <h2>2. Configurations Under Test</h2>
    <p>Five backend configurations are compared. All use the same Docker image built from one
       source tree; only runtime environment variables differ.</p>
    <table>
      <thead>
        <tr><th>Variant</th><th>Server Placement</th><th>GPU Visibility</th>
            <th>IPC Namespace</th><th>Output Copy</th><th>MQ Transport</th><th>NCCL Transport</th></tr>
      </thead>
      <tbody>
        <tr><td><b>Baseline Docker+MP</b></td><td>Inside Docker</td><td>All GPUs per container</td>
            <td>Host</td><td>Async</td><td>SHM</td><td>NVLink P2P / PCIe P2P</td></tr>
        <tr><td><b>DockerBE Sync Output</b></td><td>Host</td><td>All GPUs per container</td>
            <td>Host</td><td>Synchronous</td><td>SHM</td><td>NVLink P2P / PCIe P2P</td></tr>
        <tr><td><b>DockerBE Hybrid SHM</b></td><td>Host</td><td>All GPUs per container</td>
            <td>Host</td><td>Async</td><td>SHM broadcast, TCP response</td><td>NVLink P2P / PCIe P2P</td></tr>
        <tr><td><b>DockerBE Single GPU/Node</b></td><td>Host</td><td>1 GPU per container</td>
            <td>Private</td><td>Async</td><td>TCP</td><td>NET/Socket (TCP fallback)</td></tr>
        <tr><td><b>DockerBE Full SHM</b></td><td>Host</td><td>All GPUs per container</td>
            <td>Host</td><td>Async</td><td>SHM</td><td>NVLink P2P / PCIe P2P</td></tr>
      </tbody>
    </table>
    <p><b>Key design point:</b> In the baseline, hybrid, and full-SHM variants, every worker
       container receives <code>--gpus all</code> and <code>CUDA_VISIBLE_DEVICES=0,1</code>,
       so <code>cudaDeviceCanAccessPeer()</code> succeeds and NCCL selects NVLink P2P (or PCIe P2P)
       transport. The single-GPU variant restricts each container to one physical GPU with a private
       IPC namespace; NCCL cannot establish P2P or SHM channels and falls back to NET/Socket (TCP),
       resulting in dramatically higher TP&gt;1 latency.</p>

    <h2>3. Experimental Setup</h2>
    <table class="config-table">
      <tr><td>Benchmark tool</td><td><code>vllm bench serve</code></td></tr>
      <tr><td>Dataset</td><td>ShareGPT_V3_unfiltered_cleaned_split.json, first 500 prompts</td></tr>
      <tr><td>Request rate</td><td>10 req/s (Poisson), 3 warmup requests, <code>--ignore-eos</code></td></tr>
      <tr><td>Models</td><td>Qwen3-4B, Qwen3-8B, Qwen3-14B</td></tr>
      <tr><td>Max model length</td><td>512 tokens</td></tr>
      <tr><td>GPU memory utilization</td><td>0.5 (0.9 for Qwen3-14B at TP=1)</td></tr>
      <tr><td>Failed requests</td><td>147 per run (prompts exceeding max_model_len), consistent across all configs</td></tr>
    </table>
    <p><b>Hardware:</b></p>
    <ul>
      <li><b>node192:</b> 2&times; NVIDIA A100-SXM4-40GB (GPU 0,1), NVLink NV12 interconnect. TP=1, TP=2.</li>
      <li><b>node196:</b> 2&times; NVIDIA A100-PCIE-40GB (GPU 1,2), PCIe 4.0 only. TP=1, TP=2, TP=4.</li>
    </ul>

    <h2>4. Results</h2>

    {"".join(results_sections)}

    {cross_node_html}

    <h2>5. Analysis</h2>

    <h3>5.1 DockerBE Full SHM achieves near-zero overhead (&lt;1%)</h3>
    <p>Across all 3 model sizes (4B, 8B, 14B) and all TP configurations on both NVLink
       and PCIe hardware, the DockerBE full_shm variant shows less than 1% median TPOT
       increase compared to the baseline Docker+MP configuration. This confirms that the
       Docker container boundary itself introduces negligible latency when GPU visibility
       and IPC namespace are shared.</p>

    <h3>5.2 Async output copy is the critical optimization</h3>
    <p>The sync_output ablation consistently adds 2&ndash;7% TPOT overhead across all
       model sizes and TP configurations. This is the only variant with measurable
       performance impact among the shared-GPU configurations. The fix is that Docker
       workers now use async output copy (<code>VLLM_DOCKER_ASYNC_OUTPUT_COPY=1</code>),
       decoupling result delivery from the main worker execution path&mdash;the same
       optimization the multiprocess executor already had.</p>

    <h3>5.3 SHM vs TCP response transport has zero measurable impact</h3>
    <p>The hybrid_shm (TCP responses) and full_shm (SHM responses) variants perform
       identically, proving that worker response transport choice is not on the critical
       path. The response payload is small relative to the model computation time.</p>

    <h3>5.4 Per-GPU isolation without SHM causes 3&ndash;4&times; TP&gt;1 regression</h3>
    <p>The single-GPU variant restricts each container to one physical GPU with
       <code>--ipc private</code>. At TP=1, where no inter-GPU communication occurs,
       performance is identical to baseline. At TP=2, NCCL cannot establish P2P or SHM
       channels between containers with private IPC namespaces and falls back to
       NET/Socket (TCP) transport, causing severe latency regression:</p>
    <ul>
      <li>Qwen3-4B TP=2: 23.75 ms vs 6.66 ms baseline (+257%)</li>
      <li>Qwen3-8B TP=2: 46.09 ms vs 9.02 ms baseline (+411%)</li>
      <li>Qwen3-14B TP=2: 60.19 ms vs 15.81 ms baseline (+281%)</li>
    </ul>
    <p>This demonstrates that preserving GPU visibility across containers (the approach
       used by all other DockerBE variants) is essential for multi-GPU performance.
       A potential mitigation for per-GPU isolation is to mount a shared
       <code>/dev/shm</code> volume and configure NCCL SHM transport
       (<code>NCCL_P2P_DISABLE=1</code>, <code>NCCL_NET_DISABLE_INTRA=1</code>),
       which achieves ~22 GB/s on PCIe hardware (see Experiment 2).</p>

    <h3>5.5 NVLink P2P transport is preserved in DockerBE</h3>
    <p>Both baseline and DockerBE (non-single-GPU variants) expose all GPUs to every
       container via <code>CUDA_VISIBLE_DEVICES=0,1</code>, enabling
       <code>cudaDeviceCanAccessPeer()</code> and NCCL NVLink P2P transport. The cross-node
       comparison confirms this: at TP=1 (no communication), node192 (NVLink) and node196
       (PCIe) show identical TPOT; at TP=2, NVLink provides an 11% TPOT advantage, and
       DockerBE preserves this advantage fully.</p>

    <h2>6. Conclusions</h2>
    <ol>
      <li><code>DockerDistributedExecutor</code> with full SHM transport introduces &lt;1%
          serving-latency overhead compared to the in-Docker multiprocess baseline, across
          model sizes from 4B to 14B parameters and TP=1 through TP=4.</li>
      <li>The only significant optimization required was async output copy. SHM-vs-TCP
          response transport has no measurable impact.</li>
      <li>Per-GPU container isolation (one visible GPU per container) causes 3&ndash;4&times;
          TP&gt;1 latency regression due to NCCL falling back to TCP socket transport.
          This can potentially be mitigated with shared <code>/dev/shm</code> and NCCL SHM
          transport configuration.</li>
      <li>The Docker executor correctly preserves NVLink P2P transport when all GPUs are
          visible to each container.</li>
    </ol>

    <hr class="appendix">

    <h2>Appendix A: Per-Run Metrics</h2>
    <p>Complete benchmark metrics for each configuration. Machine-readable data available
       in <code>{html.escape(CSV_REPORT.name)}</code>.</p>

    {"".join(appendix_sections)}

  </main>
</body>
</html>
"""

    HTML_REPORT.write_text(html_text, encoding="utf-8")
    return HTML_REPORT


def main() -> None:
    ASSET_DIR.mkdir(exist_ok=True)

    all_records: list[Record] = []
    node_data: list[tuple[dict, list[Record], dict[int, list[tuple[str, str]]]]] = []
    figures: dict[str, dict[str, Path]] = {}

    for node in NODES:
        records, available = load_node_records(node)
        if not records:
            print(f"  Skipping {node['key']}: no complete results found")
            continue
        all_records.extend(records)
        node_data.append((node, records, available))

        nk = node["key"]
        print(f"  {nk}: {len(records)} records loaded")
        figures[nk] = {}

        figures[nk]["tpot"] = plot_absolute_metric(
            records, available, node["variants"],
            metric_name="median_tpot_ms",
            ylabel="Median TPOT (ms)",
            title=f"Median TPOT \u2014 {node['label']}",
            filename=f"{nk}_median_tpot.png",
            precision=2,
        )
        figures[nk]["throughput"] = plot_absolute_metric(
            records, available, node["variants"],
            metric_name="output_token_throughput_tps",
            ylabel="Output token throughput (tok/s)",
            title=f"Output Throughput \u2014 {node['label']}",
            filename=f"{nk}_output_throughput.png",
            precision=1,
        )
        figures[nk]["delta"] = plot_deltas(
            records, available, node["variants"],
            filename=f"{nk}_delta_vs_baseline.png",
        )

    if not all_records:
        print("No results found in any node directory!")
        return

    write_csv(all_records)
    cross_node_fig = plot_cross_node(node_data, "cross_node_comparison.png") if len(node_data) > 1 else None
    generate_html(node_data, figures, cross_node_fig)
    print(f"\nReport written to {HTML_REPORT}")
    print(f"CSV written to {CSV_REPORT}")


if __name__ == "__main__":
    main()
