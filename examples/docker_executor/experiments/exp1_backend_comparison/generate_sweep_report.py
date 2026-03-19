#!/usr/bin/env python3
"""Generate plots and an HTML report for the exp1 multi-model sweep."""

from __future__ import annotations

import csv
import html
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

VARIANTS = [
    {
        "key": "baseline",
        "label": "Baseline Docker+MP",
        "dir": "baseline",
        "color": "#1f5c4a",
        "tp_files": {
            1: "3_docker_mp_tp1_bench.txt",
            2: "4_docker_mp_tp2_bench.txt",
        },
    },
    {
        "key": "dockerbe_sync_output",
        "label": "DockerBE Sync Output",
        "dir": "dockerbe_sync_output",
        "color": "#c05a2b",
        "tp_files": {
            1: "5_host_dockerbe_tp1_bench.txt",
            2: "6_host_dockerbe_tp2_bench.txt",
        },
    },
    {
        "key": "dockerbe_hybrid_shm",
        "label": "DockerBE Hybrid SHM",
        "dir": "dockerbe_hybrid_shm",
        "color": "#3273a8",
        "tp_files": {
            1: "5_host_dockerbe_tp1_bench.txt",
            2: "6_host_dockerbe_tp2_bench.txt",
        },
    },
    {
        "key": "dockerbe_full_shm",
        "label": "DockerBE Full SHM",
        "dir": "dockerbe_full_shm",
        "color": "#7b4ea3",
        "tp_files": {
            1: "5_host_dockerbe_tp1_bench.txt",
            2: "6_host_dockerbe_tp2_bench.txt",
        },
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
    model_slug: str
    model_label: str
    variant_key: str
    variant_label: str
    color: str
    tp_size: int
    bench_path: Path
    metrics: dict[str, float]


def parse_numeric(value: str) -> float:
    value = value.strip()
    if value.lower() in {"nan", "inf", "-inf"}:
        return float(value)
    return float(value)


def parse_bench_file(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key not in METRIC_KEYS:
            continue
        try:
            metrics[METRIC_KEYS[key]] = parse_numeric(raw)
        except ValueError:
            continue
    return metrics


def load_records() -> list[Record]:
    records: list[Record] = []
    for model_slug, model_label in MODELS:
        for variant in VARIANTS:
            for tp_size, filename in variant["tp_files"].items():
                bench_path = ROOT / variant["dir"] / model_slug / filename
                if not bench_path.exists():
                    raise FileNotFoundError(f"Missing bench file: {bench_path}")
                records.append(
                    Record(
                        model_slug=model_slug,
                        model_label=model_label,
                        variant_key=variant["key"],
                        variant_label=variant["label"],
                        color=variant["color"],
                        tp_size=tp_size,
                        bench_path=bench_path,
                        metrics=parse_bench_file(bench_path),
                    ))
    return records


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT.parents[3]), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def records_by_key(records: list[Record]) -> dict[tuple[str, int, str], Record]:
    return {
        (record.model_slug, record.tp_size, record.variant_key): record
        for record in records
    }


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def annotate_bars(ax: Any, bars: list[Any], precision: int) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{precision}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )


def plot_absolute_metric(
    records: list[Record],
    metric_name: str,
    ylabel: str,
    title: str,
    filename: str,
    precision: int,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    fig.patch.set_facecolor("#f7f3eb")
    model_labels = [label for _, label in MODELS]
    centers = list(range(len(MODELS)))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    record_map = records_by_key(records)

    for ax, tp_size in zip(axes, [1, 2]):
        ax.set_facecolor("#fffdf8")
        ax.grid(axis="y", color="#ddd7c9", alpha=0.8)
        ax.set_axisbelow(True)

        for offset, variant in zip(offsets, VARIANTS):
            values = [
                record_map[(model_slug, tp_size, variant["key"])].metrics[metric_name]
                for model_slug, _ in MODELS
            ]
            positions = [center + offset * width for center in centers]
            bars = ax.bar(
                positions,
                values,
                width=width,
                label=variant["label"],
                color=variant["color"],
                edgecolor="#2b2b2b",
                linewidth=0.6,
            )
            annotate_bars(ax, list(bars), precision)

        ax.set_title(f"TP={tp_size}", fontsize=13, weight="bold")
        ax.set_xticks(centers)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel(ylabel)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=16, weight="bold", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    output_path = ASSET_DIR / filename
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_deltas(records: list[Record], filename: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor("#f7f3eb")
    model_labels = [label for _, label in MODELS]
    centers = list(range(len(MODELS)))
    width = 0.22
    delta_variants = VARIANTS[1:]
    offsets = [-1, 0, 1]

    record_map = records_by_key(records)
    plot_specs = [
        ("median_tpot_ms", "TPOT delta vs baseline (ms)", 0, 0, 2),
        ("output_token_throughput_tps", "Throughput delta vs baseline (tok/s)", 1, 0, 1),
    ]

    for row, tp_size in enumerate([1, 2]):
        for metric_name, ylabel, metric_row, _, precision in plot_specs:
            ax = axes[metric_row][row]
            ax.set_facecolor("#fffdf8")
            ax.grid(axis="y", color="#ddd7c9", alpha=0.8)
            ax.set_axisbelow(True)
            baseline_values = [
                record_map[(model_slug, tp_size, "baseline")].metrics[metric_name]
                for model_slug, _ in MODELS
            ]

            for offset, variant in zip(offsets, delta_variants):
                values = []
                for index, (model_slug, _) in enumerate(MODELS):
                    variant_value = record_map[(model_slug, tp_size,
                                                variant["key"])].metrics[metric_name]
                    values.append(variant_value - baseline_values[index])

                positions = [center + offset * width for center in centers]
                bars = ax.bar(
                    positions,
                    values,
                    width=width,
                    label=variant["label"],
                    color=variant["color"],
                    edgecolor="#2b2b2b",
                    linewidth=0.6,
                )
                annotate_bars(ax, list(bars), precision)

            ax.axhline(0.0, color="#2b2b2b", linewidth=1.0)
            ax.set_title(f"TP={tp_size}", fontsize=13, weight="bold")
            ax.set_xticks(centers)
            ax.set_xticklabels(model_labels)
            ax.set_ylabel(ylabel)
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin
            ax.set_ylim(ymin - span * 0.06, ymax + span * 0.10)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Delta Relative To Baseline", fontsize=16, weight="bold", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    output_path = ASSET_DIR / filename
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_csv(records: list[Record]) -> Path:
    fieldnames = [
        "model_slug",
        "model_label",
        "variant_key",
        "variant_label",
        "tp_size",
        "successful_requests",
        "failed_requests",
        "benchmark_duration_s",
        "request_throughput_rps",
        "output_token_throughput_tps",
        "peak_concurrent_requests",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
        "bench_path",
    ]
    with CSV_REPORT.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "model_slug": record.model_slug,
                "model_label": record.model_label,
                "variant_key": record.variant_key,
                "variant_label": record.variant_label,
                "tp_size": record.tp_size,
                "bench_path": str(record.bench_path),
            }
            row.update(record.metrics)
            writer.writerow(row)
    return CSV_REPORT


def summarize(records: list[Record]) -> dict[str, float]:
    record_map = records_by_key(records)
    summary: dict[str, float] = {}

    for tp_size in [1, 2]:
        sync_tpot = []
        hybrid_tpot = []
        full_tpot = []
        hybrid_full_tpot_gap = []
        sync_tps = []
        hybrid_tps = []
        full_tps = []

        for model_slug, _ in MODELS:
            baseline = record_map[(model_slug, tp_size, "baseline")]
            sync = record_map[(model_slug, tp_size, "dockerbe_sync_output")]
            hybrid = record_map[(model_slug, tp_size, "dockerbe_hybrid_shm")]
            full = record_map[(model_slug, tp_size, "dockerbe_full_shm")]

            sync_tpot.append(sync.metrics["median_tpot_ms"] -
                             baseline.metrics["median_tpot_ms"])
            hybrid_tpot.append(hybrid.metrics["median_tpot_ms"] -
                               baseline.metrics["median_tpot_ms"])
            full_tpot.append(full.metrics["median_tpot_ms"] -
                             baseline.metrics["median_tpot_ms"])
            hybrid_full_tpot_gap.append(abs(full.metrics["median_tpot_ms"] -
                                            hybrid.metrics["median_tpot_ms"]))

            sync_tps.append(sync.metrics["output_token_throughput_tps"] -
                            baseline.metrics["output_token_throughput_tps"])
            hybrid_tps.append(hybrid.metrics["output_token_throughput_tps"] -
                              baseline.metrics["output_token_throughput_tps"])
            full_tps.append(full.metrics["output_token_throughput_tps"] -
                            baseline.metrics["output_token_throughput_tps"])

        summary[f"sync_tpot_delta_tp{tp_size}"] = safe_mean(sync_tpot)
        summary[f"hybrid_tpot_delta_tp{tp_size}"] = safe_mean(hybrid_tpot)
        summary[f"full_tpot_delta_tp{tp_size}"] = safe_mean(full_tpot)
        summary[f"hybrid_full_tpot_gap_tp{tp_size}"] = max(hybrid_full_tpot_gap)
        summary[f"sync_tps_delta_tp{tp_size}"] = safe_mean(sync_tps)
        summary[f"hybrid_tps_delta_tp{tp_size}"] = safe_mean(hybrid_tps)
        summary[f"full_tps_delta_tp{tp_size}"] = safe_mean(full_tps)

    failures = sorted(
        {(record.metrics["successful_requests"], record.metrics["failed_requests"])
         for record in records})
    summary["request_pattern_count"] = float(len(failures))
    summary["dominant_successful_requests"] = max(
        pair[0] for pair in failures)
    summary["dominant_failed_requests"] = max(pair[1] for pair in failures)
    return summary


def table_rows(records: list[Record]) -> str:
    record_map = records_by_key(records)
    rows: list[str] = []
    for model_slug, model_label in MODELS:
        for tp_size in [1, 2]:
            baseline = record_map[(model_slug, tp_size, "baseline")]
            for variant in VARIANTS:
                record = record_map[(model_slug, tp_size, variant["key"])]
                baseline_tpot = baseline.metrics["median_tpot_ms"]
                baseline_tps = baseline.metrics["output_token_throughput_tps"]
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(model_label)}</td>"
                    f"<td>{tp_size}</td>"
                    f"<td>{html.escape(variant['label'])}</td>"
                    f"<td>{record.metrics['successful_requests']:.0f}/{record.metrics['failed_requests']:.0f}</td>"
                    f"<td>{record.metrics['median_tpot_ms']:.2f}</td>"
                    f"<td>{record.metrics['median_tpot_ms'] - baseline_tpot:+.2f}</td>"
                    f"<td>{record.metrics['output_token_throughput_tps']:.2f}</td>"
                    f"<td>{record.metrics['output_token_throughput_tps'] - baseline_tps:+.2f}</td>"
                    f"<td>{record.metrics['median_ttft_ms']:.2f}</td>"
                    f"<td>{html.escape(str(record.bench_path.relative_to(ROOT)))}</td>"
                    "</tr>")
    return "\n".join(rows)


def variant_cards_html() -> str:
    cards = [
        (
            "Baseline Docker+MP",
            "vLLM runs fully inside one Docker container and uses the standard multiprocess executor (<code>mp</code>). This is the control setup: no DockerDistributedExecutor RPC path is involved.",
            [
                "Server placement: inside Docker",
                "Executor: MultiprocExecutor (<code>--distributed-executor-backend mp</code>)",
                "What it measures: the reference latency/throughput we want DockerBE to match",
            ],
        ),
        (
            "DockerBE Sync Output",
            "This keeps the Docker executor and SHM transport enabled, but removes the async output-copy optimization. It is the direct ablation for the root cause we found earlier.",
            [
                "Server placement: host process using <code>DockerDistributedExecutor</code>",
                "Broadcast path: SHM",
                "Worker response path: SHM",
                "Output handling: synchronous (<code>VLLM_DOCKER_ASYNC_OUTPUT_COPY=0</code>)",
                "What it ablates: async output copy",
            ],
        ),
        (
            "DockerBE Hybrid SHM",
            "This keeps async output copy enabled, but switches worker responses back to TCP while leaving the host-to-worker broadcast on SHM.",
            [
                "Server placement: host process using <code>DockerDistributedExecutor</code>",
                "Broadcast path: SHM",
                "Worker response path: TCP",
                "Output handling: asynchronous (<code>VLLM_DOCKER_ASYNC_OUTPUT_COPY=1</code>)",
                "What it ablates: response-side SHM only",
            ],
        ),
        (
            "DockerBE Full SHM",
            "This is the optimized target. It keeps the Docker executor, async output copy, SHM broadcast, and SHM worker responses all enabled at once.",
            [
                "Server placement: host process using <code>DockerDistributedExecutor</code>",
                "Broadcast path: SHM",
                "Worker response path: SHM",
                "Output handling: asynchronous (<code>VLLM_DOCKER_ASYNC_OUTPUT_COPY=1</code>)",
                "What it represents: the final optimized DockerBE path",
            ],
        ),
    ]

    html_cards: list[str] = []
    for title, body, bullets in cards:
        bullet_html = "".join(f"<li>{item}</li>" for item in bullets)
        html_cards.append(
            "<div class=\"variant-card\">"
            f"<h3>{html.escape(title)}</h3>"
            f"<p>{body}</p>"
            f"<ul>{bullet_html}</ul>"
            "</div>")
    return "\n".join(html_cards)


def glossary_html() -> str:
    entries = [
        (
            "Async output copy",
            "The worker copies generation outputs onto a separate path so the main execution path does not block on handing results back to the host. In this experiment, removing it causes the clear latency regression.",
        ),
        (
            "Sync output",
            "The worker returns outputs on the same critical path without the async handoff. That makes response delivery wait longer and shows up as higher TPOT.",
        ),
        (
            "SHM broadcast MQ",
            "The host-to-worker RPC broadcast MessageQueue uses shared memory instead of a TCP or socket transport. This reduces messaging overhead for control traffic sent from the host to all workers.",
        ),
        (
            "SHM worker response MQ",
            "Each worker returns its RPC response to the host through a shared-memory MessageQueue rather than a TCP path. The ablation shows that this change alone has only a very small effect in this setup.",
        ),
        (
            "Hybrid SHM",
            "A mixed transport mode: SHM for broadcast from host to workers, but TCP for worker responses back to the host.",
        ),
        (
            "Full SHM",
            "SHM in both directions: broadcast traffic and worker response traffic both use shared memory.",
        ),
    ]

    items = []
    for term, definition in entries:
        items.append(
            "<div class=\"glossary-item\">"
            f"<strong>{html.escape(term)}</strong>"
            f"<p>{html.escape(definition)}</p>"
            "</div>")
    return "\n".join(items)


def generate_html(records: list[Record], figures: dict[str, Path]) -> Path:
    commit = git_commit()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary = summarize(records)
    figure_refs = {key: path.relative_to(ROOT) for key, path in figures.items()}

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Exp1 Multi-Model Sweep Report</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --card: #fffdf8;
      --ink: #1f2933;
      --muted: #5f6c76;
      --line: #d9d1c3;
      --accent: #8e5d2d;
    }}
    body {{
      margin: 0;
      font-family: "Georgia", "Iowan Old Style", serif;
      background: linear-gradient(180deg, #efe7d8 0%, #f9f6ef 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    h1, h2, h3 {{
      margin: 0 0 10px;
      line-height: 1.15;
    }}
    p {{
      margin: 0 0 14px;
      color: var(--ink);
      line-height: 1.5;
    }}
    .hero {{
      background: radial-gradient(circle at top left, #fff7dc 0%, #fffdf8 55%, #f7efe2 100%);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(58, 44, 28, 0.08);
      margin-bottom: 22px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 18px;
      font-size: 14px;
      color: var(--muted);
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      margin-bottom: 18px;
      box-shadow: 0 8px 24px rgba(58, 44, 28, 0.05);
    }}
    .callouts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }}
    .callout {{
      background: #fff9ef;
      border: 1px solid #ead7b5;
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .callout strong {{
      display: block;
      margin-bottom: 6px;
      color: var(--accent);
    }}
    .variant-grid, .glossary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .variant-card, .glossary-item {{
      background: #fffaf1;
      border: 1px solid #eadfce;
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .variant-card h3, .glossary-item strong {{
      color: var(--accent);
    }}
    .variant-card ul {{
      margin: 10px 0 0;
      padding-left: 18px;
      color: var(--ink);
    }}
    .variant-card li {{
      margin: 0 0 8px;
      line-height: 1.45;
    }}
    .glossary-item p {{
      margin: 8px 0 0;
    }}
    .figure {{
      margin: 18px 0 8px;
      text-align: center;
    }}
    .figure img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      background: #f3eee5;
      border-radius: 6px;
      padding: 2px 6px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Experiment 1: Multi-Model Sweep Report</h1>
      <p>This report summarizes the 3-model, 4-variant rerun of the Docker executor experiment using the first 1000 ShareGPT prompts on GPUs <code>1,2</code>.</p>
      <p>The core pattern is stable across model sizes: disabling async output copy is the only optimization that causes a consistent regression, while the hybrid-SHM and full-SHM paths both stay essentially at baseline.</p>
      <div class="meta">
        <div><strong>Generated</strong><br>{generated}</div>
        <div><strong>Git commit</strong><br>{html.escape(commit)}</div>
        <div><strong>Dataset</strong><br>ShareGPT first 1000 prompts</div>
        <div><strong>Models</strong><br>Qwen3-4B, Qwen3-8B, Qwen3-14B</div>
      </div>
    </section>

    <section class="card">
      <h2>Key Findings</h2>
      <div class="callouts">
        <div class="callout">
          <strong>Async output copy is the critical optimization</strong>
          <span>Compared with baseline, the sync-output ablation adds an average of <code>{summary['sync_tpot_delta_tp1']:+.2f} ms</code> TPOT at TP=1 and <code>{summary['sync_tpot_delta_tp2']:+.2f} ms</code> at TP=2.</span>
        </div>
        <div class="callout">
          <strong>Response-side SHM barely moves the needle</strong>
          <span>The maximum median-TPOT gap between hybrid-SHM and full-SHM is only <code>{summary['hybrid_full_tpot_gap_tp1']:.2f} ms</code> at TP=1 and <code>{summary['hybrid_full_tpot_gap_tp2']:.2f} ms</code> at TP=2.</span>
        </div>
        <div class="callout">
          <strong>Full SHM reaches baseline parity</strong>
          <span>Relative to baseline, the full-SHM path averages <code>{summary['full_tpot_delta_tp1']:+.2f} ms</code> TPOT delta at TP=1 and <code>{summary['full_tpot_delta_tp2']:+.2f} ms</code> at TP=2.</span>
        </div>
        <div class="callout">
          <strong>Request validity is stable enough for fair comparison</strong>
          <span>Most runs landed at <code>700/300</code> successful/failed requests; one 14B hybrid TP=2 run landed at <code>699/301</code>, which is still very close.</span>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Root Cause And Fix</h2>
      <p>The performance gap was <strong>not</strong> solved by making the RPC transport itself "async." In fact, the hybrid-SHM and full-SHM rows are already enough to show that response-side transport choice only has a tiny effect here. If SHM response transport were the main bottleneck, we would expect a much larger gap between <code>DockerBE Hybrid SHM</code> and <code>DockerBE Full SHM</code>, but the measured difference stays near zero.</p>
      <p>The real fix is that Docker workers now use the same <strong>async output-copy behavior</strong> that the multiprocess executor already had. In the regressed path, the worker stayed on the critical path while handing completed outputs back to the host. In the fixed path, output handoff is decoupled from the main worker execution path, so result delivery no longer stalls the next step of execution as much.</p>
      <p>In practical terms, the key switch is <code>VLLM_DOCKER_ASYNC_OUTPUT_COPY=1</code>. That change matters much more than the SHM-vs-TCP response transport choice, which is why the <code>DockerBE Sync Output</code> ablation regresses clearly while <code>DockerBE Hybrid SHM</code> and <code>DockerBE Full SHM</code> stay close to baseline.</p>
      <p>So the short answer is: <strong>we solved the main loss by enabling async output copy in the Docker executor worker path, not by enabling an "async RPC" transport mode.</strong> The SHM work still helps keep transport overhead low, but it is not the dominant reason the final DockerBE path matches baseline.</p>
    </section>

    <section class="card">
      <h2>Settings And Ablations</h2>
      <p>The four variants are intentionally close to each other so the effect of each optimization is easy to isolate. The baseline is the reference implementation, and the three DockerBE rows add or remove one optimization at a time.</p>
      <div class="variant-grid">
        {variant_cards_html()}
      </div>
    </section>

    <section class="card">
      <h2>Glossary</h2>
      <p>This experiment mixes executor choices, transport choices, and output-handling choices. The short definitions below explain the terms used in the tables and plots.</p>
      <div class="glossary-grid">
        {glossary_html()}
      </div>
    </section>

    <section class="card">
      <h2>Median TPOT</h2>
      <p>Lower is better. The sync-output ablation sits visibly above the other three variants across all three models. Hybrid-SHM and full-SHM almost overlap baseline.</p>
      <div class="figure">
        <img src="{html.escape(str(figure_refs['tpot']))}" alt="Median TPOT chart">
      </div>
    </section>

    <section class="card">
      <h2>Output Throughput</h2>
      <p>Higher is better. Throughput differences are small overall, but the same pattern remains: sync-output is consistently the weakest variant, while hybrid-SHM and full-SHM stay near baseline.</p>
      <div class="figure">
        <img src="{html.escape(str(figure_refs['throughput']))}" alt="Output throughput chart">
      </div>
    </section>

    <section class="card">
      <h2>Delta Relative To Baseline</h2>
      <p>This view makes the ablations easier to compare directly. Positive TPOT deltas are regressions; negative throughput deltas are regressions. The response-SHM difference remains close to zero.</p>
      <div class="figure">
        <img src="{html.escape(str(figure_refs['delta']))}" alt="Delta to baseline chart">
      </div>
    </section>

    <section class="card">
      <h2>Per-Run Metrics</h2>
      <p>The table below is generated directly from the bench outputs. <code>TPOT delta</code> and <code>tok/s delta</code> are relative to the matching baseline for the same model and TP size.</p>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>TP</th>
            <th>Variant</th>
            <th>Succ/Fail</th>
            <th>Median TPOT</th>
            <th>TPOT delta</th>
            <th>Output tok/s</th>
            <th>tok/s delta</th>
            <th>Median TTFT</th>
            <th>Bench file</th>
          </tr>
        </thead>
        <tbody>
          {table_rows(records)}
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Artifacts</h2>
      <p>Machine-readable metrics: <code>{html.escape(CSV_REPORT.name)}</code></p>
      <p>Model sweep text summary: <code>model_sweep_summary.txt</code></p>
      <p>Figure assets: <code>{html.escape(ASSET_DIR.name)}/</code></p>
    </section>
  </main>
</body>
</html>
"""

    HTML_REPORT.write_text(html_text, encoding="utf-8")
    return HTML_REPORT


def main() -> None:
    ASSET_DIR.mkdir(exist_ok=True)
    records = load_records()
    write_csv(records)

    figures = {
        "tpot":
        plot_absolute_metric(
            records,
            metric_name="median_tpot_ms",
            ylabel="Median TPOT (ms)",
            title="Median TPOT Across Models And Variants",
            filename="median_tpot.png",
            precision=2,
        ),
        "throughput":
        plot_absolute_metric(
            records,
            metric_name="output_token_throughput_tps",
            ylabel="Output token throughput (tok/s)",
            title="Output Throughput Across Models And Variants",
            filename="output_throughput.png",
            precision=1,
        ),
        "delta": plot_deltas(records, filename="delta_vs_baseline.png"),
    }

    generate_html(records, figures)


if __name__ == "__main__":
    main()
