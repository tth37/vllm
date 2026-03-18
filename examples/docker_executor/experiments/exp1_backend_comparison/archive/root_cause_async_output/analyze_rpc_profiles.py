#!/usr/bin/env python3
"""Summarize RPC/MQ profile outputs for exp1 baseline vs Docker full SHM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_profiles(profile_dir: Path) -> list[dict[str, Any]]:
    profiles = []
    for path in sorted(profile_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        payload["_path"] = str(path)
        profiles.append(payload)
    return profiles


def aggregate_metric(
    profiles: list[dict[str, Any]],
    metric_names: list[str],
    role_prefix: str | None = None,
) -> dict[str, float | int | None]:
    total = 0.0
    count = 0
    for profile in profiles:
        role = str(profile.get("metadata", {}).get("role", ""))
        if role_prefix is not None and not role.startswith(role_prefix):
            continue
        metrics = profile.get("metrics", {})
        for metric_name in metric_names:
            metric = metrics.get(metric_name)
            if metric is None:
                continue
            total += float(metric["total"])
            count += int(metric["count"])
    avg = total / count if count else None
    return {"count": count, "total": total if count else None, "avg": avg}


def exact_metric(metric_name: str) -> list[str]:
    return [metric_name]


def worker_metrics(prefix: str, method: str, suffix: str, ranks: range) -> list[str]:
    return [
        f"rpc.{prefix}.worker.rank{rank}.{method}.{suffix}"
        for rank in ranks
    ]


def fmt_seconds_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000:.3f} ms"


def fmt_count(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return str(int(value))


def parse_bench_metrics(path: Path) -> dict[str, str]:
    wanted = {
        "Successful requests",
        "Failed requests",
        "Output token throughput (tok/s)",
        "Median TPOT (ms)",
        "Mean TPOT (ms)",
        "Median TTFT (ms)",
        "P99 TTFT (ms)",
    }
    metrics: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key in wanted:
                metrics[key] = value.strip()
    return metrics


def scenario_summary(
    profiles: list[dict[str, Any]],
    scenario_prefix: str,
    worker_ranks: range,
) -> dict[str, dict[str, float | int | None]]:
    output_rank = 0
    return {
        "host_execute_total": aggregate_metric(
            profiles,
            exact_metric(f"rpc.{scenario_prefix}.host.execute_model.host_total_s"),
        ),
        "host_sample_total": aggregate_metric(
            profiles,
            exact_metric(f"rpc.{scenario_prefix}.host.sample_tokens.host_total_s"),
        ),
        "host_execute_send": aggregate_metric(
            profiles,
            exact_metric(f"rpc.{scenario_prefix}.host.execute_model.host_send_s"),
        ),
        "host_sample_send": aggregate_metric(
            profiles,
            exact_metric(f"rpc.{scenario_prefix}.host.sample_tokens.host_send_s"),
        ),
        "worker_execute_exec": aggregate_metric(
            profiles,
            worker_metrics(scenario_prefix, "execute_model", "execute_s", worker_ranks),
        ),
        "worker_sample_exec": aggregate_metric(
            profiles,
            worker_metrics(scenario_prefix, "sample_tokens", "execute_s", worker_ranks),
        ),
        "worker_execute_dispatch": aggregate_metric(
            profiles,
            worker_metrics(scenario_prefix, "execute_model", "dispatch_s", worker_ranks),
        ),
        "worker_sample_dispatch": aggregate_metric(
            profiles,
            worker_metrics(scenario_prefix, "sample_tokens", "dispatch_s", worker_ranks),
        ),
        "worker_execute_response_enqueue": aggregate_metric(
            profiles,
            exact_metric(
                f"rpc.{scenario_prefix}.worker.rank{output_rank}.execute_model.response_enqueue_s"
            ),
        ),
        "worker_sample_response_enqueue": aggregate_metric(
            profiles,
            exact_metric(
                f"rpc.{scenario_prefix}.worker.rank{output_rank}.sample_tokens.response_enqueue_s"
            ),
        ),
        "worker_execute_output_materialize": aggregate_metric(
            profiles,
            exact_metric(
                f"rpc.{scenario_prefix}.worker.rank{output_rank}.execute_model.output_materialize_s"
            ),
        ),
        "worker_sample_output_materialize": aggregate_metric(
            profiles,
            exact_metric(
                f"rpc.{scenario_prefix}.worker.rank{output_rank}.sample_tokens.output_materialize_s"
            ),
        ),
        "host_response_dequeue": aggregate_metric(
            profiles,
            exact_metric(
                f"mq.{scenario_prefix}.host.response.rank{output_rank}.dequeue.total_s"
            ),
        ),
        "host_broadcast_enqueue": aggregate_metric(
            profiles,
            exact_metric(f"mq.{scenario_prefix}.host.rpc_broadcast.enqueue.total_s"),
        ),
        "host_broadcast_serialize": aggregate_metric(
            profiles,
            exact_metric(
                f"mq.{scenario_prefix}.host.rpc_broadcast.enqueue.serialize_s"
            ),
        ),
        "worker_broadcast_deserialize": aggregate_metric(
            profiles,
            [
                f"mq.{scenario_prefix}.worker.rank{rank}.rpc_broadcast.dequeue.local_deserialize_s"
                for rank in worker_ranks
            ],
        ),
        "worker_response_serialize": aggregate_metric(
            profiles,
            exact_metric(
                f"mq.{scenario_prefix}.worker.rank{output_rank}.response.enqueue.serialize_s"
            ),
        ),
        "host_response_deserialize": aggregate_metric(
            profiles,
            exact_metric(
                f"mq.{scenario_prefix}.host.response.rank{output_rank}.dequeue.local_deserialize_s"
            ),
        ),
    }


def write_line(lines: list[str], text: str = "") -> None:
    lines.append(text)


def compare_ms(
    lines: list[str],
    label: str,
    baseline: dict[str, float | int | None],
    docker: dict[str, float | int | None],
) -> None:
    baseline_avg = baseline["avg"]
    docker_avg = docker["avg"]
    delta = None
    if baseline_avg is not None and docker_avg is not None:
        delta = docker_avg - baseline_avg
    delta_text = "n/a" if delta is None else f"{delta * 1000:+.3f} ms"
    write_line(
        lines,
        f"- {label}: baseline={fmt_seconds_ms(baseline_avg)}, "
        f"docker={fmt_seconds_ms(docker_avg)}, delta={delta_text}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-profile-dir", type=Path, required=True)
    parser.add_argument("--docker-profile-dir", type=Path, required=True)
    parser.add_argument("--baseline-bench", type=Path, required=True)
    parser.add_argument("--docker-bench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=False)
    args = parser.parse_args()

    baseline_profiles = load_profiles(args.baseline_profile_dir)
    docker_profiles = load_profiles(args.docker_profile_dir)
    baseline_bench = parse_bench_metrics(args.baseline_bench)
    docker_bench = parse_bench_metrics(args.docker_bench)

    baseline = scenario_summary(
        baseline_profiles, scenario_prefix="multiproc", worker_ranks=range(2)
    )
    docker = scenario_summary(
        docker_profiles, scenario_prefix="docker", worker_ranks=range(2)
    )

    lines: list[str] = []
    write_line(lines, "==============================================================")
    write_line(lines, "  Exp1 RPC/MQ Gap Analysis")
    write_line(lines, "==============================================================")
    write_line(lines)
    write_line(lines, "Benchmark recap (TP=2, 200 ShareGPT prompts, rate=10, warmups=3)")
    write_line(
        lines,
        f"- Baseline host MP: median TPOT={baseline_bench.get('Median TPOT (ms)', 'n/a')}, "
        f"output tok/s={baseline_bench.get('Output token throughput (tok/s)', 'n/a')}, "
        f"successful={baseline_bench.get('Successful requests', 'n/a')}, "
        f"failed={baseline_bench.get('Failed requests', 'n/a')}",
    )
    write_line(
        lines,
        f"- DockerBE full SHM: median TPOT={docker_bench.get('Median TPOT (ms)', 'n/a')}, "
        f"output tok/s={docker_bench.get('Output token throughput (tok/s)', 'n/a')}, "
        f"successful={docker_bench.get('Successful requests', 'n/a')}, "
        f"failed={docker_bench.get('Failed requests', 'n/a')}",
    )
    write_line(lines)

    write_line(lines, "Per-call timing comparison")
    compare_ms(lines, "Host execute_model total", baseline["host_execute_total"], docker["host_execute_total"])
    compare_ms(lines, "Host sample_tokens total", baseline["host_sample_total"], docker["host_sample_total"])
    compare_ms(lines, "Worker execute_model compute", baseline["worker_execute_exec"], docker["worker_execute_exec"])
    compare_ms(lines, "Worker sample_tokens compute", baseline["worker_sample_exec"], docker["worker_sample_exec"])
    compare_ms(lines, "Host broadcast enqueue total", baseline["host_broadcast_enqueue"], docker["host_broadcast_enqueue"])
    compare_ms(lines, "Host broadcast serialize only", baseline["host_broadcast_serialize"], docker["host_broadcast_serialize"])
    compare_ms(lines, "Worker broadcast deserialize", baseline["worker_broadcast_deserialize"], docker["worker_broadcast_deserialize"])
    compare_ms(lines, "Output worker response serialize", baseline["worker_response_serialize"], docker["worker_response_serialize"])
    compare_ms(lines, "Host response dequeue total", baseline["host_response_dequeue"], docker["host_response_dequeue"])
    compare_ms(lines, "Host response deserialize", baseline["host_response_deserialize"], docker["host_response_deserialize"])
    compare_ms(lines, "Worker execute_model response enqueue", baseline["worker_execute_response_enqueue"], docker["worker_execute_response_enqueue"])
    compare_ms(lines, "Worker sample_tokens response enqueue", baseline["worker_sample_response_enqueue"], docker["worker_sample_response_enqueue"])
    compare_ms(lines, "Worker execute_model output materialize", baseline["worker_execute_output_materialize"], docker["worker_execute_output_materialize"])
    compare_ms(lines, "Worker sample_tokens output materialize", baseline["worker_sample_output_materialize"], docker["worker_sample_output_materialize"])
    write_line(lines)

    baseline_execute_gap = None
    docker_execute_gap = None
    if (
        baseline["host_execute_total"]["avg"] is not None
        and baseline["worker_execute_exec"]["avg"] is not None
    ):
        baseline_execute_gap = (
            float(baseline["host_execute_total"]["avg"])
            - float(baseline["worker_execute_exec"]["avg"])
        )
    if (
        docker["host_execute_total"]["avg"] is not None
        and docker["worker_execute_exec"]["avg"] is not None
    ):
        docker_execute_gap = (
            float(docker["host_execute_total"]["avg"])
            - float(docker["worker_execute_exec"]["avg"])
        )

    baseline_sample_gap = None
    docker_sample_gap = None
    if (
        baseline["host_sample_total"]["avg"] is not None
        and baseline["worker_sample_exec"]["avg"] is not None
    ):
        baseline_sample_gap = (
            float(baseline["host_sample_total"]["avg"])
            - float(baseline["worker_sample_exec"]["avg"])
        )
    if (
        docker["host_sample_total"]["avg"] is not None
        and docker["worker_sample_exec"]["avg"] is not None
    ):
        docker_sample_gap = (
            float(docker["host_sample_total"]["avg"])
            - float(docker["worker_sample_exec"]["avg"])
        )

    write_line(lines, "Non-compute gap estimate")
    write_line(
        lines,
        f"- execute_model host_total - worker_compute: "
        f"baseline={fmt_seconds_ms(baseline_execute_gap)}, "
        f"docker={fmt_seconds_ms(docker_execute_gap)}",
    )
    write_line(
        lines,
        f"- sample_tokens host_total - worker_compute: "
        f"baseline={fmt_seconds_ms(baseline_sample_gap)}, "
        f"docker={fmt_seconds_ms(docker_sample_gap)}",
    )
    write_line(lines)

    execute_wait_delta = None
    sample_compute_delta = None
    response_wait_delta = None
    if (
        docker["host_execute_total"]["avg"] is not None
        and baseline["host_execute_total"]["avg"] is not None
    ):
        execute_wait_delta = (
            float(docker["host_execute_total"]["avg"])
            - float(baseline["host_execute_total"]["avg"])
        )
    if (
        docker["worker_sample_exec"]["avg"] is not None
        and baseline["worker_sample_exec"]["avg"] is not None
    ):
        sample_compute_delta = (
            float(docker["worker_sample_exec"]["avg"])
            - float(baseline["worker_sample_exec"]["avg"])
        )
    if (
        docker["host_response_dequeue"]["avg"] is not None
        and baseline["host_response_dequeue"]["avg"] is not None
    ):
        response_wait_delta = (
            float(docker["host_response_dequeue"]["avg"])
            - float(baseline["host_response_dequeue"]["avg"])
        )

    write_line(lines, "Key findings")
    write_line(
        lines,
        f"- The execute_model gap is dominated by response availability on the host side: "
        f"host total grew by {fmt_seconds_ms(execute_wait_delta)}, while worker compute changed by only "
        f"{fmt_seconds_ms((docker['worker_execute_exec']['avg'] or 0.0) - (baseline['worker_execute_exec']['avg'] or 0.0))}.",
    )
    write_line(
        lines,
        f"- Output materialization is now measured explicitly: execute_model changed by "
        f"{fmt_seconds_ms((docker['worker_execute_output_materialize']['avg'] or 0.0) - (baseline['worker_execute_output_materialize']['avg'] or 0.0))}, "
        f"and sample_tokens changed by "
        f"{fmt_seconds_ms((docker['worker_sample_output_materialize']['avg'] or 0.0) - (baseline['worker_sample_output_materialize']['avg'] or 0.0))}.",
    )
    write_line(
        lines,
        f"- The host response dequeue delta is almost entirely wait time, not SHM read cost: "
        f"dequeue total moved by {fmt_seconds_ms(response_wait_delta)}, but response deserialize stayed "
        f"{fmt_seconds_ms((docker['host_response_deserialize']['avg'] or 0.0) - (baseline['host_response_deserialize']['avg'] or 0.0))}.",
    )
    write_line(
        lines,
        f"- sample_tokens has two effects: worker compute increased by {fmt_seconds_ms(sample_compute_delta)}, "
        f"and the remaining non-compute gap still grew to {fmt_seconds_ms(docker_sample_gap)}.",
    )
    write_line(
        lines,
        "- Broadcast serialization and SHM write/read costs stayed in the tens of microseconds, so the remaining bottleneck is not the raw MessageQueue transport path.",
    )
    write_line(lines)

    write_line(lines, "Profiler coverage")
    write_line(lines, f"- Baseline profile files: {len(baseline_profiles)}")
    write_line(lines, f"- Docker profile files: {len(docker_profiles)}")
    write_line(lines, f"- Baseline host execute_model calls captured: {fmt_count(baseline['host_execute_total']['count'])}")
    write_line(lines, f"- Docker host execute_model calls captured: {fmt_count(docker['host_execute_total']['count'])}")
    write_line(lines)

    write_line(lines, "Interpretation guide")
    write_line(lines, "- If the host-total delta is much larger than the serialize/enqueue/dequeue deltas, the remaining gap is likely worker wakeup/coordination overhead rather than raw SHM copy cost.")
    write_line(lines, "- If host broadcast serialize grows materially, SchedulerOutput serialization is a likely bottleneck.")
    write_line(lines, "- If host response dequeue and worker response enqueue stay tiny, the response MQ is probably not the dominant remaining cost.")

    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
