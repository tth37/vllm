# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class _MetricSummary:
    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def to_dict(self) -> dict[str, float | int | None]:
        avg = self.total / self.count if self.count else 0.0
        return {
            "count": self.count,
            "total": self.total,
            "avg": avg,
            "min": None if self.count == 0 else self.min,
            "max": None if self.count == 0 else self.max,
        }


class RpcProfiler:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.role = os.environ.get("VLLM_RPC_PROFILE_ROLE", "process")
        self.metadata: dict[str, Any] = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "role": self.role,
            "start_time_unix_s": time.time(),
        }
        self._metrics: dict[str, _MetricSummary] = {}
        self._lock = threading.Lock()
        atexit.register(self.dump)

    def set_metadata(self, **kwargs: Any) -> None:
        with self._lock:
            if "role" in kwargs and kwargs["role"]:
                self.role = str(kwargs["role"])
                self.metadata["role"] = self.role
            self.metadata.update(kwargs)

    def record(self, metric_name: str, value: float) -> None:
        with self._lock:
            metric = self._metrics.get(metric_name)
            if metric is None:
                metric = _MetricSummary()
                self._metrics[metric_name] = metric
            metric.add(value)

    def dump(self) -> str:
        with self._lock:
            payload = {
                "metadata": {
                    **self.metadata,
                    "end_time_unix_s": time.time(),
                },
                "metrics": {
                    name: metric.to_dict()
                    for name, metric in sorted(self._metrics.items())
                },
            }

        output_path = self.output_dir / f"{self.role}_pid{os.getpid()}.json"
        tmp_path = output_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, output_path)
        return str(output_path)


_PROFILER: RpcProfiler | None | bool = False


def get_rpc_profiler() -> RpcProfiler | None:
    global _PROFILER
    if _PROFILER is False:
        output_dir = os.environ.get("VLLM_RPC_PROFILE_DIR")
        _PROFILER = RpcProfiler(output_dir) if output_dir else None
    return _PROFILER


def set_rpc_profile_metadata(**kwargs: Any) -> None:
    profiler = get_rpc_profiler()
    if profiler is not None:
        profiler.set_metadata(**kwargs)


def flush_rpc_profile() -> str | None:
    profiler = get_rpc_profiler()
    if profiler is None:
        return None
    return profiler.dump()


def flush_rpc_profile_worker(_: Any = None) -> None:
    flush_rpc_profile()
