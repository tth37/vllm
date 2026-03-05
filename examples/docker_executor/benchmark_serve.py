#!/usr/bin/env python3
"""Simple benchmark for vLLM OpenAI API server.

This script benchmarks the vLLM serve OpenAI-compatible API endpoint.
It measures throughput (requests/sec, tokens/sec) under concurrent load.

Usage:
    # Start vLLM serve first (in another terminal):
    vllm serve Qwen/Qwen3-0.6B \
        --tensor-parallel-size 2 \
        --distributed-executor-backend docker \
        --port 8000

    # Run benchmark:
    python examples/docker_executor/benchmark_serve.py

    # Or with custom options:
    python examples/docker_executor/benchmark_serve.py \
        --url http://localhost:8000/v1/completions \
        --model Qwen/Qwen3-0.6B \
        --num-requests 100 \
        --concurrency 10 \
        --max-tokens 128

Options:
    --url               API endpoint URL (default: http://localhost:8000/v1/completions)
    --model             Model name (default: Qwen/Qwen3-0.6B)
    --num-requests      Total number of requests to send (default: 50)
    --concurrency       Number of concurrent requests (default: 5)
    --max-tokens        Max tokens to generate per request (default: 100)
    --prompt-len        Input prompt length in tokens (default: 50)
    --temperature       Sampling temperature (default: 0.0)
"""
import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import List

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip install aiohttp")
    raise


@dataclass
class RequestResult:
    """Result of a single request."""
    success: bool
    latency: float  # Total time in seconds
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str = ""


class Benchmark:
    """Benchmark vLLM OpenAI API server."""

    def __init__(
        self,
        url: str,
        model: str,
        num_requests: int,
        concurrency: int,
        max_tokens: int,
        prompt_len: int,
        temperature: float,
    ):
        self.url = url
        self.model = model
        self.num_requests = num_requests
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.prompt_len = prompt_len
        self.temperature = temperature

        # Generate a dummy prompt of approximately prompt_len tokens
        # (approximate: ~4 chars per token for English text)
        words = [
            "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
            "machine", "learning", "artificial", "intelligence", "neural", "network",
            "transformer", "attention", "mechanism", "language", "model", "generation",
            "token", "embedding", "vector", "matrix", "computation", "parallel",
            "distributed", "system", "architecture", "performance", "optimization",
        ]
        prompt_words = []
        current_len = 0
        while current_len < prompt_len:
            for word in words:
                prompt_words.append(word)
                current_len += 1
                if current_len >= prompt_len:
                    break
        self.prompt = " ".join(prompt_words) + "."

    async def _send_request(self, session: aiohttp.ClientSession, req_id: int) -> RequestResult:
        """Send a single completion request."""
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        start_time = time.perf_counter()
        try:
            async with session.post(self.url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return RequestResult(
                        success=False,
                        latency=time.perf_counter() - start_time,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        error=f"HTTP {response.status}: {error_text[:200]}",
                    )

                data = await response.json()
                latency = time.perf_counter() - start_time

                # Extract usage stats if available
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", self.prompt_len)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                return RequestResult(
                    success=True,
                    latency=latency,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
        except Exception as e:
            return RequestResult(
                success=False,
                latency=time.perf_counter() - start_time,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error=str(e),
            )

    async def _worker(
        self,
        session: aiohttp.ClientSession,
        queue: asyncio.Queue,
        results: List[RequestResult],
    ):
        """Worker that processes requests from queue."""
        while True:
            try:
                req_id = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            result = await self._send_request(session, req_id)
            results.append(result)
            queue.task_done()

    async def run(self) -> None:
        """Run the benchmark."""
        print("=" * 70)
        print("vLLM OpenAI API Benchmark")
        print("=" * 70)
        print(f"URL:              {self.url}")
        print(f"Model:            {self.model}")
        print(f"Total requests:   {self.num_requests}")
        print(f"Concurrency:      {self.concurrency}")
        print(f"Max tokens:       {self.max_tokens}")
        print(f"Prompt length:    ~{self.prompt_len} tokens")
        print(f"Temperature:      {self.temperature}")
        print(f"Prompt preview:   {self.prompt[:60]}...")
        print("=" * 70)
        print()

        # Create request queue
        queue = asyncio.Queue()
        for i in range(self.num_requests):
            queue.put_nowait(i)

        results: List[RequestResult] = []

        # Warmup - send one request first
        print("Warming up...")
        async with aiohttp.ClientSession() as session:
            warmup_result = await self._send_request(session, 0)
            if not warmup_result.success:
                print(f"Warmup failed: {warmup_result.error}")
                print("\nMake sure vLLM serve is running:")
                print(f"  vllm serve {self.model} --port 8000")
                return
            print(f"Warmup successful (latency: {warmup_result.latency:.2f}s)")
            print()

        # Run benchmark
        print(f"Running benchmark with {self.concurrency} concurrent clients...")
        print()

        start_time = time.perf_counter()

        async with aiohttp.ClientSession() as session:
            # Start workers
            workers = [
                asyncio.create_task(self._worker(session, queue, results))
                for _ in range(self.concurrency)
            ]

            # Wait for all workers to complete
            await asyncio.gather(*workers)

        total_time = time.perf_counter() - start_time

        # Calculate statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if not successful:
            print("ERROR: All requests failed!")
            for r in failed[:5]:
                print(f"  - {r.error}")
            return

        latencies = [r.latency for r in successful]
        total_prompt_tokens = sum(r.prompt_tokens for r in successful)
        total_completion_tokens = sum(r.completion_tokens for r in successful)
        total_tokens = sum(r.total_tokens for r in successful)

        # Print results
        print("=" * 70)
        print("Benchmark Results")
        print("=" * 70)
        print()
        print(f"Total time:           {total_time:.2f} seconds")
        print(f"Successful requests:  {len(successful)}/{self.num_requests}")
        print(f"Failed requests:      {len(failed)}")
        print()
        print("Throughput:")
        print(f"  Requests/sec:       {len(successful) / total_time:.2f}")
        print(f"  Input tokens/sec:   {total_prompt_tokens / total_time:.2f}")
        print(f"  Output tokens/sec:  {total_completion_tokens / total_time:.2f}")
        print(f"  Total tokens/sec:   {total_tokens / total_time:.2f}")
        print()
        print("Latency (seconds):")
        print(f"  Mean:               {sum(latencies) / len(latencies):.3f}")
        print(f"  Min:                {min(latencies):.3f}")
        print(f"  Max:                {max(latencies):.3f}")
        print(f"  P50:                {self._percentile(latencies, 0.50):.3f}")
        print(f"  P90:                {self._percentile(latencies, 0.90):.3f}")
        print(f"  P99:                {self._percentile(latencies, 0.99):.3f}")
        print()
        print("Token Statistics:")
        avg_prompt = total_prompt_tokens / len(successful)
        avg_completion = total_completion_tokens / len(successful)
        print(f"  Avg prompt tokens:      {avg_prompt:.1f}")
        print(f"  Avg completion tokens:  {avg_completion:.1f}")
        print(f"  Total prompt tokens:    {total_prompt_tokens}")
        print(f"  Total completion tokens:{total_completion_tokens}")
        print()

        if failed:
            print("Errors (first 5):")
            for i, r in enumerate(failed[:5]):
                print(f"  {i+1}. {r.error[:100]}")
            print()

        print("=" * 70)

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of a list."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM OpenAI API server"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1/completions",
        help="API endpoint URL (default: http://localhost:8000/v1/completions)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Total number of requests to send (default: 50)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens to generate per request (default: 100)",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=50,
        help="Input prompt length in tokens (default: 50)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )

    args = parser.parse_args()

    benchmark = Benchmark(
        url=args.url,
        model=args.model,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        prompt_len=args.prompt_len,
        temperature=args.temperature,
    )

    asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()
