"""
Performance benchmarks for unified CPU/GPU reduction.

This script measures the performance of unified reduction kernels across
different CPU/GPU page distributions to validate the 85-95% performance target.
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vortex_torch.cache import Mean, Max, Min, L2Norm
from vortex_torch.cache.unified_view import UnifiedCacheView
from vortex_torch.cache.context import Context


def benchmark_scenario(
    scenario_name: str,
    cpu_ratio: float,
    ctx: Context,
    num_pages: int = 100,
    num_iterations: int = 50,
    warmup_iterations: int = 10
):
    """
    Benchmark a specific CPU/GPU ratio scenario.

    Args:
        scenario_name: Human-readable name for this scenario
        cpu_ratio: Fraction of pages in CPU (0.0 to 1.0)
        ctx: Context object with page configuration
        num_pages: Total number of TOKEN pages (will be multiplied by num_heads)
        num_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations before timing

    Returns:
        dict: Performance metrics including time, throughput, and relative performance
    """
    # Calculate page distribution
    # Note: buffer is [num_pages * num_heads, page_size, head_dim]
    total_pages = num_pages * ctx.head_num
    num_cpu_pages = int(total_pages * cpu_ratio)
    num_gpu_pages = total_pages - num_cpu_pages

    # Create buffers
    if num_cpu_pages > 0:
        cpu_buffer = torch.randn(
            (num_cpu_pages, ctx.page_size, ctx.head_dim),
            dtype=torch.bfloat16,
            pin_memory=True
        )
    else:
        cpu_buffer = None

    gpu_buffer = torch.randn(
        (num_gpu_pages, ctx.page_size, ctx.head_dim),
        dtype=torch.bfloat16,
        device='cuda'
    )

    # Create slot map
    # Convention: slot_map >= 0 means GPU slot, == -1 means CPU slot
    if num_cpu_pages > 0:
        slot_map = torch.cat([
            torch.arange(num_gpu_pages, dtype=torch.int32),  # First pages in GPU
            torch.full((num_cpu_pages,), -1, dtype=torch.int32)  # Rest in CPU
        ]).cuda()
    else:
        slot_map = torch.arange(num_gpu_pages, dtype=torch.int32, device='cuda')  # All in GPU

    # Create unified view
    unified_view = UnifiedCacheView(cpu_buffer, gpu_buffer, slot_map)

    # Create location tensor (page boundaries for tokens, not including head dimension)
    loc = torch.arange(0, num_pages, dtype=torch.int64, device='cuda') * ctx.page_size + (ctx.page_size - 1)

    # Setup Mean operator
    mean_op = Mean(dim=1)
    mean_op.profile(unified_view, None, loc, ctx)
    output = mean_op.output_buffer.clone()

    # Warmup
    for _ in range(warmup_iterations):
        mean_op.execute(unified_view, output, loc, ctx)

    torch.cuda.synchronize()

    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        mean_op.execute(unified_view, output, loc, ctx)
    end_event.record()

    torch.cuda.synchronize()

    # Calculate metrics
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iterations

    # Throughput: pages processed per second
    throughput_gpages_per_sec = (num_pages * num_iterations) / (elapsed_ms / 1000.0) / 1e9

    return {
        'scenario': scenario_name,
        'cpu_ratio': cpu_ratio,
        'num_cpu_pages': num_cpu_pages,
        'num_gpu_pages': num_gpu_pages,
        'avg_time_ms': avg_time_ms,
        'throughput_gpages_s': throughput_gpages_per_sec,
        'total_time_ms': elapsed_ms
    }


def print_results(results: list, baseline_time: float):
    """
    Print benchmark results in a formatted table.

    Args:
        results: List of result dictionaries from benchmark_scenario
        baseline_time: Time for all-GPU baseline (for relative performance)
    """
    print("\n" + "="*100)
    print("UNIFIED REDUCTION PERFORMANCE BENCHMARK")
    print("="*100)
    print(f"{'Scenario':<20} {'CPU%':>6} {'GPU%':>6} {'Time(ms)':>10} {'Throughput':>12} {'Relative':>10}")
    print(f"{'':20} {'':>6} {'':>6} {'':>10} {'(Gpages/s)':>12} {'Perf':>10}")
    print("-"*100)

    for result in results:
        cpu_pct = result['cpu_ratio'] * 100
        gpu_pct = 100 - cpu_pct
        relative_perf = (baseline_time / result['avg_time_ms']) * 100

        print(f"{result['scenario']:<20} {cpu_pct:>5.0f}% {gpu_pct:>5.0f}% "
              f"{result['avg_time_ms']:>10.3f} {result['throughput_gpages_s']:>12.3f} "
              f"{relative_perf:>9.1f}%")

    print("="*100)

    # Print target validation
    print("\nPERFORMANCE TARGETS:")
    fifty_pct_result = next((r for r in results if r['cpu_ratio'] == 0.5), None)
    if fifty_pct_result:
        fifty_pct_relative = (baseline_time / fifty_pct_result['avg_time_ms']) * 100
        target_met = 85.0 <= fifty_pct_relative <= 95.0
        status = "✓ MET" if target_met else "✗ MISSED"
        print(f"  50% CPU/GPU: {fifty_pct_relative:.1f}% (Target: 85-95%) ... {status}")

    ninety_pct_result = next((r for r in results if r['cpu_ratio'] == 0.9), None)
    if ninety_pct_result:
        ninety_pct_relative = (baseline_time / ninety_pct_result['avg_time_ms']) * 100
        target_met = 80.0 <= ninety_pct_relative <= 90.0
        status = "✓ MET" if target_met else "✗ MISSED"
        print(f"  90% CPU/GPU: {ninety_pct_relative:.1f}% (Target: 80-90%) ... {status}")

    print()


def main():
    """Run comprehensive performance benchmark suite."""
    print("Initializing unified reduction benchmarks...")
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch Version: {torch.__version__}")

    # Setup context
    ctx = Context()
    ctx.page_size = 16
    ctx.head_num = 8
    ctx.head_dim = 128
    ctx.max_new_tokens_per_batch = 32
    ctx._created = True

    print(f"\nConfiguration:")
    print(f"  Page Size: {ctx.page_size}")
    print(f"  Num Heads: {ctx.head_num}")
    print(f"  Head Dim: {ctx.head_dim}")
    print(f"  Total Pages: 100")
    print(f"  Iterations: 50 (+ 10 warmup)")

    # Define scenarios
    scenarios = [
        ("All GPU (baseline)", 0.0),
        ("10% CPU / 90% GPU", 0.1),
        ("25% CPU / 75% GPU", 0.25),
        ("50% CPU / 50% GPU", 0.5),
        ("75% CPU / 25% GPU", 0.75),
        ("90% CPU / 10% GPU", 0.9),
        ("All CPU", 1.0),
    ]

    results = []

    print("\nRunning benchmarks...")
    for scenario_name, cpu_ratio in scenarios:
        print(f"  [{len(results)+1}/{len(scenarios)}] {scenario_name}...", end=" ", flush=True)
        result = benchmark_scenario(scenario_name, cpu_ratio, ctx)
        results.append(result)
        print(f"{result['avg_time_ms']:.3f} ms")

    # Print formatted results
    baseline_time = results[0]['avg_time_ms']
    print_results(results, baseline_time)

    # Additional analysis
    print("\nANALYSIS:")

    # PCIe overhead estimation
    ten_pct = next((r for r in results if r['cpu_ratio'] == 0.1), None)
    if ten_pct:
        overhead = ((ten_pct['avg_time_ms'] - baseline_time) / baseline_time) * 100
        print(f"  PCIe overhead at 10% CPU: {overhead:.1f}%")

    # Scalability check
    fifty_pct = next((r for r in results if r['cpu_ratio'] == 0.5), None)
    ninety_pct = next((r for r in results if r['cpu_ratio'] == 0.9), None)
    if fifty_pct and ninety_pct:
        time_increase = ((ninety_pct['avg_time_ms'] - fifty_pct['avg_time_ms']) /
                        fifty_pct['avg_time_ms']) * 100
        print(f"  Time increase from 50% to 90% CPU: {time_increase:.1f}%")

    # All CPU vs All GPU comparison
    all_cpu = next((r for r in results if r['cpu_ratio'] == 1.0), None)
    if all_cpu:
        slowdown = (all_cpu['avg_time_ms'] / baseline_time)
        print(f"  All-CPU slowdown vs All-GPU: {slowdown:.2f}x")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
