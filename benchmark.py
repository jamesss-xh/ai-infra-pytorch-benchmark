"""
AI Infra 入门项目：PyTorch CPU 推理性能与批处理测试
环境：Python 3.11+、PyTorch、Matplotlib
默认设备：CPU（不需要 CUDA 或昇腾 NPU）

运行：
    python benchmark.py
可选：
    python benchmark.py --iterations 100 --warmup 20 --threads 16
    python benchmark.py --batch-sizes 1 4 8 16 32 64
"""

import argparse
import csv
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 保存图片，不依赖弹出式图形窗口
import matplotlib.pyplot as plt
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


class TinyMLP(nn.Module):
    """用于基准测试的小型全连接网络，不代表真实生产模型。"""

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def percentile(sorted_values, pct):
    """计算简单的最近秩百分位数，返回单位与输入相同的值。"""
    if not sorted_values:
        return float("nan")
    idx = max(0, min(len(sorted_values) - 1,
                     int(round((pct / 100) * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def benchmark_batch(model, batch_size, input_dim, warmup, iterations, device):
    """测量一个批次大小的单次 forward 延迟与吞吐量。"""
    x = torch.randn(batch_size, input_dim, device=device)

    # 预热：让运行过程不只包含第一次调用的开销
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)

        # 同步接口留在这里，方便未来迁移到异步设备时扩展。
        # 当前 CPU 运行时 torch.cuda.is_available() 为 False，无需 CUDA 同步。
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            output = model(x)
            elapsed = time.perf_counter() - start
            samples.append(elapsed)
            # 数值检查放在计时区间之外，避免把检查开销计入推理延迟
            if not torch.isfinite(output).all().item():
                raise RuntimeError(f"模型输出出现 NaN/Inf，batch_size={batch_size}")

    ordered = sorted(samples)
    mean_seconds = statistics.mean(samples)
    median_seconds = statistics.median(samples)
    p95_seconds = percentile(ordered, 95)

    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup": warmup,
        "mean_batch_latency_ms": mean_seconds * 1000,
        "median_batch_latency_ms": median_seconds * 1000,
        "p95_batch_latency_ms": p95_seconds * 1000,
        "throughput_samples_per_sec": batch_size / mean_seconds,
        "mean_latency_per_sample_ms": mean_seconds * 1000 / batch_size,
    }


def check_batch_consistency(model, input_dim, device):
    """对同一批输入比较批量计算与逐条计算的结果，检查数值一致性。"""
    x = torch.randn(8, input_dim, device=device)
    with torch.inference_mode():
        batched = model(x)
        single = torch.cat([model(x[i:i + 1]) for i in range(x.shape[0])], dim=0)
    return bool(torch.allclose(batched, single, atol=1e-5, rtol=1e-4))


def write_csv(rows, output_path):
    fieldnames = [
        "batch_size", "iterations", "warmup",
        "mean_batch_latency_ms", "median_batch_latency_ms",
        "p95_batch_latency_ms", "throughput_samples_per_sec",
        "mean_latency_per_sample_ms",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows, output_dir):
    batch_sizes = [r["batch_size"] for r in rows]
    latencies = [r["mean_batch_latency_ms"] for r in rows]
    throughputs = [r["throughput_samples_per_sec"] for r in rows]
    per_sample = [r["mean_latency_per_sample_ms"] for r in rows]

    # 图 1：每次 forward 处理一个 batch 的总延迟
    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, latencies, marker="o")
    plt.xlabel("Batch size (samples per forward)")
    plt.ylabel("Mean batch latency (ms)")
    plt.title("PyTorch CPU Inference: Batch Latency")
    plt.xticks(batch_sizes)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "batch_latency.png", dpi=160)
    plt.close()

    # 图 2：每秒处理的样本数
    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, throughputs, marker="o")
    plt.xlabel("Batch size (samples per forward)")
    plt.ylabel("Throughput (samples/second)")
    plt.title("PyTorch CPU Inference: Throughput")
    plt.xticks(batch_sizes)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "throughput.png", dpi=160)
    plt.close()

    # 图 3：平均到每个样本的时间（不等同于单个请求的端到端延迟）
    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, per_sample, marker="o")
    plt.xlabel("Batch size (samples per forward)")
    plt.ylabel("Mean batch latency / batch size (ms)")
    plt.title("Average Compute Time per Sample")
    plt.xticks(batch_sizes)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "per_sample_time.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PyTorch CPU batch inference benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    if any(b <= 0 for b in args.batch_sizes):
        parser.error("batch sizes must all be positive integers")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be >= 0 and iterations must be >= 1")

    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)

    input_dim = 256
    model = TinyMLP(input_dim=input_dim).to(device)
    model.eval()

    print("=" * 72)
    print("AI Infra 项目：PyTorch CPU 推理性能与批处理测试")
    print(f"时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"OS：{platform.platform()}")
    print(f"Python：{platform.python_version()}")
    print(f"PyTorch：{torch.__version__}")
    print(f"设备：{device}")
    print(f"CUDA available：{torch.cuda.is_available()}")
    print(f"PyTorch CPU threads：{torch.get_num_threads()}")
    print(f"warmup={args.warmup}, iterations={args.iterations}")
    print("=" * 72)

    consistent = check_batch_consistency(model, input_dim, device)
    print(f"批量计算与逐条计算结果一致性检查：{consistent}")
    if not consistent:
        raise RuntimeError("一致性检查失败，请先排查模型或环境问题。")

    rows = []
    for batch_size in args.batch_sizes:
        row = benchmark_batch(
            model=model,
            batch_size=batch_size,
            input_dim=input_dim,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        rows.append(row)
        print(
            f"batch={row['batch_size']:>3} | "
            f"平均批次延迟={row['mean_batch_latency_ms']:.4f} ms | "
            f"P95={row['p95_batch_latency_ms']:.4f} ms | "
            f"吞吐量={row['throughput_samples_per_sec']:.2f} samples/s | "
            f"平均每样本计算时间={row['mean_latency_per_sample_ms']:.4f} ms"
        )

    csv_path = RESULTS_DIR / "benchmark_results.csv"
    write_csv(rows, csv_path)
    make_plots(rows, RESULTS_DIR)

    print("\n完成。以下文件由本次实际运行生成：")
    print(f"CSV 数据：{csv_path}")
    print(f"批次延迟图：{RESULTS_DIR / 'batch_latency.png'}")
    print(f"吞吐量图：{RESULTS_DIR / 'throughput.png'}")
    print(f"每样本计算时间图：{RESULTS_DIR / 'per_sample_time.png'}")
    print("\n提醒：本实验是合成输入 + 小型 MLP 的 CPU 微基准测试，不代表真实大模型或生产服务性能。")


if __name__ == "__main__":
    main()
