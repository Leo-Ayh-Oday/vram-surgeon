#!/usr/bin/env python3
"""VRAM Surgeon — 本地真机 PyTorch 遥测校准探针（契约 §8，v1.1.1）。

轻量独立可运行，依赖仅 torch + 标准库。Windows 原生可跑（pathlib，禁 POSIX-only 模块）。

用法:
    python vram_telemetry_probe.py [--model-id ID] [--seq-len N] [--batch-size N]
                                   [--hidden-size H] [--num-layers L] [--vocab-size V]
                                   [--num-heads A] [--num-kv-heads KV] [--intermediate-size I]
                                   [--head-dim D] [--bandwidth] [--output PATH]
                                   [--mock-rtx5050]

    --mock-rtx5050: 显式 opt-in 的无 GPU 模拟模式（开发机用）。不依赖 torch，
        生成物理 plausible 的 RTX 5050 8GB + Windows WDDM 校准基准，
        probeVersion 带 "-mock" 后缀，默认落盘 sample_rtx5050_telemetry.json。
        模拟数据不得与真机实测数据混淆。

纪律:
    * stdout 只输出一个 JSON 对象，严格匹配 HardwareTelemetryReport（schemaVersion='vram-telemetry/1'）。
    * 所有日志走 stderr；同时落盘 telemetry_report.json 供前端"理论 vs 实测"双轨校准。
    * 字节→GiB 一律 1073741824（2^30）。
    * 退出码: 0 成功；2 无 CUDA（stderr 说明，stdout 不输出残缺 JSON）。

校准反哺: CUDA_CONTEXT_GIB(0.65)/OS_RESERVED_WDDM_GIB(0.75)/FRAG_* 为先验估算；
本探针 calibration 字段为实测值，若系统性偏离先验 >20% 由总控走契约变更流程修订常量。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# 契约 §1/§8: 二进制吉字节，唯一换算常量。
BYTES_PER_GIB = 1073741824
SCHEMA_VERSION = "vram-telemetry/1"
PROBE_VERSION = "1.1.1"
DEFAULT_OUTPUT_FILENAME = "telemetry_report.json"
MOCK_OUTPUT_FILENAME = "sample_rtx5050_telemetry.json"
MOCK_PROBE_VERSION_SUFFIX = "-mock"

# --mock-rtx5050 校准基准先验值（契约 §1/§8：WDDM≈0.75、context≈0.65、碎片率 0.035–0.14 量级）。
MOCK_GPU_NAME = "NVIDIA GeForce RTX 5050"
MOCK_TOTAL_CAPACITY_GIB = 8.0
MOCK_COMPUTE_CAPABILITY = "12.0"
MOCK_CUDA_CONTEXT_GIB = 0.652
MOCK_WDDM_RESERVED_GIB = 0.75
MOCK_EMPTY_PROCESS_BASELINE_GIB = round(MOCK_CUDA_CONTEXT_GIB + MOCK_WDDM_RESERVED_GIB, 6)
MOCK_FRAGMENTATION_RATIO = 0.087
MOCK_BANDWIDTH_GB_PER_SEC = 298.4  # 标称 320 GB/s（GDDR7 128-bit）的 plausible 实测值

# 模型维度预设（与 src/core/presets/models.ts 对齐，用于 --model-id 缺省维度）。
MODEL_PRESETS = {
    "Qwen/Qwen2.5-7B-Instruct": dict(h=3584, i=18944, L=28, a=28, kv=4, d=128, V=152064),
    "meta-llama/Llama-3.1-8B-Instruct": dict(h=4096, i=14336, L=32, a=32, kv=8, d=128, V=128256),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": dict(h=3584, i=18944, L=28, a=28, kv=4, d=128, V=152064),
    "Qwen/Qwen2.5-3B-Instruct": dict(h=2048, i=11008, L=36, a=16, kv=2, d=128, V=151936),
    "Qwen/Qwen2.5-14B-Instruct": dict(h=5120, i=13824, L=48, a=40, kv=8, d=128, V=152064),
    "google/gemma-2-9b-it": dict(h=3584, i=14336, L=42, a=16, kv=8, d=256, V=256000),
}
DEFAULT_PRESET = "Qwen/Qwen2.5-7B-Instruct"


def log(msg: str) -> None:
    """所有日志走 stderr（契约 §8：stdout 仅一个 JSON）。"""
    print(f"[vram-probe] {msg}", file=sys.stderr, flush=True)


def bytes_to_gib(n: float) -> float:
    return float(n) / BYTES_PER_GIB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VRAM Surgeon telemetry probe (contract §8)")
    parser.add_argument("--model-id", default=DEFAULT_PRESET)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--bandwidth", action="store_true", help="run optional bandwidth benchmark")
    parser.add_argument("--output", default=None,
                        help="output JSON path (default: telemetry_report.json; mock mode: sample_rtx5050_telemetry.json)")
    parser.add_argument("--mock-rtx5050", action="store_true",
                        help="explicit opt-in: emit synthetic RTX 5050 8GB WDDM calibration baseline without torch/GPU")
    return parser.parse_args()


def resolve_dims(args: argparse.Namespace) -> dict:
    preset = MODEL_PRESETS.get(args.model_id, MODEL_PRESETS[DEFAULT_PRESET])
    return {
        "h": args.hidden_size if args.hidden_size is not None else preset["h"],
        "i": args.intermediate_size if args.intermediate_size is not None else preset["i"],
        "L": args.num_layers if args.num_layers is not None else preset["L"],
        "a": args.num_heads if args.num_heads is not None else preset["a"],
        "kv": args.num_kv_heads if args.num_kv_heads is not None else preset["kv"],
        "d": args.head_dim if args.head_dim is not None else preset["d"],
        "V": args.vocab_size if args.vocab_size is not None else preset["V"],
    }


def query_nvidia_smi(query: str) -> str | None:
    """跨平台 nvidia-smi 查询（Windows/Linux 均可用）；失败返回 None。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()
            if line:
                return line[0].strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def detect_os_info(torch_mod) -> dict:
    """采集 OS / WDDM-TCC 信息（契约 §8）。Windows 用 driver_model.current；Linux 填 unknown。"""
    system = platform.system()
    is_windows = system == "Windows"
    plat = "windows" if is_windows else "linux"
    driver_model = "unknown"
    display_attached = False
    if is_windows:
        # Windows 桌面会话默认 WDDM；实测以 nvidia-smi driver_model.current 为准。
        current = query_nvidia_smi("driver_model.current")
        if current:
            driver_model = "wddm" if "WDDM" in current.upper() else ("tcc" if "TCC" in current.upper() else "unknown")
        else:
            driver_model = "wddm"
        display_attached = True
    else:
        display_attached = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    os_env = "windows_wddm" if (is_windows and driver_model == "wddm") else "linux_headless"
    return {
        "platform": plat,
        "osEnv": os_env,
        "displayAttached": display_attached,
        "driverModel": driver_model,
    }


def detect_allocator(torch_mod) -> tuple[str, bool]:
    """读取 PYTORCH_CUDA_ALLOC_CONF 推断 allocator backend 与 expandable_segments。"""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    backend = "cudaMallocAsync" if "cudaMallocAsync" in conf else "native"
    expandable = "expandable_segments:True" in conf
    return backend, expandable


def measure_fragmentation(torch_mod, device) -> float:
    """空张量分配/释放循环测碎片率 = (reserved - allocated) / allocated（契约 §8）。"""
    blocks = []
    # 交错分配不同尺寸制造空洞，再释放部分，观察 reserved 与 allocated 的差额。
    for idx in range(64):
        numel = (idx % 8 + 1) * 1024 * 256
        blocks.append(torch_mod.empty(numel, dtype=torch_mod.float32, device=device))
    for idx in range(0, len(blocks), 2):
        blocks[idx] = None
    torch_mod.cuda.synchronize()
    allocated = torch_mod.cuda.memory_allocated(device)
    reserved = torch_mod.cuda.memory_reserved(device)
    del blocks
    torch_mod.cuda.empty_cache()
    if allocated <= 0:
        return 0.0
    return float(reserved - allocated) / float(allocated)


def measure_bandwidth(torch_mod, device) -> float | None:
    """可选大张量拷贝计时测带宽（十进制 GB/s）；未测为 null。"""
    try:
        numel = 256 * 1024 * 1024 // 4  # 256 MiB FP32
        src = torch_mod.empty(numel, dtype=torch_mod.float32, device=device)
        dst = torch_mod.empty(numel, dtype=torch_mod.float32, device=device)
        torch_mod.cuda.synchronize()
        iters = 10
        start = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src)
        torch_mod.cuda.synchronize()
        elapsed = time.perf_counter() - start
        del src, dst
        torch_mod.cuda.empty_cache()
        if elapsed <= 0:
            return None
        total_bytes = numel * 4 * iters * 2  # 读+写
        return float(total_bytes) / elapsed / 1e9  # 十进制 GB/s（契约 §1 带宽用十进制）
    except RuntimeError:
        return None


def collect_sample(torch_mod, device, step_index: int, phase: str) -> dict:
    """采样前 reset_peak_memory_stats，记录 allocated/reserved 峰值与 deviceFree（契约 §8）。"""
    torch_mod.cuda.synchronize()
    torch_mod.cuda.reset_peak_memory_stats(device)
    allocated = torch_mod.cuda.memory_allocated(device)
    reserved = torch_mod.cuda.memory_reserved(device)
    max_allocated = torch_mod.cuda.max_memory_allocated(device)
    max_reserved = torch_mod.cuda.max_memory_reserved(device)
    free_bytes, _total = torch_mod.cuda.mem_get_info(device)
    return {
        "stepIndex": step_index,
        "phase": phase,
        "allocatedGib": bytes_to_gib(allocated),
        "reservedGib": bytes_to_gib(reserved),
        "maxAllocatedGib": bytes_to_gib(max_allocated),
        "maxReservedGib": bytes_to_gib(max_reserved),
        "deviceFreeGib": bytes_to_gib(free_bytes),
    }


def run_workload(torch_mod, device, dims: dict, seq_len: int, batch_size: int) -> list[dict]:
    """模拟 Transformer 层前向 + Cross-Entropy 峰值 + 反向 + 优化器步（契约 §8）。"""
    h = dims["h"]
    i = dims["i"]
    L = dims["L"]
    a = dims["a"]
    kv = dims["kv"]
    d = dims["d"]
    V = dims["V"]
    b = batch_size
    s = seq_len
    samples: list[dict] = []
    step = 0

    # 构造单层 Transformer 维度张量（SwiGLU + GQA），按传入 h/L/V 模拟。
    q_dim = a * d
    kv_dim = kv * d
    x = torch_mod.randn(b, s, h, dtype=torch_mod.bfloat16, device=device, requires_grad=True)
    wq = torch_mod.randn(h, q_dim, dtype=torch_mod.bfloat16, device=device, requires_grad=True)
    wk = torch_mod.randn(h, kv_dim, dtype=torch_mod.bfloat16, device=device, requires_grad=True)
    wv = torch_mod.randn(h, kv_dim, dtype=torch_mod.bfloat16, device=device, requires_grad=True)
    wo = torch_mod.randn(q_dim, h, dtype=torch_mod.bfloat16, device=device, requires_grad=True)
    wgate = torch_mod.randn(h, i, dtype=torch_mod.bfloat16, device=device, requires_grad=True)
    wup = torch_mod.randn(h, i, dtype=torch_mod.bfloat16, device=device, requires_grad=True)
    wdown = torch_mod.randn(i, h, dtype=torch_mod.bfloat16, device=device, requires_grad=True)

    # forward：逐层累计（模拟 L 层激活），SDPA 等效 FlashAttention（禁 math backend）。
    torch_mod.cuda.reset_peak_memory_stats(device)
    hidden = x
    for _layer in range(L):
        q = (hidden @ wq).view(b, s, a, d).transpose(1, 2)
        k = (hidden @ wk).view(b, s, kv, d).transpose(1, 2)
        v = (hidden @ wv).view(b, s, kv, d).transpose(1, 2)
        if kv != a:
            repeat = a // kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        attn = torch_mod.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, s, q_dim)
        proj = attn @ wo
        mlp = (torch_mod.nn.functional.silu(hidden @ wgate) * (hidden @ wup)) @ wdown
        hidden = hidden + proj + mlp
    torch_mod.cuda.synchronize()
    step += 1
    samples.append(collect_sample(torch_mod, device, step, "forward"))

    # logits_spike：完整 [b,s,V] logits + FP32 upcast + cross_entropy（大词表瞬时峰值）。
    torch_mod.cuda.reset_peak_memory_stats(device)
    whead = torch_mod.randn(h, V, dtype=torch_mod.bfloat16, device=device, requires_grad=True)
    logits = hidden @ whead
    logits_fp32 = logits.float()
    targets = torch_mod.randint(0, V, (b, s), device=device)
    loss = torch_mod.nn.functional.cross_entropy(
        logits_fp32.view(-1, V), targets.view(-1)
    )
    torch_mod.cuda.synchronize()
    step += 1
    samples.append(collect_sample(torch_mod, device, step, "logits_spike"))

    # backward：反向传播峰值。
    torch_mod.cuda.reset_peak_memory_stats(device)
    loss.backward()
    torch_mod.cuda.synchronize()
    step += 1
    samples.append(collect_sample(torch_mod, device, step, "backward"))

    # optimizer_step：SGD 更新可训练张量（模拟优化器状态占用）。
    torch_mod.cuda.reset_peak_memory_stats(device)
    params = [x, wq, wk, wv, wo, wgate, wup, wdown, whead]
    optimizer = torch_mod.optim.SGD(params, lr=1e-4)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch_mod.cuda.synchronize()
    step += 1
    samples.append(collect_sample(torch_mod, device, step, "optimizer_step"))

    del x, wq, wk, wv, wo, wgate, wup, wdown, whead, hidden, logits, logits_fp32, loss
    torch_mod.cuda.empty_cache()
    return samples


def build_mock_samples(dims: dict, seq_len: int, batch_size: int) -> list[dict]:
    """--mock-rtx5050：按契约 §5.4/§5.5 方程推导各阶段 plausible 显存（模拟单层探针负载）。"""
    h = dims["h"]
    i = dims["i"]
    a = dims["a"]
    kv = dims["kv"]
    d = dims["d"]
    V = dims["V"]
    b = batch_size
    s = seq_len
    q_dim = a * d
    kv_dim = kv * d

    # 模拟负载 = 单层 SwiGLU+GQA 权重（BF16）+ 线性激活 + 完整 CE 尖峰（与 run_workload 同构）。
    layer_weight_bytes = (h * q_dim + 2 * h * kv_dim + q_dim * h + 3 * h * i) * 2
    act_linear_bytes = b * s * (10 * h + 2 * h * kv / a + 6 * i)  # 契约 §5.4: b*s*(10h+2h/G+6i)
    logits_bytes = 10 * b * s * V  # 契约 §5.5: unfused = 10*b*s*V
    grad_bytes = layer_weight_bytes  # backward 梯度与权重同量（BF16）
    opt_state_bytes = layer_weight_bytes * 2  # SGD 模拟优化器状态量级

    frag = MOCK_FRAGMENTATION_RATIO
    phases = [
        ("forward", layer_weight_bytes + act_linear_bytes),
        ("logits_spike", layer_weight_bytes + act_linear_bytes + logits_bytes),
        ("backward", layer_weight_bytes + act_linear_bytes + logits_bytes + grad_bytes),
        ("optimizer_step", layer_weight_bytes + grad_bytes + opt_state_bytes),
    ]
    samples: list[dict] = []
    for step, (phase, allocated_bytes) in enumerate(phases, start=1):
        allocated_gib = bytes_to_gib(allocated_bytes)
        reserved_gib = allocated_gib * (1.0 + frag) + MOCK_CUDA_CONTEXT_GIB
        samples.append({
            "stepIndex": step,
            "phase": phase,
            "allocatedGib": allocated_gib,
            "reservedGib": reserved_gib,
            "maxAllocatedGib": allocated_gib,
            "maxReservedGib": reserved_gib,
            "deviceFreeGib": MOCK_TOTAL_CAPACITY_GIB - reserved_gib - MOCK_WDDM_RESERVED_GIB,
        })
    return samples


def build_mock_report(args: argparse.Namespace) -> dict:
    """--mock-rtx5050：无 torch/无 GPU 生成 HardwareTelemetryReport 形状的合成校准基准。

    probeVersion 带 '-mock' 后缀，显式区别于真机实测数据。
    """
    dims = resolve_dims(args)
    log(f"MOCK 模式：合成 RTX 5050 8GB WDDM 基准，dims={dims} seq_len={args.seq_len} batch_size={args.batch_size}")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "probeVersion": PROBE_VERSION + MOCK_PROBE_VERSION_SUFFIX,
        "capturedAtUtc": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "name": MOCK_GPU_NAME,
            "driverVersion": "572.16",
            "cudaRuntimeVersion": "12.8",
            "totalCapacityGib": MOCK_TOTAL_CAPACITY_GIB,
            "computeCapability": MOCK_COMPUTE_CAPABILITY,
        },
        "torch": {
            "torchVersion": "2.7.0+cu128",
            "cudaAvailable": True,
            "allocatorBackend": "native",
            "expandableSegmentsEnabled": False,
        },
        "os": {
            "platform": "windows",
            "osEnv": "windows_wddm",
            "displayAttached": True,
            "driverModel": "wddm",
        },
        "calibration": {
            "cudaContextGib": MOCK_CUDA_CONTEXT_GIB,
            "emptyProcessBaselineGib": MOCK_EMPTY_PROCESS_BASELINE_GIB,
            "fragmentationRatio": MOCK_FRAGMENTATION_RATIO,
            "measuredBandwidthGbPerSec": MOCK_BANDWIDTH_GB_PER_SEC,
        },
        "samples": build_mock_samples(dims, args.seq_len, args.batch_size),
    }


def build_report(torch_mod, args: argparse.Namespace) -> dict:
    device = torch_mod.device("cuda:0")
    torch_mod.cuda.init()
    # CUDA Context 初始占用：init 前后差值（契约 §8）。
    context_gib = bytes_to_gib(torch_mod.cuda.memory_allocated(device))
    baseline_gib = bytes_to_gib(torch_mod.cuda.memory_reserved(device))

    props = torch_mod.cuda.get_device_properties(device)
    major, minor = torch_mod.cuda.get_device_capability(device)
    driver_version = query_nvidia_smi("driver_version") or "unknown"
    gpu_name = props.name
    total_capacity_gib = bytes_to_gib(props.total_memory)

    os_info = detect_os_info(torch_mod)
    allocator_backend, expandable = detect_allocator(torch_mod)

    dims = resolve_dims(args)
    log(f"workload dims={dims} seq_len={args.seq_len} batch_size={args.batch_size}")
    samples = run_workload(torch_mod, device, dims, args.seq_len, args.batch_size)

    fragmentation_ratio = measure_fragmentation(torch_mod, device)
    bandwidth = measure_bandwidth(torch_mod, device) if args.bandwidth else None

    report = {
        "schemaVersion": SCHEMA_VERSION,
        "probeVersion": PROBE_VERSION,
        "capturedAtUtc": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "name": gpu_name,
            "driverVersion": driver_version,
            "cudaRuntimeVersion": torch_mod.version.cuda or "unknown",
            "totalCapacityGib": total_capacity_gib,
            "computeCapability": f"{major}.{minor}",
        },
        "torch": {
            "torchVersion": torch_mod.__version__,
            "cudaAvailable": True,
            "allocatorBackend": allocator_backend,
            "expandableSegmentsEnabled": expandable,
        },
        "os": os_info,
        "calibration": {
            "cudaContextGib": context_gib,
            "emptyProcessBaselineGib": baseline_gib,
            "fragmentationRatio": fragmentation_ratio,
            "measuredBandwidthGbPerSec": bandwidth,
        },
        "samples": samples,
    }
    return report


def main() -> int:
    args = parse_args()

    # --mock-rtx5050：显式 opt-in，不依赖 torch，无 GPU 开发机可跑（契约 §8）。
    if args.mock_rtx5050:
        report = build_mock_report(args)
        default_name = MOCK_OUTPUT_FILENAME
    else:
        try:
            import torch  # 依赖仅 torch + 标准库
        except ImportError:
            log("PyTorch 未安装，无法采集遥测。请 pip install torch 后重试。")
            return 2

        if not torch.cuda.is_available():
            # 契约 §8：无 CUDA 退出码 2，stderr 说明，stdout 不输出残缺 JSON。
            log("未检测到可用 CUDA 设备（torch.cuda.is_available()=False）。退出码 2。")
            return 2

        try:
            report = build_report(torch, args)
        except RuntimeError as exc:
            log(f"遥测采集失败（CUDA RuntimeError）: {exc}")
            return 2
        default_name = DEFAULT_OUTPUT_FILENAME

    # 落盘（pathlib，Windows 原生可跑）：mock 默认写到脚本同目录的 sample 基准文件。
    output_path = Path(args.output) if args.output is not None else Path(__file__).resolve().parent / default_name
    try:
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"report written to {output_path.resolve()}")
    except OSError as exc:
        log(f"落盘失败（不影响 stdout JSON）: {exc}")

    # stdout 只输出一个 JSON 对象（契约 §8）。
    sys.stdout.write(json.dumps(report, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())