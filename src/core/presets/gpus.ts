/**
 * VRAM Surgeon — GPU 预设库（契约 §3.1，v1.1.1）
 *
 * 纯数据，无依赖。fp16Tflops / int8Tflops 均为 dense（非稀疏）峰值。
 * 容量语义：vramGb 数值上直接等于 GiB（二进制容量），禁止 ×10⁹÷2³⁰（契约 §1）。
 */
import type { GpuSpec } from '../types';

/** 核心默认基准卡：校准/训练机 RTX 5050 8GB（纯 Windows WDDM）。 */
export const DEFAULT_GPU_ID = 'rtx-5050-8g';

export const GPU_PRESETS: readonly GpuSpec[] = [
  {
    id: 'rtx-5050-8g',
    name: 'NVIDIA GeForce RTX 5050',
    vramGb: 8,
    architecture: 'Blackwell',
    memoryBusWidthBits: 128,
    // 契约 §3.1 修正：官方 GDDR7 128-bit @20Gbps = 320 GB/s（原规范 256 有误）。
    memoryBandwidthGbPerSec: 320,
    fp16Tflops: 52.7,
    int8Tflops: 105.4,
    supportsBf16: true,
    supportsFp8: true,
  },
  {
    id: 'rtx-4060-8g',
    name: 'NVIDIA GeForce RTX 4060',
    vramGb: 8,
    architecture: 'Ada',
    memoryBusWidthBits: 128,
    memoryBandwidthGbPerSec: 272,
    fp16Tflops: 60.7,
    int8Tflops: 121.4,
    supportsBf16: true,
    supportsFp8: true,
  },
  {
    id: 'rtx-4070-12g',
    name: 'NVIDIA GeForce RTX 4070',
    vramGb: 12,
    architecture: 'Ada',
    memoryBusWidthBits: 192,
    memoryBandwidthGbPerSec: 504,
    fp16Tflops: 116.6,
    int8Tflops: 233.2,
    supportsBf16: true,
    supportsFp8: true,
  },
  {
    id: 'rtx-4060ti-16g',
    name: 'NVIDIA GeForce RTX 4060 Ti 16GB',
    vramGb: 16,
    architecture: 'Ada',
    memoryBusWidthBits: 128,
    memoryBandwidthGbPerSec: 288,
    fp16Tflops: 60.7,
    int8Tflops: 121.4,
    supportsBf16: true,
    supportsFp8: true,
  },
  {
    // v1.1.1 新增：开发机 RTX 3060 12GB（Ampere, 192-bit, 360 GB/s）。
    id: 'rtx-3060-12g',
    name: 'NVIDIA GeForce RTX 3060 12GB',
    vramGb: 12,
    architecture: 'Ampere',
    memoryBusWidthBits: 192,
    memoryBandwidthGbPerSec: 360,
    fp16Tflops: 51.2,
    int8Tflops: 102.4,
    supportsBf16: true,
    supportsFp8: false,
  },
  {
    id: 'rtx-3090-24g',
    name: 'NVIDIA GeForce RTX 3090',
    vramGb: 24,
    architecture: 'Ampere',
    memoryBusWidthBits: 384,
    memoryBandwidthGbPerSec: 936,
    fp16Tflops: 142.2,
    int8Tflops: 284.4,
    supportsBf16: true,
    supportsFp8: false,
  },
  {
    id: 'rtx-4090-24g',
    name: 'NVIDIA GeForce RTX 4090',
    vramGb: 24,
    architecture: 'Ada',
    memoryBusWidthBits: 384,
    memoryBandwidthGbPerSec: 1008,
    fp16Tflops: 330.3,
    int8Tflops: 660.6,
    supportsBf16: true,
    supportsFp8: true,
  },
  {
    id: 'a100-40g',
    name: 'NVIDIA A100 40GB',
    vramGb: 40,
    architecture: 'Ampere',
    memoryBusWidthBits: 5120,
    memoryBandwidthGbPerSec: 1555,
    fp16Tflops: 312,
    int8Tflops: 624,
    supportsBf16: true,
    supportsFp8: false,
  },
  {
    id: 'a100-80g',
    name: 'NVIDIA A100 80GB',
    vramGb: 80,
    architecture: 'Ampere',
    memoryBusWidthBits: 5120,
    memoryBandwidthGbPerSec: 2039,
    fp16Tflops: 312,
    int8Tflops: 624,
    supportsBf16: true,
    supportsFp8: false,
  },
];

export function getGpuById(id: string): GpuSpec | undefined {
  return GPU_PRESETS.find((gpu) => gpu.id === id);
}