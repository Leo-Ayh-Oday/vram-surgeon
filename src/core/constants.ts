/**
 * VRAM Surgeon — 物理常数法典（契约 §1，v1.1.1 冻结值）
 *
 * 所有物理魔法数字必须在此命名常量化，各模块统一 import，禁止裸数字散落。
 * 单位纪律：1 GiB = 2^30 Bytes（BYTES_PER_GIB 定义于 types.ts）；
 * GPU 容量语义为二进制 GiB（gpuCapacityGib = gpu.vramGb，禁止 ×10⁹÷2³⁰）。
 * 十进制 BYTES_PER_GB_DECIMAL 仅用于带宽等厂商标称量，不参与容量计算。
 */

// --- 数值精度字节宽度（契约 §1） ---
/** FP32 单参数字节。 */
export const BYTES_PER_FP32 = 4;
/** BF16/FP16 单参数字节。 */
export const BYTES_PER_BF16 = 2;
/** LLM.int8() 权重 + absmax 综合开销（Bytes/param）。 */
export const INT8_BYTES_PER_PARAM = 1.05;
/** NF4 单参数字节（4 bit = 0.5 Byte）。 */
export const BYTES_PER_NF4 = 0.5;
/** bitsandbytes NF4 量化块大小。 */
export const NF4_BLOCK_SIZE = 64;
/** 双量化 absmax 分组大小。 */
export const NF4_DOUBLE_QUANT_BLOCK = 256;
/** 双量化开销 = 1/64 + 4/(64×256) ≈ 0.127 bits/param（契约钉死值）。 */
export const NF4_DQ_OVERHEAD_BYTES = 0.015869140625;
/** 未开双量化：FP32 absmax / 64 = 0.0625 Bytes/param。 */
export const NF4_NO_DQ_OVERHEAD_BYTES = 0.0625;

// --- 梯度 / 优化器状态（契约 §1、§5.3） ---
/** LoRA 适配器 BF16 副本(2) + BF16 梯度(2) = 4 Bytes/param，归 gradAndOpt。 */
export const LORA_COPY_GRAD_BYTES_PER_PARAM = 4;
/** full 模式 BF16 梯度（权重已在 weights 计）= 2 Bytes/param。 */
export const FULL_GRAD_BYTES_PER_PARAM = 2;
/** adamw_32bit：FP32 master(4) + m(4) + v(4) = 12 Bytes/param。 */
export const ADAMW32_OPT_BYTES_PER_PARAM = 12;
/** adamw_8bit / paged_adamw_8bit：FP32 master(4) + INT8 m/v(2) + 2×(4/2048) absmax = 6.00390625。 */
export const ADAMW8_OPT_BYTES_PER_PARAM = 6.00390625;
/** 8bit 优化器 absmax 块大小。 */
export const ADAMW8_STATE_BLOCK = 2048;
/** sgd：仅 FP32 master，无动量 = 4 Bytes/param。 */
export const SGD_OPT_BYTES_PER_PARAM = 4;
/** paged AdamW 最高可分页到 CPU 的优化器状态比例（封顶 60%）。 */
export const PAGED_OPT_MAX_CPU_FRACTION = 0.6;

// --- 激活模型（契约 §1、§5.4，保守工程上界） ---
/** 每层 hidden 侧激活因子（Bytes / (b·s·h)）。 */
export const ACT_HIDDEN_FACTOR = 10;
/** GQA KV 项因子（Bytes / (b·s·h/G)）。 */
export const ACT_KV_FACTOR = 2;
/** SwiGLU gate/up/act 因子（Bytes / (b·s·i)）。 */
export const ACT_MLP_FACTOR = 6;
/** eager 注意力 FP32 softmax 矩阵（Bytes / (b·a·s²)）。 */
export const EAGER_ATTN_BYTES_PER_ELEM = 4;
/** FlashAttention 片上分块，S² 项归零。 */
export const FLASH_ATTN_MATRIX_BYTES = 0;
/** checkpoint 反向重算单层的峰值放大系数。 */
export const CHECKPOINT_RECOMPUTE_FACTOR = 1.25;

// --- 输出层 logits 峰值（契约 §1、§5.5） ---
/** unfused：BF16 logits(2) + FP32 upcast(4) + FP32 softmax/grad(4) = 10 Bytes/elem。 */
export const LOGITS_UNFUSED_BYTES_PER_ELEM = 10;
/** Liger/Unsloth 融合 CE 的词表维分块大小（切词表维，非 token 维）。 */
export const FUSED_CE_CHUNK = 4096;

// --- 系统 / 碎片（契约 §1、§5.6） ---
/** CUDA Runtime Context + cuBLAS/Triton workspace（GiB）。 */
export const CUDA_CONTEXT_GIB = 0.65;
/** Windows WDDM 桌面合成器与驱动抢占（GiB）；linux_headless = 0。 */
export const OS_RESERVED_WDDM_GIB = 0.75;
/** dynamicPadding 且未开 expandable 的动态张量碎片率。 */
export const FRAG_DYNAMIC_PADDING = 0.14;
/** expandable_segments 或定长 Packing 的碎片率。 */
export const FRAG_STABLE = 0.035;

// --- 诊断 / 手术阈值（契约 §1、§6） ---
/** 手术停止条件：reservedPeak ≤ 0.85×capacity（安全绿区）。 */
export const HEAL_TARGET_RATIO = 0.85;
/** utilizationRatio ≥ 此值判 MEMORY_BOUND。 */
export const MEMORY_BOUND_THRESHOLD = 0.85;
/** 碎片诊断绝对阈值（GiB）。 */
export const FRAG_GAP_ABS_GIB = 1.0;
/** 碎片诊断相对阈值（×capacity）。 */
export const FRAG_GAP_RATIO = 0.10;
/** triedToAllocate 与 logits 候选峰值匹配容差 ±25%。 */
export const LOGITS_SPIKE_MATCH_TOLERANCE = 0.25;
/** 末位有损手段：序列长度缩减因子。 */
export const SEQ_LEN_LAST_RESORT_FACTOR = 0.75;

// --- 代码生成 / 解析（契约 §1、§6.2、§7） ---
/** launch 脚本注入的 max_split_size_mb。 */
export const MAX_SPLIT_SIZE_MB = 128;
/** OOM 日志 MiB→GiB 归一。 */
export const MIB_PER_GIB = 1024;
/** HF 镜像端点（降级重试 / 可选 HF_ENDPOINT）。 */
export const HF_MIRROR_ENDPOINT = 'https://hf-mirror.com';
/** HF 官方端点。 */
export const HF_PRIMARY_ENDPOINT = 'https://huggingface.co';
/** fetchModelSpec 默认网络超时（ms），命名常量（契约 §4.1）。 */
export const HF_FETCH_TIMEOUT_MS = 10_000;
/** LLaMA-Factory 输出目录命名常量（契约 §7.2）。 */
export const LLAMA_FACTORY_OUTPUT_DIR = 'saves/vram-surgeon';
/** 8GB 及以下小显存卡阈值（契约 §7.3：自动 load_in_4bit）。 */
export const SMALL_VRAM_GB_THRESHOLD = 8;
/** 大词表阈值（契约 §7.5：V>100000 触发 logits 尖峰告警）。 */
export const LARGE_VOCAB_THRESHOLD = 100_000;
/** FlashAttention-2 最低 compute capability 主版本（Ampere ≥ 8.0，契约 §7.4）。 */
export const FA2_MIN_COMPUTE_CAPABILITY_MAJOR = 8;
/** bitsandbytes 官方 Windows 支持版本线（契约 §7.5）。 */
export const BITSANDBYTES_WINDOWS_MIN_VERSION = '0.43';