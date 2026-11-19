/**
 * VRAM Surgeon — 全局类型数据契约（冻结版 v1.1.0）
 *
 * 本文件是全项目唯一的数据契约源（Single Source of Truth）。
 * 执行层 / 审核层不得私自增删字段；契约变更必须走 docs/01_ARCH_CONTRACT.md 的变更流程。
 * 纪律：零 `any`；所有显存字段必须标注单位（bytes 或 GiB，1 GiB = 1073741824 Bytes）。
 *
 * v1.1.0 变更：GpuSpec 增 memoryBusWidthBits/int8Tflops；vramGb 语义改为二进制 GiB；
 * ModelArchitectureSpec 增 headDim（支持 Gemma-2）；GeneratedArtifacts 增 launchPowerShellScript；
 * TelemetryOsInfo 增 driverModel（WDDM/TCC）。
 */

/** 二进制吉字节：1 GiB = 2^30 Bytes。全项目显存换算唯一合法常量。 */
export const BYTES_PER_GIB = 1_073_741_824 as const;

/** 十进制吉字节：1 GB = 10^9 Bytes。仅用于带宽/厂商标称算力等十进制边界，禁止用于显存容量换算。 */
export const BYTES_PER_GB_DECIMAL = 1_000_000_000 as const;

// ---------------------------------------------------------------------------
// 1. 硬件规格
// ---------------------------------------------------------------------------

export type GpuArchitecture = 'Ada' | 'Blackwell' | 'Ampere' | 'Hopper';

export interface GpuSpec {
  id: string;
  name: string;
  /** 标称容量。GPU 显存颗粒为二进制容量，数值上直接等于 GiB（8 → 8.00 GiB，torch 实测报告 ≈7.99–8.00）。 */
  vramGb: number;
  architecture: GpuArchitecture;
  /** 显存位宽，单位 bit。 */
  memoryBusWidthBits: number;
  /** 显存带宽，单位十进制 GB/s（10^9 Bytes/s）。 */
  memoryBandwidthGbPerSec: number;
  /** FP16 Tensor Core 稠密（非稀疏）峰值算力，单位 TFLOPS（10^12 FLOP/s）。 */
  fp16Tflops: number;
  /** INT8 Tensor Core 稠密峰值算力，单位 TOPS（10^12 OP/s）。 */
  int8Tflops: number;
  supportsBf16: boolean;
  supportsFp8: boolean;
}

// ---------------------------------------------------------------------------
// 2. 归一化模型架构
// ---------------------------------------------------------------------------

export interface ModelArchitectureSpec {
  /** HuggingFace 风格模型 ID，如 "Qwen/Qwen2.5-7B-Instruct"。 */
  modelId: string;
  /** 反推得到的总参数量 P（含 embedding / lm_head，按 tie 规则计入）。 */
  numParams: number;
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  /** GQA 的 KV 头数；MHA 时等于 numAttentionHeads。 */
  numKeyValueHeads: number;
  /** 单头维度。通常 = hiddenSize/numAttentionHeads；Gemma-2 等架构显式给定（如 256），不可想当然推导。 */
  headDim: number;
  vocabSize: number;
  /** true 时 lm_head 与 embedding 共享权重，P 只计一份 V*h。 */
  tieWordEmbeddings: boolean;
  maxPositionEmbeddings: number;
}

// ---------------------------------------------------------------------------
// 3. 训练 / 微调超参数
// ---------------------------------------------------------------------------

export type FinetuneMode = 'full' | 'lora' | 'qlora_4bit' | 'qlora_8bit';

export type OptimizerKind =
  | 'adamw_32bit'
  | 'adamw_8bit'
  | 'paged_adamw_8bit'
  | 'sgd';

export type OsEnv = 'linux_headless' | 'windows_wddm';

export interface TrainingConfig {
  mode: FinetuneMode;
  /** 单卡 micro-batch size（不是全局 batch，不乘 gradAccumSteps）。 */
  batchSize: number;
  seqLen: number;
  /** 梯度累积步数：只影响等效 batch 与吞吐，不增加显存峰值。 */
  gradAccumSteps: number;
  loraRank: number;
  loraAlpha: number;
  /** 如 ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]。 */
  loraTargetModules: string[];
  optimizer: OptimizerKind;
  gradCheckpointing: boolean;
  flashAttention2: boolean;
  fusedCrossEntropy: boolean;
  osEnv: OsEnv;
  /** PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，抑制 allocator 碎片。 */
  expandableSegments: boolean;
  dynamicPadding: boolean;
}

/** 手术补丁：只允许覆盖 TrainingConfig 的合法字段。 */
export type ConfigPatch = Partial<TrainingConfig>;

// ---------------------------------------------------------------------------
// 4. 五大显存分项与汇总（字节级精度）
// ---------------------------------------------------------------------------

export type VramCategoryKey =
  | 'weights'
  | 'gradAndOpt'
  | 'activations'
  | 'logitsPeak'
  | 'systemAndFrag';

/** 主干权重 + 量化常数开销（LoRA 适配器副本归入 gradAndOpt，此处不重复计）。 */
export interface WeightsComponent {
  bytes: number;
  gib: number;
  formulaNote: string;
}

/** 可训练参数的 BF16 副本（LoRA）+ 梯度 + 优化器状态（含 FP32 master weights）。 */
export interface GradAndOptComponent {
  bytes: number;
  gib: number;
  trainableParams: number;
  formulaNote: string;
}

/** 前向中间激活值；attentionMatrixBytes 单列 eager attention 的 S^2 矩阵开销。 */
export interface ActivationsComponent {
  bytes: number;
  gib: number;
  attentionMatrixBytes: number;
  formulaNote: string;
}

/** 输出层大词表 Logits 与 FP32 交叉熵瞬时峰值。 */
export interface LogitsPeakComponent {
  bytes: number;
  gib: number;
  isFused: boolean;
  formulaNote: string;
}

/** CUDA 上下文、OS/驱动抢占（WDDM）与 PyTorch CachingAllocator 碎片。 */
export interface SystemAndFragComponent {
  cudaContextGib: number;
  osReservedGib: number;
  fragmentationGib: number;
  totalGib: number;
  formulaNote: string;
}

export type BottleneckType = 'MEMORY_BOUND' | 'COMPUTE_BOUND';

export interface VramSummary {
  /** 四大计算分项（weights+gradAndOpt+activations+logitsPeak）GiB 之和。 */
  totalAllocatedGib: number;
  /** allocated * (1 + 碎片率) + CUDA 上下文 + OS 抢占，即真实峰值占用。 */
  totalReservedPeakGib: number;
  /** 数值上等于 gpu.vramGb（显存颗粒二进制容量），严禁再乘 10^9/2^30。 */
  gpuCapacityGib: number;
  /** totalReservedPeakGib / gpuCapacityGib，可 > 1。 */
  utilizationRatio: number;
  isOom: boolean;
  /** gpuCapacityGib - totalReservedPeakGib，OOM 时为负值。 */
  headRoomGib: number;
  bottleneckType: BottleneckType;
}

export interface VramBreakdown {
  weights: WeightsComponent;
  gradAndOpt: GradAndOptComponent;
  activations: ActivationsComponent;
  logitsPeak: LogitsPeakComponent;
  systemAndFrag: SystemAndFragComponent;
  summary: VramSummary;
}

// ---------------------------------------------------------------------------
// 5. OOM 诊断与自动手术
// ---------------------------------------------------------------------------

export type OomRootCause =
  | 'VOCAB_LOGITS_SPIKE'
  | 'ACTIVATION_EXPLOSION'
  | 'WEIGHT_OVERFLOW'
  | 'ALLOCATOR_FRAGMENTATION'
  | 'UNKNOWN';

export interface OomDiagnosisResult {
  /** 正则是否成功解析日志；false 时数值字段为 0、rootCause 为 UNKNOWN。 */
  parsed: boolean;
  rawLog: string;
  triedToAllocateGib: number;
  totalCapacityGib: number;
  /** PyTorch 已分配（allocated）GiB。 */
  allocatedGib: number;
  /** PyTorch 已预留（reserved = allocated + reserved-but-unallocated）GiB。 */
  reservedGib: number;
  /** reserved - allocated，碎片诊断的核心证据。 */
  unallocatedGapGib: number;
  rootCause: OomRootCause;
  /** 人类可读的判定依据，供审核层复核。 */
  evidence: string;
}

export interface SurgicalStep {
  /** 从 1 开始的原子手术序号。 */
  stepIndex: number;
  actionName: string;
  configPatch: ConfigPatch;
  savedGib: number;
  beforeGib: number;
  afterGib: number;
  /** 副作用说明（无损 / 吞吐损失 / 精度风险），不允许为空字符串。 */
  sideEffectNote: string;
}

/** 手术前后 Diff —— 预留 motion-pro-max 物理动效引擎的插值数据源。 */
export interface VramDiff {
  perCategoryDeltaGib: Record<VramCategoryKey, number>;
  totalAllocatedDeltaGib: number;
  reservedPeakDeltaGib: number;
  oomResolved: boolean;
  changedConfigKeys: ReadonlyArray<keyof TrainingConfig>;
}

/** 单帧显存状态 —— 预留 motion-pro-max 时间轴驱动字段。 */
export type VramPhase =
  | 'weights_load'
  | 'forward'
  | 'logits_spike'
  | 'backward'
  | 'optimizer_step'
  | 'oom'
  | 'healed';

export interface VramFrameState {
  frameIndex: number;
  timestampMs: number;
  phase: VramPhase;
  categoryGib: Record<VramCategoryKey, number>;
  allocatedGib: number;
  reservedGib: number;
  capacityGib: number;
  /** reservedGib / capacityGib，动效压力条输入。 */
  pressureRatio: number;
}

export interface SurgicalPlan {
  steps: SurgicalStep[];
  originalConfig: TrainingConfig;
  healedConfig: TrainingConfig;
  beforeBreakdown: VramBreakdown;
  afterBreakdown: VramBreakdown;
  diff: VramDiff;
  totalSavedGib: number;
  /** 手术后 summary.isOom === false 时为 true。 */
  isHealed: boolean;
  /** 预留：由动效层按 before/after 生成的关键帧序列，核心层可不填充。 */
  animationTimeline?: VramFrameState[];
}

// ---------------------------------------------------------------------------
// 6. 导出工件
// ---------------------------------------------------------------------------

export type ChecklistSeverity = 'critical' | 'warning' | 'info';

export interface ChecklistItem {
  id: string;
  title: string;
  detail: string;
  severity: ChecklistSeverity;
}

export interface GeneratedArtifacts {
  llamaFactoryYaml: string;
  unslothPythonScript: string;
  launchShellScript: string;
  /** Windows PowerShell 启动脚本（$env:PYTORCH_CUDA_ALLOC_CONF 注入）。 */
  launchPowerShellScript: string;
  checklist: ChecklistItem[];
}

// ---------------------------------------------------------------------------
// 7. 真机遥测校准（对应 python-probe/vram_telemetry_probe.py 的 stdout JSON）
// ---------------------------------------------------------------------------

export interface TelemetryGpuInfo {
  name: string;
  driverVersion: string;
  cudaRuntimeVersion: string;
  totalCapacityGib: number;
  computeCapability: string;
}

export interface TelemetryTorchInfo {
  torchVersion: string;
  cudaAvailable: boolean;
  allocatorBackend: 'native' | 'cudaMallocAsync';
  expandableSegmentsEnabled: boolean;
}

export interface TelemetryOsInfo {
  platform: 'linux' | 'windows';
  osEnv: OsEnv;
  displayAttached: boolean;
  /** Windows 驱动模型（WDDM 抢占显存 / TCC 无抢占）；Linux 为 unknown。 */
  driverModel: 'wddm' | 'tcc' | 'unknown';
}

export interface TelemetryCalibration {
  /** CUDA 上下文初始化实测开销 GiB。 */
  cudaContextGib: number;
  /** 空进程 + 上下文建立后的基线占用 GiB。 */
  emptyProcessBaselineGib: number;
  /** 实测 (reserved - allocated) / allocated 碎片率。 */
  fragmentationRatio: number;
  /** 实测显存带宽 GB/s（十进制）；未跑带宽基准时为 null。 */
  measuredBandwidthGbPerSec: number | null;
}

export interface TelemetrySample {
  stepIndex: number;
  phase: VramPhase;
  allocatedGib: number;
  reservedGib: number;
  maxAllocatedGib: number;
  maxReservedGib: number;
  deviceFreeGib: number;
}

export interface HardwareTelemetryReport {
  /** 契约版本锁：探针与前端必须严格匹配该字面量。 */
  schemaVersion: 'vram-telemetry/1';
  probeVersion: string;
  /** ISO 8601 UTC 时间戳。 */
  capturedAtUtc: string;
  gpu: TelemetryGpuInfo;
  torch: TelemetryTorchInfo;
  os: TelemetryOsInfo;
  calibration: TelemetryCalibration;
  samples: TelemetrySample[];
}