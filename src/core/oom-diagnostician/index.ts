/**
 * VRAM Surgeon — OOM 诊断与自动手术（契约 §6）
 *
 * parseOomTraceback：PyTorch OOM 正则解析（GiB/MiB/bytes 三单位，兼容 "0 bytes is free"）。
 * diagnoseOom：叠加 breakdown 上下文的 rootCause 判定（短路求值，evidence 写明命中依据）。
 * solveAutoHeal：五阶段无损优先削峰，停止条件 reservedPeak ≤ 0.85×capacity（安全绿区）。
 */
import type {
  GpuSpec,
  ModelArchitectureSpec,
  TrainingConfig,
  ConfigPatch,
  VramBreakdown,
  VramCategoryKey,
  OomDiagnosisResult,
  OomRootCause,
  SurgicalPlan,
  SurgicalStep,
} from '../types';
import { BYTES_PER_GIB } from '../types';
import { computeVramBreakdown } from '../physics-engine';
import {
  MIB_PER_GIB,
  FRAG_GAP_ABS_GIB,
  FRAG_GAP_RATIO,
  LOGITS_SPIKE_MATCH_TOLERANCE,
  HEAL_TARGET_RATIO,
  SEQ_LEN_LAST_RESORT_FACTOR,
  PAGED_OPT_MAX_CPU_FRACTION,
  LOGITS_UNFUSED_BYTES_PER_ELEM,
} from '../constants';

const MODULE_TAG = '[oom-diagnostician]';

// ---------------------------------------------------------------------------
// §6.2 PyTorch OOM 正则（命名捕获组；free 字段兼容 "0 bytes"）
// ---------------------------------------------------------------------------

/** 主正则（PyTorch ≥ 2.1）：free 单位 GiB|MiB|bytes 三兼容。 */
const OOM_REGEX =
  /CUDA out of memory\. Tried to allocate (?<tried>[\d.]+) (?<triedUnit>GiB|MiB)\. GPU \d+ has a total capacity of (?<cap>[\d.]+) (?<capUnit>GiB|MiB) of which (?<free>[\d.]+) (?<freeUnit>GiB|MiB|bytes) is free\.(?: Including non-PyTorch memory, this process has (?<proc>[\d.]+) (?<procUnit>GiB|MiB) memory in use\.)? Of the allocated memory (?<alloc>[\d.]+) (?<allocUnit>GiB|MiB) is allocated by PyTorch, and (?<gap>[\d.]+) (?<gapUnit>GiB|MiB) is reserved by PyTorch but unallocated\./;

/** 旧格式兜底正则。 */
const LEGACY_OOM_REGEX =
  /Tried to allocate ([\d.]+) (GiB|MiB).*?([\d.]+) (GiB|MiB) already allocated/;

/** 单位归一：MiB ÷ MIB_PER_GIB → GiB；bytes ÷ BYTES_PER_GIB（契约 §6.2）。 */
function toGib(value: number, unit: string): number {
  if (unit === 'MiB') {
    return value / MIB_PER_GIB;
  }
  if (unit === 'bytes') {
    return value / BYTES_PER_GIB;
  }
  return value; // GiB
}

function failedDiagnosis(rawLog: string): OomDiagnosisResult {
  // 契约 §6.2：解析失败 parsed=false，数值置 0，rootCause=UNKNOWN，禁止抛异常中断手术。
  return {
    parsed: false,
    rawLog,
    triedToAllocateGib: 0,
    totalCapacityGib: 0,
    allocatedGib: 0,
    reservedGib: 0,
    unallocatedGapGib: 0,
    rootCause: 'UNKNOWN',
    evidence: `${MODULE_TAG} no PyTorch OOM pattern matched; numeric fields zeroed`,
  };
}

interface ParsedOomNumbers {
  triedToAllocateGib: number;
  totalCapacityGib: number;
  allocatedGib: number;
  reservedGib: number;
  unallocatedGapGib: number;
}

/** §6.3 rootCause 判定顺序（短路求值，evidence 写明命中依据）。 */
function classifyRootCause(
  nums: ParsedOomNumbers,
  breakdown: VramBreakdown | undefined,
): { rootCause: OomRootCause; evidence: string } {
  // 1. 碎片：gap ≥ max(1.0 GiB, 0.10×capacity)（空闲缓存够但不连续）。
  const fragThreshold = Math.max(FRAG_GAP_ABS_GIB, FRAG_GAP_RATIO * nums.totalCapacityGib);
  if (nums.unallocatedGapGib >= fragThreshold) {
    return {
      rootCause: 'ALLOCATOR_FRAGMENTATION',
      evidence:
        `unallocatedGapGib=${nums.unallocatedGapGib} >= max(${FRAG_GAP_ABS_GIB}, ` +
        `${FRAG_GAP_RATIO}*${nums.totalCapacityGib})=${fragThreshold} GiB ` +
        '(reserved-but-unallocated cache is sufficient but non-contiguous)',
    };
  }
  // 2. 大词表 logits 尖峰：tried 与 {2,4,10}×b·s·V 之一相对误差 ≤ 25%（需 breakdown 上下文）。
  if (breakdown !== undefined) {
    const candidates = logitsSpikeCandidates(breakdown);
    for (const candidate of candidates) {
      if (candidate.gib <= 0) {
        continue;
      }
      const relErr = Math.abs(nums.triedToAllocateGib - candidate.gib) / candidate.gib;
      if (relErr <= LOGITS_SPIKE_MATCH_TOLERANCE) {
        return {
          rootCause: 'VOCAB_LOGITS_SPIKE',
          evidence:
            `triedToAllocateGib=${nums.triedToAllocateGib} matches ${candidate.multiplier}*b*s*V=` +
            `${candidate.gib} GiB (relErr=${relErr} <= ${LOGITS_SPIKE_MATCH_TOLERANCE}); ` +
            'FP32 upcast single tensor (4*b*s*V) is the most common trigger',
        };
      }
    }
  }
  // 3. 权重溢出：weights.gib ≥ capacity（无 breakdown 时用日志 allocated 近似主干占用）。
  const weightsGib = breakdown !== undefined ? breakdown.weights.gib : nums.allocatedGib;
  if (weightsGib >= nums.totalCapacityGib && nums.totalCapacityGib > 0) {
    return {
      rootCause: 'WEIGHT_OVERFLOW',
      evidence: `weights.gib=${weightsGib} >= totalCapacityGib=${nums.totalCapacityGib} (backbone alone exceeds capacity)`,
    };
  }
  // 4. 激活爆炸：breakdown 中 activations 为最大分项。
  if (breakdown !== undefined) {
    const entries: Array<[VramCategoryKey, number]> = [
      ['weights', breakdown.weights.gib],
      ['gradAndOpt', breakdown.gradAndOpt.gib],
      ['activations', breakdown.activations.gib],
      ['logitsPeak', breakdown.logitsPeak.gib],
    ];
    let maxKey: VramCategoryKey = entries[0][0];
    let maxVal = entries[0][1];
    for (const [key, val] of entries) {
      if (val > maxVal) {
        maxKey = key;
        maxVal = val;
      }
    }
    if (maxKey === 'activations') {
      return {
        rootCause: 'ACTIVATION_EXPLOSION',
        evidence:
          `activations.gib=${breakdown.activations.gib} is the largest component ` +
          `(weights=${breakdown.weights.gib}, gradAndOpt=${breakdown.gradAndOpt.gib}, logitsPeak=${breakdown.logitsPeak.gib})`,
      };
    }
  }
  // 5. 其余 → UNKNOWN。
  return {
    rootCause: 'UNKNOWN',
    evidence: `${MODULE_TAG} no specific root cause rule matched`,
  };
}

/** logits 尖峰候选 {2,4,10}×b·s·V（GiB）。由 unfused logits bytes=10*b*s*V 反推 b*s*V。 */
function logitsSpikeCandidates(
  breakdown: VramBreakdown,
): Array<{ multiplier: number; gib: number }> {
  // unfused: bytes = 10*b*s*V → b*s*V = bytes/10。fused 时 bytes 不含完整 V，候选不可靠，返回空。
  if (breakdown.logitsPeak.isFused) {
    return [];
  }
  const bsV = breakdown.logitsPeak.bytes / LOGITS_UNFUSED_BYTES_PER_ELEM;
  return [2, 4, 10].map((multiplier) => ({
    multiplier,
    gib: (multiplier * bsV) / BYTES_PER_GIB,
  }));
}

/**
 * 契约 §6.1/§6.2：解析 PyTorch OOM 日志。无 breakdown 上下文（rootCause 仅碎片/权重/UNKNOWN）。
 * 解析失败不抛异常。
 */
export function parseOomTraceback(logText: string): OomDiagnosisResult {
  return diagnoseOom(logText);
}

/**
 * 契约 §6.1/§6.3：解析 + 上下文 rootCause 判定。
 * context.breakdown 提供时候选 logits 尖峰与激活爆炸判据方可生效。
 */
export function diagnoseOom(
  logText: string,
  context?: { breakdown: VramBreakdown },
): OomDiagnosisResult {
  const breakdown = context?.breakdown;
  const match = OOM_REGEX.exec(logText);
  if (match !== null && match.groups !== undefined) {
    const g = match.groups;
    const triedToAllocateGib = toGib(Number(g.tried), g.triedUnit);
    const totalCapacityGib = toGib(Number(g.cap), g.capUnit);
    const allocatedGib = toGib(Number(g.alloc), g.allocUnit);
    const unallocatedGapGib = toGib(Number(g.gap), g.gapUnit);
    // 契约 §6.2: reservedGib = allocatedGib + gapGib。
    const reservedGib = allocatedGib + unallocatedGapGib;
    const nums: ParsedOomNumbers = {
      triedToAllocateGib,
      totalCapacityGib,
      allocatedGib,
      reservedGib,
      unallocatedGapGib,
    };
    const { rootCause, evidence } = classifyRootCause(nums, breakdown);
    return { parsed: true, rawLog: logText, ...nums, rootCause, evidence };
  }

  const legacy = LEGACY_OOM_REGEX.exec(logText);
  if (legacy !== null) {
    const triedToAllocateGib = toGib(Number(legacy[1]), legacy[2]);
    const allocatedGib = toGib(Number(legacy[3]), legacy[4]);
    const nums: ParsedOomNumbers = {
      triedToAllocateGib,
      totalCapacityGib: 0,
      allocatedGib,
      reservedGib: allocatedGib, // 旧格式无 gap 信息
      unallocatedGapGib: 0,
    };
    const { rootCause, evidence } = classifyRootCause(nums, breakdown);
    return {
      parsed: true,
      rawLog: logText,
      ...nums,
      rootCause,
      evidence: `${evidence} (legacy OOM format; capacity/gap unavailable)`,
    };
  }

  return failedDiagnosis(logText);
}

// ---------------------------------------------------------------------------
// §6.4 solveAutoHeal 五阶段无损优先削峰
// ---------------------------------------------------------------------------

/** 末位有损手段的序列长度下限（避免无限缩减；纯数学护栏）。 */
const MIN_SEQ_LEN = 1;

/**
 * 契约 §6.4（v1.1.3 附录 D #24）：五阶段手术（S1 fused → S2 FA2+expandable 合并 → S3 ckpt →
 * S4 QLoRA+paged 合并 → S5 batch 守恒）。策略表一行 = 一条 SurgicalStep，耦合字段同 patch 同时应用。
 * 停止条件：每条 SurgicalStep 后重算 breakdown，totalReservedPeakGib ≤ HEAL_TARGET_RATIO×gpuCapacityGib
 * （0.85 安全绿区）即停；S5 每次减半后也重查。
 * 每个实际生效步骤产出一条 SurgicalStep（savedGib>0 才记录），每步后立即重算 breakdown。
 * 若 diagnosis 指向 ALLOCATOR_FRAGMENTATION，优先注入 expandableSegments 修复（Case 3 首步）。
 */
export function solveAutoHeal(
  gpu: GpuSpec,
  model: ModelArchitectureSpec,
  config: TrainingConfig,
  diagnosis?: OomDiagnosisResult,
): SurgicalPlan {
  const originalConfig: TrainingConfig = { ...config, loraTargetModules: [...config.loraTargetModules] };
  const beforeBreakdown = computeVramBreakdown(gpu, model, originalConfig);
  const gpuCapacityGib = gpu.vramGb; // 契约 §1：二进制容量语义
  const greenZoneGib = HEAL_TARGET_RATIO * gpuCapacityGib; // 0.85 安全绿区

  let currentConfig: TrainingConfig = { ...originalConfig, loraTargetModules: [...originalConfig.loraTargetModules] };
  let currentBreakdown = beforeBreakdown;
  const steps: SurgicalStep[] = [];
  let stepIndex = 0;

  const inGreenZone = (): boolean => currentBreakdown.summary.totalReservedPeakGib <= greenZoneGib;

  const applyStep = (actionName: string, patch: ConfigPatch, sideEffectNote: string): void => {
    const beforeGib = currentBreakdown.summary.totalReservedPeakGib;
    const nextConfig: TrainingConfig = { ...currentConfig, ...patch };
    const nextBreakdown = computeVramBreakdown(gpu, model, nextConfig);
    const afterGib = nextBreakdown.summary.totalReservedPeakGib;
    const savedGib = beforeGib - afterGib;
    currentConfig = nextConfig;
    currentBreakdown = nextBreakdown;
    // 契约 §6.4：savedGib > 0 才记录。
    if (savedGib > 0) {
      stepIndex += 1;
      steps.push({ stepIndex, actionName, configPatch: patch, savedGib, beforeGib, afterGib, sideEffectNote });
    }
  };

  // 碎片诊断优先：allocator 整形（无损），作为首步（契约 Case 3）。
  if (diagnosis?.rootCause === 'ALLOCATOR_FRAGMENTATION' && !currentConfig.expandableSegments) {
    applyStep(
      'S0-expandable-segments (fragmentation fix)',
      { expandableSegments: true },
      '无损：PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 抑制 allocator 碎片，整形 reserved-but-unallocated 缓存',
    );
  }

  // S1 融合交叉熵（无损，斩大词表尖峰）。
  if (!inGreenZone() && !currentConfig.fusedCrossEntropy) {
    applyStep(
      'S1-fused-cross-entropy',
      { fusedCrossEntropy: true },
      '无损：数学等价，Liger/Unsloth 融合 CE 切词表维，斩大词表 logits 尖峰',
    );
  }
  // S2 注意力 + 分配器（契约 v1.1.3 §6.4 附录 D #24：耦合字段合并为同一 SurgicalStep，仅变更者入 patch）。
  if (!inGreenZone()) {
    const s2Patch: ConfigPatch = {};
    if (!currentConfig.flashAttention2) {
      s2Patch.flashAttention2 = true;
    }
    if (!currentConfig.expandableSegments) {
      s2Patch.expandableSegments = true;
    }
    if (Object.keys(s2Patch).length > 0) {
      applyStep(
        'S2-flash-attention-2+expandable-segments',
        s2Patch,
        '无损：FlashAttention-2/SDPA 片上分块斩 S² 矩阵（Windows 可走 SDPA fallback）+ expandable_segments allocator 整形（碎片率 14%→3.5%）',
      );
    }
  }
  // S3 梯度检查点（无损，吞吐降约 20–35%）。
  if (!inGreenZone() && !currentConfig.gradCheckpointing) {
    applyStep(
      'S3-gradient-checkpointing',
      { gradCheckpointing: true },
      '无损：只存层入口输入 + 1.25×单层重算峰值，吞吐降约 20–35%',
    );
  }
  // S4 量化 + 分页优化器（契约 v1.1.3 §6.4 附录 D #24：耦合字段合并为同一 SurgicalStep）。
  if (!inGreenZone()) {
    const s4Patch: ConfigPatch = {};
    if (currentConfig.mode !== 'qlora_4bit') {
      s4Patch.mode = 'qlora_4bit';
    }
    if (currentConfig.optimizer !== 'paged_adamw_8bit') {
      s4Patch.optimizer = 'paged_adamw_8bit';
    }
    if (Object.keys(s4Patch).length > 0) {
      applyStep(
        'S4-qlora-4bit+paged-adamw-8bit',
        s4Patch,
        `近无损：NF4 双量化主干权重（量化误差风险，适配器仍 BF16）+ 8bit 优化器 CPU 分页（封顶 ${PAGED_OPT_MAX_CPU_FRACTION * 100}%，CPU↔GPU 分页延迟）`,
      );
    }
  }
  // S5 等效 batch 守恒（无损）：batchSize 减半、gradAccumSteps 翻倍，循环至 1；每次减半后重查绿区。
  while (!inGreenZone() && currentConfig.batchSize > 1) {
    const nextBatch = Math.max(1, Math.floor(currentConfig.batchSize / 2));
    const nextAccum = currentConfig.gradAccumSteps * 2;
    applyStep(
      'S5-batch-conservation',
      { batchSize: nextBatch, gradAccumSteps: nextAccum },
      '无损：等效全局 batch 守恒（batchSize×gradAccumSteps 不变），micro-batch 减半降激活峰值，吞吐下降',
    );
  }
  // 末位有损手段：序列长度缩减（必须显式警告长文截断风险）。
  while (!inGreenZone() && currentConfig.seqLen > MIN_SEQ_LEN) {
    const nextSeq = Math.max(MIN_SEQ_LEN, Math.floor(currentConfig.seqLen * SEQ_LEN_LAST_RESORT_FACTOR));
    if (nextSeq >= currentConfig.seqLen) {
      break; // 护栏：无法再缩减
    }
    applyStep(
      'S6-seq-len-last-resort',
      { seqLen: nextSeq },
      `有损：序列长度 ×${SEQ_LEN_LAST_RESORT_FACTOR} 缩减，长文截断风险（上下文信息丢失），仅末位手段`,
    );
  }

  const afterBreakdown = currentBreakdown;
  const healedConfig = currentConfig;

  // diff（契约 VramDiff）。
  const perCategoryDeltaGib: Record<VramCategoryKey, number> = {
    weights: afterBreakdown.weights.gib - beforeBreakdown.weights.gib,
    gradAndOpt: afterBreakdown.gradAndOpt.gib - beforeBreakdown.gradAndOpt.gib,
    activations: afterBreakdown.activations.gib - beforeBreakdown.activations.gib,
    logitsPeak: afterBreakdown.logitsPeak.gib - beforeBreakdown.logitsPeak.gib,
    systemAndFrag: afterBreakdown.systemAndFrag.totalGib - beforeBreakdown.systemAndFrag.totalGib,
  };
  const changedConfigKeys = (Object.keys(originalConfig) as Array<keyof TrainingConfig>).filter((key) => {
    if (key === 'loraTargetModules') {
      return originalConfig.loraTargetModules.join(',') !== healedConfig.loraTargetModules.join(',');
    }
    return originalConfig[key] !== healedConfig[key];
  });

  const diff = {
    perCategoryDeltaGib,
    totalAllocatedDeltaGib: afterBreakdown.summary.totalAllocatedGib - beforeBreakdown.summary.totalAllocatedGib,
    reservedPeakDeltaGib: afterBreakdown.summary.totalReservedPeakGib - beforeBreakdown.summary.totalReservedPeakGib,
    oomResolved: beforeBreakdown.summary.isOom && !afterBreakdown.summary.isOom,
    changedConfigKeys,
  };

  const totalSavedGib = steps.reduce((sum, step) => sum + step.savedGib, 0);
  // 契约：手术后 summary.isOom === false 时为 true。
  const isHealed = !afterBreakdown.summary.isOom;

  return {
    steps,
    originalConfig,
    healedConfig,
    beforeBreakdown,
    afterBreakdown,
    diff,
    totalSavedGib,
    isHealed,
  };
}