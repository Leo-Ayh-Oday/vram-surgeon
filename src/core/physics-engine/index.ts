/**
 * VRAM Surgeon — 物理引擎（契约 §5，字节级显存方程）
 *
 * 纯函数、零副作用、零网络。五大分项严格按 §5.2–§5.6。
 * 铁律：gradAccumSteps 不出现在任何显存方程；dynamicPadding 不改激活峰值，只经碎片率影响 reserved。
 * 容量语义：gpuCapacityGib = gpu.vramGb（二进制 GiB），禁止 ×10⁹÷2³⁰（契约 §1）。
 */
import type {
  GpuSpec,
  ModelArchitectureSpec,
  TrainingConfig,
  VramBreakdown,
  OptimizerKind,
} from '../types';
import { BYTES_PER_GIB } from '../types';
import {
  BYTES_PER_BF16,
  BYTES_PER_FP32,
  INT8_BYTES_PER_PARAM,
  BYTES_PER_NF4,
  NF4_DQ_OVERHEAD_BYTES,
  LORA_COPY_GRAD_BYTES_PER_PARAM,
  FULL_GRAD_BYTES_PER_PARAM,
  ADAMW32_OPT_BYTES_PER_PARAM,
  ADAMW8_OPT_BYTES_PER_PARAM,
  SGD_OPT_BYTES_PER_PARAM,
  PAGED_OPT_MAX_CPU_FRACTION,
  ACT_HIDDEN_FACTOR,
  ACT_KV_FACTOR,
  ACT_MLP_FACTOR,
  EAGER_ATTN_BYTES_PER_ELEM,
  FLASH_ATTN_MATRIX_BYTES,
  CHECKPOINT_RECOMPUTE_FACTOR,
  LOGITS_UNFUSED_BYTES_PER_ELEM,
  FUSED_CE_CHUNK,
  CUDA_CONTEXT_GIB,
  OS_RESERVED_WDDM_GIB,
  FRAG_DYNAMIC_PADDING,
  FRAG_STABLE,
  MEMORY_BOUND_THRESHOLD,
} from '../constants';

const MODULE_TAG = '[physics-engine]';

/** 契约 §5.1：唯一显存换算入口 bytes / BYTES_PER_GIB。 */
export function bytesToGib(bytes: number): number {
  return bytes / BYTES_PER_GIB;
}

// ---------------------------------------------------------------------------
// 输入边界校验（契约 §5.1：非法输入抛 RangeError('[physics-engine] ...')）
// ---------------------------------------------------------------------------

function requirePositiveInt(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${MODULE_TAG} ${name} must be a positive integer, got ${value}`);
  }
}

function validateInputs(gpu: GpuSpec, model: ModelArchitectureSpec, config: TrainingConfig): void {
  if (!(gpu.vramGb > 0) || !Number.isFinite(gpu.vramGb)) {
    throw new RangeError(`${MODULE_TAG} gpu.vramGb must be a positive finite number, got ${gpu.vramGb}`);
  }
  requirePositiveInt(model.hiddenSize, 'model.hiddenSize');
  requirePositiveInt(model.intermediateSize, 'model.intermediateSize');
  requirePositiveInt(model.numHiddenLayers, 'model.numHiddenLayers');
  requirePositiveInt(model.numAttentionHeads, 'model.numAttentionHeads');
  requirePositiveInt(model.numKeyValueHeads, 'model.numKeyValueHeads');
  requirePositiveInt(model.headDim, 'model.headDim');
  requirePositiveInt(model.vocabSize, 'model.vocabSize');
  requirePositiveInt(model.numParams, 'model.numParams');
  if (model.numAttentionHeads % model.numKeyValueHeads !== 0) {
    throw new RangeError(
      `${MODULE_TAG} numAttentionHeads (${model.numAttentionHeads}) must be divisible by numKeyValueHeads (${model.numKeyValueHeads})`,
    );
  }
  requirePositiveInt(config.batchSize, 'config.batchSize');
  requirePositiveInt(config.seqLen, 'config.seqLen');
  requirePositiveInt(config.gradAccumSteps, 'config.gradAccumSteps');
  requirePositiveInt(config.loraRank, 'config.loraRank');
  requirePositiveInt(config.loraAlpha, 'config.loraAlpha');
}

// ---------------------------------------------------------------------------
// §5.3 可训练参数量
// ---------------------------------------------------------------------------

/** LoRA 模块形状表（契约 §5.3）：q/o 用 qDim 而非 h（Gemma-2 才体现差异）。 */
interface LoraModuleShape {
  dIn: number;
  dOut: number;
}

function loraModuleShapes(model: ModelArchitectureSpec): Record<string, LoraModuleShape> {
  const h = model.hiddenSize;
  const i = model.intermediateSize;
  const qDim = model.numAttentionHeads * model.headDim;
  const kvDim = model.numKeyValueHeads * model.headDim;
  return {
    q_proj: { dIn: h, dOut: qDim },
    k_proj: { dIn: h, dOut: kvDim },
    v_proj: { dIn: h, dOut: kvDim },
    o_proj: { dIn: qDim, dOut: h },
    gate_proj: { dIn: h, dOut: i },
    up_proj: { dIn: h, dOut: i },
    down_proj: { dIn: i, dOut: h },
  };
}

/**
 * 契约 §5.3：
 *   full: T = P
 *   lora/qlora_*: T = L * Σ_{m∈targets} r*(d_in(m)+d_out(m))
 */
export function computeTrainableParams(model: ModelArchitectureSpec, config: TrainingConfig): number {
  if (config.mode === 'full') {
    return model.numParams;
  }
  requirePositiveInt(config.loraRank, 'config.loraRank');
  if (config.loraTargetModules.length === 0) {
    throw new RangeError(`${MODULE_TAG} loraTargetModules must be non-empty for mode '${config.mode}'`);
  }
  const shapes = loraModuleShapes(model);
  const r = config.loraRank;
  let perLayer = 0;
  for (const moduleName of config.loraTargetModules) {
    const shape = shapes[moduleName];
    if (shape === undefined) {
      throw new RangeError(`${MODULE_TAG} unknown lora target module: '${moduleName}'`);
    }
    // 契约 §5.3: r*(d_in + d_out)
    perLayer += r * (shape.dIn + shape.dOut);
  }
  return model.numHiddenLayers * perLayer;
}

/** 契约 §5.3：optBytes（8bit AdamW 含 FP32 master，裁决已确认）。 */
function optimizerBytesPerParam(optimizer: OptimizerKind): number {
  switch (optimizer) {
    case 'adamw_32bit':
      return ADAMW32_OPT_BYTES_PER_PARAM; // 12
    case 'adamw_8bit':
    case 'paged_adamw_8bit':
      return ADAMW8_OPT_BYTES_PER_PARAM; // 6.00390625
    case 'sgd':
      return SGD_OPT_BYTES_PER_PARAM; // 4
  }
}

// ---------------------------------------------------------------------------
// §5.2–§5.6 五大分项
// ---------------------------------------------------------------------------

/**
 * 契约 §5：字节级显存解剖。纯函数。
 * paged 分页（§5.3）：仅当 optimizer==='paged_adamw_8bit' 且不分页即 OOM 时启用，
 * 比例 = min(PAGED_OPT_MAX_CPU_FRACTION, 超出量/优化器状态GiB)，只扣减 optBytes 部分。
 */
export function computeVramBreakdown(
  gpu: GpuSpec,
  model: ModelArchitectureSpec,
  config: TrainingConfig,
): VramBreakdown {
  validateInputs(gpu, model, config);

  const b = config.batchSize; // micro-batch（不乘 gradAccumSteps）
  const s = config.seqLen;
  const P = model.numParams;
  const h = model.hiddenSize;
  const i = model.intermediateSize;
  const L = model.numHiddenLayers;
  const a = model.numAttentionHeads;
  const V = model.vocabSize;
  const G = model.numAttentionHeads / model.numKeyValueHeads; // GQA 分组比
  const T = computeTrainableParams(model, config);

  // --- 分项一 weights（契约 §5.2）：主干权重 + 量化常数；LoRA 副本不在此计 ---
  let weightBytesPerParam: number;
  let weightsLabel: string;
  switch (config.mode) {
    case 'full':
    case 'lora':
      // 契约 §5.2: W_bytes = P*2（BF16 主干）
      weightBytesPerParam = BYTES_PER_BF16;
      weightsLabel = `BF16 backbone (mode=${config.mode})`;
      break;
    case 'qlora_8bit':
      // 契约 §5.2: W_bytes = P*1.05（LLM.int8() 权重+absmax）
      weightBytesPerParam = INT8_BYTES_PER_PARAM;
      weightsLabel = 'QLoRA 8-bit (LLM.int8())';
      break;
    case 'qlora_4bit':
      // 契约 §5.2: W_bytes = P*(0.5+NF4_DQ_OVERHEAD_BYTES)（NF4 双量化默认开启）
      weightBytesPerParam = BYTES_PER_NF4 + NF4_DQ_OVERHEAD_BYTES;
      weightsLabel = 'QLoRA 4-bit (NF4 double-quant)';
      break;
  }
  const weightsBytes = P * weightBytesPerParam;
  const weightsGib = bytesToGib(weightsBytes);
  // formulaNote 动态填入当前数值推导（契约 §5.6：供审核层与 UI 公式透视复核）。
  const weightsFormula = `${weightsLabel}: ${P} × ${weightBytesPerParam} = ${weightsBytes} B = ${weightsGib} GiB`;

  // --- 分项二 gradAndOpt（契约 §5.3）：副本 + 梯度 + 优化器状态 ---
  const optBytes = optimizerBytesPerParam(config.optimizer);
  // full: 梯度 2T（权重已在 weights）；lora/qlora: 适配器 BF16 副本 2T + 梯度 2T = 4T。
  const copyGradBytesPerParam =
    config.mode === 'full' ? FULL_GRAD_BYTES_PER_PARAM : LORA_COPY_GRAD_BYTES_PER_PARAM;
  const gradAndOptBytesNoPaging = T * (copyGradBytesPerParam + optBytes);
  const optimizerStateBytes = T * optBytes;
  const gradAndOptFormulaBase =
    `T=${T}; ${config.mode === 'full' ? 'full' : 'lora'}: T×(${copyGradBytesPerParam}+${optBytes}) ` +
    `(optimizer=${config.optimizer})`;

  // --- 分项三 activations（契约 §5.4） ---
  // 契约 §5.4: A_layer_linear = b*s*(10h + 2h/G + 6i)
  const actHidden = ACT_HIDDEN_FACTOR * h;
  const actKv = (ACT_KV_FACTOR * h) / G;
  const actMlp = ACT_MLP_FACTOR * i;
  const aLayerLinear = b * s * (actHidden + actKv + actMlp);
  // 契约 §5.4: flash → S² 项=0；eager → L*4*b*a*s²（全 L 层累计）。
  const attentionMatrixBytes = config.flashAttention2
    ? L * FLASH_ATTN_MATRIX_BYTES
    : L * EAGER_ATTN_BYTES_PER_ELEM * b * a * s * s;
  const aLayerAttn = attentionMatrixBytes / L;
  let activationsBytes: number;
  let activationsFormula: string;
  if (config.gradCheckpointing) {
    // 契约 §5.4: L*(2*b*s*h) + 1.25*(A_layer_linear + A_layer_attn)（只存层入口 + 单层重算峰值）
    const storedInputs = L * (BYTES_PER_BF16 * b * s * h);
    const recomputePeak = CHECKPOINT_RECOMPUTE_FACTOR * (aLayerLinear + aLayerAttn);
    activationsBytes = storedInputs + recomputePeak;
    activationsFormula =
      `checkpoint: ${L}×(2×${b}×${s}×${h}) + ${CHECKPOINT_RECOMPUTE_FACTOR}×(${aLayerLinear}+${aLayerAttn}) = ` +
      `${activationsBytes} B = ${bytesToGib(activationsBytes)} GiB; attentionMatrixBytes=${attentionMatrixBytes}`;
  } else {
    // 契约 §5.4: L*(A_layer_linear + A_layer_attn)
    activationsBytes = L * (aLayerLinear + aLayerAttn);
    activationsFormula =
      `${L}×(${aLayerLinear}+${aLayerAttn}) = ${activationsBytes} B = ${bytesToGib(activationsBytes)} GiB ` +
      `(linear=b*s*(10h+2h/G+6i)=${b}*${s}*(${actHidden}+${actKv}+${actMlp}); ` +
      `attentionMatrixBytes=${attentionMatrixBytes})`;
  }

  // --- 分项四 logitsPeak（契约 §5.5）：大词表输出层瞬时峰值 ---
  let logitsBytes: number;
  let logitsFormula: string;
  if (config.fusedCrossEntropy) {
    // 契约 §5.5: fused 切词表维 b*s*min(V,4096)*4（不实例化完整 [b,s,V]）
    const chunk = Math.min(V, FUSED_CE_CHUNK);
    logitsBytes = b * s * chunk * BYTES_PER_FP32;
    logitsFormula = `fused: ${b}×${s}×min(${V},${FUSED_CE_CHUNK})×${BYTES_PER_FP32} = ${logitsBytes} B = ${bytesToGib(logitsBytes)} GiB (V=${V}, chunk=${chunk})`;
  } else {
    // 契约 §5.5: unfused = 10*b*s*V（BF16 logits2 + FP32 upcast4 + FP32 softmax/grad4）
    logitsBytes = LOGITS_UNFUSED_BYTES_PER_ELEM * b * s * V;
    logitsFormula = `unfused: ${LOGITS_UNFUSED_BYTES_PER_ELEM}×${b}×${s}×${V} = ${logitsBytes} B = ${bytesToGib(logitsBytes)} GiB (V=${V})`;
  }

  // --- 分项五 systemAndFrag 与 summary（契约 §5.6） ---
  const cudaContextGib = CUDA_CONTEXT_GIB; // 0.65
  const osReservedGib = config.osEnv === 'windows_wddm' ? OS_RESERVED_WDDM_GIB : 0; // 0.75 / 0
  // 契约 §5.6: expandable 或 定长(!dynamicPadding) → 稳定碎片率；否则动态碎片率。
  const fragRatio = config.expandableSegments || !config.dynamicPadding ? FRAG_STABLE : FRAG_DYNAMIC_PADDING;
  // 契约 §1/§5.6: 容量直接取 vramGb（二进制 GiB 语义）。
  const gpuCapacityGib = gpu.vramGb;

  const activationsGib = bytesToGib(activationsBytes);
  const logitsGib = bytesToGib(logitsBytes);

  // 不分页基线（用于 paged 判定）。
  const gradAndOptGibNoPaging = bytesToGib(gradAndOptBytesNoPaging);
  const totalAllocatedNoPaging = weightsGib + gradAndOptGibNoPaging + activationsGib + logitsGib;
  const reservedNoPaging = totalAllocatedNoPaging * (1 + fragRatio) + cudaContextGib + osReservedGib;

  // 契约 v1.1.3 §5.3 paged 分页确定性规则：仅 paged_adamw_8bit 且不分页即 OOM 才启用。
  let pagingFraction = 0;
  let gradAndOptBytes = gradAndOptBytesNoPaging;
  let pagingNote = '';
  if (config.optimizer === 'paged_adamw_8bit' && reservedNoPaging > gpuCapacityGib) {
    const excessGib = reservedNoPaging - gpuCapacityGib;
    const optimizerStateGib = bytesToGib(optimizerStateBytes);
    if (optimizerStateGib > 0) {
      // 契约 v1.1.3 §5.3（附录 D #23）反解：优化器扣减经汇总层 (1+fragRatio) 乘子，
      // fraction = min(0.6, excess / (optStateGib × (1 + fragRatio)))；
      // 未触封顶时 reservedPeak 精确等于 capacity（浮点误差 ≤1e-6），即"刚好脱险、封顶 60%"。
      const pagedExcessDenominator = optimizerStateGib * (1 + fragRatio);
      pagingFraction = Math.min(PAGED_OPT_MAX_CPU_FRACTION, excessGib / pagedExcessDenominator);
      // 分页只扣减 optBytes 部分（仅优化器状态，不含 4T 副本/梯度）。
      gradAndOptBytes = gradAndOptBytesNoPaging - pagingFraction * optimizerStateBytes;
      // formulaNote 必须写明实际分页比例与反解算式（契约 §5.3）。
      pagingNote =
        `; paged fraction=${pagingFraction} = min(${PAGED_OPT_MAX_CPU_FRACTION}, ` +
        `excess(${excessGib} GiB) / [optState(${optimizerStateGib} GiB)×(1+frag ${fragRatio})` +
        `=${pagedExcessDenominator} GiB])`;
    }
  }
  const gradAndOptGib = bytesToGib(gradAndOptBytes);
  const gradAndOptFormula = `${gradAndOptFormulaBase}${pagingNote} = ${gradAndOptBytes} B = ${gradAndOptGib} GiB`;

  const totalAllocatedGib = weightsGib + gradAndOptGib + activationsGib + logitsGib;
  const fragmentationGib = fragRatio * totalAllocatedGib;
  const systemTotalGib = cudaContextGib + osReservedGib + fragmentationGib;
  // 契约 §5.6: reservedPeak = allocated*(1+fragRatio) + cudaContext + osReserved。
  const totalReservedPeakGib = totalAllocatedGib * (1 + fragRatio) + cudaContextGib + osReservedGib;
  const utilizationRatio = totalReservedPeakGib / gpuCapacityGib;
  const isOom = totalReservedPeakGib > gpuCapacityGib;
  const headRoomGib = gpuCapacityGib - totalReservedPeakGib;
  const bottleneckType = utilizationRatio >= MEMORY_BOUND_THRESHOLD ? 'MEMORY_BOUND' : 'COMPUTE_BOUND';

  const systemFormula =
    `cudaContext=${cudaContextGib} + osReserved=${osReservedGib} (osEnv=${config.osEnv}) + ` +
    `frag=${fragRatio}*${totalAllocatedGib}=${fragmentationGib}; ` +
    `fragRatio=${fragRatio} (expandable=${config.expandableSegments}, dynamicPadding=${config.dynamicPadding})`;

  return {
    weights: { bytes: weightsBytes, gib: weightsGib, formulaNote: weightsFormula },
    gradAndOpt: {
      bytes: gradAndOptBytes,
      gib: gradAndOptGib,
      trainableParams: T,
      formulaNote: gradAndOptFormula,
    },
    activations: {
      bytes: activationsBytes,
      gib: activationsGib,
      attentionMatrixBytes,
      formulaNote: activationsFormula,
    },
    logitsPeak: {
      bytes: logitsBytes,
      gib: logitsGib,
      isFused: config.fusedCrossEntropy,
      formulaNote: logitsFormula,
    },
    systemAndFrag: {
      cudaContextGib,
      osReservedGib,
      fragmentationGib,
      totalGib: systemTotalGib,
      formulaNote: systemFormula,
    },
    summary: {
      totalAllocatedGib,
      totalReservedPeakGib,
      gpuCapacityGib,
      utilizationRatio,
      isOom,
      headRoomGib,
      bottleneckType,
    },
  };
}