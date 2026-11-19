/**
 * VRAM Surgeon — 执行层验收测试（契约 §9.2，v1.1.1）
 *
 * Case 1–4 硬核用例 + 黄金值回归 A3–A15（A16 真机选测跳过）。
 * 单元测试是纯函数模拟，与宿主机操作系统无关（osEnv 只是配置参数，契约附录 B#21）。
 */
import { describe, expect, it } from 'vitest';
import { parse as yamlParse } from 'yaml';
import {
  ADAMW8_OPT_BYTES_PER_PARAM,
  DEFAULT_GPU_ID,
  GPU_PRESETS,
  MODEL_PRESETS,
  bytesToGib,
  computeTrainableParams,
  computeVramBreakdown,
  diagnoseOom,
  estimateParamCount,
  generateArtifacts,
  getGpuById,
  getModelById,
  normalizeHfConfig,
  parseLocalConfig,
  parseOomTraceback,
  solveAutoHeal,
} from '../index';
import type {
  GpuSpec,
  HardwareTelemetryReport,
  ModelArchitectureSpec,
  OsEnv,
  TrainingConfig,
  VramPhase,
} from '../types';
import { BYTES_PER_GIB } from '../types';
import telemetrySample from '../../../python-probe/sample_rtx5050_telemetry.json';

// ---------------------------------------------------------------------------
// 测试夹具
// ---------------------------------------------------------------------------

function requireGpu(id: string): GpuSpec {
  const gpu = getGpuById(id);
  if (gpu === undefined) {
    throw new Error(`[test] missing GPU preset: ${id}`);
  }
  return gpu;
}

function requireModel(id: string): ModelArchitectureSpec {
  const model = getModelById(id);
  if (model === undefined) {
    throw new Error(`[test] missing model preset: ${id}`);
  }
  return model;
}

/** Case 1 场景钉死：b=1, s=2048, r=16 全 7 模块, adamw_32bit，全优化关闭，WDDM，dynamicPadding。 */
function caseOneConfig(overrides: Partial<TrainingConfig> = {}): TrainingConfig {
  return {
    mode: 'lora',
    batchSize: 1,
    seqLen: 2048,
    gradAccumSteps: 1,
    loraRank: 16,
    loraAlpha: 32,
    loraTargetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    optimizer: 'adamw_32bit',
    gradCheckpointing: false,
    flashAttention2: false,
    fusedCrossEntropy: false,
    osEnv: 'windows_wddm',
    expandableSegments: false,
    dynamicPadding: true,
    ...overrides,
  };
}

const RTX_5050 = requireGpu(DEFAULT_GPU_ID);
const QWEN_7B = requireModel('Qwen/Qwen2.5-7B-Instruct');
const QWEN_3B = requireModel('Qwen/Qwen2.5-3B-Instruct');
const QWEN_14B = requireModel('Qwen/Qwen2.5-14B-Instruct');

/** Case 1b/Case 2 共用状态：qlora_4bit + gradCheckpointing，未开 fused，其余全关。 */
const CASE_1B_OVERRIDES: Partial<TrainingConfig> = {
  mode: 'qlora_4bit',
  gradCheckpointing: true,
};

// ---------------------------------------------------------------------------
// Case 1 — RTX 5050 8GB 默认微调 Qwen2.5-7B 爆显存复现（契约 §9.2）
// ---------------------------------------------------------------------------

describe('Case 1: RTX 5050 8GB + Qwen2.5-7B OOM reproduction', () => {
  it('1a lora 裸配置 reserved > 18 GiB（理论 ≈44.7 GiB）', () => {
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig());
    expect(breakdown.summary.totalReservedPeakGib).toBeGreaterThan(18);
    expect(breakdown.summary.isOom).toBe(true);
    // 理论 ≈44.69 GiB（契约 Case 1a）。
    expect(breakdown.summary.totalReservedPeakGib).toBeCloseTo(44.69, 1);
  });

  it('1b qlora_4bit + checkpoint：2.90 GiB logits 尖峰 + WDDM + 14% 碎片 → reserved ≈11.0 > 8.00', () => {
    const config = caseOneConfig(CASE_1B_OVERRIDES);
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, config);
    // 未开 fused：unfused logits 尖峰 = 2.900390625 GiB（契约 §5.5 黄金值）。
    expect(breakdown.logitsPeak.gib).toBeCloseTo(2.900390625, 6);
    expect(breakdown.logitsPeak.isFused).toBe(false);
    // WDDM 抢占 0.75 GiB。
    expect(breakdown.systemAndFrag.osReservedGib).toBe(0.75);
    // dynamicPadding 且未开 expandable → 碎片率 14%。
    expect(breakdown.systemAndFrag.fragmentationGib).toBeCloseTo(
      0.14 * breakdown.summary.totalAllocatedGib,
      6,
    );
    expect(breakdown.summary.totalReservedPeakGib).toBeCloseTo(11.03, 1);
    expect(breakdown.summary.totalReservedPeakGib).toBeGreaterThan(8.0);
    expect(breakdown.summary.isOom).toBe(true);
    expect(breakdown.summary.headRoomGib).toBeLessThan(0);
  });
});

// ---------------------------------------------------------------------------
// Case 2 — Auto-Heal 一键抢救（契约 §9.2；osEnv=linux_headless 纯函数模拟）
// ---------------------------------------------------------------------------

describe('Case 2: solveAutoHeal one-click rescue', () => {
  it('S1 fused → S2 FA2+expandable 后进入 [5.4, 6.2] GiB 绿区停止（理论 ≈5.859）', () => {
    const config = caseOneConfig({ ...CASE_1B_OVERRIDES, osEnv: 'linux_headless' });
    const plan = solveAutoHeal(RTX_5050, QWEN_7B, config);

    expect(plan.isHealed).toBe(true);
    expect(plan.afterBreakdown.summary.isOom).toBe(false);
    // 黄金窗口：reserved ∈ [5.4, 6.2]（契约钉死 HEAL_TARGET_RATIO=0.85 → 绿区 ≤6.8）。
    expect(plan.afterBreakdown.summary.totalReservedPeakGib).toBeGreaterThanOrEqual(5.4);
    expect(plan.afterBreakdown.summary.totalReservedPeakGib).toBeLessThanOrEqual(6.2);
    expect(plan.afterBreakdown.summary.totalReservedPeakGib).toBeCloseTo(5.859, 2);
    // 手术路径（契约 v1.1.3 §6.4）：S2 合并为单条 SurgicalStep，全程恰 2 条步骤。
    expect(plan.steps.length).toBe(2);
    expect(plan.steps.map((step) => step.actionName)).toEqual([
      'S1-fused-cross-entropy',
      'S2-flash-attention-2+expandable-segments',
    ]);
    // S2 的耦合字段合并进同一 configPatch 同时应用（附录 D #24）。
    expect(plan.steps[1].configPatch).toEqual({
      flashAttention2: true,
      expandableSegments: true,
    });
    // 步骤账单：S1 后 7.011 > 6.8 继续，S2 合并步后 5.859 进绿区停止。
    expect(plan.steps[0].beforeGib).toBeCloseTo(10.28, 1);
    expect(plan.steps[0].afterGib).toBeCloseTo(7.011, 2);
    expect(plan.steps[1].afterGib).toBeCloseTo(5.859, 2);
    expect(plan.healedConfig.fusedCrossEntropy).toBe(true);
    expect(plan.healedConfig.flashAttention2).toBe(true);
    expect(plan.healedConfig.expandableSegments).toBe(true);
    // 等效全局 batch 守恒：batchSize × gradAccumSteps 不变。
    expect(plan.healedConfig.batchSize * plan.healedConfig.gradAccumSteps).toBe(
      plan.originalConfig.batchSize * plan.originalConfig.gradAccumSteps,
    );
    // 无损步先于有损步：本场景不应出现任何有损（S6 seq-len）步骤。
    expect(plan.steps.some((step) => step.actionName.startsWith('S6'))).toBe(false);
    // 差值账单完整性：每步 savedGib > 0、before/after/sideEffectNote 必填。
    for (const step of plan.steps) {
      expect(step.savedGib).toBeGreaterThan(0);
      expect(step.sideEffectNote.length).toBeGreaterThan(0);
      expect(step.beforeGib - step.afterGib).toBeCloseTo(step.savedGib, 9);
    }
    expect(plan.steps.map((step) => step.stepIndex)).toEqual([1, 2]);
    // VramDiff 填充。
    expect(plan.diff.oomResolved).toBe(true);
    expect(plan.diff.reservedPeakDeltaGib).toBeLessThan(0);
    expect(plan.diff.changedConfigKeys).toEqual(
      expect.arrayContaining(['fusedCrossEntropy', 'flashAttention2', 'expandableSegments']),
    );
    expect(plan.totalSavedGib).toBeCloseTo(
      plan.beforeBreakdown.summary.totalReservedPeakGib - plan.afterBreakdown.summary.totalReservedPeakGib,
      9,
    );
  });

  it('WDDM 变体（真机 5050 Windows 场景）：只验 isHealed=true（契约附录 B#21）', () => {
    const config = caseOneConfig(CASE_1B_OVERRIDES); // osEnv=windows_wddm
    const plan = solveAutoHeal(RTX_5050, QWEN_7B, config);
    expect(plan.isHealed).toBe(true);
    // 术后 ≈6.6 GiB < 8.00 GiB（不在 linux 黄金窗口内，属预期）。
    expect(plan.afterBreakdown.summary.totalReservedPeakGib).toBeLessThan(8.0);
  });

  it('审核 D3：LoRA 16.754 GiB 起，S4 单条合并步骤直达 5.626 GiB 绿区且其后无多余步骤', () => {
    // 全优化已开启的 LoRA 裸配置（fused/flash/ckpt/expandable 均 true），仅剩 S4 可救。
    const config = caseOneConfig({
      mode: 'lora',
      gradCheckpointing: true,
      flashAttention2: true,
      fusedCrossEntropy: true,
      expandableSegments: true,
      osEnv: 'linux_headless',
    });
    const plan = solveAutoHeal(RTX_5050, QWEN_7B, config);
    expect(plan.beforeBreakdown.summary.totalReservedPeakGib).toBeCloseTo(16.754, 2);
    // S4 为单条合并步骤（mode + optimizer 同 patch），非 S4a/S4b 两条。
    expect(plan.steps.length).toBe(1);
    expect(plan.steps[0].actionName).toBe('S4-qlora-4bit+paged-adamw-8bit');
    expect(plan.steps[0].configPatch).toEqual({
      mode: 'qlora_4bit',
      optimizer: 'paged_adamw_8bit',
    });
    // 术后 5.626 GiB 进绿区停止，isHealed=true。
    expect(plan.afterBreakdown.summary.totalReservedPeakGib).toBeCloseTo(5.626, 2);
    expect(plan.afterBreakdown.summary.isOom).toBe(false);
    expect(plan.isHealed).toBe(true);
    // 其后无多余步骤（S5/S6 均未触发）。
    expect(
      plan.steps.some(
        (step) => step.actionName.startsWith('S5') || step.actionName.startsWith('S6'),
      ),
    ).toBe(false);
    expect(plan.steps[0].savedGib).toBeCloseTo(16.754 - 5.626, 2);
  });

  it('S5 等效 batch 守恒 + 末位有损手段显式警告；不可治愈时如实 isHealed=false', () => {
    // Qwen2.5-14B qlora_4bit b=4 在 8GB 卡上权重即 7.1 GiB，任何手段都无法进绿区。
    const config = caseOneConfig({
      ...CASE_1B_OVERRIDES,
      osEnv: 'linux_headless',
      batchSize: 4,
    });
    const plan = solveAutoHeal(RTX_5050, QWEN_14B, config);
    // 等效全局 batch 守恒：4×1 → 1×4。
    expect(plan.healedConfig.batchSize * plan.healedConfig.gradAccumSteps).toBe(
      plan.originalConfig.batchSize * plan.originalConfig.gradAccumSteps,
    );
    expect(plan.healedConfig.batchSize).toBe(1);
    expect(plan.healedConfig.gradAccumSteps).toBe(4);
    // 无损步（S1–S5）全部先于有损步（S6 seq-len）。
    const firstLossyIndex = plan.steps.findIndex((step) => step.sideEffectNote.startsWith('有损'));
    const lastLosslessIndex = plan.steps.reduce(
      (acc, step, idx) => (step.sideEffectNote.startsWith('有损') ? acc : idx),
      -1,
    );
    expect(firstLossyIndex).toBeGreaterThan(lastLosslessIndex);
    // 有损步必须显式警告长文截断风险。
    for (const step of plan.steps.filter((s) => s.sideEffectNote.startsWith('有损'))) {
      expect(step.sideEffectNote).toContain('截断');
    }
    // 全部手段用尽仍 OOM：如实返回。
    expect(plan.isHealed).toBe(false);
    expect(plan.afterBreakdown.summary.isOom).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Case 3 — PyTorch OOM 真实报错解析（契约 §9.2）
// ---------------------------------------------------------------------------

const CASE3_FIXTURE =
  'CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 8.00 GiB ' +
  'of which 0 bytes is free. Of the allocated memory 5.12 GiB is allocated by PyTorch, and ' +
  '1.18 GiB is reserved by PyTorch but unallocated.';

describe('Case 3: PyTorch OOM traceback parsing & fragmentation surgery', () => {
  it('精确提取 tried/capacity/allocated/gap/reserved（含 "0 bytes is free"）', () => {
    const diagnosis = parseOomTraceback(CASE3_FIXTURE);
    expect(diagnosis.parsed).toBe(true);
    expect(diagnosis.triedToAllocateGib).toBeCloseTo(2.32, 9);
    expect(diagnosis.totalCapacityGib).toBeCloseTo(8.0, 9);
    expect(diagnosis.allocatedGib).toBeCloseTo(5.12, 9);
    expect(diagnosis.unallocatedGapGib).toBeCloseTo(1.18, 9);
    // reserved = allocated + gap。
    expect(diagnosis.reservedGib).toBeCloseTo(6.3, 9);
  });

  it('gap=1.18 ≥ max(1.0, 0.10×8.0)=1.0 → ALLOCATOR_FRAGMENTATION', () => {
    const diagnosis = parseOomTraceback(CASE3_FIXTURE);
    expect(diagnosis.rootCause).toBe('ALLOCATOR_FRAGMENTATION');
    expect(diagnosis.evidence).toContain('unallocatedGapGib');
  });

  it('手术方案首步含 expandableSegments: true 修复', () => {
    const diagnosis = parseOomTraceback(CASE3_FIXTURE);
    const config = caseOneConfig(CASE_1B_OVERRIDES);
    const plan = solveAutoHeal(RTX_5050, QWEN_7B, config, diagnosis);
    expect(plan.steps.length).toBeGreaterThan(0);
    expect(plan.steps[0].configPatch.expandableSegments).toBe(true);
  });

  it('MiB 单位归一（÷1024）', () => {
    const mibLog =
      'CUDA out of memory. Tried to allocate 2375.68 MiB. GPU 0 has a total capacity of 8192.00 MiB ' +
      'of which 0 bytes is free. Of the allocated memory 5242.88 MiB is allocated by PyTorch, and ' +
      '1208.32 MiB is reserved by PyTorch but unallocated.';
    const diagnosis = parseOomTraceback(mibLog);
    expect(diagnosis.parsed).toBe(true);
    expect(diagnosis.triedToAllocateGib).toBeCloseTo(2.32, 6);
    expect(diagnosis.totalCapacityGib).toBeCloseTo(8.0, 6);
    expect(diagnosis.allocatedGib).toBeCloseTo(5.12, 6);
    expect(diagnosis.unallocatedGapGib).toBeCloseTo(1.18, 6);
    expect(diagnosis.rootCause).toBe('ALLOCATOR_FRAGMENTATION');
  });

  it('旧格式兜底正则（"... already allocated"）', () => {
    const legacyLog =
      'CUDA out of memory. Tried to allocate 2.32 GiB (GPU 0; 8.00 GiB total capacity; ' +
      '5.12 GiB already allocated; 1.20 GiB free; 6.30 GiB reserved in total)';
    const diagnosis = parseOomTraceback(legacyLog);
    expect(diagnosis.parsed).toBe(true);
    expect(diagnosis.triedToAllocateGib).toBeCloseTo(2.32, 9);
    expect(diagnosis.allocatedGib).toBeCloseTo(5.12, 9);
  });

  it('解析失败 parsed=false 不抛异常，rootCause=UNKNOWN', () => {
    const diagnosis = diagnoseOom('some unrelated RuntimeError: cuDNN error');
    expect(diagnosis.parsed).toBe(false);
    expect(diagnosis.triedToAllocateGib).toBe(0);
    expect(diagnosis.totalCapacityGib).toBe(0);
    expect(diagnosis.rootCause).toBe('UNKNOWN');
  });

  it('diagnoseOom + breakdown 上下文：tried 与 4×b·s·V 候选 ±25% 匹配 → VOCAB_LOGITS_SPIKE', () => {
    // Qwen2.5-7B b1 s2048 unfused：FP32 upcast 单张量 4*b*s*V = 1.16 GiB。
    const gapFreeLog =
      'CUDA out of memory. Tried to allocate 1.20 GiB. GPU 0 has a total capacity of 8.00 GiB ' +
      'of which 0.31 GiB is free. Of the allocated memory 7.02 GiB is allocated by PyTorch, and ' +
      '0.42 GiB is reserved by PyTorch but unallocated.';
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig());
    const diagnosis = diagnoseOom(gapFreeLog, { breakdown });
    expect(diagnosis.parsed).toBe(true);
    // gap=0.42 < max(1.0, 0.8) → 非碎片；tried=1.20 命中 4*b*s*V=1.1602 GiB（relErr≈3.4% ≤25%）。
    expect(diagnosis.rootCause).toBe('VOCAB_LOGITS_SPIKE');
    expect(diagnosis.evidence).toContain('4*b*s*V');
  });
});

// ---------------------------------------------------------------------------
// Case 4 — 代码编译器输出校验（契约 §9.2）
// ---------------------------------------------------------------------------

describe('Case 4: codegen artifacts validation', () => {
  const healedLikeConfig = caseOneConfig({
    ...CASE_1B_OVERRIDES,
    osEnv: 'linux_headless',
    optimizer: 'paged_adamw_8bit',
    flashAttention2: true,
    fusedCrossEntropy: true,
    expandableSegments: true,
  });

  it('YAML 可 parse 且含 quantization_bit:4 / flash_attn:fa2 / enable_liger_kernel:true / optim:paged_adamw_8bit', () => {
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, healedLikeConfig);
    const artifacts = generateArtifacts(QWEN_7B, RTX_5050, healedLikeConfig, breakdown);
    const parsed = yamlParse(artifacts.llamaFactoryYaml) as Record<string, unknown>;
    expect(parsed.model_name_or_path).toBe('Qwen/Qwen2.5-7B-Instruct');
    expect(parsed.finetuning_type).toBe('lora');
    expect(parsed.quantization_bit).toBe(4);
    expect(parsed.quantization_method).toBe('bitsandbytes');
    expect(parsed.double_quantization).toBe(true);
    expect(parsed.flash_attn).toBe('fa2');
    expect(parsed.enable_liger_kernel).toBe(true);
    expect(parsed.optim).toBe('paged_adamw_8bit');
    expect(parsed.gradient_checkpointing).toBe(true);
    expect(parsed.per_device_train_batch_size).toBe(1);
    expect(parsed.gradient_accumulation_steps).toBe(1);
    expect(parsed.cutoff_len).toBe(2048);
    expect(parsed.bf16).toBe(true);
    // 原始字符串层面同样包含关键字段。
    expect(artifacts.llamaFactoryYaml).toContain('quantization_bit: 4');
    expect(artifacts.llamaFactoryYaml).toContain('flash_attn: fa2');
    expect(artifacts.llamaFactoryYaml).toContain('enable_liger_kernel: true');
    expect(artifacts.llamaFactoryYaml).toContain('optim: paged_adamw_8bit');
  });

  it('Unsloth 脚本含 load_in_4bit=True 与 use_gradient_checkpointing="unsloth"，import 齐全无 TODO', () => {
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, healedLikeConfig);
    const artifacts = generateArtifacts(QWEN_7B, RTX_5050, healedLikeConfig, breakdown);
    expect(artifacts.unslothPythonScript).toContain('load_in_4bit=True');
    expect(artifacts.unslothPythonScript).toContain('use_gradient_checkpointing="unsloth"');
    expect(artifacts.unslothPythonScript).toContain('from unsloth import FastLanguageModel');
    expect(artifacts.unslothPythonScript).toContain('from trl import SFTTrainer');
    expect(artifacts.unslothPythonScript).not.toContain('TODO');
  });

  it('launch.sh 与 launch.ps1 均注入 expandable_segments:True,max_split_size_mb:128', () => {
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, healedLikeConfig);
    const artifacts = generateArtifacts(QWEN_7B, RTX_5050, healedLikeConfig, breakdown);
    expect(artifacts.launchShellScript).toContain('expandable_segments:True,max_split_size_mb:128');
    expect(artifacts.launchPowerShellScript).toContain('expandable_segments:True,max_split_size_mb:128');
    expect(artifacts.launchShellScript).toContain('#!/usr/bin/env bash');
    expect(artifacts.launchShellScript).toContain('set -euo pipefail');
    expect(artifacts.launchPowerShellScript).toContain('$env:PYTORCH_CUDA_ALLOC_CONF');
    // FA2 架构兼容检查（Ampere+）。
    expect(artifacts.launchShellScript).toContain('compute_cap');
    // Windows SDPA 等效 fallback 注明（ps1 产物）。
    expect(artifacts.launchPowerShellScript).toContain('SDPA');
  });

  it('传入 plan 时一律以 healedConfig 为准生成（契约 §7.1）', () => {
    const preHealConfig = caseOneConfig({ ...CASE_1B_OVERRIDES, osEnv: 'linux_headless' });
    const plan = solveAutoHeal(RTX_5050, QWEN_7B, preHealConfig);
    const beforeBreakdown = plan.beforeBreakdown;
    const artifacts = generateArtifacts(QWEN_7B, RTX_5050, preHealConfig, beforeBreakdown, plan);
    // 术前 config 未开 flash/fused，但术后 healedConfig 已开 → 产物必须反映术后状态。
    expect(artifacts.llamaFactoryYaml).toContain('flash_attn: fa2');
    expect(artifacts.llamaFactoryYaml).toContain('enable_liger_kernel: true');
    expect(artifacts.launchShellScript).toContain('expandable_segments:True,max_split_size_mb:128');
  });

  it('Windows Checklist 专属避坑项必出（契约 §7.5 v1.1.1）', () => {
    const config = caseOneConfig(CASE_1B_OVERRIDES); // windows_wddm
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, config);
    const artifacts = generateArtifacts(QWEN_7B, RTX_5050, config, breakdown);
    const ids = artifacts.checklist.map((item) => item.id);
    expect(ids).toEqual(
      expect.arrayContaining([
        'wddm-preemption',
        'vocab-logits-spike',
        'win-flash-attn-sdpa',
        'win-liger-triton',
        'win-bitsandbytes-version',
        'win-wddm-taskmgr',
        'win-prevent-sleep',
      ]),
    );
    const bnb = artifacts.checklist.find((item) => item.id === 'win-bitsandbytes-version');
    expect(bnb?.severity).toBe('critical');
    expect(bnb?.detail).toContain('0.43');
    // 每项 sideEffect/severity 合法、detail 非空。
    for (const item of artifacts.checklist) {
      expect(['critical', 'warning', 'info']).toContain(item.severity);
      expect(item.detail.length).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// 黄金值回归 A3–A15（契约 §9.2 表；A16 真机选测跳过）
// ---------------------------------------------------------------------------

describe('Golden value regressions A3-A15', () => {
  it('A3: bytesToGib(1073741824) === 1（精确，唯一换算入口）', () => {
    expect(bytesToGib(BYTES_PER_GIB)).toBe(1);
    expect(bytesToGib(0)).toBe(0);
    expect(bytesToGib(2 * BYTES_PER_GIB + BYTES_PER_GIB / 2)).toBe(2.5);
  });

  it('A4: 8GB 卡 gpuCapacityGib === 8.00（二进制容量语义，禁止 ×10⁹÷2³⁰）', () => {
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig());
    expect(breakdown.summary.gpuCapacityGib).toBeCloseTo(8.0, 2);
    expect(RTX_5050.vramGb).toBe(8);
  });

  it('A5: 五个预设模型参数量（含 Gemma-2 泛化式）±0.5%', () => {
    const golden: ReadonlyArray<[string, number]> = [
      ['Qwen/Qwen2.5-7B-Instruct', 7_615_487_488],
      ['meta-llama/Llama-3.1-8B-Instruct', 8_030_261_248],
      ['deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 7_615_487_488],
      ['Qwen/Qwen2.5-3B-Instruct', 3_085_846_528],
      ['Qwen/Qwen2.5-14B-Instruct', 14_769_689_600],
      ['google/gemma-2-9b-it', 9_241_404_928],
    ];
    for (const [modelId, expected] of golden) {
      const model = requireModel(modelId);
      expect(Math.abs(model.numParams - expected) / expected).toBeLessThanOrEqual(0.005);
      // 自洽性（契约 §4.2）：预设 numParams === estimateParamCount 结果（此处应为精确相等）。
      expect(model.numParams).toBe(estimateParamCount(model));
    }
  });

  it('A6: tie=true/false 双分支差 = V*h（精确）', () => {
    const tied: ModelArchitectureSpec = { ...QWEN_3B, tieWordEmbeddings: true };
    const untied: ModelArchitectureSpec = { ...QWEN_3B, tieWordEmbeddings: false };
    expect(estimateParamCount(untied) - estimateParamCount(tied)).toBe(
      QWEN_3B.vocabSize * QWEN_3B.hiddenSize,
    );
    // Qwen2.5-7B tie=false 分解：7.615B = 主干 + 2×V*h + final norm。
    expect(QWEN_7B.tieWordEmbeddings).toBe(false);
  });

  it('A7: Qwen2.5-7B b1 s2048 unfused logits = 2.900390625 GiB（±1%）', () => {
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig());
    expect(breakdown.logitsPeak.bytes).toBe(3_114_270_720);
    expect(breakdown.logitsPeak.gib).toBeCloseTo(2.900390625, 6);
  });

  it('A8: eager attentionMatrixBytes = 13,153,337,344 B（精确）；flash 时 = 0', () => {
    const eager = computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig());
    expect(eager.activations.attentionMatrixBytes).toBe(13_153_337_344);
    // flash 全层线性激活黄金值：8,631,877,632 B = 8.0390625 GiB（契约 §5.4）。
    const flashNoCkpt = computeVramBreakdown(
      RTX_5050,
      QWEN_7B,
      caseOneConfig({ flashAttention2: true }),
    );
    expect(flashNoCkpt.activations.attentionMatrixBytes).toBe(0);
    expect(flashNoCkpt.activations.bytes).toBe(8_631_877_632);
    // flash + checkpoint：411,041,792 + 1.25×308,281,344 = 796,393,472 B（±35% 容差内取精确式）。
    const flashCkpt = computeVramBreakdown(
      RTX_5050,
      QWEN_7B,
      caseOneConfig({ flashAttention2: true, gradCheckpointing: true }),
    );
    expect(flashCkpt.activations.bytes).toBe(796_393_472);
  });

  it('A9: gradAccumSteps 不变性（1→8→64 逐字段相等，铁律）', () => {
    const base = computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig({ gradAccumSteps: 1 }));
    const ga8 = computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig({ gradAccumSteps: 8 }));
    const ga64 = computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig({ gradAccumSteps: 64 }));
    expect(ga8).toEqual(base);
    expect(ga64).toEqual(base);
  });

  it('A10: LoRA T（Qwen2.5-7B r16 全 7 模块）= 40,370,176（精确）', () => {
    expect(computeTrainableParams(QWEN_7B, caseOneConfig())).toBe(40_370_176);
    // full 模式 T = P。
    expect(computeTrainableParams(QWEN_7B, caseOneConfig({ mode: 'full' }))).toBe(QWEN_7B.numParams);
  });

  it('A11: fused logits = 33,554,432 B = 0.03125 GiB（精确，切词表维）', () => {
    const breakdown = computeVramBreakdown(
      RTX_5050,
      QWEN_7B,
      caseOneConfig({ fusedCrossEntropy: true }),
    );
    expect(breakdown.logitsPeak.bytes).toBe(33_554_432);
    expect(breakdown.logitsPeak.gib).toBe(0.03125);
    expect(breakdown.logitsPeak.isFused).toBe(true);
  });

  it('A12: lora+adamw32 gradAndOpt = 645,922,816 B（精确，16T = 副本2+梯度2+opt12）', () => {
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig());
    expect(breakdown.gradAndOpt.bytes).toBe(645_922_816);
    expect(breakdown.gradAndOpt.gib).toBe(0.6015625);
    expect(breakdown.gradAndOpt.trainableParams).toBe(40_370_176);
  });

  it('A13: RTX 4060 qlora_4bit 全优化 linux reservedPeak ≈5.63 GiB 且 isOom=false（±10%）', () => {
    const rtx4060 = requireGpu('rtx-4060-8g');
    const config = caseOneConfig({
      mode: 'qlora_4bit',
      optimizer: 'adamw_8bit',
      gradCheckpointing: true,
      flashAttention2: true,
      fusedCrossEntropy: true,
      expandableSegments: true,
      osEnv: 'linux_headless',
      dynamicPadding: false,
    });
    const breakdown = computeVramBreakdown(rtx4060, QWEN_7B, config);
    expect(Math.abs(breakdown.summary.totalReservedPeakGib - 5.63) / 5.63).toBeLessThanOrEqual(0.1);
    expect(breakdown.summary.isOom).toBe(false);
    expect(breakdown.summary.bottleneckType).toBeDefined();
  });

  it('A14①: 不 OOM 时 paging fraction=0（不启用分页）', () => {
    const rtx4060 = requireGpu('rtx-4060-8g');
    const config = caseOneConfig({
      mode: 'qlora_4bit',
      optimizer: 'paged_adamw_8bit',
      gradCheckpointing: true,
      flashAttention2: true,
      fusedCrossEntropy: true,
      expandableSegments: true,
      osEnv: 'linux_headless',
      dynamicPadding: false,
    });
    const breakdown = computeVramBreakdown(rtx4060, QWEN_7B, config);
    // 未分页：gradAndOpt = T×(4 + 6.00390625) 原值，formulaNote 无分页记录 → fraction=0。
    expect(breakdown.gradAndOpt.bytes).toBe(40_370_176 * (4 + ADAMW8_OPT_BYTES_PER_PARAM));
    expect(breakdown.gradAndOpt.formulaNote).not.toContain('paged fraction');
    expect(breakdown.summary.isOom).toBe(false);
  });

  it('A14②: OOM 且未触封顶时 |reserved − capacity| ≤ 1e-6（反解公式含 1+fragRatio）', () => {
    // RTX 4090 24GB + Qwen2.5-3B full + paged：optState 大，反解比例 <0.6 的真分支。
    const rtx4090 = requireGpu('rtx-4090-24g');
    const config = caseOneConfig({
      mode: 'full',
      optimizer: 'paged_adamw_8bit',
      gradCheckpointing: true,
      flashAttention2: true,
      fusedCrossEntropy: true,
      osEnv: 'linux_headless',
    });
    const paged = computeVramBreakdown(rtx4090, QWEN_3B, config);
    const match = /paged fraction=([\d.eE+-]+)/.exec(paged.gradAndOpt.formulaNote);
    expect(match).not.toBeNull();
    const fraction = Number(match?.[1] ?? 'NaN');

    // 独立重算期望 fraction = excess / (optStateGib × (1+fragRatio))，fragRatio=0.14。
    const T = QWEN_3B.numParams; // full: T = P
    const fragRatio = 0.14;
    const optStateGib = (T * ADAMW8_OPT_BYTES_PER_PARAM) / BYTES_PER_GIB;
    const gradUnpagedGib = (T * (2 + ADAMW8_OPT_BYTES_PER_PARAM)) / BYTES_PER_GIB;
    const reservedUnpaged =
      (paged.weights.gib + gradUnpagedGib + paged.activations.gib + paged.logitsPeak.gib) *
        (1 + fragRatio) +
      0.65;
    const expectedFraction = (reservedUnpaged - 24) / (optStateGib * (1 + fragRatio));
    expect(expectedFraction).toBeGreaterThan(0);
    expect(expectedFraction).toBeLessThan(0.6);
    expect(fraction).toBeCloseTo(expectedFraction, 9);
    // 未触封顶 → reservedPeak 精确等于容量（刚好脱险，≤1e-6 GiB）。
    expect(Math.abs(paged.summary.totalReservedPeakGib - 24)).toBeLessThanOrEqual(1e-6);
    expect(paged.summary.isOom).toBe(false);
    // formulaNote 写明反解算式。
    expect(paged.gradAndOpt.formulaNote).toContain('(1+frag');
  });

  it('A14③: 极端 OOM 时 fraction=0.6 封顶且如实报告仍 OOM', () => {
    // RTX 4090 24GB + Qwen2.5-7B full：所需分页 ≫ 60%，必然触顶。
    const rtx4090 = requireGpu('rtx-4090-24g');
    const config = caseOneConfig({
      mode: 'full',
      optimizer: 'paged_adamw_8bit',
      gradCheckpointing: true,
      flashAttention2: true,
      fusedCrossEntropy: true,
      expandableSegments: true,
      osEnv: 'linux_headless',
    });
    const paged = computeVramBreakdown(rtx4090, QWEN_7B, config);
    const match = /paged fraction=([\d.eE+-]+)/.exec(paged.gradAndOpt.formulaNote);
    expect(match).not.toBeNull();
    expect(Number(match?.[1] ?? 'NaN')).toBe(0.6);
    // 封顶后仍 OOM：如实报告，不做假脱险。
    expect(paged.summary.isOom).toBe(true);
    expect(paged.summary.totalReservedPeakGib).toBeGreaterThan(24);
  });

  it('A15: probe sample JSON 与 HardwareTelemetryReport 字段一一对应（类型级 satisfies）', () => {
    // 类型级：逐字段收窄后 satisfies 契约类型（含 driverModel）。
    const report = {
      schemaVersion: telemetrySample.schemaVersion as 'vram-telemetry/1',
      probeVersion: telemetrySample.probeVersion,
      capturedAtUtc: telemetrySample.capturedAtUtc,
      gpu: telemetrySample.gpu,
      torch: {
        ...telemetrySample.torch,
        allocatorBackend: telemetrySample.torch.allocatorBackend as 'native' | 'cudaMallocAsync',
      },
      os: {
        ...telemetrySample.os,
        platform: telemetrySample.os.platform as 'linux' | 'windows',
        osEnv: telemetrySample.os.osEnv as OsEnv,
        driverModel: telemetrySample.os.driverModel as 'wddm' | 'tcc' | 'unknown',
      },
      calibration: telemetrySample.calibration,
      samples: telemetrySample.samples.map((sample) => ({
        ...sample,
        phase: sample.phase as VramPhase,
      })),
    } satisfies HardwareTelemetryReport;

    // 运行时锁：schemaVersion 字面量、mock 后缀、WDDM 与校准先验量级。
    expect(report.schemaVersion).toBe('vram-telemetry/1');
    expect(report.probeVersion.endsWith('-mock')).toBe(true);
    expect(report.os.driverModel).toBe('wddm');
    expect(report.os.osEnv).toBe('windows_wddm');
    expect(report.gpu.totalCapacityGib).toBeCloseTo(8.0, 2);
    // 契约 §8 mock 基准：context≈0.65、WDDM 基线≈0.75 量级、碎片率 0.035–0.14。
    expect(Math.abs(report.calibration.cudaContextGib - 0.65) / 0.65).toBeLessThanOrEqual(0.2);
    expect(report.calibration.fragmentationRatio).toBeGreaterThanOrEqual(0.035);
    expect(report.calibration.fragmentationRatio).toBeLessThanOrEqual(0.14);
    // 四阶段采样齐全。
    expect(report.samples.map((sample) => sample.phase)).toEqual([
      'forward',
      'logits_spike',
      'backward',
      'optimizer_step',
    ]);
    for (const sample of report.samples) {
      expect(sample.stepIndex).toBeGreaterThan(0);
      expect(Number.isFinite(sample.deviceFreeGib)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// 预设库 / 解析器补充回归（契约 §3、§4）
// ---------------------------------------------------------------------------

describe('Presets & hf-parser', () => {
  it('GPU 预设完整（9 卡，5050 为默认基准，320 GB/s）', () => {
    expect(DEFAULT_GPU_ID).toBe('rtx-5050-8g');
    expect(GPU_PRESETS.length).toBe(9);
    const ids = GPU_PRESETS.map((gpu) => gpu.id);
    expect(ids).toEqual(
      expect.arrayContaining([
        'rtx-5050-8g',
        'rtx-4060-8g',
        'rtx-4070-12g',
        'rtx-4060ti-16g',
        'rtx-3060-12g',
        'rtx-3090-24g',
        'rtx-4090-24g',
        'a100-40g',
        'a100-80g',
      ]),
    );
    expect(RTX_5050.architecture).toBe('Blackwell');
    expect(RTX_5050.memoryBandwidthGbPerSec).toBe(320);
    expect(RTX_5050.memoryBusWidthBits).toBe(128);
    expect(getGpuById('no-such-gpu')).toBeUndefined();
  });

  it('模型预设完整（6 模型，Qwen2.5-3B tie=true 且词表 151936；Gemma-2 headDim=256）', () => {
    expect(MODEL_PRESETS.length).toBe(6);
    expect(QWEN_3B.tieWordEmbeddings).toBe(true);
    expect(QWEN_3B.vocabSize).toBe(151_936);
    expect(QWEN_7B.vocabSize).toBe(152_064);
    const gemma2 = requireModel('google/gemma-2-9b-it');
    expect(gemma2.headDim).toBe(256);
    expect(gemma2.headDim).not.toBe(gemma2.hiddenSize / gemma2.numAttentionHeads);
    expect(gemma2.tieWordEmbeddings).toBe(true);
    expect(getModelById('no/such-model')).toBeUndefined();
  });

  it('normalizeHfConfig：异构别名归一化 + head_dim 缺省 = h/n + numParams 自洽', () => {
    const spec = normalizeHfConfig(
      {
        n_embd: 2048,
        n_inner: 11008,
        n_layer: 36,
        n_head: 16,
        n_head_kv: 2,
        n_vocab: 151936,
        tie_word_embeddings: true,
        n_positions: 32768,
      },
      'local/test-gpt-aliases',
    );
    expect(spec.hiddenSize).toBe(2048);
    expect(spec.intermediateSize).toBe(11008);
    expect(spec.numHiddenLayers).toBe(36);
    expect(spec.numAttentionHeads).toBe(16);
    expect(spec.numKeyValueHeads).toBe(2);
    expect(spec.headDim).toBe(128); // 缺省 = h/n
    expect(spec.vocabSize).toBe(151936);
    expect(spec.tieWordEmbeddings).toBe(true);
    expect(spec.maxPositionEmbeddings).toBe(32768);
    expect(spec.numParams).toBe(estimateParamCount(spec));
    expect(spec.numParams).toBe(QWEN_3B.numParams);
  });

  it('normalizeHfConfig：显式 head_dim 尊重（Gemma-2 试金石）；MHA 缺省 kv=heads', () => {
    const spec = normalizeHfConfig(
      {
        hidden_size: 3584,
        intermediate_size: 14336,
        num_hidden_layers: 42,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 256,
        vocab_size: 256000,
        tie_word_embeddings: true,
        max_position_embeddings: 8192,
      },
      'google/gemma-2-9b-it',
    );
    // Gemma-2 走泛化公式 qDim = n*d = 4096 ≠ h。
    expect(spec.headDim).toBe(256);
    expect(spec.numParams).toBe(9_241_404_928);

    const mha = normalizeHfConfig(
      {
        hidden_size: 4096,
        intermediate_size: 14336,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        vocab_size: 128256,
        max_position_embeddings: 131072,
      },
      'local/mha-test',
    );
    expect(mha.numKeyValueHeads).toBe(32); // 缺省 = heads（MHA）
    expect(mha.tieWordEmbeddings).toBe(false); // 缺省 false
  });

  it('契约 v1.1.2 §4.1：真实 Gemma-2 原始 config（hidden_activation=gelu_pytorch_tanh）放行', () => {
    // 真实 HF config 形态：GeGLU（gated 三矩阵）走 hidden_activation 白名单，不因 gelu 误杀。
    const gemmaRaw: Record<string, unknown> = {
      hidden_size: 3584,
      intermediate_size: 14336,
      num_hidden_layers: 42,
      num_attention_heads: 16,
      num_key_value_heads: 8,
      head_dim: 256,
      vocab_size: 256000,
      tie_word_embeddings: true,
      hidden_activation: 'gelu_pytorch_tanh',
      max_position_embeddings: 8192,
    };
    const viaNormalize = normalizeHfConfig(gemmaRaw, 'google/gemma-2-9b-it');
    expect(viaNormalize.headDim).toBe(256);
    expect(Math.abs(viaNormalize.numParams - 9_241_404_928) / 9_241_404_928).toBeLessThanOrEqual(0.005);

    const viaLocal = parseLocalConfig(JSON.stringify(gemmaRaw), 'google/gemma-2-9b-it');
    expect(viaLocal.headDim).toBe(256);
    expect(viaLocal.numParams).toBe(viaNormalize.numParams);
    expect(viaLocal.numParams).toBe(9_241_404_928); // 泛化公式精确值
  });

  it('契约 v1.1.2 §4.1：hidden_act 白名单优先于 activation_function（同时存在时放行）', () => {
    const spec = normalizeHfConfig(
      {
        hidden_size: 4096,
        intermediate_size: 14336,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 128256,
        max_position_embeddings: 131072,
        hidden_act: 'silu',
        activation_function: 'gelu_new',
      },
      'local/precedence-test',
    );
    expect(spec.vocabSize).toBe(128256);
    expect(spec.headDim).toBe(128);
  });

  it('parseLocalConfig：modelId 缺省 local/custom；非 SwiGLU（GPT-2 GELU）抛带前缀 Error', () => {
    const spec = parseLocalConfig(
      JSON.stringify({
        hidden_size: 3584,
        intermediate_size: 18944,
        num_hidden_layers: 28,
        num_attention_heads: 28,
        num_key_value_heads: 4,
        vocab_size: 152064,
        max_position_embeddings: 32768,
        hidden_act: 'silu',
      }),
    );
    expect(spec.modelId).toBe('local/custom');
    expect(spec.numParams).toBe(QWEN_7B.numParams);

    expect(() =>
      parseLocalConfig(
        JSON.stringify({
          n_embd: 768,
          n_inner: 3072,
          n_layer: 12,
          n_head: 12,
          n_vocab: 50257,
          n_positions: 1024,
          activation_function: 'gelu_new',
        }),
        'gpt2',
      ),
    ).toThrow(/\[hf-parser\] unsupported MLP architecture/);

    expect(() => parseLocalConfig('{ not json')).toThrow(/\[hf-parser\]/);
  });

  it('physics-engine 输入边界校验抛 RangeError（契约 §5.1）', () => {
    expect(() =>
      computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig({ batchSize: 0 })),
    ).toThrow(RangeError);
    expect(() =>
      computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig({ seqLen: -1 })),
    ).toThrow(/\[physics-engine\]/);
    expect(() =>
      computeTrainableParams(QWEN_7B, caseOneConfig({ loraTargetModules: ['lm_head'] })),
    ).toThrow(RangeError);
  });

  it('formulaNote 动态数值推导（供前端公式透视卡片渲染）', () => {
    const breakdown = computeVramBreakdown(RTX_5050, QWEN_7B, caseOneConfig(CASE_1B_OVERRIDES));
    // QLoRA 4-bit 权重：P × 0.515869140625 = bytes = GiB 完整推导链。
    expect(breakdown.weights.formulaNote).toContain('7615487488');
    expect(breakdown.weights.formulaNote).toContain('0.515869140625');
    expect(breakdown.weights.formulaNote).toContain('B =');
    expect(breakdown.weights.formulaNote).toContain('GiB');
    expect(breakdown.gradAndOpt.formulaNote).toContain('T=40370176');
    expect(breakdown.gradAndOpt.formulaNote).toContain('GiB');
    expect(breakdown.logitsPeak.formulaNote).toContain('V=152064');
    expect(breakdown.activations.formulaNote).toContain('attentionMatrixBytes=');
    expect(breakdown.systemAndFrag.formulaNote).toContain('fragRatio=0.14');
  });
});
