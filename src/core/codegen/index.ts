/**
 * VRAM Surgeon — 零报错启动包编译器（契约 §7）
 *
 * 纯函数。若传入 plan，一律以 plan.healedConfig 为准生成。
 * 产物：LLaMA-Factory YAML / Unsloth Python / launch.sh / launch.ps1 / 动态避坑 Checklist。
 */
import { stringify as yamlStringify } from 'yaml';
import type {
  GpuSpec,
  ModelArchitectureSpec,
  TrainingConfig,
  VramBreakdown,
  SurgicalPlan,
  GeneratedArtifacts,
  ChecklistItem,
  OptimizerKind,
} from '../types';
import {
  MAX_SPLIT_SIZE_MB,
  HF_MIRROR_ENDPOINT,
  LLAMA_FACTORY_OUTPUT_DIR,
  SMALL_VRAM_GB_THRESHOLD,
  LARGE_VOCAB_THRESHOLD,
  FA2_MIN_COMPUTE_CAPABILITY_MAJOR,
  BITSANDBYTES_WINDOWS_MIN_VERSION,
  OS_RESERVED_WDDM_GIB,
} from '../constants';

/** 契约 §7.2：optim 映射（adamw_32bit → adamw_torch）。 */
function llamaFactoryOptim(optimizer: OptimizerKind): string {
  switch (optimizer) {
    case 'adamw_32bit':
      return 'adamw_torch';
    case 'adamw_8bit':
      return 'adamw_8bit';
    case 'paged_adamw_8bit':
      return 'paged_adamw_8bit';
    case 'sgd':
      return 'sgd';
  }
}

/** Unsloth/TRL optim 字符串（与 HF TrainingArguments 对齐）。 */
function unslothOptim(optimizer: OptimizerKind): string {
  switch (optimizer) {
    case 'adamw_32bit':
      return 'adamw_torch';
    case 'adamw_8bit':
      return 'adamw_8bit';
    case 'paged_adamw_8bit':
      return 'paged_adamw_8bit';
    case 'sgd':
      return 'sgd';
  }
}

/** 契约 §7.4：FlashAttention-2 要求 Ampere 及以上（compute capability ≥ 8.0）。 */
function architectureComputeCapability(gpu: GpuSpec): string {
  switch (gpu.architecture) {
    case 'Ampere':
      return '8.0';
    case 'Ada':
      return '8.9';
    case 'Hopper':
      return '9.0';
    case 'Blackwell':
      return '12.0';
  }
}

// ---------------------------------------------------------------------------
// §7.2 LLaMA-Factory train_config.yaml
// ---------------------------------------------------------------------------

function buildLlamaFactoryYaml(model: ModelArchitectureSpec, config: TrainingConfig): string {
  const isLoraFamily = config.mode !== 'full';
  // 键映射（契约 §7.2）。用 Record<string, unknown> 承载，yaml 库无损序列化。
  const yamlConfig: Record<string, unknown> = {
    model_name_or_path: model.modelId,
    finetuning_type: config.mode === 'full' ? 'full' : 'lora',
  };
  if (config.mode === 'qlora_4bit' || config.mode === 'qlora_8bit') {
    yamlConfig.quantization_bit = config.mode === 'qlora_4bit' ? 4 : 8;
    yamlConfig.quantization_method = 'bitsandbytes';
    yamlConfig.double_quantization = true;
  }
  if (isLoraFamily) {
    yamlConfig.lora_rank = config.loraRank;
    yamlConfig.lora_alpha = config.loraAlpha;
    yamlConfig.lora_target = config.loraTargetModules.join(',');
  }
  yamlConfig.per_device_train_batch_size = config.batchSize;
  yamlConfig.gradient_accumulation_steps = config.gradAccumSteps;
  yamlConfig.cutoff_len = config.seqLen;
  yamlConfig.gradient_checkpointing = config.gradCheckpointing;
  if (config.flashAttention2) {
    yamlConfig.flash_attn = 'fa2';
  }
  if (config.fusedCrossEntropy) {
    // 契约 §7.2：fusedCrossEntropy 开启 → enable_liger_kernel: true。
    yamlConfig.enable_liger_kernel = true;
  }
  yamlConfig.optim = llamaFactoryOptim(config.optimizer);
  yamlConfig.bf16 = true;
  yamlConfig.output_dir = LLAMA_FACTORY_OUTPUT_DIR;

  const header =
    '# VRAM Surgeon — LLaMA-Factory train_config.yaml (auto-generated, contract §7.2)\n' +
    `# model=${model.modelId} mode=${config.mode} optimizer=${config.optimizer}\n`;
  return header + yamlStringify(yamlConfig);
}

// ---------------------------------------------------------------------------
// §7.3 Unsloth 极速微调独立 Python 脚本
// ---------------------------------------------------------------------------

function buildUnslothScript(
  model: ModelArchitectureSpec,
  gpu: GpuSpec,
  config: TrainingConfig,
): string {
  // 契约 §7.3：8GB 卡（vramGb≤8）自动 load_in_4bit。
  const loadIn4bit = config.mode === 'qlora_4bit' || gpu.vramGb <= SMALL_VRAM_GB_THRESHOLD;
  const loadIn4bitLiteral = loadIn4bit ? 'True' : 'False';
  const checkpointLiteral = config.gradCheckpointing ? '"unsloth"' : 'False';
  const targetModules = config.loraTargetModules.map((m) => `"${m}"`).join(', ');
  const optim = unslothOptim(config.optimizer);
  const isLoraFamily = config.mode !== 'full';

  // 脚本可直接 python 运行：import 齐全、无 TODO 占位符。
  return `#!/usr/bin/env python3
# VRAM Surgeon — Unsloth 极速微调独立脚本 (auto-generated, contract §7.3)
# 可直接运行: python train_unsloth.py
# Unsloth 内置 chunked cross-entropy（切词表维），等效 fusedCrossEntropy=${config.fusedCrossEntropy}，
# 无需手动实例化完整 [b, s, V] logits，自动斩大词表尖峰。
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

model_id = "${model.modelId}"
max_seq_length = ${config.seqLen}

# 8GB 及以下小显存卡自动 load_in_4bit（契约 §7.3）；dtype=None 自动选择 BF16/FP16。
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=${loadIn4bitLiteral},
)

model = FastLanguageModel.get_peft_model(
    model,
    r=${config.loraRank},
    lora_alpha=${config.loraAlpha},
    target_modules=[${targetModules}],
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing=${checkpointLiteral},
    random_state=3407,
)

# 公开指令数据集（alpaca 格式），格式化为纯文本供 SFTTrainer 消费。
dataset = load_dataset("yahma/alpaca-cleaned", split="train")


def format_example(example):
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]
    if inp:
        text = f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{inp}\\n\\n### Response:\\n{output}"
    else:
        text = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}"
    return {"text": text}


dataset = dataset.map(format_example)

training_args = TrainingArguments(
    per_device_train_batch_size=${config.batchSize},
    gradient_accumulation_steps=${config.gradAccumSteps},
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    optim="${optim}",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)

trainer.train()
model.save_pretrained("outputs/lora_adapter")
tokenizer.save_pretrained("outputs/lora_adapter")
${isLoraFamily ? '# LoRA/QLoRA 适配器已保存至 outputs/lora_adapter。' : '# 注意：full 模式下 Unsloth 仍以 PEFT 包装训练；如需全参微调请改用 LLaMA-Factory YAML。'}
`;
}

// ---------------------------------------------------------------------------
// §7.4 环境避坑启动脚本（launch.sh + launch.ps1，双份必出）
// ---------------------------------------------------------------------------

function buildLaunchShell(config: TrainingConfig, gpu: GpuSpec): string {
  const allocConf = config.expandableSegments
    ? `export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:${MAX_SPLIT_SIZE_MB}"`
    : '# expandableSegments 未开启：不注入 PYTORCH_CUDA_ALLOC_CONF（如需抑制碎片请在配置中开启）';
  const cc = architectureComputeCapability(gpu);
  return `#!/usr/bin/env bash
# VRAM Surgeon — 环境避坑启动脚本 (auto-generated, contract §7.4)
set -euo pipefail

# --- 显存分配器整形（expandableSegments=${config.expandableSegments}）---
${allocConf}

# --- 可选 HF 镜像（中国大陆网络环境取消注释）---
# export HF_ENDPOINT="${HF_MIRROR_ENDPOINT}"

# --- 显卡架构兼容性检查：FlashAttention-2 要求 Ampere 及以上（compute capability >= 8.0）---
# 预设显卡 ${gpu.name} (${gpu.architecture}) 标称 compute capability ${cc}。
if command -v nvidia-smi >/dev/null 2>&1; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | tr -d ' ' || true)"
  if [ -n "\${CC}" ]; then
    MAJOR="\${CC%%.*}"
    if [ "\${MAJOR}" -lt ${FA2_MIN_COMPUTE_CAPABILITY_MAJOR} ]; then
      echo "[VRAM Surgeon][WARN] compute capability \${CC} < 8.0: FlashAttention-2 需要 Ampere 及以上，请改用 SDPA fallback。" >&2
    fi
  else
    echo "[VRAM Surgeon][WARN] 无法查询 compute capability；请确认显卡 >= Ampere (8.0) 以使用 FlashAttention-2。" >&2
  fi
else
  echo "[VRAM Surgeon][WARN] 未检测到 nvidia-smi；请确认显卡 >= Ampere (8.0) 以使用 FlashAttention-2。" >&2
fi

# --- 启动训练（LLaMA-Factory）---
llamafactory-cli train train_config.yaml
`;
}

function buildLaunchPowerShell(config: TrainingConfig, gpu: GpuSpec): string {
  const allocConf = config.expandableSegments
    ? `$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:${MAX_SPLIT_SIZE_MB}"`
    : '# expandableSegments 未开启：不注入 PYTORCH_CUDA_ALLOC_CONF';
  const cc = architectureComputeCapability(gpu);
  return `# VRAM Surgeon — Windows PowerShell 启动脚本 (auto-generated, contract §7.4)
# 用法: powershell -ExecutionPolicy Bypass -File launch.ps1
$ErrorActionPreference = "Stop"

# --- 显存分配器整形（expandableSegments=${config.expandableSegments}）---
${allocConf}

# --- 可选 HF 镜像（中国大陆网络环境取消注释）---
# $env:HF_ENDPOINT = "${HF_MIRROR_ENDPOINT}"

# --- 显卡架构兼容性检查：FlashAttention-2 要求 Ampere 及以上（compute capability >= 8.0）---
# 预设显卡 ${gpu.name} (${gpu.architecture}) 标称 compute capability ${cc}。
# Windows WDDM 保底路径：flash-attn 轮子难装时用 PyTorch 原生 SDPA（Flash/Mem-efficient backend）。
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $cc = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    if ($cc) {
        $major = [int]($cc.Split('.')[0])
        if ($major -lt ${FA2_MIN_COMPUTE_CAPABILITY_MAJOR}) {
            Write-Warning "[VRAM Surgeon] compute capability $cc < 8.0: FlashAttention-2 需要 Ampere 及以上，请改用 SDPA fallback。"
        }
    } else {
        Write-Warning "[VRAM Surgeon] 无法查询 compute capability；请确认显卡 >= Ampere (8.0)。"
    }
} else {
    Write-Warning "[VRAM Surgeon] 未检测到 nvidia-smi；请确认显卡 >= Ampere (8.0) 以使用 FlashAttention-2。"
}

# --- 启动训练（LLaMA-Factory）---
llamafactory-cli train train_config.yaml
`;
}

// ---------------------------------------------------------------------------
// §7.5 避坑 Checklist（动态生成）
// ---------------------------------------------------------------------------

function buildChecklist(
  model: ModelArchitectureSpec,
  gpu: GpuSpec,
  config: TrainingConfig,
): ChecklistItem[] {
  const items: ChecklistItem[] = [];
  const isWindows = config.osEnv === 'windows_wddm';
  const isSmallVram = gpu.vramGb <= SMALL_VRAM_GB_THRESHOLD;

  // WDDM 抢占 0.75 GiB（windows_wddm 时 critical）。
  if (isWindows) {
    items.push({
      id: 'wddm-preemption',
      title: `Windows WDDM 桌面/驱动抢占约 ${OS_RESERVED_WDDM_GIB} GiB`,
      detail:
        `WDDM 模式下桌面合成器与驱动抢占约 ${OS_RESERVED_WDDM_GIB} GiB 显存，实际可用容量低于标称 ${gpu.vramGb} GiB。` +
        '显存预算已扣除该项；训练时尽量关闭占用 GPU 的桌面应用/浏览器硬件加速。',
      severity: 'critical',
    });
  }
  // 大词表 logits 尖峰（V>100000 且未 fused 时 critical）。
  if (model.vocabSize > LARGE_VOCAB_THRESHOLD && !config.fusedCrossEntropy) {
    items.push({
      id: 'vocab-logits-spike',
      title: `大词表 logits 尖峰（V=${model.vocabSize}）`,
      detail:
        `词表 V=${model.vocabSize} > ${LARGE_VOCAB_THRESHOLD}，未开融合 CE 时输出层瞬时峰值 = 10*b*s*V，` +
        '是 8GB 卡头号隐形杀手。建议开启 fusedCrossEntropy（Liger/Unsloth chunked CE，切词表维）。',
      severity: 'critical',
    });
  }
  // eager 注意力 S² 爆炸（未开 FA2 时 warning）。
  if (!config.flashAttention2) {
    items.push({
      id: 'eager-attn-s2',
      title: 'eager 注意力 S² 矩阵爆炸',
      detail:
        '未开 FlashAttention-2，eager 注意力需实例化 L*4*b*a*s² 的 FP32 softmax 矩阵，随序列长度平方增长。' +
        '建议开启 flashAttention2（Windows 可用 PyTorch 原生 SDPA 等效替代，禁 math backend）。',
      severity: 'warning',
    });
  }
  // gradAccum 不省显存认知纠偏（info）。
  items.push({
    id: 'grad-accum-myth',
    title: '梯度累积不省显存（认知纠偏）',
    detail:
      'gradAccumSteps 只影响等效全局 batch 与吞吐，不增加也不减少显存峰值；显存峰值由 micro-batch（batchSize）决定。' +
      '想省显存应减小 batchSize 并同步增大 gradAccumSteps 以保持等效 batch。',
    severity: 'info',
  });
  // 8GB 卡建议 qlora_4bit（vramGb≤8 且 mode=full 时 critical）。
  if (isSmallVram && config.mode === 'full') {
    items.push({
      id: 'small-vram-full',
      title: `${gpu.vramGb}GB 小显存卡不建议全参微调`,
      detail:
        `vramGb=${gpu.vramGb} ≤ ${SMALL_VRAM_GB_THRESHOLD} 且 mode=full，全参微调权重+梯度+优化器状态极易 OOM。` +
        '强烈建议改用 qlora_4bit（NF4 双量化主干 + LoRA 适配器）。',
      severity: 'critical',
    });
  }
  // tie_word_embeddings 对参数量影响（info）。
  items.push({
    id: 'tie-embeddings',
    title: `tie_word_embeddings=${model.tieWordEmbeddings} 对参数量的影响`,
    detail: model.tieWordEmbeddings
      ? `embedding 与 lm_head 共享权重，参数量只计一份 V*h（V=${model.vocabSize}, h=${model.hiddenSize}）。`
      : `embedding 与 lm_head 不共享，参数量计两份 V*h（V=${model.vocabSize}, h=${model.hiddenSize}），` +
        `lm_head 额外占用 V*h=${model.vocabSize * model.hiddenSize} 参数。`,
    severity: 'info',
  });
  // Gemma-2 headDim≠h/n 提示（info）。
  if (model.headDim !== model.hiddenSize / model.numAttentionHeads) {
    items.push({
      id: 'gemma2-headdim',
      title: `headDim=${model.headDim} ≠ hiddenSize/heads=${model.hiddenSize / model.numAttentionHeads}`,
      detail:
        `该架构（如 Gemma-2）Q/O 投影维度为 n*d=${model.numAttentionHeads * model.headDim} 而非 h，` +
        '参数量与 LoRA 形状必须走泛化公式 qDim=n*d，不可想当然用 h 推导。',
      severity: 'info',
    });
  }

  // --- Windows 专属避坑项（windows_wddm 时必出，契约 §7.5 v1.1.1）---
  if (isWindows) {
    items.push({
      id: 'win-flash-attn-sdpa',
      title: 'flash-attn 需预编译轮子，装不上走 SDPA fallback',
      detail:
        'Windows 上 flash-attn 官方轮子稀缺。装不上时用 PyTorch 原生 ' +
        'torch.nn.functional.scaled_dot_product_attention（Flash/Mem-efficient backend，禁 math backend）等效替代，' +
        '显存模型不变（S² 项归零），是 5050 Windows 机的保底路径。',
      severity: 'warning',
    });
    items.push({
      id: 'win-liger-triton',
      title: 'Liger-Kernel 依赖 Triton，Windows 需 triton-windows',
      detail:
        'Liger-Kernel 融合 CE 依赖 Triton；Windows 需安装 triton-windows。不可用时 fused CE 退化为 ' +
        'Unsloth chunked CE 或分块手动实现（切词表维），数学等价。',
      severity: 'warning',
    });
    items.push({
      id: 'win-bitsandbytes-version',
      title: `bitsandbytes 需 ≥${BITSANDBYTES_WINDOWS_MIN_VERSION}（官方 Windows 支持版本线）`,
      detail:
        `Windows 量化需 bitsandbytes ≥${BITSANDBYTES_WINDOWS_MIN_VERSION}；旧版量化会静默失败（不报错但权重未量化），` +
        '导致显存预算失真。务必 pip install -U bitsandbytes 并验证版本。',
      severity: 'critical',
    });
    items.push({
      id: 'win-wddm-taskmgr',
      title: '任务管理器"专用 GPU 内存"与 torch reserved 口径差异',
      detail:
        'WDDM 下任务管理器"专用 GPU 内存"含桌面合成/驱动抢占，与 torch.cuda.memory_reserved() 口径不同；' +
        '以 torch reserved/allocated 与 nvidia-smi 为准做显存校准，勿直接采信任务管理器数值。',
      severity: 'info',
    });
    items.push({
      id: 'win-prevent-sleep',
      title: '长训防 Windows 休眠/自动更新重启',
      detail:
        '长时间训练前关闭 Windows 休眠/睡眠（电源计划设为"从不"），暂停自动更新重启，' +
        '避免训练中断与 checkpoint 丢失；必要时用 powercfg /change standby-timeout-ac 0。',
      severity: 'info',
    });
  }

  return items;
}

// ---------------------------------------------------------------------------
// §7.1 入口
// ---------------------------------------------------------------------------

/**
 * 契约 §7.1：生成全部启动工件。纯函数；若传入 plan，一律以 plan.healedConfig 为准。
 * breakdown 参数用于上下文展示（核心产物以 config 为准）。
 */
export function generateArtifacts(
  model: ModelArchitectureSpec,
  gpu: GpuSpec,
  config: TrainingConfig,
  breakdown: VramBreakdown,
  plan?: SurgicalPlan,
): GeneratedArtifacts {
  // 契约 §7.1：若传入 plan，一律以 plan.healedConfig 为准生成。
  const effectiveConfig = plan !== undefined ? plan.healedConfig : config;
  void breakdown; // 上下文参数：产物以 effectiveConfig 为准，breakdown 供调用方展示复核。
  return {
    llamaFactoryYaml: buildLlamaFactoryYaml(model, effectiveConfig),
    unslothPythonScript: buildUnslothScript(model, gpu, effectiveConfig),
    launchShellScript: buildLaunchShell(effectiveConfig, gpu),
    launchPowerShellScript: buildLaunchPowerShell(effectiveConfig, gpu),
    checklist: buildChecklist(model, gpu, effectiveConfig),
  };
}