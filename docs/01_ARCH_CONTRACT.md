# VRAM Surgeon — 核心架构与数学契约书

> 版本：**v1.1.3（冻结）** · 赛道：2026 Qoder 高校创造力大赛 StudyHack
> 地位：本文件是【执行层】施工与【审核层】验收的最高法典。类型契约见 `src/core/types.ts`（同为冻结件）。
> 冲突裁决顺序：`src/core/types.ts` 字段定义 > 本契约书公式 > 执行层自由裁量。
> v1.1.0 依据用户《核心模块详细工程规范》修订，经总控对账修复原规范 3 处内部矛盾（见附录 A 变更日志）。

---

## 0. 三层接力边界

| 层 | 职责 | 禁区 |
|---|---|---|
| 总控 | 工程底座、类型契约、数学契约 | 不写业务函数体、不写 UI |
| 执行层 | 按本契约实现 6 个目标模块 + 单元测试 | 不改契约字段、不写 UI、不用 `any` |
| 审核层 | 按第 9 节验收标准逐条核验 | 不放宽公式与容差 |

契约变更流程：任何字段/公式变更必须先修订本文件并升版本号，再同步 `types.ts`，最后才允许改实现。

---

## 1. 单位与常量法典（全局强制）

**1 GiB = 1073741824 Bytes = 2^30 Bytes。严禁在任何显存换算中混用 10^9 与 2^30。**

**容量语义（v1.1.0 重大修正）**：GPU 显存颗粒是二进制容量，标称 `vramGb` 数值上直接等于 GiB
（nvidia-smi 报 8192 MiB / 24576 MiB；PyTorch OOM 日志报 "8.00 GiB" / "23.99 GiB"）。
因此 `gpuCapacityGib = gpu.vramGb`，**禁止**再做 `×10⁹÷2³⁰` 换算（v1.0 的 7.45 GiB 算法已撤销）。
十进制 `BYTES_PER_GB_DECIMAL` 仅保留给带宽等厂商标称量，不参与容量计算。

执行层必须命名常量化的物理常数（命名不可省略，值为本契约钉死值）：

| 常量名 | 值 | 含义 |
|---|---|---|
| `BYTES_PER_FP32` | 4 | FP32 单参数字节 |
| `BYTES_PER_BF16` | 2 | BF16/FP16 单参数字节 |
| `INT8_BYTES_PER_PARAM` | 1.05 | LLM.int8() 权重 + absmax 综合开销 |
| `BYTES_PER_NF4` | 0.5 | NF4 单参数字节 |
| `NF4_BLOCK_SIZE` | 64 | bitsandbytes NF4 量化块 |
| `NF4_DOUBLE_QUANT_BLOCK` | 256 | 双量化 absmax 分组 |
| `NF4_DQ_OVERHEAD_BYTES` | 0.015869140625 | 1/64 + 4/(64×256)，双量化开销（≈0.127 bits/param） |
| `NF4_NO_DQ_OVERHEAD_BYTES` | 0.0625 | 未开双量化：FP32 absmax / 64 |
| `LORA_COPY_GRAD_BYTES_PER_PARAM` | 4 | LoRA 适配器 BF16 副本(2) + BF16 梯度(2) |
| `FULL_GRAD_BYTES_PER_PARAM` | 2 | full 模式 BF16 梯度（权重已在 weights 计） |
| `ADAMW32_OPT_BYTES_PER_PARAM` | 12 | FP32 master(4) + m(4) + v(4) |
| `ADAMW8_OPT_BYTES_PER_PARAM` | 6.00390625 | FP32 master(4) + INT8 m/v(2) + 2×(4/2048) absmax |
| `ADAMW8_STATE_BLOCK` | 2048 | 8bit 优化器 absmax 块 |
| `SGD_OPT_BYTES_PER_PARAM` | 4 | 仅 FP32 master，无动量 |
| `PAGED_OPT_MAX_CPU_FRACTION` | 0.6 | paged AdamW 最高可分页到 CPU 的优化器状态比例 |
| `ACT_HIDDEN_FACTOR` | 10 | 每层 hidden 侧激活（Bytes / (b·s·h)） |
| `ACT_KV_FACTOR` | 2 | GQA KV 项（Bytes / (b·s·h/G)） |
| `ACT_MLP_FACTOR` | 6 | SwiGLU gate/up/act（Bytes / (b·s·i)） |
| `EAGER_ATTN_BYTES_PER_ELEM` | 4 | eager 注意力 FP32 softmax 矩阵（Bytes / (b·a·s²)） |
| `FLASH_ATTN_MATRIX_BYTES` | 0 | FlashAttention 片上分块，S² 项归零 |
| `CHECKPOINT_RECOMPUTE_FACTOR` | 1.25 | checkpoint 反向重算单层的峰值放大系数 |
| `LOGITS_UNFUSED_BYTES_PER_ELEM` | 10 | BF16 logits(2) + FP32 upcast(4) + FP32 softmax/grad(4) |
| `FUSED_CE_CHUNK` | 4096 | Liger/Unsloth 融合 CE 的**词表维**分块大小 |
| `CUDA_CONTEXT_GIB` | 0.65 | CUDA Runtime Context + cuBLAS/Triton workspace |
| `OS_RESERVED_WDDM_GIB` | 0.75 | Windows WDDM 桌面合成器与驱动抢占（linux_headless = 0） |
| `FRAG_DYNAMIC_PADDING` | 0.14 | dynamicPadding 且未开 expandable 的动态张量碎片率 |
| `FRAG_STABLE` | 0.035 | expandable_segments 或定长 Packing 的碎片率 |
| `HEAL_TARGET_RATIO` | 0.85 | 手术停止条件：reservedPeak ≤ 0.85×capacity（安全绿区） |
| `MEMORY_BOUND_THRESHOLD` | 0.85 | utilizationRatio ≥ 此值判 MEMORY_BOUND |
| `FRAG_GAP_ABS_GIB` | 1.0 | 碎片诊断绝对阈值 |
| `FRAG_GAP_RATIO` | 0.10 | 碎片诊断相对阈值（×capacity） |
| `LOGITS_SPIKE_MATCH_TOLERANCE` | 0.25 | triedToAllocate 与 logits 候选峰值匹配容差 ±25% |
| `SEQ_LEN_LAST_RESORT_FACTOR` | 0.75 | 末位有损手段：序列长度缩减因子 |
| `MAX_SPLIT_SIZE_MB` | 128 | launch 脚本注入的 max_split_size_mb |
| `MIB_PER_GIB` | 1024 | OOM 日志 MiB→GiB 归一 |

> 激活模型是**保守工程上界**，不是逐算子仿真；真机探针（第 8 节）负责校准。审核层对激活项容差 ±35%，对其余分项容差 ±2%。

---

## 2. 目标模块总览与依赖图

```
presets/gpus.ts ─┐
presets/models.ts┤（纯数据，无依赖）
                 ▼
hf-parser/index.ts ──► ModelArchitectureSpec
                 ▼
physics-engine/index.ts ──► VramBreakdown        （纯函数，零副作用）
                 ▼
oom-diagnostician/index.ts ──► OomDiagnosisResult / SurgicalPlan（内部调用 physics-engine）
                 ▼
codegen/index.ts ──► GeneratedArtifacts          （纯函数）

python-probe/vram_telemetry_probe.py ──► HardwareTelemetryReport（独立进程，stdout JSON）
```

全部核心模块只依赖 `src/core/types.ts`。禁止核心层 import 任何 UI 框架、DOM API（`hf-parser` 的 `fetch` 除外）。

---

## 3. 模块一：预设库

### 3.1 `src/core/presets/gpus.ts`

```ts
export const GPU_PRESETS: readonly GpuSpec[];
export const DEFAULT_GPU_ID = 'rtx-5050-8g';            // 核心默认基准卡
export function getGpuById(id: string): GpuSpec | undefined;
```

必须内置（`fp16Tflops`/`int8Tflops` 均为 **dense** 峰值，禁止填稀疏值；估算值容差 ±15%，仅用于瓶颈启发式）：

| id | name | vramGb | arch | 位宽 bit | 带宽 GB/s | fp16 dense | int8 dense | bf16 | fp8 |
|---|---|---|---|---|---|---|---|---|---|
| `rtx-5050-8g` | NVIDIA GeForce RTX 5050 | 8 | Blackwell | 128 | **320** | ≈52.7 | ≈105.4 | true | true |
| `rtx-4060-8g` | NVIDIA GeForce RTX 4060 | 8 | Ada | 128 | 272 | ≈60.7 | ≈121.4 | true | true |
| `rtx-4070-12g` | NVIDIA GeForce RTX 4070 | 12 | Ada | 192 | 504 | ≈116.6 | ≈233.2 | true | true |
| `rtx-4060ti-16g` | NVIDIA GeForce RTX 4060 Ti 16GB | 16 | Ada | 128 | 288 | ≈60.7 | ≈121.4 | true | true |
| `rtx-3060-12g` | NVIDIA GeForce RTX 3060 12GB | 12 | Ampere | 192 | 360 | ≈51.2 | ≈102.4 | true | false |
| `rtx-3090-24g` | NVIDIA GeForce RTX 3090 | 24 | Ampere | 384 | 936 | ≈142.2 | ≈284.4 | true | false |
| `rtx-4090-24g` | NVIDIA GeForce RTX 4090 | 24 | Ada | 384 | 1008 | ≈330.3 | ≈660.6 | true | true |
| `a100-40g` | NVIDIA A100 40GB | 40 | Ampere | 5120 | 1555 | 312 | 624 | true | false |
| `a100-80g` | NVIDIA A100 80GB | 80 | Ampere | 5120 | 2039 | 312 | 624 | true | false |

> 修正记录：原规范 RTX 5050 带宽 256 GB/s 有误，官方 GDDR7 128-bit @20 Gbps = **320 GB/s**，本契约以 320 为准。
> 容量自检：8 → 8.00 GiB；12 → 12.00 GiB；24 → 24.00 GiB（torch 实测 23.99，容差内）。
> **真实部署环境（v1.1.1 钉死）**：开发机 = RTX 3060 12GB（若实为 8GB  variant 需回报总控改预设）；校准/训练机 = RTX 5050 8GB，**纯 Windows（WDDM），无 Linux**。真机探针与实测验收一律走 `windows_wddm` 分支。

### 3.2 `src/core/presets/models.ts`

```ts
export const MODEL_PRESETS: readonly ModelArchitectureSpec[];
export function getModelById(modelId: string): ModelArchitectureSpec | undefined;
```

必须严格按照 HuggingFace 官方 config.json 录入（`numParams` 必须与第 4.2 节公式自洽，`headDim` 必填）：

| modelId | h | i | L | heads | kv | headDim | vocab | tie | maxPos | P |
|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | 3584 | 18944 | 28 | 28 | 4 | 128 | **152064** | false | 32768 | 7,615,487,488 |
| `meta-llama/Llama-3.1-8B-Instruct` | 4096 | 14336 | 32 | 32 | 8 | 128 | **128256** | false | 131072 | 8,030,261,248 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 3584 | 18944 | 28 | 28 | 4 | 128 | 152064 | false | 32768 | 7,615,487,488 |
| `Qwen/Qwen2.5-3B-Instruct` | 2048 | 11008 | 36 | 16 | 2 | 128 | **151936** | **true** | 32768 | 3,085,846,528 |
| `Qwen/Qwen2.5-14B-Instruct` | 5120 | 13824 | 48 | 40 | 8 | 128 | 152064 | false | 32768 | 14,769,689,600 |
| `google/gemma-2-9b-it` | 3584 | 14336 | 42 | 16 | 8 | **256** | 256000 | **true** | 8192 | ≈9,241,404,928 |

> 陷阱备忘：① Qwen2.5-3B 词表 151936 ≠ 7B 的 152064；② Qwen2.5-7B tie=false（7.615B = 6.525B 主干 + 2×0.545B 双份词表），3B 才 tie=true；③ DeepSeek-R1-Distill-Qwen-7B 架构与 Qwen2.5-7B 完全相同；④ **Gemma-2 的 headDim=256 ≠ h/n=224**，Q/O 投影维度是 n×d=4096 而非 h，必须走泛化公式；Gemma-2 每层 4 个 RMSNorm（公式按 2h 计，差值 301,056 参数 <0.004%，落入容差）、滑窗注意力按全 S² 保守计。

---

## 4. 模块二：HF 解析器与参数量反推

### 4.1 `src/core/hf-parser/index.ts` 导出签名

```ts
export function normalizeHfConfig(raw: Record<string, unknown>, modelId: string): ModelArchitectureSpec;
export function parseLocalConfig(jsonText: string, modelId?: string): ModelArchitectureSpec;  // 粘贴/拖拽上传模式，modelId 缺省 'local/custom'
export function estimateParamCount(spec: ModelArchitectureSpec): number;
export interface FetchModelSpecOptions {
  endpoint?: 'huggingface' | 'hf-mirror';   // 默认 'huggingface'，失败自动降级 hf-mirror 重试一次
  timeoutMs?: number;                        // 默认 10_000，命名常量
}
export async function fetchModelSpec(modelId: string, options?: FetchModelSpecOptions): Promise<ModelArchitectureSpec>;
```

- 双输入模式：repoId 拉取 `https://huggingface.co/{modelId}/resolve/main/config.json`（镜像 `https://hf-mirror.com/...` 降级重试）；或本地 config.json 字符串/对象直接归一化。
- 原始 JSON 一律按 `Record<string, unknown>` 接收，逐字段类型收窄，**禁止 `any`、禁止非空断言兜底**。
- 异构字段名兼容映射（取第一个存在且类型合法者）：

| 归一化字段 | 兼容别名 |
|---|---|
| hiddenSize | `hidden_size`, `n_embd`, `d_model` |
| intermediateSize | `intermediate_size`, `n_inner`, `ffn_dim` |
| numHiddenLayers | `num_hidden_layers`, `n_layer`, `num_layers` |
| numAttentionHeads | `num_attention_heads`, `n_head`, `num_heads` |
| numKeyValueHeads | `num_key_value_heads`, `n_head_kv`, `num_kv_heads`, `multi_query_group_num`（缺省 = heads，即 MHA） |
| headDim | `head_dim`（缺省 = hiddenSize / numAttentionHeads） |
| vocabSize | `vocab_size`, `n_vocab` |
| tieWordEmbeddings | `tie_word_embeddings`, `tie_embeddings`（缺省 false） |
| maxPositionEmbeddings | `max_position_embeddings`, `n_positions`, `max_seq_len` |

- 边界规则（**v1.1.2 裁决修订**）：v1.1 起支持 `head_dim ≠ h/n`（Gemma-2）。非 gated MLP 检测**按激活字段命名谱系判定，禁止按激活函数名含 gelu 判定**：
  - 白名单：config 含 `hidden_act`（LLaMA/Qwen/Mistral 谱系，silu/swiglu）或 `hidden_activation`（Gemma 谱系，**含 `gelu_pytorch_tanh`，即 GeGLU**）→ gated 三矩阵 MLP，放行。GeGLU 与 SwiGLU 仅激活函数不同，矩阵数（3hi）与激活保存张量数完全一致，参数量与显存模型不变。
  - 抛错：含 `activation_function` 且**无** `hidden_act`/`hidden_activation`（GPT-2/GPT-NeoX 谱系，c_fc/c_proj 双矩阵 + 学习式位置编码）→ 抛 `Error('[hf-parser] unsupported MLP architecture: ...')`，禁止静默错算。
  - 三类字段皆无：默认放行（预设库覆盖已验证架构）。
  - 修订依据：v1.1.1 的"gelu 且无 SwiGLU 标志抛错"措辞会误杀真实 Gemma-2 config（`hidden_activation: gelu_pytorch_tanh`），执行层审计发现，总控认领契约措辞错误。
- 网络/解析失败抛出带 `[hf-parser]` 前缀的 Error；不做静默假数据兜底。

### 4.2 参数量反推公式（泛化版，数学契约）

记 `h, i, L, n, n_kv, V` 同名字段，`d = headDim`，`qDim = n*d`，`kvDim = n_kv*d`，`G = n/n_kv`。

单层 Transformer（SwiGLU MLP + RMSNorm，GQA）：

```
P_layer = h*qDim         # Q 投影
        + 2*h*kvDim      # K、V 投影
        + qDim*h         # O 投影
        + 3*h*i          # gate_proj / up_proj / down_proj
        + 2*h            # 2× RMSNorm（Gemma-2 的 4 norm 差值落入容差）
```

总参数量（**tie 分支是验收必考点**）：

```
P = V*h * (tieWordEmbeddings ? 1 : 2)   # tie=true：embedding 与 lm_head 共享，只计一份
  + L * P_layer                          # tie=false：额外计入 lm_head 的 V*h
  + h                                    # final norm
```

- 标准架构（d = h/n）退化为 `P_layer = 2h² + 2h·kvDim + 3hi + 2h`；Gemma-2（d=256）必须用泛化式。
- 忽略项（合计 <0.01%）：QKV bias、RoPE 无参数、Gemma-2 embedding 缩放与 logits softcap（无参数）。
- 黄金值：Llama-3.1-8B `8,030,261,248`；Qwen2.5-7B `7,615,487,488`；Qwen2.5-3B（tie=true）`3,085,846,528`；Qwen2.5-14B `14,769,689,600`；Gemma-2-9b `≈9,241,404,928`。误差 ≤0.5%。
- 自洽性：`normalizeHfConfig` 输出的 `numParams` 必须等于 `estimateParamCount` 结果。

---

## 5. 模块三：物理引擎（字节级显存方程）

### 5.1 `src/core/physics-engine/index.ts` 导出签名

```ts
export function bytesToGib(bytes: number): number;   // bytes / BYTES_PER_GIB，唯一换算入口
export function computeTrainableParams(model: ModelArchitectureSpec, config: TrainingConfig): number;
export function computeVramBreakdown(gpu: GpuSpec, model: ModelArchitectureSpec, config: TrainingConfig): VramBreakdown;
```

纯函数、零副作用、零网络。输入边界校验（正整数等），非法输入抛 `RangeError('[physics-engine] ...')`。

记 `b = batchSize`（micro-batch），`s = seqLen`，`P = numParams`，`T = computeTrainableParams(...)`，`a = numAttentionHeads`，`G = n/n_kv`。
**铁律：`gradAccumSteps` 不出现在任何显存方程中**；`dynamicPadding` 不改变激活峰值（峰值按满序列 s 计），只通过碎片率（5.6）影响 reserved。

### 5.2 分项一：weights（主干权重 + 量化常数）

```
full / lora:      W_bytes = P * 2                       # BF16 主干
qlora_8bit:       W_bytes = P * INT8_BYTES_PER_PARAM    # 1.05P
qlora_4bit:       W_bytes = P * (0.5 + NF4_DQ_OVERHEAD_BYTES)   # ≈0.515869140625P（双量化默认开启）
```

**归属规则（v1.1 变更）**：LoRA 适配器的 BF16 副本不在 weights 计，统一归入 gradAndOpt（5.3 的 4T），避免重复计费。
黄金值：Qwen2.5-7B BF16 = 15,230,974,976 B ≈ 14.18 GiB；QLoRA-4bit ≈ 3.66 GiB。

### 5.3 分项二：gradAndOpt（副本 + 梯度 + 优化器状态）

可训练参数量：

```
full:  T = P
lora / qlora_*:  T = L * Σ_{m ∈ loraTargetModules} r * (d_in(m) + d_out(m))
```

模块形状表（`r = loraRank`，注意 q/o 用 qDim 而非 h，Gemma-2 才体现差异）：

| module | d_in | d_out |
|---|---|---|
| q_proj | h | qDim |
| k_proj / v_proj | h | kvDim |
| o_proj | qDim | h |
| gate_proj / up_proj | h | i |
| down_proj | i | h |

字节方程（**8bit AdamW 含 FP32 master，裁决已确认**）：

```
optBytes: adamw_32bit = 12;  adamw_8bit = paged_adamw_8bit = 6.00390625;  sgd = 4

full:        G_bytes = T * (2 + optBytes)          # 梯度 2T（权重已在 weights）
lora/qlora:  G_bytes = T * (4 + optBytes)          # 适配器 BF16 副本 2T + 梯度 2T + 优化器
```

即：full+adamw32 = 14T（经典 16B/param = 权重 2 + 14）；lora+adamw32 = 16T；lora+adamw8 = 10.00390625T。

**paged 分页规则（确定性，v1.1.3 公式修正）**：仅当 `optimizer === 'paged_adamw_8bit'` 且**不分页时** `totalReservedPeakGib > gpuCapacityGib` 才启用。优化器扣减会流经汇总层 `(1+fragRatio)` 乘子，故分页比例必须反解方程：

```
excess   = unpagedReservedPeakGib − gpuCapacityGib
fraction = min(PAGED_OPT_MAX_CPU_FRACTION, excess / (optStateGib × (1 + fragRatio)))
```

`optStateGib = T × optBytes / 2^30`（仅优化器状态部分，不含副本与梯度）。未触封顶时pag后 `totalReservedPeakGib == gpuCapacityGib`（浮点误差 ≤1e-6），即"刚好脱险"；触 0.6 封顶仍 OOM 则如实报告。分页只扣减 optBytes 部分，副本与梯度不动。formulaNote 必须写明实际分页比例与反解算式。
> v1.1.2 字面公式 `excess/optStateGib` 漏乘碎片乘子导致过分页（审核 D1：24GiB 卡多降 1.40 GiB），总控认领公式错误，执行层无责。

黄金值：Qwen2.5-7B r=16 全 7 模块 → `T = 40,370,176`；lora+adamw32 `G_bytes = 645,922,816 B = 0.6015625 GiB`（精确）。

### 5.4 分项三：activations（前向激活）

单层线性激活（Bytes，因子已含 BF16 字节宽度）：

```
A_layer_linear = b*s*h * (ACT_HIDDEN_FACTOR + ACT_KV_FACTOR/G + ACT_MLP_FACTOR*i/h)
               = b*s * (10h + 2h/G + 6i)
```

注意力矩阵分项（全 L 层累计，写入 `attentionMatrixBytes`）：

```
flashAttention2 = true:   M_bytes = L * FLASH_ATTN_MATRIX_BYTES = 0     # 片上分块，≈0
flashAttention2 = false:  M_bytes = L * EAGER_ATTN_BYTES_PER_ELEM*b*a*s² = L*4*b*a*s²
```

> **Windows 弹性条款（v1.1.1）**：`flashAttention2=true` 的物理语义是"无 S² 注意力路径"。Windows 上 flash-attn 轮子难装时，
> 用 PyTorch 原生 `scaled_dot_product_attention`（Flash/Mem-efficient backend，禁 math backend）等效替代，显存模型不变。
> codegen 在 Windows 产物中必须注明该 fallback（SDPA 无需额外安装，是 5050 Windows 机的保底路径）。

总激活：

```
gradCheckpointing = false:  A_bytes = L * (A_layer_linear + A_layer_attn)
gradCheckpointing = true:   A_bytes = L*(2*b*s*h) + CHECKPOINT_RECOMPUTE_FACTOR*(A_layer_linear + A_layer_attn)
                            # 只存各层入口输入 + 1.25×单层重算峰值
```

黄金值（Qwen2.5-7B, b=1, s=2048）：
- `A_layer_linear = 2048×(35840+1024+113664) = 308,281,344 B`；flash 无 checkpoint 全层 = 8,631,877,632 B = **8.0390625 GiB**（精确）。
- eager `M_bytes = 28×4×1×28×2048² = 13,153,337,344 B = 12.25 GiB`（精确，必考）。
- flash + checkpoint：`411,041,792 + 1.25×308,281,344 = 796,393,472 B ≈ 0.7417 GiB`（±35%）。

### 5.5 分项四：logitsPeak（大词表输出层瞬时峰值）——关键工程亮点

```
unfused:  L_bytes = LOGITS_UNFUSED_BYTES_PER_ELEM * b*s*V = 10*b*s*V
fused:    L_bytes = b*s * min(V, FUSED_CE_CHUNK) * BYTES_PER_FP32      # 切词表维，不实例化完整 [b,s,V]
```

> v1.1 裁决：原规范 `min(b·s,4096)·V·4` 是切 token 维，b·s=2048 时算出 1.16 GiB，与其宣称的 ≈60 MB 及 Case 2 绿区矛盾。
> 真实 Liger-Kernel FusedLinearCrossEntropy / Unsloth Chunked CE 切的是**词表维**，故钉死为 `b·s·min(V,4096)·4`。

黄金值（必考）：Qwen2.5-7B, b=1, s=2048, V=152064：
- unfused = 3,114,270,720 B = **2.900390625 GiB**（精确）——8GB 卡头号隐形杀手；
- fused = 2048×4096×4 = 33,554,432 B = **0.03125 GiB（32 MiB）**（精确）；
- Llama-3.1-8B 同条件 unfused = 2,626,682,880 B ≈ 2.4463 GiB。
- `isFused = config.fusedCrossEntropy`；formulaNote 必须写明 V 值与 chunk 算式。

### 5.6 分项五：systemAndFrag 与 summary

```
cudaContextGib   = CUDA_CONTEXT_GIB                              # 0.65
osReservedGib    = osEnv === 'windows_wddm' ? OS_RESERVED_WDDM_GIB : 0   # 0.75 / 0
fragRatio        = (expandableSegments || !dynamicPadding) ? FRAG_STABLE : FRAG_DYNAMIC_PADDING   # 0.035 / 0.14
fragmentationGib = fragRatio * totalAllocatedGib
totalGib         = cudaContextGib + osReservedGib + fragmentationGib

totalAllocatedGib    = weights.gib + gradAndOpt.gib + activations.gib + logitsPeak.gib
totalReservedPeakGib = totalAllocatedGib * (1 + fragRatio) + cudaContextGib + osReservedGib
gpuCapacityGib       = gpu.vramGb                                # 二进制容量语义，见第 1 节
utilizationRatio     = totalReservedPeakGib / gpuCapacityGib
isOom                = totalReservedPeakGib > gpuCapacityGib
headRoomGib          = gpuCapacityGib - totalReservedPeakGib     # OOM 时为负
bottleneckType       = utilizationRatio >= MEMORY_BOUND_THRESHOLD ? 'MEMORY_BOUND' : 'COMPUTE_BOUND'
```

每个分项的 `formulaNote` 必须写出代入数值后的算式（如 `"10*1*2048*152064/2^30"`），供审核层与 UI 展示复核。

---

## 6. 模块四：OOM 诊断与自动手术

### 6.1 `src/core/oom-diagnostician/index.ts` 导出签名

```ts
export function parseOomTraceback(logText: string): OomDiagnosisResult;
export function diagnoseOom(logText: string, context?: { breakdown: VramBreakdown }): OomDiagnosisResult;
export function solveAutoHeal(gpu: GpuSpec, model: ModelArchitectureSpec, config: TrainingConfig, diagnosis?: OomDiagnosisResult): SurgicalPlan;
```

### 6.2 PyTorch OOM 正则（命名捕获组，GiB/MiB/bytes 三单位）

主正则（PyTorch ≥ 2.1；**free 字段可能是 "0 bytes"，必须兼容**）：

```regex
/CUDA out of memory\. Tried to allocate (?<tried>[\d.]+) (?<triedUnit>GiB|MiB)\. GPU \d+ has a total capacity of (?<cap>[\d.]+) (?<capUnit>GiB|MiB) of which (?<free>[\d.]+) (?<freeUnit>GiB|MiB|bytes) is free\.(?: Including non-PyTorch memory, this process has (?<proc>[\d.]+) (?<procUnit>GiB|MiB) memory in use\.)? Of the allocated memory (?<alloc>[\d.]+) (?<allocUnit>GiB|MiB) is allocated by PyTorch, and (?<gap>[\d.]+) (?<gapUnit>GiB|MiB) is reserved by PyTorch but unallocated\./
```

旧格式兜底：`/Tried to allocate ([\d.]+) (GiB|MiB).*?([\d.]+) (GiB|MiB) already allocated/`。
单位归一：MiB ÷ MIB_PER_GIB → GiB；bytes ÷ BYTES_PER_GIB。`reservedGib = allocatedGib + gapGib`。
解析失败：`parsed=false`，数值置 0，`rootCause='UNKNOWN'`，禁止抛异常中断手术流程。

### 6.3 rootCause 判定顺序（短路求值，evidence 必须写明命中依据）

1. `unallocatedGapGib >= max(FRAG_GAP_ABS_GIB, FRAG_GAP_RATIO * totalCapacityGib)` → `ALLOCATOR_FRAGMENTATION`
   （修复原规范 "gap > tried" 判据与其样例 gap=1.18 < tried=2.32 的自相矛盾；语义：空闲缓存够但不连续）
2. 有 breakdown 上下文且 `triedToAllocateGib` 与候选 `{2,4,10}×b·s·V` 之一相对误差 ≤ `LOGITS_SPIKE_MATCH_TOLERANCE` → `VOCAB_LOGITS_SPIKE`（FP32 upcast 单张量 = 4bsV 是最常见触发）
3. `weights.gib >= totalCapacityGib` → `WEIGHT_OVERFLOW`
4. breakdown 中 activations 为最大分项 → `ACTIVATION_EXPLOSION`
5. 其余 → `UNKNOWN`

### 6.4 solveAutoHeal 五阶段无损优先削峰策略（v1.1 重排）

**原子手术单位（v1.1.3 裁决，解决审核 D3/D4 契约冲突）**：下表**一行 = 一个原子手术 = 一条 `SurgicalStep`**。同行耦合字段（S2 的 flash+expandable、S4 的 mode+optimizer）**合并进同一个 `configPatch` 同时应用**，禁止拆成两条步骤；若其中仅一个字段需变更，patch 只含该字段。S5 每次减半为独立一条步骤。
**停止条件：每条 `SurgicalStep` 应用后立即重算 `computeVramBreakdown`，`totalReservedPeakGib ≤ HEAL_TARGET_RATIO × gpuCapacityGib`（0.85 安全绿区）即停**，禁止继续任何后续步骤。
每条实际生效的步骤产出 `SurgicalStep`（`stepIndex` 全局递增，`beforeGib/afterGib` 取 `totalReservedPeakGib`，`savedGib > 0` 才记录，`sideEffectNote` 必填）。
> 裁决依据：字段级原子停会使 Case 2 停在 S2a 后 6.388 GiB，脱离用户钉死的 [5.4,6.2] 黄金窗口；阶段原子既保验收数又合"耦合疗法"语义。此裁决下阶段边界检查语义正确，执行层仅需合并步骤记录。

| 阶段 | 动作 | configPatch | 副作用等级 |
|---|---|---|---|
| S1 融合交叉熵 | `fusedCrossEntropy: true`（若 false） | 无损 | 数学等价，斩大词表尖峰 |
| S2 注意力+分配器 | `flashAttention2: true` + `expandableSegments: true`（**合并为同一 SurgicalStep**，仅变更者为 false 的字段） | 无损 | 精确注意力斩 S²；allocator 整形 |
| S3 梯度检查点 | `gradCheckpointing: true`（若 false） | 无损 | 吞吐降约 20–35% |
| S4 量化+分页优化器 | `mode → 'qlora_4bit'`（若当前非 qlora_4bit）+ `optimizer → 'paged_adamw_8bit'`（若当前非 paged），**合并为同一 SurgicalStep** | 近无损 | NF4 双量化误差风险；CPU 分页延迟 |
| S5 等效 batch 守恒 | `batchSize → max(1, floor(batchSize/2))` 且 `gradAccumSteps × 2`，循环至 1 | 无损 | 等效全局 batch 不变，吞吐下降 |

S5 用尽仍未进绿区时的末位手段（有损，必须显式警告）：`seqLen → floor(seqLen × SEQ_LEN_LAST_RESORT_FACTOR)`，`sideEffectNote` 注明长文截断风险。
全部手段用尽仍 OOM：`isHealed=false` 如实返回。`diff` 按 `VramDiff` 填充；`animationTimeline` 核心层可省略。

---

## 7. 模块五：零报错启动包编译器

### 7.1 `src/core/codegen/index.ts` 导出签名

```ts
export function generateArtifacts(
  model: ModelArchitectureSpec,
  gpu: GpuSpec,
  config: TrainingConfig,
  breakdown: VramBreakdown,
  plan?: SurgicalPlan,
): GeneratedArtifacts;
```

纯函数；若传入 plan，一律以 `plan.healedConfig` 为准生成。

### 7.2 LLaMA-Factory train_config.yaml 映射规范

必须可被 `yaml` 库无损 parse。键映射：`model_name_or_path`；`finetuning_type: full|lora`；qlora 时 `quantization_bit: 4|8` + `quantization_method: bitsandbytes` + `double_quantization: true`；`lora_rank/lora_alpha/lora_target`；`per_device_train_batch_size`；`gradient_accumulation_steps`；`cutoff_len`；`gradient_checkpointing`；`flash_attn: fa2`（开启时）；**`enable_liger_kernel: true`（fusedCrossEntropy 开启时）**；`optim: adamw_torch|adamw_8bit|paged_adamw_8bit|sgd`；`bf16: true`；`output_dir` 命名常量。

### 7.3 Unsloth 极速微调独立 Python 脚本（train_unsloth.py）

骨架：`FastLanguageModel.from_pretrained(model_name=..., max_seq_length=seqLen, dtype=None, load_in_4bit=(mode==='qlora_4bit'))` → `FastLanguageModel.get_peft_model(r=loraRank, lora_alpha=loraAlpha, target_modules=[...], use_gradient_checkpointing='unsloth' if gradCheckpointing else False)` → `SFTTrainer + TrainingArguments(per_device_train_batch_size, gradient_accumulation_steps, optim, bf16=True, ...)`。8GB 卡（vramGb≤8）自动 load_in_4bit。脚本必须可直接 `python` 运行（import 齐全、无 TODO 占位符）；Unsloth 内置 chunked CE 以注释注明。

### 7.4 环境避坑启动脚本（launch.sh + launch.ps1，双份必出）

- sh：`#!/usr/bin/env bash` + `set -euo pipefail`；ps1：PowerShell 语法 `$env:PYTORCH_CUDA_ALLOC_CONF = "..."`。
- `expandableSegments=true` 时注入 `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"`（MAX_SPLIT_SIZE_MB 常量化）。
- 显卡架构兼容性检查：FlashAttention-2 要求 Ampere 及以上（compute capability ≥ 8.0），脚本内以 nvidia-smi 查询 + 条件警告实现。
- 可选 `HF_ENDPOINT=https://hf-mirror.com`（命名常量）；末尾调用 `llamafactory-cli train` 或 python 脚本。

### 7.5 避坑 Checklist（动态生成，至少覆盖）

WDDM 抢占 0.75 GiB（windows_wddm 时 critical）；大词表 logits 尖峰（V>100000 且未 fused 时 critical）；eager 注意力 S² 爆炸（未开 FA2 时 warning）；gradAccum 不省显存认知纠偏（info）；8GB 卡建议 qlora_4bit（vramGb≤8 且 mode=full 时 critical）；tie_word_embeddings 对参数量影响（info）；Gemma-2 headDim≠h/n 提示（info）。

**Windows 专属避坑项（windows_wddm 时必出，v1.1.1）**：
- flash-attn 需预编译轮子，装不上则走 SDPA fallback（warning，附 `torch.nn.functional.scaled_dot_product_attention` 说明）；
- Liger-Kernel 依赖 Triton，Windows 需 `triton-windows`；不可用时 fused CE 退化为 Unsloth chunked CE 或分块手动实现（warning）；
- bitsandbytes 需 ≥0.43（官方 Windows 支持版本线），旧版量化会静默失败（critical）；
- WDDM 下任务管理器"专用 GPU 内存"与 torch reserved 的口径差异说明（info）；
- 长训防 Windows 休眠/自动更新重启（info）。

---

## 8. 模块六：本地真机 PyTorch 遥测校准探针

### 8.1 `python-probe/vram_telemetry_probe.py` 规范

- 轻量独立可运行（命令行 `python vram_telemetry_probe.py [--model-id ...] [--seq-len ...] [--batch-size ...]`），依赖仅 `torch` + 标准库。
- **stdout 只输出一个 JSON 对象**，严格匹配 `HardwareTelemetryReport`（`schemaVersion: 'vram-telemetry/1'` 字面量锁死）；所有日志走 stderr；同时落盘 `telemetry_report.json` 供前端工作台拖入做"理论 vs 实测"双轨校准。
- 采集项：GPU 型号/总显存/compute capability、driver 与 CUDA 版本、**WDDM/TCC 模式**（Windows 用 `nvidia-smi --query-gpu=driver_model.current`，Linux 填 unknown）、CUDA Context 初始占用（init 前后差值）。
- 工作负载：**模拟或真实加载指定维度的 Transformer 层与 Cross-Entropy 峰值**（按传入 h/L/V 构造张量前向 + CE），调用 `torch.cuda.memory_stats()` 记录 `allocated_bytes.all.peak` 与 `reserved_bytes.all.peak`，配合 `mem_get_info` 采 deviceFree。
- 每阶段（forward/logits_spike/backward/optimizer_step）一条 `TelemetrySample`，采样前 `reset_peak_memory_stats()`；空张量分配/释放循环测 `fragmentationRatio`；可选大张量拷贝计时测带宽（未测为 null）。
- 退出码：0 成功；2 无 CUDA（stderr 说明，stdout 不输出残缺 JSON）。字节→GiB 一律 `1073741824`。
- **部署目标（v1.1.1）**：首要运行环境为 RTX 5050 8GB + Windows（WDDM），脚本必须 Windows 原生可跑（路径用 `pathlib`，禁止依赖 fcntl/termios 等 POSIX-only 模块）；`driverModel` 实测填 `wddm`。
- **校准反哺机制**：`CUDA_CONTEXT_GIB`(0.65)、`OS_RESERVED_WDDM_GIB`(0.75)、`FRAG_*` 均为先验估算；探针实测值（calibration 字段）用于"理论 vs 实测"双轨对比。若实测系统性偏离先验 >20%，由总控走契约变更流程修订常量，执行层不得私自改数。

---

## 9. 施工纪律与验收标准

### 9.1 执行层施工纪律（违反即审核不通过）

1. **零 UI**：不创建任何组件/页面/样式文件，不引入 react/vue 等依赖。
2. **零 `any`**：包括 `as any`、`<any>`、隐式 any；外部 JSON 用 `unknown` + 类型收窄。
3. **魔法数字常量化**：第 1 节常量表全部落地为命名导出常量；除 0/1/2 等纯数学恒等量外禁止裸数字。
4. **单位纪律**：显存换算只走 `bytesToGib` / `BYTES_PER_GIB`；容量直接取 vramGb（GiB 语义）。
5. **所有数学公式必须附物理意义注释**（本契约公式编号可引用，如 `// 契约 §5.5: 10*b*s*V`）。
6. 核心模块纯函数化（hf-parser 网络与 probe 除外）；错误带模块前缀；不静默兜底假数据。

### 9.2 审核层验收清单

**单元测试文件钉死：`src/core/__tests__/engine.test.ts`，至少包含以下 4 个硬核用例，`npx vitest run` 100% 绿灯。**

**Case 1 — RTX 5050 8GB 默认微调 Qwen2.5-7B 爆显存复现**
场景钉死：`rtx-5050-8g` + Qwen2.5-7B，b=1, s=2048, r=16 全 7 模块, adamw_32bit，所有优化开关 false，`windows_wddm`，dynamicPadding=true，expandable=false。
- 1a（mode=lora）：`totalReservedPeakGib > 18`（理论 ≈44.7 GiB）。
- 1b（mode=qlora_4bit + gradCheckpointing=true，其余不变，**未开 fusedCrossEntropy**）：2.90 GiB logits 尖峰 + WDDM 0.75 + 碎片 14% → reserved ≈11.0 GiB > 8.00 GiB 红线，`isOom=true`。

**Case 2 — Auto-Heal 一键抢救**
场景钉死：Case 1b 状态但 `osEnv='linux_headless'`。**单元测试是纯函数模拟，与宿主机操作系统无关**——在 Windows 开发机上跑 vitest 同样必须通过（osEnv 只是配置参数）。真机验证（5050 Windows）走 WDDM 变体：术后 ≈6.6 GiB < 8.00 GiB，安全但不在 5.4~6.2 窗口内，故黄金窗口验收以 linux 模拟为准，WDDM 变体只验 `isHealed=true`。
- `solveAutoHeal` 自动开启 fusedCrossEntropy（S1）+ flashAttention2/expandableSegments（S2）后进入绿区停止；
- `totalReservedPeakGib ∈ [5.4, 6.2]`（理论 ≈5.86 GiB）；`isHealed=true`；
- 全程 `batchSize × gradAccumSteps` 等效全局 batch 守恒；S1/S2 无损步先于任何有损步。

**Case 3 — PyTorch OOM 真实报错解析**
fixture（含 "0 bytes is free" 与碎片）：
`CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Of the allocated memory 5.12 GiB is allocated by PyTorch, and 1.18 GiB is reserved by PyTorch but unallocated.`
- 精确提取 tried=2.32 / cap=8.00 / alloc=5.12 / gap=1.18 / reserved=6.30；
- gap=1.18 ≥ max(1.0, 0.8) → `rootCause='ALLOCATOR_FRAGMENTATION'`；
- 手术方案首步含 `expandableSegments: true` 环境变量修复。另测 MiB 单位与旧格式 fixture。

**Case 4 — 代码编译器输出校验**
- YAML 可 parse 且含 `quantization_bit: 4`、`flash_attn: fa2`、`enable_liger_kernel: true`、`optim: paged_adamw_8bit`（对应术后配置）；
- Unsloth 脚本含 `load_in_4bit=True` 与 `use_gradient_checkpointing="unsloth"`；
- launch.sh 与 launch.ps1 均含 `expandable_segments:True,max_split_size_mb:128`。

**黄金值回归测试（同文件或者 tests/ 补充）**：

| # | 验收项 | 期望 | 容差 |
|---|---|---|---|
| A1 | `tsc --noEmit` strict 全量 | 0 报错 | 精确 |
| A2 | src/core 类型位置 `any` | 0 命中 | 精确 |
| A3 | bytesToGib(1073741824) | 1 | 精确 |
| A4 | 8GB 卡 gpuCapacityGib | 8.00 | ±0.01 |
| A5 | 五个预设模型参数量（含 Gemma-2 泛化式） | §3.2 表列值 | ±0.5% |
| A6 | tie=true/false 双分支 | lm_head 差异 = V*h | 精确 |
| A7 | Qwen2.5-7B b1 s2048 unfused logits | 2.900390625 GiB | ±1% |
| A8 | eager attentionMatrixBytes（同条件） | 13,153,337,344 B | 精确 |
| A9 | gradAccumSteps 不变性（1→8→64） | breakdown 逐字段相等 | 精确 |
| A10 | LoRA T（Qwen2.5-7B r16 全模块） | 40,370,176 | 精确 |
| A11 | fused logits（同条件） | 33,554,432 B = 0.03125 GiB | 精确 |
| A12 | lora+adamw32 gradAndOpt | 645,922,816 B | 精确 |
| A13 | 4060 qlora4bit 全优化 linux 场景 reservedPeak | ≈5.63 GiB 且 isOom=false | ±10% |
| A14 | paged 分页确定性三断言：① 不 OOM 时 fraction=0；② OOM 且未触封顶时 \|reserved−capacity\| ≤ 1e-6 GiB（刚好脱险，公式含 (1+fragRatio) 乘子）；③ 极端 OOM 时 fraction==0.6 封顶且如实报告仍 OOM | 通过 | 精确 |
| A15 | probe JSON 与 HardwareTelemetryReport 字段一一对应（含 driverModel） | 类型级 satisfies 测试 | 精确 |
| A16 | 激活估算 vs 真机探针（5050 Windows WDDM，选测） | 偏差 ≤35%，对比配置用 windows_wddm | ±35% |

### 9.3 交付物清单（执行层完成标志）

`src/core/presets/gpus.ts`、`src/core/presets/models.ts`、`src/core/hf-parser/index.ts`、`src/core/physics-engine/index.ts`、`src/core/oom-diagnostician/index.ts`、`src/core/codegen/index.ts`、`python-probe/vram_telemetry_probe.py`、`src/core/__tests__/engine.test.ts`；`npm test` 与 `npm run typecheck` 全绿。

---

## 附录 A：v1.0.0 → v1.1.0 变更日志

| # | 变更 | 依据 |
|---|---|---|
| 1 | 容量语义：vramGb 直接按 GiB（8→8.00），撤销 ×10⁹÷2³⁰ | GPU 颗粒二进制容量；torch 日志实证；总控认领 v1.0 错误 |
| 2 | GpuSpec 增 memoryBusWidthBits / int8Tflops；预设扩至 8 卡，ID 连字符化，5050 为默认基准 | 新规范 §1 |
| 3 | RTX 5050 带宽 256→**320 GB/s** | 官方 GDDR7 128-bit@20Gbps；修复原规范笔误 |
| 4 | ModelArchitectureSpec 增 headDim；P 公式泛化（qDim=n·d）；预设扩至 6 模型含 Gemma-2 | 新规范 §1；Gemma-2 head_dim=256≠h/n |
| 5 | hf-parser 增 parseLocalConfig 本地输入模式；撤销 head_dim 抛错规则；非 SwiGLU 仍抛错 | 新规范 §2 |
| 6 | INT8 权重 1.0625P→1.05P；LoRA 副本归属 weights→gradAndOpt（4T） | 新规范 §3 |
| 7 | 8bit AdamW 保留 FP32 master：optBytes=6.00390625（用户裁决） | bitsandbytes 物理实现 |
| 8 | paged 分页：确定性"刚好脱险、封顶 60%"规则 | 新规范 §3 + 总控确定性补全 |
| 9 | 激活公式改 b·s·(10h+2h/G+6i)；eager 4bs²a；checkpoint 1.25 系数；flash S²=0 | 新规范 §3 |
| 10 | 融合 CE 改切词表维 b·s·min(V,4096)·4（用户裁决） | 修复原规范 min(b·s,4096)·V·4 与 60MB/Case2 的矛盾 |
| 11 | 系统项 0.65/0.75 GiB；碎片率 dynamicPadding 14% / 稳定 3.5% | 新规范 §3 |
| 12 | 手术五步重排：S1 fused → S2 FA2+expandable → S3 ckpt → S4 QLoRA+paged → S5 batch 守恒；停止条件 0.85 绿区 | 新规范 §4 + Case2 可达性推导 |
| 13 | OOM 正则兼容 "0 bytes is free"；碎片判据改 gap≥max(1GiB,10%cap)；函数更名 parseOomTraceback | 修复原规范判据与样例矛盾 |
| 14 | codegen 增 launch.ps1、enable_liger_kernel、max_split_size_mb:128、架构兼容检查 | 新规范 §5 |
| 15 | 探针增 WDDM/TCC 采集、Transformer 层+CE 模拟负载、telemetry_report.json 落盘 | 新规范 §6 |
| 16 | 验收钉死 src/core/__tests__/engine.test.ts 与 Case1–4 精确场景（Case2 钉死 linux_headless） | 新规范验收标准 + 总控数值推演 |

*本契约书 v1.1.0 由总控架构师冻结。执行层开工前请通读第 1、5、6、9 节；审核层以 9.2 表为唯一判据。*

---

## 附录 B：v1.1.0 → v1.1.1 环境适配补丁（types.ts 无字段变更）

| # | 变更 | 依据 |
|---|---|---|
| 17 | 预设增 `rtx-3060-12g`（Ampere, 192-bit, 360 GB/s）；钉死真实部署环境：开发机 3060、校准/训练机 5050 纯 Windows WDDM | 用户环境情报 |
| 18 | `flashAttention2=true` 语义弹性化：Windows 上 SDPA 等效 fallback，显存模型不变 | flash-attn Windows 轮子现实约束 |
| 19 | Checklist 增 Windows 专属避坑项（triton-windows / bitsandbytes≥0.43 / WDDM 口径 / 防休眠） | 5050 Windows 训练机实际部署 |
| 20 | 探针钉死 Windows 原生可跑（pathlib、禁 POSIX-only 依赖）；确立校准反哺机制（实测偏离先验 >20% 走契约变更修常量） | 无 Linux 环境约束 |
| 21 | Case 2 澄清：vitest 纯模拟与宿主 OS 无关；真机 WDDM 变体只验 isHealed，黄金窗口以 linux 模拟为准 | 避免审核层在 Windows 上误判 Case 2 |

---

## 附录 C：v1.1.1 → v1.1.2 裁决补丁（types.ts 无字段变更）

| # | 变更 | 依据 |
|---|---|---|
| 22 | §4.1 非 gated MLP 检测改为激活字段命名谱系判定：`hidden_act`/`hidden_activation`（含 GeGLU gelu_pytorch_tanh）白名单放行；仅 `activation_function`（GPT-2/NeoX 谱系）抛错。撤销"含 gelu 即抛错"措辞 | 执行层审计发现真实 Gemma-2 config 被误杀；GeGLU 与 SwiGLU 矩阵拓扑相同，公式无需变更。总控认领措辞错误 |

**配套施工指令（执行层微补丁）**：修正 `src/core/hf-parser/index.ts` 的 `assertSwiGluArchitecture`（约 L100–110）按附录 C #22 实现；新增回归测试：真实 Gemma-2 原始 config（`hidden_activation: 'gelu_pytorch_tanh'`, `head_dim: 256`）经 `normalizeHfConfig`/`parseLocalConfig` 必须成功且 headDim=256、P≈9,241,404,928（±0.5%）；既有 GPT-2（`activation_function: 'gelu_new'`）抛错测试保持通过。

---

## 附录 D：v1.1.2 → v1.1.3 审核裁决补丁（types.ts 无字段变更）

| # | 变更 | 依据 |
|---|---|---|
| 23 | §5.3 paged 分页比例公式修正为 `excess/(optStateGib×(1+fragRatio))`，未封顶时 reserved 精确等于 capacity（≤1e-6）；A14 升级为三断言精确验收 | 审核 D1/D2：v1.1.2 字面公式漏碎片乘子致过分页 1.40 GiB，测试仅断言 ≤capacity 属弱断言。总控认领公式错误 |
| 24 | §6.4 原子手术单位裁决：策略表一行=一条 SurgicalStep，S2/S4 耦合字段合并同一 configPatch 同时应用；每条步骤后查绿区愈即停；S5 每次减半独立成步 | 审核 D3/D4：字段级原子停与 Case 2 黄金窗口 [5.4,6.2] 冲突（S2a 后 6.388 出窗）。裁决保用户钉死验收数，阶段原子语义自洽 |

**配套返工指令（执行层）**：
1. physics-engine：paged fraction 按 #23 反解公式实现，formulaNote 写反解算式；
2. oom-diagnostician：S2/S4 各自合并为单条 SurgicalStep（多键 configPatch），绿区检查置于每条步骤之后（含 S5 每次减半后）；Case 2 预期不变（S1→7.011>6.8 继续，S2 合并步→5.859∈窗口停）；
3. engine.test.ts：A14 升级三断言（含 |reserved−capacity|≤1e-6 与 fraction==0.6 封顶场景）；新增步骤结构回归（Case 2 恰 2 条步骤、S2 patch 同时含 flashAttention2 与 expandableSegments；构造 D3 场景断言 S4 为单条步骤且其后无多余步骤）；
4. 冻结件禁改，完成标准 tsc 零报错 + vitest 全绿。