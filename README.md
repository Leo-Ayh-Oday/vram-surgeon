# VRAM Surgeon

> 在你按下 `train` 之前，先算清楚这 8 GB 到底够不够。

[![CI](https://github.com/Leo-Ayh-Oday/vram-surgeon/actions/workflows/ci.yml/badge.svg)](https://github.com/Leo-Ayh-Oday/vram-surgeon/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-42%20passed-brightgreen)](src/core/__tests__/engine.test.ts)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![contract](https://img.shields.io/badge/contract-v1.1.3%20frozen-orange)](docs/01_ARCH_CONTRACT.md)

**大模型微调与部署的显存解剖 + 自动抢救引擎。** 输入「显卡 × 模型 × 训练配置」，输出字节级显存账本、OOM 根因诊断、五阶段无损优先手术方案，以及一套开箱即用的零报错启动包。

2026 Qoder 高校创造力大赛 · StudyHack 赛道参赛项目。

---

## 这个项目解决什么问题

小显存机器（8 GB / 12 GB）微调大模型时，OOM 几乎必然发生，而报错信息只告诉你「tried to allocate 2.32 GiB」，不会告诉你：

- 是 **大词表 logits 尖峰**（Qwen2.5-7B 在 `b=1, s=2048` 下单这一项就是 **2.900390625 GiB**，8 GB 卡的头号隐形杀手）
- 是 **eager 注意力 S² 爆炸**（同配置 **12.25 GiB**）
- 是 **allocator 碎片**（有 1.18 GiB 空闲缓存却不连续）
- 还是 **权重本身就装不下**

VRAM Surgeon 把这些拆成五个可加的分项，每一项给出代入数值的公式串，然后按「无损优先」顺序自动手术，直到进入 `reservedPeak ≤ 0.85 × capacity` 的安全绿区。

典型效果：RTX 5050 8 GB 跑 Qwen2.5-7B，裸 QLoRA 配置理论峰值 **≈11.0 GiB**（红），自动手术后降到 **≈5.86 GiB**（绿），且等效全局 batch 守恒。

---

## 核心能力

| # | 模块 | 能力 |
|---|---|---|
| 1 | `presets/` | 9 张显卡 × 6 个模型架构的实测参数（含 Gemma-2 `headDim=256` 这类坑） |
| 2 | `hf-parser/` | 从 HuggingFace 拉 `config.json` 或本地粘贴，异构字段名归一化后反推参数量（误差 ≤0.5%） |
| 3 | `physics-engine/` | 字节级显存方程：weights / gradAndOpt / activations / logitsPeak / systemAndFrag |
| 4 | `oom-diagnostician/` | 解析真实 PyTorch OOM traceback，判定 5 类根因，生成五阶段手术方案 |
| 5 | `codegen/` | 一键产出 LLaMA-Factory YAML、Unsloth 脚本、`launch.sh`、`launch.ps1`、避坑清单 |
| 6 | `python-probe/` | 真机 PyTorch 遥测探针，输出 JSON 做「理论 vs 实测」双轨校准 |

所有核心模块是**纯函数、零副作用、零 `any`**，只依赖 `src/core/types.ts` 这一份数据契约。

---

## 快速开始

环境：Node ≥ 18（开发机实测 Node 22 / 24），Python ≥ 3.10（仅探针需要）。

```bash
git clone https://github.com/Leo-Ayh-Oday/vram-surgeon.git
cd vram-surgeon
npm install

npm run typecheck   # tsc --noEmit，strict 全开
npm test            # vitest，42 个用例
```

### 算一次显存账

```ts
import { getGpuById, getModelById, computeVramBreakdown, solveAutoHeal } from './src/core';

const gpu = getGpuById('rtx-5050-8g')!;
const model = getModelById('Qwen/Qwen2.5-7B-Instruct')!;

const config = {
  mode: 'qlora_4bit' as const,
  batchSize: 1,
  seqLen: 2048,
  gradAccumSteps: 8,
  loraRank: 16,
  loraAlpha: 32,
  loraTargetModules: ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
  optimizer: 'adamw_32bit' as const,
  gradCheckpointing: true,
  flashAttention2: false,
  fusedCrossEntropy: false,
  osEnv: 'windows_wddm' as const,
  expandableSegments: false,
  dynamicPadding: true,
};

const breakdown = computeVramBreakdown(gpu, model, config);
console.log(breakdown.summary.totalReservedPeakGib); // ≈11.0 → isOom: true
console.log(breakdown.logitsPeak.formulaNote);       // 10*1*2048*152064/2^30

const plan = solveAutoHeal(gpu, model, config);
console.log(plan.isHealed, plan.afterBreakdown.summary.totalReservedPeakGib); // true, ≈5.86–6.6
console.log(plan.steps.map(s => s.actionName));
```

### 真机探针（可选，需 CUDA）

```bash
python python-probe/vram_telemetry_probe.py --model-id Qwen/Qwen2.5-7B-Instruct --seq-len 2048
# stdout 只输出一个 HardwareTelemetryReport JSON；日志全走 stderr
# 无 GPU 时可显式跑模拟基准：
python python-probe/vram_telemetry_probe.py --mock-rtx5050
```

---

## 架构

```
presets/gpus.ts ─┐
presets/models.ts┤（纯数据，无依赖）
                 ▼
hf-parser ──► ModelArchitectureSpec
                 ▼
physics-engine ──► VramBreakdown        （纯函数）
                 ▼
oom-diagnostician ──► OomDiagnosisResult / SurgicalPlan
                 ▼
codegen ──► GeneratedArtifacts

python-probe/vram_telemetry_probe.py ──► HardwareTelemetryReport（独立进程）
```

依赖方向单向，核心层禁止 import 任何 UI 框架或 DOM API（`hf-parser` 的 `fetch` 除外）。

## 目录

```
src/core/
  types.ts              # 数据契约（冻结件，唯一 SSOT）
  constants.ts          # 物理常数法典（冻结值）
  presets/              # 显卡 / 模型预设
  hf-parser/            # config.json 归一化 + 参数量反推
  physics-engine/       # 字节级显存方程
  oom-diagnostician/    # OOM 解析 + 五阶段手术
  codegen/              # 启动包编译器
  __tests__/engine.test.ts  # 42 用例，含 16 项黄金值回归 A1–A16
docs/
  01_ARCH_CONTRACT.md   # 架构与数学契约书 v1.1.3（冻结）
python-probe/
  vram_telemetry_probe.py
  sample_rtx5050_telemetry.json
```

## 工程纪律（不是建议，是硬约束）

1. **契约先行**：`docs/01_ARCH_CONTRACT.md` 与 `src/core/types.ts` 是冻结件。改字段/公式必须先升版本号、改契约、再改实现，顺序不能反。
2. **零 `any`**：外部 JSON 一律 `unknown` + 类型收窄，禁止非空断言兜底。
3. **魔法数字常量化**：所有物理常数必须在 `constants.ts` 具名导出。
4. **单位纪律**：显存换算只走 `bytesToGib`；`gpu.vramGb` 是二进制 GiB，禁止 `×10⁹÷2³⁰`。
5. **不静默兜底**：解析失败抛带模块前缀的 Error，不返回假数据。
6. **黄金值回归**：A1–A16 是验收基线，改动导致任何一项红就是回归，不是「容差问题」。

冲突裁决顺序：`types.ts` 字段定义 > 契约书公式 > 实现自由裁量。

## 参与协作

欢迎一起写。开工前请读 [CONTRIBUTING.md](CONTRIBUTING.md)，核心就三条：

- 从 `main` 切 `feat/*` 或 `fix/*` 分支，**不要直推 main**（已开分支保护）
- 提交前本地跑 `npm run typecheck && npm test` 全绿
- 动冻结件必须先开 issue 讨论版本号，PR 里说明契约变更理由

## License

[MIT](LICENSE) © 2026 Leo-Ayh-Oday

---

> 免责：显存模型是**保守工程上界**，不是逐算子仿真。激活项容差 ±35%，其余分项 ±2%，真机探针负责校准。若实测系统性偏离先验 >20%，走契约变更流程修常量——不允许在执行层偷偷改数。
