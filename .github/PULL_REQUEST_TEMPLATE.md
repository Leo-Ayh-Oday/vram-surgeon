## 这个 PR 做了什么

<!-- 一句话说清楚。例如：修正 paged AdamW 分页比例漏乘 (1+fragRatio) 的问题 -->

## 类型

- [ ] `feat` 新功能
- [ ] `fix` 修复 bug
- [ ] `docs` 文档
- [ ] `test` 测试
- [ ] `refactor` 重构（不改行为）
- [ ] `chore` 工程杂务

## 影响范围

<!-- 勾选改动的模块 -->
- [ ] `presets`（显卡 / 模型预设）
- [ ] `hf-parser`
- [ ] `physics-engine`
- [ ] `oom-diagnostician`
- [ ] `codegen`
- [ ] `python-probe`
- [ ] 冻结件：`types.ts` / `constants.ts` / 契约书 / `engine.test.ts` ⚠️

## 自检（必须全绿才能合并）

- [ ] `npm run typecheck` 0 报错
- [ ] `npm test` 全绿（42+ passed）
- [ ] `src/core` 内无 `any`（`grep -rnE ':\s*any\b|as any' src/core`）
- [ ] 新增/修改的公式带契约编号注释
- [ ] 新增的 `formulaNote` 写的是代入数值后的真实算式

## 黄金值影响（有则填，无则写「无」）

| 验收项 | 变更前 | 变更后 | 是否在容差内 |
|---|---|---|---|
| 例：A10 LoRA T | 40,370,176 | 40,370,176 | 精确一致 |

<!--
动了冻结件？必须补：
1. 关联 issue：[契约变更] v1.1.3 → vX.Y.Z
2. 契约书版本号已升：是 / 否
3. 变更理由与证据（官方文档 / 实测日志 / 数值反推）
-->

## 备注

<!-- 踩过的坑、需要 reviewer 重点看的地方 -->
