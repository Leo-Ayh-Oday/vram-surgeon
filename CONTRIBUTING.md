# 协作指南

> 本仓库不接受「先写代码，后改契约」。顺序错了，PR 会被退回重做。

## 1. 开工准备

```bash
git clone https://github.com/Leo-Ayh-Oday/vram-surgeon.git
cd vram-surgeon
npm install
npm run typecheck && npm test    # 应当全绿；不绿先报 issue，别在红的基础上开工
```

建议用 VS Code + ESLint 之外的零配置即可，本项目不强制 Prettier，但**缩进 2 空格、单引号、行尾 LF**（见 `.editorconfig`）。

## 2. 分支与 PR 流程

`main` 已开分支保护，**禁止直推**。所有改动走：

```bash
git switch -c feat/your-topic      # 或 fix/xxx、docs/xxx、test/xxx
# ... 改动 ...
npm run typecheck && npm test      # 必须全绿
git commit -m "feat(physics-engine): 修正 paged 分页反解公式"
git push -u origin feat/your-topic
gh pr create --fill                # 或网页开 PR
```

- CI（typecheck + test + python 语法检查）必须绿，才能合并。
- 合并方式用 **Squash**，提交信息遵循 Conventional Commits：`feat(scope): ...` / `fix(scope): ...` / `docs: ...` / `test: ...` / `chore: ...`。
- 合并后源分支自动删除。

## 3. 冻结件的红灯区

以下文件是**冻结件**，动它们不是写代码，是改宪法：

| 文件 | 约束 |
|---|---|
| `src/core/types.ts` | 数据契约 SSOT，字段增删 = 破坏性变更 |
| `docs/01_ARCH_CONTRACT.md` | 架构与数学契约书，当前 v1.1.3 |
| `src/core/constants.ts` | 物理常数，值由契约钉死 |

想改冻结件？按这个顺序，一步都不能跳：

1. 开 issue，标题写明 `[契约变更] vX.Y.Z → vX.Y+1.0：变更点`，说明**为什么现有公式是错的**（要证据：官方文档、实测日志、数值反推）。
2. 在 PR 里先改 `docs/01_ARCH_CONTRACT.md`（升版本号 + 附录写变更日志 + 认领责任）。
3. 再改 `types.ts` / `constants.ts`。
4. 最后才改实现和测试。
5. PR 描述里附「变更前 vs 变更后」的黄金值对比表。

**不允许**：为了让测试变绿而改黄金值；不允许在执行层偷偷调常数；不允许用 `any` 或非空断言把类型错误按住。

## 4. 代码规范

- **零 `any`**：外部数据 `unknown` → 逐字段收窄。`noUnusedLocals` / `noUnusedParameters` 已开，别用下划线糊过去。
- **魔法数字常量化**：除 0/1/2 这类纯数学恒等量外，一律进 `constants.ts` 并具名。
- **公式必须带物理意义注释**：引用契约编号，例如 `// 契约 §5.5: 10*b*s*V 大词表 logits 瞬时峰值`。
- **每个分项必须填 `formulaNote`**：写代入数值后的算式字符串，UI 和审核层都靠它复核。
- **错误带模块前缀**：`throw new RangeError('[physics-engine] batchSize must be a positive integer')`。
- **纯函数优先**：核心层不碰 DOM、不读环境变量、不发网络请求（`hf-parser.fetchModelSpec` 与 python 探针除外）。

## 5. 测试要求

新增能力必须带测试，并且写清楚**断言的是物理量还是构造值**：

```ts
it('A7: Qwen2.5-7B b1 s2048 unfused logits = 2.900390625 GiB（±1%）', () => {
  // 契约 §5.5 黄金值，改动此断言等同于改契约
});
```

- 黄金值（A1–A16）的容差收紧可以，放宽不行。
- 修 bug 的 PR 必须先有一个能复现 bug 的红色测试，再让它变绿。
- `gradAccumSteps` 不变性（A9）是铁律：任何改动导致显存随 `gradAccumSteps` 变化，直接拒绝。

## 6. 提交前自检清单

```bash
npm run typecheck     # 0 报错
npm test              # 42+ passed
grep -rn ": any" src/core || echo "no any ✓"
python -m py_compile python-probe/vram_telemetry_probe.py
```

四条都过，再推。

## 7. 沟通

- 有疑问优先开 issue，不要私聊解决——别人的 PR 里可能踩过同一个坑。
- 认领任务在 issue 里留言「我来做」，避免两人同时改同一模块。
- 评审意见就事论事：指出「这个公式少了 (1+fragRatio) 乘子」比说「感觉不对」有用一百倍。
