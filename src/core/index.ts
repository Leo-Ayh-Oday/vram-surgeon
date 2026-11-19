/**
 * VRAM Surgeon — 核心层统一出口（契约 §2 依赖图）。
 *
 * re-export 六大模块公开 API + 类型契约 + 物理常数法典。
 * 上层（UI/工作台）只允许从本入口 import 核心层能力。
 */

// 类型契约（冻结件，types.ts 为唯一数据契约源）。
export type {
  GpuArchitecture,
  GpuSpec,
  ModelArchitectureSpec,
  FinetuneMode,
  OptimizerKind,
  OsEnv,
  TrainingConfig,
  ConfigPatch,
  VramCategoryKey,
  WeightsComponent,
  GradAndOptComponent,
  ActivationsComponent,
  LogitsPeakComponent,
  SystemAndFragComponent,
  BottleneckType,
  VramSummary,
  VramBreakdown,
  OomRootCause,
  OomDiagnosisResult,
  SurgicalStep,
  VramDiff,
  VramPhase,
  VramFrameState,
  SurgicalPlan,
  ChecklistSeverity,
  ChecklistItem,
  GeneratedArtifacts,
  TelemetryGpuInfo,
  TelemetryTorchInfo,
  TelemetryOsInfo,
  TelemetryCalibration,
  TelemetrySample,
  HardwareTelemetryReport,
} from './types';
export { BYTES_PER_GIB, BYTES_PER_GB_DECIMAL } from './types';

// 物理常数法典（契约 §1）。
export * from './constants';

// 模块一：预设库（契约 §3）。
export { GPU_PRESETS, DEFAULT_GPU_ID, getGpuById } from './presets/gpus';
export { MODEL_PRESETS, getModelById } from './presets/models';

// 模块二：HF 解析器与参数量反推（契约 §4）。
export {
  normalizeHfConfig,
  parseLocalConfig,
  estimateParamCount,
  fetchModelSpec,
} from './hf-parser';
export type { FetchModelSpecOptions } from './hf-parser';

// 模块三：物理引擎（契约 §5）。
export { bytesToGib, computeTrainableParams, computeVramBreakdown } from './physics-engine';

// 模块四：OOM 诊断与自动手术（契约 §6）。
export { parseOomTraceback, diagnoseOom, solveAutoHeal } from './oom-diagnostician';

// 模块五：零报错启动包编译器（契约 §7）。
export { generateArtifacts } from './codegen';
