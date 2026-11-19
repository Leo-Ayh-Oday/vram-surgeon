/**
 * VRAM Surgeon — HF 解析器与参数量反推（契约 §4）
 *
 * 双输入模式：repoId 网络拉取（huggingface→hf-mirror 降级重试）或本地 config.json 归一化。
 * 原始 JSON 一律 Record<string, unknown> + 逐字段类型收窄，禁止 any / 非空断言兜底。
 * 非 gated MLP 检测按激活字段命名谱系判定（GPT-2/NeoX 谱系）抛错，禁止静默错算（契约 v1.1.2 §4.1）。
 */
import type { ModelArchitectureSpec } from '../types';
import {
  HF_FETCH_TIMEOUT_MS,
  HF_MIRROR_ENDPOINT,
  HF_PRIMARY_ENDPOINT,
} from '../constants';

const MODULE_TAG = '[hf-parser]';
const CONFIG_PATH = 'resolve/main/config.json';
const LOCAL_MODEL_ID = 'local/custom';

// ---------------------------------------------------------------------------
// 类型收窄工具（外部 JSON 一律 unknown，禁止 any）
// ---------------------------------------------------------------------------

/** 取第一个存在且为有限数字的别名值。 */
function pickNumber(raw: Record<string, unknown>, keys: readonly string[]): number | undefined {
  for (const key of keys) {
    const value = raw[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
}

/** 取第一个存在且为布尔的别名值。 */
function pickBoolean(raw: Record<string, unknown>, keys: readonly string[]): boolean | undefined {
  for (const key of keys) {
    const value = raw[key];
    if (typeof value === 'boolean') {
      return value;
    }
  }
  return undefined;
}

/** 取第一个存在且为字符串的别名值。 */
function pickString(raw: Record<string, unknown>, keys: readonly string[]): string | undefined {
  for (const key of keys) {
    const value = raw[key];
    if (typeof value === 'string') {
      return value;
    }
  }
  return undefined;
}

/** 收窄为正整数，否则抛带模块前缀的 Error。 */
function requirePositiveInt(value: number | undefined, field: string): number {
  if (value === undefined || !Number.isInteger(value) || value <= 0) {
    throw new Error(`${MODULE_TAG} missing or invalid positive integer field: ${field}`);
  }
  return value;
}

// ---------------------------------------------------------------------------
// §4.2 参数量反推公式（泛化版，数学契约）
// ---------------------------------------------------------------------------

/**
 * 契约 §4.2：记 d=headDim，qDim=n*d，kvDim=n_kv*d。
 * 单层 SwiGLU + RMSNorm + GQA：
 *   P_layer = h*qDim(Q) + 2*h*kvDim(K,V) + qDim*h(O) + 3*h*i(gate/up/down) + 2*h(2×RMSNorm)
 * 总量（tie 双分支为验收必考点）：
 *   P = V*h*(tie?1:2) + L*P_layer + h(final norm)
 * 标准架构 d=h/n 退化为 2h²+2h·kvDim+3hi+2h；Gemma-2(d=256) 必须用泛化式。
 */
export function estimateParamCount(spec: ModelArchitectureSpec): number {
  const h = spec.hiddenSize;
  const i = spec.intermediateSize;
  const L = spec.numHiddenLayers;
  const n = spec.numAttentionHeads;
  const nKv = spec.numKeyValueHeads;
  const d = spec.headDim;
  const V = spec.vocabSize;

  const qDim = n * d; // Q/O 投影维度（Gemma-2 ≠ h）
  const kvDim = nKv * d; // K/V 投影维度

  // 契约 §4.2: P_layer = h*qDim + 2*h*kvDim + qDim*h + 3*h*i + 2*h
  const pLayer = h * qDim + 2 * h * kvDim + qDim * h + 3 * h * i + 2 * h;
  // 契约 §4.2: tie=true 只计一份 V*h；tie=false 额外计 lm_head 的 V*h。
  const embeddingCount = spec.tieWordEmbeddings ? 1 : 2;
  return V * h * embeddingCount + L * pLayer + h;
}

// ---------------------------------------------------------------------------
// §4.1 归一化
// ---------------------------------------------------------------------------

/**
 * 非 gated MLP 检测（契约 v1.1.2 §4.1，附录 C #22）：按激活字段**命名谱系**判定，禁止按激活函数名含 gelu 判定。
 * - 白名单：含 hidden_act（LLaMA/Qwen/Mistral 谱系）或 hidden_activation（Gemma 谱系，
 *   含 GeGLU 的 gelu_pytorch_tanh）→ gated 三矩阵 MLP，放行（矩阵拓扑与 SwiGLU 相同，公式不变）；
 * - 抛错：含 activation_function 且无 hidden_act/hidden_activation（GPT-2/GPT-NeoX 谱系，
 *   c_fc/c_proj 双矩阵 + 学习式位置编码）；
 * - 三类字段皆无：默认放行（预设库覆盖已验证架构）。
 */
function assertSwiGluArchitecture(raw: Record<string, unknown>, modelId: string): void {
  // 白名单优先：hidden_act / hidden_activation（类型合法字符串）。
  const gatedActivation = pickString(raw, ['hidden_act', 'hidden_activation']);
  if (gatedActivation !== undefined) {
    return;
  }
  // 仅 activation_function（GPT-2/NeoX 谱系）→ 抛错，禁止静默错算。
  const legacyActivation = pickString(raw, ['activation_function']);
  if (legacyActivation !== undefined) {
    throw new Error(
      `${MODULE_TAG} unsupported MLP architecture: ${modelId} uses '${legacyActivation}' ` +
        'via activation_function (GPT-2/NeoX lineage, c_fc/c_proj + GELU); ' +
        'param-count formula covers gated MLPs only (contract v1.1.2 §4.1)',
    );
  }
}

/**
 * 契约 §4.1：异构字段名兼容映射（取第一个存在且类型合法者），归一化为 ModelArchitectureSpec。
 * numParams 由 estimateParamCount 计算，保证自洽。
 */
export function normalizeHfConfig(raw: Record<string, unknown>, modelId: string): ModelArchitectureSpec {
  assertSwiGluArchitecture(raw, modelId);

  const hiddenSize = requirePositiveInt(
    pickNumber(raw, ['hidden_size', 'n_embd', 'd_model']),
    'hidden_size',
  );
  const intermediateSize = requirePositiveInt(
    pickNumber(raw, ['intermediate_size', 'n_inner', 'ffn_dim']),
    'intermediate_size',
  );
  const numHiddenLayers = requirePositiveInt(
    pickNumber(raw, ['num_hidden_layers', 'n_layer', 'num_layers']),
    'num_hidden_layers',
  );
  const numAttentionHeads = requirePositiveInt(
    pickNumber(raw, ['num_attention_heads', 'n_head', 'num_heads']),
    'num_attention_heads',
  );
  // 缺省 = heads（MHA）。
  const numKeyValueHeads = requirePositiveInt(
    pickNumber(raw, ['num_key_value_heads', 'n_head_kv', 'num_kv_heads', 'multi_query_group_num']) ??
      numAttentionHeads,
    'num_key_value_heads',
  );
  // 缺省 = hiddenSize / numAttentionHeads；Gemma-2 显式 head_dim=256。
  const headDimRaw = pickNumber(raw, ['head_dim']);
  const headDim =
    headDimRaw !== undefined
      ? requirePositiveInt(headDimRaw, 'head_dim')
      : requirePositiveInt(hiddenSize / numAttentionHeads, 'head_dim (derived hidden_size/num_attention_heads)');
  const vocabSize = requirePositiveInt(pickNumber(raw, ['vocab_size', 'n_vocab']), 'vocab_size');
  const maxPositionEmbeddings = requirePositiveInt(
    pickNumber(raw, ['max_position_embeddings', 'n_positions', 'max_seq_len']),
    'max_position_embeddings',
  );
  // 缺省 false。
  const tieWordEmbeddings = pickBoolean(raw, ['tie_word_embeddings', 'tie_embeddings']) ?? false;

  if (numAttentionHeads % numKeyValueHeads !== 0) {
    throw new Error(
      `${MODULE_TAG} num_attention_heads (${numAttentionHeads}) must be divisible by num_key_value_heads (${numKeyValueHeads})`,
    );
  }

  const spec: ModelArchitectureSpec = {
    modelId,
    numParams: 0,
    hiddenSize,
    intermediateSize,
    numHiddenLayers,
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    vocabSize,
    tieWordEmbeddings,
    maxPositionEmbeddings,
  };
  // 契约 §4.2 自洽性：numParams === estimateParamCount(spec)。
  spec.numParams = estimateParamCount(spec);
  return spec;
}

/** 契约 §4.1：本地 config.json 字符串模式（粘贴/拖拽），modelId 缺省 'local/custom'。 */
export function parseLocalConfig(jsonText: string, modelId?: string): ModelArchitectureSpec {
  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonText);
  } catch (cause) {
    throw new Error(`${MODULE_TAG} invalid JSON text: ${String(cause)}`);
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error(`${MODULE_TAG} config.json must be a JSON object`);
  }
  return normalizeHfConfig(parsed as Record<string, unknown>, modelId ?? LOCAL_MODEL_ID);
}

// ---------------------------------------------------------------------------
// §4.1 网络拉取（huggingface → hf-mirror 降级重试）
// ---------------------------------------------------------------------------

export interface FetchModelSpecOptions {
  /** 默认 'huggingface'，失败自动降级 hf-mirror 重试一次。 */
  endpoint?: 'huggingface' | 'hf-mirror';
  /** 默认 HF_FETCH_TIMEOUT_MS（10_000ms），命名常量。 */
  timeoutMs?: number;
}

function endpointBaseUrl(endpoint: 'huggingface' | 'hf-mirror'): string {
  return endpoint === 'hf-mirror' ? HF_MIRROR_ENDPOINT : HF_PRIMARY_ENDPOINT;
}

/** 单端点拉取 + 超时控制 + 类型收窄；失败抛带前缀 Error。 */
async function fetchConfigJson(
  modelId: string,
  baseUrl: string,
  timeoutMs: number,
): Promise<Record<string, unknown>> {
  const url = `${baseUrl}/${modelId}/${CONFIG_PATH}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText} for ${url}`);
    }
    const body: unknown = await response.json();
    if (typeof body !== 'object' || body === null || Array.isArray(body)) {
      throw new Error(`config.json is not a JSON object: ${url}`);
    }
    return body as Record<string, unknown>;
  } catch (cause) {
    throw new Error(`${MODULE_TAG} fetch failed (${url}): ${String(cause)}`);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * 契约 §4.1：repoId 拉取 config.json 并归一化。
 * 主端点失败自动降级 hf-mirror 重试一次；全部失败抛带 [hf-parser] 前缀 Error，不静默兜底假数据。
 */
export async function fetchModelSpec(
  modelId: string,
  options?: FetchModelSpecOptions,
): Promise<ModelArchitectureSpec> {
  const primary = options?.endpoint ?? 'huggingface';
  const timeoutMs = options?.timeoutMs ?? HF_FETCH_TIMEOUT_MS;
  const fallback: 'huggingface' | 'hf-mirror' = primary === 'huggingface' ? 'hf-mirror' : 'huggingface';

  let primaryError: unknown;
  try {
    const raw = await fetchConfigJson(modelId, endpointBaseUrl(primary), timeoutMs);
    return normalizeHfConfig(raw, modelId);
  } catch (cause) {
    primaryError = cause;
  }

  try {
    const raw = await fetchConfigJson(modelId, endpointBaseUrl(fallback), timeoutMs);
    return normalizeHfConfig(raw, modelId);
  } catch (cause) {
    throw new Error(
      `${MODULE_TAG} failed to fetch model spec for '${modelId}' from both endpoints. ` +
        `primary(${primary}): ${String(primaryError)}; fallback(${fallback}): ${String(cause)}`,
    );
  }
}