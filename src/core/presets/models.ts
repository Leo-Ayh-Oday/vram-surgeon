/**
 * VRAM Surgeon — 模型架构预设库（契约 §3.2，v1.1.1）
 *
 * 严格按 HuggingFace 官方 config.json 录入；numParams 由 §4.2 泛化公式反推，
 * 保证与 estimateParamCount 自洽（契约 §4.2 自洽性要求）。
 * 陷阱备忘：Qwen2.5-3B 词表 151936≠152064 且 tie=true；7B tie=false；
 * Gemma-2 headDim=256≠h/n=224，必须走泛化式 qDim=n*d。
 */
import type { ModelArchitectureSpec } from '../types';
import { estimateParamCount } from '../hf-parser';

/** 维度输入（numParams 由公式计算，避免手填与公式不自洽）。 */
type ModelDims = Omit<ModelArchitectureSpec, 'numParams'>;

function buildModel(dims: ModelDims): ModelArchitectureSpec {
  const spec: ModelArchitectureSpec = { ...dims, numParams: 0 };
  // 契约 §4.2：numParams 必须等于 estimateParamCount 结果（自洽性）。
  spec.numParams = estimateParamCount(spec);
  return spec;
}

export const MODEL_PRESETS: readonly ModelArchitectureSpec[] = [
  buildModel({
    modelId: 'Qwen/Qwen2.5-7B-Instruct',
    hiddenSize: 3584,
    intermediateSize: 18944,
    numHiddenLayers: 28,
    numAttentionHeads: 28,
    numKeyValueHeads: 4,
    headDim: 128,
    vocabSize: 152064,
    tieWordEmbeddings: false,
    maxPositionEmbeddings: 32768,
  }),
  buildModel({
    modelId: 'meta-llama/Llama-3.1-8B-Instruct',
    hiddenSize: 4096,
    intermediateSize: 14336,
    numHiddenLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    headDim: 128,
    vocabSize: 128256,
    tieWordEmbeddings: false,
    maxPositionEmbeddings: 131072,
  }),
  buildModel({
    // 架构与 Qwen2.5-7B 完全相同（契约 §3.2 备忘③）。
    modelId: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    hiddenSize: 3584,
    intermediateSize: 18944,
    numHiddenLayers: 28,
    numAttentionHeads: 28,
    numKeyValueHeads: 4,
    headDim: 128,
    vocabSize: 152064,
    tieWordEmbeddings: false,
    maxPositionEmbeddings: 32768,
  }),
  buildModel({
    modelId: 'Qwen/Qwen2.5-3B-Instruct',
    hiddenSize: 2048,
    intermediateSize: 11008,
    numHiddenLayers: 36,
    numAttentionHeads: 16,
    numKeyValueHeads: 2,
    headDim: 128,
    vocabSize: 151936,
    tieWordEmbeddings: true,
    maxPositionEmbeddings: 32768,
  }),
  buildModel({
    modelId: 'Qwen/Qwen2.5-14B-Instruct',
    hiddenSize: 5120,
    intermediateSize: 13824,
    numHiddenLayers: 48,
    numAttentionHeads: 40,
    numKeyValueHeads: 8,
    headDim: 128,
    vocabSize: 152064,
    tieWordEmbeddings: false,
    maxPositionEmbeddings: 32768,
  }),
  buildModel({
    // Gemma-2：headDim=256（≠h/n=224），qDim=n*d=4096；tie=true；滑窗按全 S² 保守计。
    modelId: 'google/gemma-2-9b-it',
    hiddenSize: 3584,
    intermediateSize: 14336,
    numHiddenLayers: 42,
    numAttentionHeads: 16,
    numKeyValueHeads: 8,
    headDim: 256,
    vocabSize: 256000,
    tieWordEmbeddings: true,
    maxPositionEmbeddings: 8192,
  }),
];

export function getModelById(modelId: string): ModelArchitectureSpec | undefined {
  return MODEL_PRESETS.find((model) => model.modelId === modelId);
}