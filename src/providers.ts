// The benchmark matrix. Each "cell" is one model served by one provider (or one Sail tier), with its
// list price. A cell runs only when its API key env var is set.
//
// To add a provider: create a client below, then add one line per model to `cells`.
import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import type { LanguageModel } from 'ai';

/** List price in USD per 1M tokens. `cacheWrite` only for providers that charge extra to fill the cache. */
export type Price = { input: number; cached: number; output: number; cacheWrite?: number };

export type Cell = {
  id: string; // "provider/model". Shown in results; `npm run bench -- <filter>` matches on it.
  keyEnv: string; // env var holding the API key
  model: LanguageModel;
  price: Price;
  stream?: false; // set for cells that can't stream: they report total time but no time to first token
  sessionHeader?: string; // header the provider uses to route a conversation's requests to the same cache
};

/** Each run replays every trace this many times, in full passes one after another (bench.ts). */
export const PASSES = 3;

/** Open models are served by several providers; closed ones only by their maker. */
export const isOpenModel = (cellId: string) => !/^(openai|anthropic)\//.test(cellId);

/** The model part of a cell id ("sail-flex/glm-5.3" → "glm-5.3"). Each model has its own traces/<model>/. */
export const modelOf = (cellId: string) => cellId.split('/')[1];

/**
 * Every cell uses the lowest reasoning setting its model accepts, so we measure serving speed rather
 * than how long a model chooses to think. GLM-5.3 can't disable reasoning on every provider, so all GLM
 * cells use "low". For Claude and GPT the AI SDK maps "none" to their lowest level.
 */
export const reasoningFor = (cellId: string) => (cellId.includes('glm-5.3') ? 'low' : 'none');

/** USD cost of one call, from its token counts. Cache reads and writes are subsets of `input`. */
export function cost(price: Price, t: { input: number; cacheRead: number; cacheWrite: number; output: number }) {
  const uncached = t.input - t.cacheRead - t.cacheWrite;
  const cacheWritePrice = price.cacheWrite ?? price.input; // most providers bill cache writes as normal input
  return (uncached * price.input + t.cacheRead * price.cached + t.cacheWrite * cacheWritePrice + t.output * price.output) / 1e6;
}

const usd = ([input, cached, output, cacheWrite]: number[]): Price => ({ input, cached, output, cacheWrite });

// --- Provider clients -----------------------------------------------------------------------------

const openAICompatible = (name: string, baseURL: string, keyEnv: string, headers?: Record<string, string>) =>
  createOpenAICompatible({ name, baseURL, apiKey: process.env[keyEnv], headers, includeUsage: true });

// Sail sells each model in three "completion windows" (https://docs.sailresearch.com/completion-windows),
// chosen per request with a header: asap (priority, priciest), balanced, and flex (cheapest).
// asap and balanced are ordinary streaming chat completions. flex only accepts background requests to the
// Responses API, so `sailFlexFetch` submits the request and polls until the result is ready.
const SAIL_URL = 'https://api.sailresearch.com/v1';
const sail = (window: 'asap' | 'balanced') =>
  openAICompatible(`sail-${window}`, SAIL_URL, 'SAIL_API_KEY', { 'X-Sail-Completion-Window': window });
const sailFlex = (reasoningEffort: string) =>
  createOpenAI({
    baseURL: SAIL_URL,
    apiKey: process.env.SAIL_API_KEY,
    headers: { 'X-Sail-Completion-Window': 'flex' },
    fetch: (url, init) => sailFlexFetch(url, init, reasoningEffort),
  });

async function sailFlexFetch(url: string | URL | Request, init: RequestInit | undefined, reasoningEffort: string) {
  // The AI SDK's Responses client drops the reasoning setting for non-OpenAI models, so we add it here.
  const body = { ...JSON.parse(String(init?.body)), background: true, reasoning: { effort: reasoningEffort } };
  let res = await fetch(url, { ...init, body: JSON.stringify(body) });
  let json: any = await res.json();
  while (res.ok && ['queued', 'in_progress'].includes(json.status)) {
    await new Promise((r) => setTimeout(r, 250));
    res = await fetch(`${url}/${json.id}`, { headers: init?.headers, signal: init?.signal });
    json = await res.json();
  }
  return new Response(JSON.stringify(json), { status: res.status, headers: { 'content-type': 'application/json' } });
}

// Modal shared endpoints are created per model (`modal endpoint create --shared`) and each gets its own URL.
// MODAL_ENDPOINT_PREFIX is "<workspace>-<environment>", e.g. "powerset-co-inference-bench".
const modal = (endpoint: string) =>
  openAICompatible('modal', `https://${process.env.MODAL_ENDPOINT_PREFIX}--ep-${endpoint}-server.us-west.modal.direct/v1`, 'MODAL_API_KEY');

const wafer = openAICompatible('wafer', 'https://pass.wafer.ai/v1', 'WAFER_API_KEY');

// Baseten Model APIs cache automatically; an x-session-affinity header per conversation improves cache reuse
// (https://docs.baseten.co/inference/model-apis/pricing-and-limits). "-Fast" model IDs are its faster tier.
const baseten = openAICompatible('baseten', 'https://inference.baseten.co/v1', 'BASETEN_API_KEY');
const BASETEN = { keyEnv: 'BASETEN_API_KEY', sessionHeader: 'x-session-affinity' };

// Fireworks serverless (https://docs.fireworks.ai/serverless/priority-and-fast): Fast/Ultra-fast are separate
// model IDs ("routers") that aim for higher output speed at a higher price. Left out: Priority (it only
// reduces load shedding) and Spot (best-effort capacity, which would need a much longer time window to measure).
const fireworks = openAICompatible('fireworks', 'https://api.fireworks.ai/inference/v1', 'FIREWORKS_API_KEY');
const fw = (id: string) => fireworks(`accounts/fireworks/${id}`);

// Vultr Serverless Inference (https://api.vultrinference.com/docs): OpenAI-compatible, keyed per inference
// subscription. Its prefix cache is automatic and reported in usage, but cached input has no discount.
const vultr = openAICompatible('vultr', 'https://api.vultrinference.com/v1', 'VULTR_INFERENCE_API_KEY');

const together = openAICompatible('together', 'https://api.together.xyz/v1', 'TOGETHER_API_KEY');
const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });
const anthropic = createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// OpenCode Zen (https://opencode.ai/docs/zen/): a pay-as-you-go gateway that routes each model to upstream
// providers. An x-opencode-session header per conversation keeps a conversation on one route so its prompt
// cache is reused (https://opencode.ai/docs/console/harnesses).
const opencode = openAICompatible('opencode', 'https://opencode.ai/inference/openai/v1', 'OPENCODE_API_KEY');
const OPENCODE = { keyEnv: 'OPENCODE_API_KEY', sessionHeader: 'x-opencode-session' };

// --- Cells ----------------------------------------------------------------------------------------
// Prices are [input, cached input, output, cache write?] in USD per 1M tokens: list prices as of 2026-11-08 (Vultr, OpenCode: 2026-11-10).
// Providers change them, so check each pricing page (linked below) and update this date when re-checking.

// Sail: one cell per model × window. https://docs.sailresearch.com/pricing
const SAIL_MODELS = [
  { id: 'zai-org/GLM-5.3', name: 'glm-5.3', asap: [0.98, 0.18, 3.08], balanced: [0.5, 0.12, 2.5], flex: [0.4, 0.08, 1.8] },
  { id: 'moonshotai/Kimi-K3.1-Pro', name: 'kimi-k3.1-pro', asap: [1.5, 0.2, 10], balanced: [1.2, 0.15, 8], flex: [0.7, 0.1, 6] },
  { id: 'deepseek-ai/DeepSeek-V4.1-Flash-1017', name: 'deepseek-v4.1-flash', asap: [0.09, 0.006, 0.3], balanced: [0.07, 0.005, 0.22], flex: [0.05, 0.004, 0.15] },
];
const sailCells: Cell[] = SAIL_MODELS.flatMap((m) => [
  { id: `sail-asap/${m.name}`, keyEnv: 'SAIL_API_KEY', model: sail('asap')(m.id), price: usd(m.asap) },
  { id: `sail-balanced/${m.name}`, keyEnv: 'SAIL_API_KEY', model: sail('balanced')(m.id), price: usd(m.balanced) },
  { id: `sail-flex/${m.name}`, keyEnv: 'SAIL_API_KEY', model: sailFlex(reasoningFor(m.name)).responses(m.id), price: usd(m.flex), stream: false },
]);

export const cells: Cell[] = [
  ...sailCells,

  // Modal: https://modal.com/docs/guide/supported-models#shared-endpoint-models
  { id: 'modal/glm-5.3', keyEnv: 'MODAL_API_KEY', model: modal('glm-5-3')('zai-org/GLM-5.3'), price: usd([1.4, 0.26, 4.4]) },
  { id: 'modal/kimi-k3.1-pro', keyEnv: 'MODAL_API_KEY', model: modal('kimi-k3-1-pro')('moonshotai/Kimi-K3.1-Pro'), price: usd([2, 0.2, 10]) },
  { id: 'modal/deepseek-v4.1-flash', keyEnv: 'MODAL_API_KEY', model: modal('deepseek-v4-1-flash')('deepseek-ai/DeepSeek-V4.1-Flash-1017'), price: usd([0.25, 0.005, 1]) },

  // Wafer: https://pass.wafer.ai/v1/models
  { id: 'wafer/glm-5.3', keyEnv: 'WAFER_API_KEY', model: wafer('GLM-5.3'), price: usd([0.2, 0.18, 1.8]) },
  { id: 'wafer/kimi-k3.1-pro', keyEnv: 'WAFER_API_KEY', model: wafer('Kimi-K3.1-Pro'), price: usd([2, 0.2, 10]) },
  { id: 'wafer/deepseek-v4.1-flash', keyEnv: 'WAFER_API_KEY', model: wafer('DeepSeek-V4.1-Flash-1017'), price: usd([0.13, 0.008, 0.5]) },

  // Baseten: https://inference.baseten.co/v1/models
  { id: 'baseten/glm-5.3', ...BASETEN, model: baseten('zai-org/GLM-5.3'), price: usd([1.4, 0.14, 4.4]) },
  { id: 'baseten-fast/glm-5.3', ...BASETEN, model: baseten('zai-org/GLM-5.3-Fast'), price: usd([2.1, 0.21, 6.6]) },
  { id: 'baseten/kimi-k3.1-pro', ...BASETEN, model: baseten('moonshotai/Kimi-K3.1-Pro'), price: usd([2, 0.2, 10]) },
  { id: 'baseten/deepseek-v4.1-flash', ...BASETEN, model: baseten('deepseek-ai/DeepSeek-V4.1-Flash-1017'), price: usd([0.3, 0.007, 1.2]) },
  { id: 'baseten-fast/deepseek-v4.1-flash', ...BASETEN, model: baseten('deepseek-ai/DeepSeek-V4.1-Flash-1017-Fast'), price: usd([0.6, 0.014, 2.4]) },

  // Fireworks: https://docs.fireworks.ai/serverless/pricing
  { id: 'fireworks/glm-5.3', keyEnv: 'FIREWORKS_API_KEY', model: fw('models/glm-5p3'), price: usd([1.4, 0.26, 4.4]) },
  { id: 'fireworks-fast/glm-5.3', keyEnv: 'FIREWORKS_API_KEY', model: fw('routers/glm-5p3-fast'), price: usd([2.1, 0.39, 6.6]) },
  { id: 'fireworks/kimi-k3.1-pro', keyEnv: 'FIREWORKS_API_KEY', model: fw('models/kimi-k3p1-pro'), price: usd([2, 0.2, 10]) },
  { id: 'fireworks/deepseek-v4.1-flash', keyEnv: 'FIREWORKS_API_KEY', model: fw('models/deepseek-v4p1-flash-1017'), price: usd([0.3, 0.006, 1.2]) },
  { id: 'fireworks-ultrafast/deepseek-v4.1-flash', keyEnv: 'FIREWORKS_API_KEY', model: fw('routers/deepseek-v4p1-flash-1017-ultrafast'), price: usd([2.4, 0.048, 9.6]) },

  // Together AI: https://api.together.xyz/v1/models
  { id: 'together/glm-5.3', keyEnv: 'TOGETHER_API_KEY', model: together('zai-org/GLM-5.3'), price: usd([1.4, 0.26, 4.4]) },
  { id: 'together/kimi-k3.1-pro', keyEnv: 'TOGETHER_API_KEY', model: together('moonshotai/Kimi-K3.1-Pro'), price: usd([2, 0.2, 10]) },
  { id: 'together/deepseek-v4.1-flash', keyEnv: 'TOGETHER_API_KEY', model: together('deepseek-ai/DeepSeek-V4.1-Flash-1017'), price: usd([0.3, 0.006, 1.2]) },

  // Vultr: https://api.vultrinference.com/v1/models (no Kimi K3.1 Pro; both models are served at NVFP4)
  { id: 'vultr/glm-5.3', keyEnv: 'VULTR_INFERENCE_API_KEY', model: vultr('glm-5.3'), price: usd([0.75, 0.75, 3]) },
  { id: 'vultr/deepseek-v4.1-flash', keyEnv: 'VULTR_INFERENCE_API_KEY', model: vultr('deepseek-v4.1-flash-1017'), price: usd([0.15, 0.15, 0.6]) },

  // OpenCode Zen: https://opencode.ai/zen/v1/models (no Kimi K3.1 Pro on pay-as-you-go, only on the Go subscription)
  { id: 'opencode/glm-5.3', ...OPENCODE, model: opencode('glm-5.3'), price: usd([1.4, 0.26, 4.4]) },
  { id: 'opencode/deepseek-v4.1-flash', ...OPENCODE, model: opencode('deepseek-v4.1-flash'), price: usd([0.3, 0.006, 1.2]) },

  // Closed models, as reference points: https://developers.openai.com/api/docs/pricing,
  // https://platform.claude.com/docs/en/about-claude/pricing (cache writes at the 5-minute / 30-minute rate)
  { id: 'openai/gpt-6.1-sol', keyEnv: 'OPENAI_API_KEY', model: openai('gpt-6.1-sol'), price: usd([2, 0.1, 10, 2.5]) },
  { id: 'openai/gpt-6.1-terra', keyEnv: 'OPENAI_API_KEY', model: openai('gpt-6.1-terra'), price: usd([1, 0.05, 5, 1.25]) },
  { id: 'openai/gpt-6-luna', keyEnv: 'OPENAI_API_KEY', model: openai('gpt-6-luna'), price: usd([0.1, 0.01, 0.5, 0.125]) },
  { id: 'anthropic/claude-sonnet-5.5', keyEnv: 'ANTHROPIC_API_KEY', model: anthropic('claude-sonnet-5-5'), price: usd([2, 0.1, 10, 2.5]) },
  { id: 'anthropic/claude-haiku-5.5', keyEnv: 'ANTHROPIC_API_KEY', model: anthropic('claude-haiku-5-5'), price: usd([0.1, 0.01, 0.5, 0.125]) },
];
