// Replay recorded agent traces against every cell, and log one row per model call to results/<run start>.jsonl.
//
// A trace is a recorded agent conversation: the question, then assistant turns (tool calls) alternating
// with tool results. Each model has its own traces (traces/<model>/, made by record.ts), and every cell
// replays its model's traces. For each assistant turn we send the conversation up to that point, time
// the response, then throw the response away and move on. So every provider serving a model gets the same
// conversation, with the same growing context, that the agent sent. No tools run during the benchmark.
//
// Every trace is replayed PASSES times, in full passes one after another, so each cell's numbers average
// over a few hours instead of one stretch, and the spread between passes shows how much they move.
//
// Usage: npm run bench -- [filter ...]      e.g.  npm run bench -- sail   or   npm run bench -- baseten/kimi wafer
import { APICallError, generateText, streamText, type ModelMessage } from 'ai';
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync } from 'node:fs';
import { cells, cost, isOpenModel, modelOf, PASSES, reasoningFor, type Cell } from './providers.ts';
import { INSTRUCTIONS, tools } from './tools.ts';

type Trace = { id: string; messages: ModelMessage[] };
const TIMEOUT_MS = 180_000; // a call that takes longer counts as failed (the slowest success seen is 144 s)
const RETRIES = 2; // a failed task is run again from the start, up to this many times, at the end of its pass
const RETRY_DELAY_MS = 60_000;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const tracesOf = (cell: Cell): Trace[] => {
  const dir = `traces/${modelOf(cell.id)}`;
  return existsSync(dir) ? readdirSync(dir).sort().map((f) => JSON.parse(readFileSync(`${dir}/${f}`, 'utf8'))) : [];
};

const filters = process.argv.slice(2);
const matches = (id: string) => !filters.length || filters.some((f) => id.includes(f));
const active = cells.filter((c) => matches(c.id) && process.env[c.keyEnv] && tracesOf(c).length);
const runStarted = new Date().toISOString();
// One file per run, named by when it started, e.g. results/2026-11-08T14-53Z.jsonl. Every row also records when
// its request was sent.
const out = `results/${runStarted.slice(0, 16).replace(':', '-')}Z.jsonl`;
mkdirSync('results', { recursive: true });
console.log(`${active.length} cells → ${out}`);

// Providers run in parallel with each other. Within a provider, requests go one at a time, like a single
// agent would send them.
const byProvider = Object.values(Object.groupBy(active, (c) => c.id.split(/[-/]/)[0])) as Cell[][];

// First, one unmeasured warm-up request per cell, so connection setup and a provider's cold start don't land
// in whichever task happens to run first. It also checks that every account works (key, credits, quota): if
// a warm-up still fails after a few tries, the run stops before anything is timed. Its own tag means it
// can't seed the cache for real calls.
const warmUpErrors = await Promise.all(
  byProvider.map(async (providerCells) => {
    const errors: string[] = [];
    for (const cell of providerCells) {
      for (let tries = 1; ; tries++) {
        const { error } = await timeCall(cell, [{ role: 'user', content: 'Hello' }], `${runStarted} warm-up ${cell.id}`, 'warm-up');
        if (!error) break;
        if (tries === 3) { errors.push(`${cell.id}: ${error}`); break; }
        await sleep(RETRY_DELAY_MS);
      }
    }
    return errors;
  }),
);
if (warmUpErrors.flat().length) {
  console.error(`Warm-up failed, so the run was stopped:\n${warmUpErrors.flat().join('\n')}`);
  process.exit(1);
}

// The provider's cells take turns trace by trace, starting with a different cell each time, so that cells of
// the same provider (e.g. Sail's three tiers) are measured over the same stretch of time and none always
// goes first. Tasks that fail are retried at the end of the pass, after a short wait.
await Promise.all(
  byProvider.map(async (providerCells) => {
    const traces = new Map(providerCells.map((c) => [c, tracesOf(c)]));
    const rounds = Math.max(...providerCells.map((c) => traces.get(c)!.length));
    for (let pass = 0; pass < PASSES; pass++) {
      let failed: [Cell, Trace][] = [];
      for (let n = 0; n < rounds; n++) {
        const first = (n + pass) % providerCells.length;
        for (const cell of [...providerCells.slice(first), ...providerCells.slice(0, first)]) {
          const trace = traces.get(cell)![n];
          if (trace && !(await replay(cell, trace, pass, 0))) failed.push([cell, trace]);
        }
      }
      for (let attempt = 1; attempt <= RETRIES && failed.length; attempt++) {
        await sleep(RETRY_DELAY_MS);
        const again: [Cell, Trace][] = [];
        for (const [cell, trace] of failed) if (!(await replay(cell, trace, pass, attempt))) again.push([cell, trace]);
        failed = again;
      }
    }
  }),
);

/**
 * Replay one trace against one cell: one call per assistant turn, then the cold-cache probe. A failed call
 * ends the attempt there, since the agent couldn't have continued past it. Returns whether it completed.
 */
async function replay(cell: Cell, trace: Trace, pass: number, attempt: number) {
  const turns = trace.messages.flatMap((m, i) => (m.role === 'assistant' ? [i] : []));
  const log = async (step: number, tag: string, probe?: 'cold') => {
    const row = { time: new Date().toISOString(), cell: cell.id, pass, attempt, trace: trace.id, step, probe, ...(await timeCall(cell, trace.messages.slice(0, turns[step]), tag, `${tag} ${trace.id}`)) };
    appendFileSync(out, JSON.stringify(row) + '\n');
    console.log(`${cell.id} pass ${pass + 1}${attempt ? ` retry ${attempt}` : ''} ${trace.id}#${step}${probe ? ' (cold probe)' : ''}`, row.error ?? `${row.totalMs}ms $${row.cost.toFixed(5)}`);
    return !row.error;
  };

  // The agent run. The tag makes each cell's prompt unique per run, pass and attempt, so cells never hit each
  // other's prompt cache (Sail's tiers share one) and nothing reuses an earlier pass's or attempt's cache.
  // Caching across the turns of the trace still works as normal.
  const tag = `${runStarted} pass ${pass} attempt ${attempt} ${cell.id}`;
  for (const step of turns.keys()) if (!(await log(step, tag))) return false;

  // Cold-cache probe: resend the final turn with a new tag, so none of it can be cached. Compared with the
  // final call above (identical messages, prefix already cached), it shows how much caching speeds up a
  // turn deep in an agent run. Probe rows are marked `probe: "cold"` and left out of the task numbers.
  await log(turns.length - 1, `${tag} cold ${trace.id}`, 'cold');
  return true;
}

/**
 * Send one request and measure it. A failed call is logged as a data point and not retried, with one
 * exception: when a provider says our account is over its rate limit (HTTP 429), we wait for the quota to
 * refill and send it again. That's a limit on our account, not the provider's serving speed, so the wait
 * isn't timed; only the attempt that goes through is. The row records the wait as `rateLimitWaitMs`.
 */
async function timeCall(cell: Cell, messages: ModelMessage[], tag: string, session: string) {
  let rateLimitWaitMs = 0;
  for (let backoff = 5000; ; backoff = Math.min(backoff * 2, 60_000)) {
    const { status, retryAfter, ...result } = await attempt(cell, messages, tag, session);
    if (status !== 429 || rateLimitWaitMs > 15 * 60_000) return { ...result, rateLimitWaitMs: rateLimitWaitMs || undefined };
    const wait = Number(retryAfter) * 1000 || backoff;
    await new Promise((r) => setTimeout(r, wait));
    rateLimitWaitMs += wait;
  }
}

async function attempt(cell: Cell, messages: ModelMessage[], tag: string, session: string) {
  // Anthropic only caches prompts that carry an explicit breakpoint. Marking the last message gives it the
  // same automatic prefix caching the other providers do.
  if (cell.id.startsWith('anthropic/')) {
    const last = messages.at(-1)!;
    messages = [...messages.slice(0, -1), { ...last, providerOptions: { anthropic: { cacheControl: { type: 'ephemeral' } } } }];
  }
  const request = {
    model: cell.model,
    instructions: `[${tag}]\n${INSTRUCTIONS}`, // the tag at the very start decides what can be cached
    messages,
    tools, // definitions only (no execute), so the call ends after the model's first response
    reasoning: reasoningFor(cell.id),
    // Open models run at temperature 0 (always the most likely token), so every provider generates the same
    // way. top_p is left alone: some models lock it (Kimi K3.1 Pro at 0.95), and at temperature 0 it has no
    // effect. Closed models keep their defaults: OpenAI and Anthropic don't accept temperature with reasoning on.
    ...(isOpenModel(cell.id) && { temperature: 0 }),
    maxOutputTokens: 2048,
    maxRetries: 0,
    abortSignal: AbortSignal.timeout(TIMEOUT_MS),
    // Providers that route by session (Baseten, OpenCode) get one session per task, as a real agent would send.
    headers: cell.sessionHeader ? { [cell.sessionHeader]: session } : undefined,
  } as const;

  const start = performance.now();
  const elapsed = () => Math.round(performance.now() - start);
  let ttftMs: number | undefined;
  try {
    // streamText prints every error by default; ours are caught below and logged as one line per call.
    const result = cell.stream === false ? await generateText(request) : streamText({ ...request, onError: () => {} });
    if ('stream' in result) {
      for await (const part of result.stream) {
        if (part.type === 'error') throw part.error;
        // First token of any kind: text, reasoning, or a tool call.
        if (ttftMs === undefined && /delta|tool-input-start|tool-call/.test(part.type)) ttftMs = elapsed();
      }
    }
    const totalMs = elapsed();
    const usage = await result.usage;
    const tokens = {
      input: usage.inputTokens ?? 0, // includes cached tokens
      cacheRead: usage.inputTokenDetails?.cacheReadTokens ?? 0,
      cacheWrite: usage.inputTokenDetails?.cacheWriteTokens ?? 0,
      output: usage.outputTokens ?? 0, // includes reasoning tokens
      reasoning: usage.outputTokenDetails?.reasoningTokens ?? 0,
    };
    return { ttftMs, totalMs, ...tokens, cost: cost(cell.price, tokens), finish: await result.finishReason, error: undefined, status: undefined, retryAfter: undefined };
  } catch (e) {
    const http = APICallError.isInstance(e) ? e : undefined;
    return { totalMs: elapsed(), cost: 0, error: String(e).slice(0, 300), status: http?.statusCode, retryAfter: http?.responseHeaders?.['retry-after'] };
  }
}
