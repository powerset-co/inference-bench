// Summarize benchmark results. Reads the per-call rows in results/*.jsonl (one file per run) and writes:
//   results/summary.csv     one row per cell, every metric (for anyone who wants the numbers)
//   results/charts/*.svg    figures (charts.ts)
//   README.md               the figures, between the results markers
// and prints a table to the terminal. The numbers cover every run that started in the WINDOW_DAYS before the
// newest one, so running a provider again adds to its data rather than replacing it, and a provider with no
// run in that window is left out. It refuses to publish if the newest run is missing a completed task: every
// cell in it must complete every trace in every pass.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { banner, charts } from './charts.ts';
import { cells, modelOf, PASSES } from './providers.ts';

const WINDOW_DAYS = 30;

type Row = {
  run: string; time: string; cell: string; pass: number; attempt?: number; trace: string; step: number; probe?: 'cold';
  ttftMs?: number; totalMs: number; input: number; cacheRead: number; output: number; cost: number; error?: string;
};

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const mean = (xs: number[]) => sum(xs) / xs.length;
const quantile = (xs: number[], q: number) => [...xs].sort((a, b) => a - b)[Math.floor((xs.length - 1) * q)];
const round = (x: number | undefined, digits: number): number | '' => (x === undefined || !Number.isFinite(x) ? '' : +x.toFixed(digits));
const ttfts = (rows: Row[]) => rows.flatMap((r) => (r.ttftMs === undefined ? [] : [r.ttftMs / 1000]));

const readRun = (run: string): Row[] => readFileSync(`results/${run}.jsonl`, 'utf8').trim().split('\n').map((line) => ({ ...JSON.parse(line), run }));
const attemptOf = (r: Row) => `${r.run} ${r.pass} ${r.trace} ${r.attempt ?? 0}`;

/**
 * A "task" is one replay of a trace: once per pass of each run, plus a retry for each failed attempt. Every
 * metric counts only completed tasks, where every call succeeded; failed attempts are only counted, in `retries`.
 */
const tasksOf = (rows: Row[]) => {
  const attempts = Object.values(Object.groupBy(rows.filter((r) => !r.probe), attemptOf)) as Row[][];
  return { attempts, completed: attempts.filter((t) => t.every((r) => !r.error)) };
};

function summarize(allRows: Row[]) {
  const { attempts, completed: tasks } = tasksOf(allRows);
  const ok = tasks.flat(); // every call in a completed task
  const taskSeconds = (t: Row[]) => sum(t.map((r) => r.totalMs)) / 1000;
  // Mean time per task within each pass: the lowest and highest show how much the result moves on a rerun.
  const passes = Object.values(Object.groupBy(tasks, (t) => `${t[0].run} ${t[0].pass}`)).map((p) => mean(p!.map(taskSeconds)));
  // Caching's effect on latency: each trace's final turn sent cold (the probe, nothing cached) vs. warm
  // (in the agent run, prefix cached). Same messages, so the difference is caching alone.
  const done = new Set(tasks.map((t) => attemptOf(t[0])));
  const cold = allRows.filter((r) => r.probe === 'cold' && !r.error && done.has(attemptOf(r)));
  const warm = cold.flatMap((c) => ok.filter((r) => attemptOf(r) === attemptOf(c) && r.step === c.step));
  return {
    task_turns: round(mean(tasks.map((t) => t.length)), 1), // model calls per task
    task_s: round(mean(tasks.map(taskSeconds)), 1), // model time to finish a task
    task_s_low: round(Math.min(...passes), 1), // lowest and highest of the per-pass means
    task_s_high: round(Math.max(...passes), 1),
    task_ttft_s: round(mean(tasks.map((t) => sum(t.map((r) => (r.ttftMs ?? NaN) / 1000)))), 1), // of which, waiting for first tokens
    task_usd: round(mean(tasks.map((t) => sum(t.map((r) => r.cost)))), 5), // list-price cost of a task
    task_output_tokens: round(mean(tasks.map((t) => sum(t.map((r) => r.output)))), 0), // incl. reasoning
    ttft_p50_s: round(quantile(ttfts(ok), 0.5), 2),
    call_p50_s: round(quantile(ok.map((r) => r.totalMs / 1000), 0.5), 2),
    call_p90_s: round(quantile(ok.map((r) => r.totalMs / 1000), 0.9), 2),
    cache_hit: round(sum(ok.map((r) => r.cacheRead)) / sum(ok.map((r) => r.input)), 2), // share of input tokens read from cache
    // Time to first token, since caching only speeds up reading the prompt (blank for cells that can't stream).
    final_turn_ttft_cold_s: round(quantile(ttfts(cold), 0.5), 2),
    final_turn_ttft_warm_s: round(quantile(ttfts(warm), 0.5), 2),
    retries: attempts.length - tasks.length, // failed attempts, each run again
  };
}

// --- Check the newest run is complete -------------------------------------------------------------

const runs = readdirSync('results').filter((f) => f.endsWith('.jsonl')).sort().map((f) => f.replace('.jsonl', ''));
const newestRun = runs.at(-1)!;
const newestRows = readRun(newestRun);
const ran = cells.filter((cell) => newestRows.some((r) => r.cell === cell.id));
const gaps = ran.flatMap((cell) => {
  const done = new Set(tasksOf(newestRows.filter((r) => r.cell === cell.id)).completed.map((t) => `${t[0].pass} ${t[0].trace}`));
  const traces = readdirSync(`traces/${modelOf(cell.id)}`).map((f) => f.replace('.json', ''));
  const missing = Array.from({ length: PASSES }, (_, pass) => traces.filter((t) => !done.has(`${pass} ${t}`))).flat().length;
  return missing ? [`${cell.id}: ${missing} of ${PASSES * traces.length} tasks never completed`] : [];
});
if (gaps.length) {
  console.error(`${newestRun} is incomplete, so nothing was published:\n${gaps.join('\n')}`);
  process.exit(1);
}

// --- Summary CSV ----------------------------------------------------------------------------------

// Runs are named by when they started (2026-11-08T14-53Z).
const startOf = (run: string) => Date.parse(run.replace(/T(\d\d)-(\d\d)Z$/, 'T$1:$2Z'));
const inWindow = runs.filter((run) => startOf(run) >= startOf(newestRun) - WINDOW_DAYS * 86_400_000);
const rows = inWindow.flatMap(readRun);
const byCell = Object.groupBy(rows, (r) => r.cell);
const summary = Object.keys(byCell).sort().map((cell) => ({ cell, runs: new Set(byCell[cell]!.map((r) => r.run)).size, ...summarize(byCell[cell]!) }));
const columns = Object.keys(summary[0]) as (keyof (typeof summary)[0])[];
writeFileSync('results/summary.csv', [columns, ...summary.map((s) => columns.map((c) => s[c]))].map((r) => r.join(',')).join('\n') + '\n');

// --- Charts and README ----------------------------------------------------------------------------

mkdirSync('results/charts', { recursive: true });
// The charts also need every call's duration, for the latency distribution.
const callsOf = (cell: string) => tasksOf(byCell[cell]!).completed.flat().map((r) => r.totalMs / 1000);
const figures = charts(summary.map((s) => ({ ...s, calls_s: callsOf(s.cell) })) as any);
for (const [name, svg] of Object.entries(figures)) writeFileSync(`results/charts/${name}.svg`, svg);
const BANNER_ORDER = ['sail', 'wafer', 'modal', 'baseten', 'fireworks', 'together', 'vultr', 'opencode', 'openai', 'anthropic'];
const shown = new Set(summary.map((s) => s.cell.split(/[-/]/)[0]));
writeFileSync('assets/banner.svg', banner(BANNER_ORDER.filter((p) => shown.has(p))));

// A quick table for the terminal.
const secs = (x: number | '') => (x === '' ? '–' : `${x.toFixed(2)} s`);
const table = [
  '| Cell | Turns / task | Time / task | Cost / task | Median TTFT | p90 call | Final-turn TTFT, cold → cached | Retries |',
  '|--|--:|--:|--:|--:|--:|--:|--:|',
  ...summary.map((r) =>
    `| ${r.cell} | ${r.task_turns} | ${r.task_s} s | $${r.task_usd} | ${secs(r.ttft_p50_s)} | ${secs(r.call_p90_s)} | ` +
    `${r.final_turn_ttft_cold_s === '' ? '–' : `${r.final_turn_ttft_cold_s} → ${r.final_turn_ttft_warm_s} s`} | ${r.retries} |`),
];
// When the requests behind these results were sent.
const days = rows.map((r) => r.time.slice(0, 10)).sort();
const sent = `Last ${WINDOW_DAYS} days of runs: ${days[0] === days.at(-1) ? days[0] : `${days[0]} to ${days.at(-1)}`}`;
const section = [
  ...Object.keys(figures).map((name) => `<img src="results/charts/${name}.svg" alt="${name}" width="100%">\n`),
  `<sub>${sent}. Data: [summary](results/summary.csv), [raw results](results/).</sub>`,
].join('\n');
console.log(table.join('\n'));
const readme = readFileSync('README.md', 'utf8');
writeFileSync('README.md', readme.replace(/(<!-- results:start -->)[\s\S]*(<!-- results:end -->)/, `$1\n${section}\n$2`));
