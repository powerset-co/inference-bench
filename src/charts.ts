// Draw the README figures (SVG) from the latest run's summary. Called by report.ts.
//
// Each figure is a stack of panels, one per model, so providers are always compared on the same model.
// Closed models (served only by their maker) share a final "reference" panel.
import * as Plot from '@observablehq/plot';
import { JSDOM } from 'jsdom';
import { readFileSync } from 'node:fs';

export type CellSummary = {
  cell: string;
  task_s: number;
  task_s_low: number; // lowest and highest of the per-pass mean times
  task_s_high: number;
  task_ttft_s: number | ''; // the part of task_s spent waiting for first tokens (blank if the cell can't stream)
  task_usd: number;
  final_turn_ttft_cold_s: number | '';
  final_turn_ttft_warm_s: number | '';
  calls_s: number[]; // duration of every call in the agent runs, for the latency distribution
};

// --- Names and logos ------------------------------------------------------------------------------

const MODELS: Record<string, string> = {
  'glm-5.3': 'GLM-5.3', 'kimi-k3.1-pro': 'Kimi K3.1 Pro', 'deepseek-v4.1-flash': 'DeepSeek V4.1 Flash',
  'gpt-6.1-sol': 'GPT-6.1 Sol', 'gpt-6.1-terra': 'GPT-6.1 Terra', 'claude-sonnet-5.5': 'Claude Sonnet 5.5', 'claude-haiku-5.5': 'Claude Haiku 5.5',
};
const PROVIDERS: Record<string, { name: string; logo: string }> = {
  modal: { name: 'Modal', logo: 'modal' },
  'sail-asap': { name: 'Sail · asap', logo: 'sail' },
  'sail-balanced': { name: 'Sail · balanced', logo: 'sail' },
  'sail-flex': { name: 'Sail · flex', logo: 'sail' },
  wafer: { name: 'Wafer', logo: 'wafer' },
  baseten: { name: 'Baseten', logo: 'baseten' },
  'baseten-fast': { name: 'Baseten · fast', logo: 'baseten' },
  fireworks: { name: 'Fireworks', logo: 'fireworks' },
  'fireworks-fast': { name: 'Fireworks · fast', logo: 'fireworks' },
  'fireworks-ultrafast': { name: 'Fireworks · ultrafast', logo: 'fireworks' },
  together: { name: 'Together AI', logo: 'together' },
  vultr: { name: 'Vultr', logo: 'vultr' },
  openai: { name: 'OpenAI', logo: 'openai' },
  anthropic: { name: 'Anthropic', logo: 'anthropic' },
};
const CLOSED = 'Closed models (reference)';
const dataURI = (svg: string) => `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
// Modal's mark is drawn as outlines. The logo file thickens them so it stays legible at chart size (16px);
// at banner size a lighter outline matches the other marks.
const logo = (name: string, color = '#111111', large = false) => {
  const svg = readFileSync(`assets/logos/${name}.svg`, 'utf8').replaceAll('#111111', color);
  return dataURI(large ? svg.replace('stroke-width="0.8"', 'stroke-width="0.25"') : svg);
};

type Row = CellSummary & { group: string; label: string; logo: string };

/** Panels: one per open model (served by several providers), in a fixed order, then the closed models. */
function rows(summary: CellSummary[]): Row[] {
  const modelOf = (s: CellSummary) => s.cell.split('/')[1];
  const providerOf = (s: CellSummary) => s.cell.split('/')[0];
  const shared = (s: CellSummary) => summary.filter((o) => modelOf(o) === modelOf(s)).length > 1;
  const order = (s: CellSummary) => [shared(s) ? 0 : 1, Object.keys(MODELS).indexOf(modelOf(s)), Object.keys(PROVIDERS).indexOf(providerOf(s))];
  return [...summary]
    .sort((a, b) => order(a).reduce((d, x, i) => d || x - order(b)[i], 0))
    .map((s) => ({
      ...s,
      group: shared(s) ? MODELS[modelOf(s)] ?? modelOf(s) : CLOSED,
      label: shared(s) ? PROVIDERS[providerOf(s)].name : MODELS[modelOf(s)] ?? modelOf(s),
      logo: logo(PROVIDERS[providerOf(s)].logo),
    }));
}

// --- Style ----------------------------------------------------------------------------------------

const WIDTH = 780;
const LABELS = 180; // left column: logo + row label
const FONT = '"Helvetica Neue", Helvetica, Arial, sans-serif';
const INK = '#1a1a1a';
const MUTED = '#666666';
const WAIT = '#2f4f7f'; // waiting for the first token
const GENERATE = '#8fa9d0'; // generating the rest of the response
const NO_STREAM = '#9a9a9a'; // cells that can't stream, so time isn't split
const COST = '#7a8c3b';
const BOX = '#3b5b8c';
const RANGE = INK; // spread between passes
const COLD = '#d9622b';
const WARM = '#2a78d6';
const document = new JSDOM('').window.document;
const ROW = 24;

const plot = (options: Plot.PlotOptions) =>
  (Plot.plot({
    document, marginTop: 4, marginBottom: 24, marginLeft: 10, marginRight: 64,
    style: { fontFamily: FONT, fontSize: '10.5px', color: INK, background: 'white' },
    ...options,
  }) as unknown as SVGSVGElement).outerHTML;

const BAND_PADDING = 0.32;
const yBand = (data: Row[]) => ({ domain: data.map((r) => r.label), axis: null, padding: BAND_PADDING });
const panelHeight = (data: Row[]) => data.length * ROW + 28;

/** Logo + label per row, as plain SVG in the left column (aligned with Plot's band scale). */
function labelColumn(data: Row[]) {
  // Same geometry as Plot's band scale with padding p: step = range / (n + p), center = step * (i + p + (1 - p) / 2).
  const p = BAND_PADDING;
  const step = (panelHeight(data) - 28) / (data.length + p);
  return data
    .map((r, i) => {
      const cy = 4 + step * (i + p + (1 - p) / 2);
      return `<image href="${r.logo}" x="0" y="${cy - 8}" width="16" height="16"/><text x="24" y="${cy + 4}" font-size="12" fill="${INK}">${r.label}</text>`;
    })
    .join('');
}

/** Time per task, split into waiting for first tokens and generating. */
function timeBars(data: Row[], width: number) {
  const segments = data.flatMap((r) =>
    r.task_ttft_s === ''
      ? [{ label: r.label, part: 'no-stream', x1: 0, x2: r.task_s }]
      : [{ label: r.label, part: 'wait', x1: 0, x2: r.task_ttft_s }, { label: r.label, part: 'generate', x1: r.task_ttft_s, x2: r.task_s }],
  );
  const fill = { wait: WAIT, generate: GENERATE, 'no-stream': NO_STREAM } as Record<string, string>;
  return plot({
    width, height: panelHeight(data),
    // Each panel gets its own axis (providers are compared within a model), with room for the label.
    x: { label: null, grid: true, nice: true, ticks: 4, domain: [0, Math.max(...data.map((r) => r.task_s_high)) * 1.2], tickFormat: (x: number) => `${x} s` },
    y: yBand(data),
    marks: [
      Plot.barX(segments, { y: 'label', x1: 'x1', x2: 'x2', fill: (d) => fill[d.part] }),
      // The range across passes: a thin line with end caps, drawn over the bar.
      Plot.link(data, { y1: 'label', y2: 'label', x1: 'task_s_low', x2: 'task_s_high', stroke: RANGE, strokeWidth: 1 }),
      Plot.tickX(data, { y: 'label', x: 'task_s_low', stroke: RANGE, insetTop: 4, insetBottom: 4 }),
      Plot.tickX(data, { y: 'label', x: 'task_s_high', stroke: RANGE, insetTop: 4, insetBottom: 4 }),
      Plot.text(data, { y: 'label', x: (r: Row) => Math.max(r.task_s, r.task_s_high), text: (r: Row) => `${r.task_s.toFixed(1)} s`, dx: 4, textAnchor: 'start', fontSize: 10.5, fill: INK }),
      Plot.ruleX([0], { stroke: INK }),
    ],
  });
}


/** Horizontal bars of one value per row, with the value printed at the end of each bar. */
function bars(data: Row[], key: 'task_s' | 'task_usd', width: number, color: string, format: (x: number) => string, tick?: (x: number) => string) {
  return plot({
    width, height: panelHeight(data),
    x: { label: null, grid: true, nice: true, ticks: 4, tickFormat: tick },
    y: yBand(data),
    marks: [
      Plot.barX(data, { y: 'label', x: key, fill: color }),
      Plot.text(data, { y: 'label', x: key, text: (r: Row) => format(r[key]), dx: 4, textAnchor: 'start', fontSize: 10.5, fill: INK }),
      Plot.ruleX([0], { stroke: INK }),
    ],
  });
}

/** A figure: title, subtitle lines, optional legend, then one panel per group stacked vertically. */
function figure(title: string, subtitle: string[], legend: string, data: Row[], panel: (rows: Row[]) => string) {
  const parts = [
    `<text x="0" y="20" font-size="17" font-weight="700" fill="${INK}">${title}</text>`,
    ...subtitle.map((line, i) => `<text x="0" y="${42 + i * 17}" font-size="12.5" fill="${MUTED}">${line}</text>`),
  ];
  let y = 42 + subtitle.length * 17;
  if (legend) (parts.push(`<g transform="translate(0 ${y + 8})">${legend}</g>`), (y += 32)); // a gap below the description
  for (const [group, groupRows] of Object.entries(Object.groupBy(data, (r) => r.group)) as [string, Row[]][]) {
    parts.push(
      `<line x1="0" x2="${WIDTH}" y1="${y + 4}" y2="${y + 4}" stroke="#e3e3e3"/>`,
      `<text x="0" y="${y + 24}" font-size="13" font-weight="700" fill="${INK}">${group}</text>`,
      `<g transform="translate(0 ${y + 32})">${labelColumn(groupRows)}${panel(groupRows)}</g>`,
    );
    y += 32 + panelHeight(groupRows) + 6;
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH + 32}" height="${y + 16}" viewBox="-16 -12 ${WIDTH + 32} ${y + 16}" font-family='${FONT}'>
<rect x="-16" y="-12" width="${WIDTH + 32}" height="${y + 16}" fill="white"/>
${parts.join('\n')}
</svg>\n`;
}

const dot = (x: number, color: string, text: string) =>
  `<circle cx="${x + 5}" cy="0" r="5" fill="${color}"/><text x="${x + 15}" y="4" font-size="12" fill="${MUTED}">${text}</text>`;

/** A key for the latency box plot: a sample of each part, labelled in plain words. */
function boxLegend() {
  const item = (x: number, swatch: string, text: string) => `<g transform="translate(${x} 0)">${swatch}<text x="30" y="4" font-size="12" fill="${MUTED}">${text}</text></g>`;
  return [
    item(0, `<rect x="0" y="-5" width="24" height="10" fill="${GENERATE}" stroke="${BOX}"/>`, 'Typical range (middle 50%)'),
    item(215, `<line x1="0" x2="24" y1="0" y2="0" stroke="${BOX}" stroke-width="1.5"/>`, 'Most calls (P10–P90)'),
    item(386, `<line x1="12" x2="12" y1="-6" y2="6" stroke="${INK}" stroke-width="2"/>`, 'Median call'),
    item(505, `<circle cx="12" cy="0" r="3.5" fill="none" stroke="${INK}"/>`, 'Slowest call'),
  ].join('');
}

// --- Figures --------------------------------------------------------------------------------------

export function charts(summary: CellSummary[]) {
  const data = rows(summary);
  const half = (WIDTH - LABELS - 20) / 2;

  const timeAndCost = figure(
    'Time and cost per task',
    [
      'Average time and list-price cost to complete one agentic research task (multiple conversation turns).',
    ],
    `${dot(0, WAIT, 'Waiting for first token')}${dot(160, GENERATE, 'Generating')}${dot(264, NO_STREAM, 'Not split (no streaming)')}` +
      `<g transform="translate(432 0)"><line x1="0" x2="20" y1="0" y2="0" stroke="${RANGE}"/><line x1="0" x2="0" y1="-4" y2="4" stroke="${RANGE}"/><line x1="20" x2="20" y1="-4" y2="4" stroke="${RANGE}"/><text x="28" y="4" font-size="12" fill="${MUTED}">Run-to-run variation</text></g>` +
      dot(608, COST, 'USD per task'),
    data,
    (rows) => {
      const usd = (x: number) => `$${x < 0.01 ? x.toFixed(4) : x.toFixed(3)}`;
      const time = timeBars(rows, half).replace('<svg', `<svg x="${LABELS}"`);
      const cost = bars(rows, 'task_usd', half, COST, usd, (x) => `$${+x.toPrecision(2)}`).replace('<svg', `<svg x="${LABELS + half + 20}"`);
      return time + cost;
    },
  );

  const streaming = data.filter((r) => r.final_turn_ttft_cold_s !== '' && r.final_turn_ttft_warm_s !== '');
  const caching = figure(
    'Prompt caching',
    [
      'Seconds to first token on the final turn of each task (median), sent once with the conversation not',
      'cached and once with it cached. Same request, so the gap shows how much caching helped.',
    ],
    `${dot(0, COLD, 'Conversation not cached')}${dot(200, WARM, 'Conversation cached')}`,
    streaming,
    (rows) =>
      plot({
        width: WIDTH - LABELS, height: panelHeight(rows),
        x: { label: null, grid: true, nice: true, zero: true, ticks: 6, tickFormat: (x: number) => `${x} s` },
        y: yBand(rows),
        marks: [
          Plot.link(rows, { y1: 'label', y2: 'label', x1: 'final_turn_ttft_cold_s', x2: 'final_turn_ttft_warm_s', stroke: '#bdbdbd', strokeWidth: 2 }),
          Plot.dot(rows, { y: 'label', x: 'final_turn_ttft_cold_s', fill: COLD, r: 5 }),
          Plot.dot(rows, { y: 'label', x: 'final_turn_ttft_warm_s', fill: WARM, r: 5 }),
          Plot.ruleX([0], { stroke: INK }),
        ],
      }).replace('<svg', `<svg x="${LABELS}"`),
  );

  // Box plot per cell: box from the 25th to 75th percentile call, whiskers to the 10th and 90th, a tick at
  // the median, and a ring at the slowest call. Shared log axis across panels.
  const q = (xs: number[], p: number) => [...xs].sort((a, b) => a - b)[Math.floor((xs.length - 1) * p)];
  const boxes = data.map((r) => ({ ...r, p10: q(r.calls_s, 0.1), p25: q(r.calls_s, 0.25), p50: q(r.calls_s, 0.5), p75: q(r.calls_s, 0.75), p90: q(r.calls_s, 0.9), max: Math.max(...r.calls_s) }));
  const lo = Math.min(...boxes.map((b) => b.p10)) * 0.8;
  const hi = Math.max(...boxes.map((b) => b.max)) * 1.3;
  const latency = figure(
    'Latency per call',
    ['How long each model call (one turn of a task) took. Log scale.'],
    boxLegend(),
    boxes,
    (rows) =>
      plot({
        width: WIDTH - LABELS, height: panelHeight(rows),
        x: { type: 'log', label: null, grid: true, domain: [lo, hi], tickFormat: (x: number) => `${x} s` },
        y: yBand(rows),
        marks: [
          Plot.link(rows, { y1: 'label', y2: 'label', x1: 'p10', x2: 'p90', stroke: BOX, strokeWidth: 1.5 }),
          Plot.barX(rows, { y: 'label', x1: 'p25', x2: 'p75', fill: GENERATE, stroke: BOX, insetTop: 3, insetBottom: 3 }),
          Plot.tickX(rows, { y: 'label', x: 'p50', stroke: INK, strokeWidth: 2, insetTop: 3, insetBottom: 3 }),
          Plot.dot(rows, { y: 'label', x: 'max', stroke: INK, r: 3.5 }),
        ],
      }).replace('<svg', `<svg x="${LABELS}"`),
  );

  return { 'time-and-cost': timeAndCost, latency, caching };
}

/**
 * A row of provider logos with names, for the top of the README. Gray reads on light and dark themes.
 * Items are laid out from measured widths so the gap from each name to the next mark is identical.
 */
export function banner(providers: string[]) {
  const NAMES: Record<string, string> = {
    sail: 'Sail', wafer: 'Wafer', modal: 'Modal', baseten: 'Baseten', fireworks: 'Fireworks', together: 'Together AI', vultr: 'Vultr', openai: 'OpenAI', anthropic: 'Anthropic',
  };
  // Width of each mark's visible glyph as a share of its square logo box (the rest is even side padding).
  const GLYPH: Record<string, number> = { baseten: 0.62, fireworks: 0.97, together: 0.76, sail: 0.81, modal: 0.97, wafer: 0.7, vultr: 0.79, openai: 0.78, anthropic: 0.75 };
  // Helvetica Bold advance widths (per 1000 units of font size) for the letters in the names.
  const ADVANCE: Record<string, number> = {
    ' ': 278, B: 722, F: 611, T: 611, V: 667, u: 611, g: 611, k: 556, s: 556, w: 778, y: 556,
    A: 722, I: 278, M: 833, O: 778, S: 667, W: 944, a: 556, c: 556, d: 611, e: 556, f: 333, h: 611, i: 278, l: 278, n: 611, o: 611, p: 611, r: 389, t: 333,
  };
  const SIZE = 36, FONT_SIZE = 18, MARK_TO_NAME = 10, GAP = 44, GRAY = '#7d8590';
  const textWidth = (text: string) => [...text].reduce((w, c) => w + (ADVANCE[c] ?? 600), 0) * FONT_SIZE / 1000;

  // Up to five providers per row, with rows balanced (9 → 5 + 4) and each one centered.
  const PER_ROW = 5, ROW_GAP = 20, SCALE = 0.85; // SCALE sets the displayed size
  const rowCount = Math.ceil(providers.length / PER_ROW);
  const perRow = Math.ceil(providers.length / rowCount);
  const rows = Array.from({ length: rowCount }, (_, i) => {
    let x = 0; // left edge of the next visible glyph
    const items = providers.slice(i * perRow, (i + 1) * perRow).map((p) => {
      const glyph = SIZE * GLYPH[p];
      const boxX = x - (SIZE - glyph) / 2;
      const item = `<image href="${logo(p, GRAY, true)}" x="${boxX.toFixed(1)}" y="0" width="${SIZE}" height="${SIZE}"/>` +
        `<text x="${(x + glyph + MARK_TO_NAME).toFixed(1)}" y="25" font-size="${FONT_SIZE}" font-weight="700" fill="${GRAY}">${NAMES[p]}</text>`;
      x += glyph + MARK_TO_NAME + textWidth(NAMES[p]) + GAP;
      return item;
    });
    return { items, width: x - GAP };
  });
  const width = Math.ceil(Math.max(...rows.map((r) => r.width)) + 8); // a little slack for font differences
  const height = rowCount * SIZE + (rowCount - 1) * ROW_GAP;
  const body = rows.map((r, i) =>
    `<g transform="translate(${((width - 8 - r.width) / 2).toFixed(1)} ${i * (SIZE + ROW_GAP)})">\n${r.items.join('\n')}\n</g>`);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${Math.round(width * SCALE)}" height="${Math.round(height * SCALE)}" viewBox="-4 0 ${width} ${height}" font-family='${FONT}'>\n${body.join('\n')}\n</svg>\n`;
}
