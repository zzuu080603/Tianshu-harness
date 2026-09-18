/**
 * Equivalence + latency harness for `PromptEngine.buildOaiRequest`.
 *
 * Why this exists (Issue #2 / upstream #139): buildOaiRequest runs five full
 * scans per turn (prune / staleness / observation-mask / dedup+disk-budget)
 * plus `estimateOaiTokens` and `recordPrefixDivergence`. Any optimization of
 * those passes is only safe if the serialized request stays byte-identical —
 * DeepSeek's 95–99% prefix-cache hit rate lives or dies on that. This harness
 * is the hard gate: a seeded, 300-case corpus across the form matrix, hashed
 * with the repo's own `stableStringify` (gate 1) plus the engine's observable
 * side effects (gate 2).
 *
 * The corpus is explicit-shape-driven rather than purely random on purpose:
 * a random generator cannot be trusted to hit sidePath / disk-truncation /
 * observation masking, and a harness that misses a branch is a false green.
 */
import { createHash } from 'node:crypto'
import { PromptEngine, type PrefixDivergence } from '../../engine.js'
import { stableStringify } from '../../../api/stable-json.js'
import type { OaiContentPart, OaiMessage } from '../../../api/oai-types.js'

export type Rng = () => number

/** Deterministic 32-bit PRNG (mulberry32) — same seed ⇒ same corpus forever. */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function pick<T>(rng: Rng, arr: readonly T[]): T {
  return arr[Math.floor(rng() * arr.length)]!
}

function int(rng: Rng, min: number, max: number): number {
  return min + Math.floor(rng() * (max - min + 1))
}

const WORDS = [
  'cache', 'prompt', 'engine', 'token', 'prefix', 'window', 'session', 'hash',
  'scan', 'pass', 'build', 'request', 'mask', 'prune', 'stale', 'dedup',
  'snapshot', 'freeze', 'divergence', 'anchor', 'trailer', 'appendix',
] as const

const CJK = ['前缀缓存', '扫描', '剪枝', '遮蔽', '去重', '请求', '会话', '构建', '字节等价', '探针'] as const

/** Control chars / surrogate pairs / RTL override / combining marks / lone surrogate. */
const UNICODE_SNIPPETS = [
  '\u0000', '\u0001', '\u001b[31m', '\u202eRTL', 'e\u0301', '𝕏𝕐', '漢字テスト',
  '\ud83d\ude00', '\ud83e\udd16', '\ud800', 'a\u0007b', '\u200b', 'Ω≈ç√',
] as const

function wordy(rng: Rng, n: number): string {
  const out: string[] = []
  for (let i = 0; i < n; i++) out.push(pick(rng, WORDS))
  return out.join(' ')
}

function unicodeBlob(rng: Rng, n: number): string {
  const out: string[] = []
  for (let i = 0; i < n; i++) out.push(pick(rng, UNICODE_SNIPPETS))
  return out.join('')
}

/** Content whose first 2000 chars differ per index — defeats simpleHash dedup. */
function bulk(rng: Rng, index: number, size: number): string {
  const head = `# result ${index} ${wordy(rng, 4)}\n`
  const body = (wordy(rng, 8) + ' ') .repeat(Math.ceil(size / 60))
  return (head + body).slice(0, size)
}

const TOOL_NAMES = ['read_file', 'grep', 'list_dir', 'bash', 'write_file', 'edit_file', 'repo_map'] as const
const TOOL_ARGS: Record<string, (rng: Rng, i: number) => string> = {
  read_file: (_r, i) => JSON.stringify({ file_path: `src/file${i}.ts` }),
  grep: (r, i) => JSON.stringify({ pattern: pick(r, WORDS), path: `src/dir${i}` }),
  list_dir: () => JSON.stringify({ path: '.' }),
  bash: (r) => JSON.stringify({ command: `npm run ${pick(r, WORDS)}` }),
  write_file: (_r, i) => JSON.stringify({ file_path: `src/out${i}.ts` }),
  edit_file: (_r, i) => JSON.stringify({ file_path: `src/edit${i}.ts` }),
  repo_map: (r, i) => JSON.stringify({ query: `${pick(r, WORDS)}${i}` }),
}

function assistantToolCalls(rng: Rng, i: number, count: number): OaiMessage {
  const tool_calls = Array.from({ length: count }, (_, k) => {
    const name = pick(rng, TOOL_NAMES)
    return { id: `tc-${i}-${k}`, type: 'function' as const, function: { name, arguments: TOOL_ARGS[name]!(rng, i + k) } }
  })
  const msg: OaiMessage = { role: 'assistant', content: rng() < 0.5 ? null : wordy(rng, int(rng, 2, 8)), tool_calls }
  if (rng() < 0.3) (msg as { reasoning_content?: string }).reasoning_content = wordy(rng, int(rng, 5, 40))
  return msg
}

function toolResult(id: string, content: string): OaiMessage {
  return { role: 'tool', tool_call_id: id, content }
}

interface ShapeOutput {
  messages: OaiMessage[]
  contextWindow?: number
  sidePath?: boolean
}

type ShapeBuilder = (rng: Rng, seed: number) => ShapeOutput

const SHAPES: Record<string, ShapeBuilder> = {
  'plain-dialog': (rng) => {
    const messages: OaiMessage[] = []
    const turns = int(rng, 6, 14)
    for (let t = 0; t < turns; t++) {
      messages.push({ role: 'user', content: `turn ${t}: ${wordy(rng, int(rng, 3, 12))}` })
      messages.push({ role: 'assistant', content: rng() < 0.2 ? '' : wordy(rng, int(rng, 2, 20)) })
    }
    return { messages }
  },

  'tool-batch': (rng, seed) => {
    const messages: OaiMessage[] = []
    const turns = int(rng, 5, 10)
    for (let t = 0; t < turns; t++) {
      messages.push({ role: 'user', content: `batch ${t} ${wordy(rng, 4)}` })
      const count = int(rng, 1, 3)
      const a = assistantToolCalls(rng, seed * 100 + t, count)
      messages.push(a)
      for (const tc of (a as { tool_calls: { id: string }[] }).tool_calls) {
        messages.push(toolResult(tc.id, wordy(rng, int(rng, 5, 60))))
      }
    }
    return { messages }
  },

  'dedup-hit-miss': (rng, seed) => {
    const messages: OaiMessage[] = []
    const shared = bulk(rng, seed, 900)
    for (let t = 0; t < 6; t++) {
      const a = assistantToolCalls(rng, seed * 100 + t, 1)
      messages.push({ role: 'user', content: `scan ${t}` })
      messages.push(a)
      const id = (a as { tool_calls: { id: string }[] }).tool_calls[0]!.id
      // alternate identical (dedup hit) and unique (dedup miss)
      messages.push(toolResult(id, t % 2 === 0 ? shared : `${shared} tail ${t}`))
    }
    return { messages }
  },

  'observation-mask': (rng, seed) => {
    const messages: OaiMessage[] = []
    const turns = int(rng, 12, 18)
    for (let t = 0; t < turns; t++) {
      messages.push({ role: 'user', content: `old turn ${t}` })
      const a = assistantToolCalls(rng, seed * 100 + t, 1)
      messages.push(a)
      const id = (a as { tool_calls: { id: string }[] }).tool_calls[0]!.id
      messages.push(toolResult(id, t < turns - 2 ? bulk(rng, seed * 1000 + t, 400) : 'recent small'))
    }
    return { messages }
  },

  'disk-truncate': (rng, seed) => {
    const messages: OaiMessage[] = [
      { role: 'user', content: 'dump the giant log' },
      toolResult(`tc-${seed}-a`, bulk(rng, seed, 50_001)),
      { role: 'assistant', content: 'and another' },
      toolResult(`tc-${seed}-b`, bulk(rng, seed + 1, 130_000)),
    ]
    return { messages }
  },

  'huge-payload': (rng, seed) => {
    const messages: OaiMessage[] = [{ role: 'user', content: 'read the big file' }]
    const a = assistantToolCalls(rng, seed, 1)
    messages.push(a)
    const id = (a as { tool_calls: { id: string }[] }).tool_calls[0]!.id
    messages.push(toolResult(id, bulk(rng, seed, 100_000)))
    messages.push({ role: 'assistant', content: 'noted' })
    return { messages }
  },

  'unicode-control': (rng, seed) => {
    const messages: OaiMessage[] = [
      { role: 'user', content: unicodeBlob(rng, 12) },
      { role: 'assistant', content: unicodeBlob(rng, 10), reasoning_content: unicodeBlob(rng, 6) } as OaiMessage,
      { role: 'user', content: `混合 ${unicodeBlob(rng, 5)} ${CJK[seed % CJK.length]}` },
      toolResult(`tc-${seed}-u`, unicodeBlob(rng, 400)),
    ]
    return { messages }
  },

  'empty-null': (rng) => {
    return {
      messages: [
        { role: 'user', content: '' },
        { role: 'assistant', content: null },
        { role: 'user', content: '' },
        { role: 'assistant', content: '' },
        toolResult('tc-empty', ''),
        { role: 'assistant', content: null },
        { role: 'user', content: wordy(rng, 3) },
      ],
    }
  },

  'reasoning-echo': (rng, seed) => {
    const messages: OaiMessage[] = []
    for (let t = 0; t < 8; t++) {
      messages.push({ role: 'user', content: `think ${t}` })
      messages.push({ role: 'assistant', content: wordy(rng, 4), reasoning_content: `reasoning ${t} ${wordy(rng, 30)}` } as OaiMessage)
    }
    messages.push(toolResult(`tc-${seed}-r`, bulk(rng, seed, 600)))
    return { messages }
  },

  'vision-parts': (rng, seed) => {
    const parts: OaiContentPart[] = [
      { type: 'text', text: `describe this ${wordy(rng, 3)}` },
      { type: 'image_url', image_url: { url: `data:image/png;base64,${'A'.repeat(64)}` } },
    ]
    return {
      messages: [
        { role: 'user', content: parts },
        { role: 'assistant', content: 'a screenshot' },
        { role: 'user', content: `and this one ${seed}` },
        { role: 'assistant', content: null },
      ],
    }
  },

  'system-reminder': (rng, seed) => {
    return {
      messages: [
        { role: 'user', content: `main task ${seed}` },
        { role: 'user', content: `<system-reminder>\nconvergence kick ${wordy(rng, 4)}\n</system-reminder>` },
        { role: 'assistant', content: 'working' },
        { role: 'user', content: `<system-reminder>${wordy(rng, 6)}</system-reminder>` },
        { role: 'assistant', content: wordy(rng, 5) },
      ],
    }
  },

  'orphan-repair': (rng, seed) => {
    return {
      messages: [
        { role: 'user', content: `aborted batch ${seed}` },
        assistantToolCalls(rng, seed * 7, 2),
        // only ONE of the two tool_call_ids gets a result → preflight repair path
        toolResult(`tc-${seed * 7}-0`, wordy(rng, 20)),
      ],
    }
  },

  'large-window-estimate': (rng, seed) => {
    const parsed = SHAPES['tool-batch']!(rng, seed)
    return { ...parsed, contextWindow: 200_000 }
  },

  'large-window-collapse': (rng, seed) => {
    const heavy = seed < 3
    const turns = heavy ? 60 : 12
    // 48K stays under the 50K disk-budget truncation but keeps ~10 unmasked
    // turns × 48K chars above the 200K window's 0.5 fill floor, so the T7
    // estimate/collapse block actually fires (observable: old reasoning stripped).
    const size = heavy ? 48_000 : 800
    const messages: OaiMessage[] = []
    for (let t = 0; t < turns; t++) {
      messages.push({ role: 'user', content: `turn ${t}` })
      messages.push({
        role: 'assistant',
        content: '',
        tool_calls: [{ id: `c-${t}`, type: 'function', function: { name: 'grep', arguments: JSON.stringify({ pattern: `p${t}` }) } }],
        reasoning_content: `old reasoning ${t} ${wordy(rng, 20)}`,
      } as OaiMessage)
      messages.push(toolResult(`c-${t}`, Array.from({ length: Math.ceil(size / 20) + 4 }, (_, k) => `src/f${t}-${k}.ts:1: match p${t}`).join('\n').slice(0, size)))
    }
    return { messages, contextWindow: 200_000 }
  },

  'side-path': (rng, seed) => {
    const parsed = SHAPES['dedup-hit-miss']!(rng, seed)
    return {
      messages: parsed.messages,
      sidePath: true,
      ...(seed % 2 === 0 ? { contextWindow: 200_000 } : {}),
    }
  },
}

export const SHAPE_NAMES = Object.keys(SHAPES)
const SEEDS_PER_SHAPE = 20

export interface EquivalenceCase {
  id: string
  shape: string
  seed: number
  messages: OaiMessage[]
  contextWindow?: number
  sidePath?: boolean
}

/** The 300-case corpus (15 shapes × 20 seeds), deterministic across processes. */
export function buildEquivalenceCorpus(): EquivalenceCase[] {
  const cases: EquivalenceCase[] = []
  const index = new Map(SHAPE_NAMES.map((s, i) => [s, i]))
  for (let i = 0; i < SHAPE_NAMES.length * SEEDS_PER_SHAPE; i++) {
    const shape = SHAPE_NAMES[i % SHAPE_NAMES.length]!
    const seed = Math.floor(i / SHAPE_NAMES.length)
    const rng = mulberry32(0x9e3779b9 ^ (index.get(shape)! * 0x85ebca6b) ^ (seed * 0x27d4eb2f))
    const out = SHAPES[shape]!(rng, seed)
    cases.push({
      id: `${shape}#${String(seed).padStart(2, '0')}`,
      shape,
      seed,
      messages: out.messages,
      contextWindow: out.contextWindow,
      sidePath: out.sidePath,
    })
  }
  return cases
}

export function sha256(value: string): string {
  return createHash('sha256').update(value).digest('hex')
}

/**
 * The host-derived byte regions of the built request have to be canonicalized,
 * because the fixture is a *cross-version* byte gate that is hashed on every
 * machine (author laptop, CI runner, Windows dev box). All of them live in the
 * frozen volatile block and are selected by `process.platform` / `os.type()` /
 * `os.release()` (see `src/platform.ts::getTargetPlatform` and
 * `src/prompt/volatile.ts`):
 *
 *  1. `<environment platform=… cwd=… os=… />` — `platform` comes from
 *     `process.platform`, `os` from `os.type()`/`os.release()`. The kernel
 *     release alone differs between the authoring host and the ubuntu runner,
 *     which made the first CI run of this gate fail with all 300 cases drifting
 *     while it passed locally.
 *  2. `<platform-note>` (target platform ≠ host platform),
 *     `<path-style-note>` (target platform is win32) and `<shell-note>`
 *     (resolved shell ≠ `sh`) are emitted *only* on some hosts — on a Windows
 *     host the built request carries two elements a Linux host does not, which
 *     drifted all 300 cases again on a win32 simulation.
 *
 * Pinning/removing exactly those regions keeps the golden host-independent
 * without weakening anything else: message order/bytes, the five pass rewrites,
 * dedup, masking, disk truncation, appendix and collapse all stay verbatim
 * under the hash. `cwd` is preserved (corpus-controlled, not host-derived) and a
 * re-pinned `platform` is still a pinned string, so a shape that stops carrying
 * the element is reported by `hostTags` instead of silently passing. The three
 * note bodies are static literals whose own bytes are covered by
 * `src/prompt/__tests__/volatile.test.ts` (`windowsShellNote`, path-style-note
 * presence); they are dropped here because their mere presence is host-shaped.
 */
const HOST_ENV_TAG = /<environment platform=\\"[^"\\]*\\"(?: host=\\"[^"\\]*\\")? cwd=\\"([^"\\]*)\\" os=\\"[^"\\]*\\" \/>/g
// The leading `\n\n` separator is consumed with the element: the volatile block joins
// its parts with it, so dropping only the element would leave a host-shaped blank run.
const HOST_ONLY_BLOCK = /(?:\\n\\n)?<(platform-note|path-style-note|shell-note)>[\s\S]*?<\/\1>/g

export interface CanonicalBytes {
  bytes: string
  /** How many `<environment … />` tags were canonicalized (0 = regex went stale). */
  hostTags: number
}

/** Canonicalize the host-derived `<environment>` attributes before hashing. */
export function canonicalizeHostBytes(serialized: string): CanonicalBytes {
  let hostTags = 0
  const withPinnedEnv = serialized.replace(HOST_ENV_TAG, (_match: string, cwd: string) => {
    hostTags++
    return `<environment platform=\\"<host>\\" cwd=\\"${cwd}\\" os=\\"<host>\\" />`
  })
  return { bytes: withPinnedEnv.replace(HOST_ONLY_BLOCK, ''), hostTags }
}

export interface CaseResult {
  id: string
  shape: string
  /** sha256(stableStringify(request)) — the hard byte gate. */
  requestHash: string
  /** sha256(stableStringify(request.messages)) — diagnosis granularity. */
  messagesHash: string
  /** Engine-side observable side effects over a two-build sequence. */
  sideEffectHash: string
  /** Host-derived `<environment>` tags canonicalized out of the hash input. */
  hostTags: number
  /** Host-independent sanity check on the (unhashed) numeric part of the
   *  divergence breadcrumbs — see {@link divergenceBreadcrumb}. */
  divergenceOffsetsBounded: boolean
}

/**
 * The divergence breadcrumb without its absolute character offset.
 *
 * `approxCharPos` sums the engine's own message lengths, and those lengths embed
 * the host-derived `<environment>` element (kernel/platform strings differ per
 * machine). Hashing the raw number would re-bind this fixture to the machine that
 * generated it — the exact class of the round-1 CI failure, where the runner's
 * kernel release made 300/300 cases drift while the test passed locally.
 *
 * What gate 2 hashes is the breadcrumb's contract (which message diverged, in
 * which direction, how many messages on each side); the offset stays verified
 * through a relational bound computed from bytes of the same host
 * (`divergenceOffsetsBounded`) — a 0, negative or past-the-end offset still
 * fails the gate.
 */
function divergenceBreadcrumb(d: PrefixDivergence | null): Omit<PrefixDivergence, 'approxCharPos'> | null {
  if (!d) return null
  const { approxCharPos: _hostBound, ...breadcrumb } = d
  return breadcrumb
}

function offsetBounded(d: PrefixDivergence | null, serializedLength: number): boolean {
  if (!d) return true
  return d.approxCharPos >= 0 && (d.idx === 0 || d.approxCharPos > 0) && d.approxCharPos < serializedLength
}

function makeEngine(): PromptEngine {
  return new PromptEngine({
    model: 'test-model',
    maxTokens: 4096,
    staticCtx: { tools: [] },
    volatileCtx: { cwd: '/test/project', rivetMd: '# Test Project' },
  })
}

/**
 * Run one corpus case on a fresh engine with a two-build sequence (build the
 * session, then append one assistant turn). The second build is what exercises
 * frozen-snapshot retrieval and the append-only fast path; hashing both catches
 * an optimization that emits identical bytes once but corrupts engine state.
 */
export function runCase(c: EquivalenceCase): CaseResult {
  const engine = makeEngine()
  const repairs: Array<{ count: number; messages: OaiMessage[] }> = []
  const onOrphanRepair = (messages: OaiMessage[], count: number): void => { repairs.push({ count, messages }) }

  const first = engine.buildOaiRequest(c.messages, undefined, c.contextWindow, { sidePath: c.sidePath, onOrphanRepair })
  const div1 = engine.consumePrefixDivergence()

  const extended = [...c.messages, { role: 'assistant' as const, content: 'extension' }]
  const second = engine.buildOaiRequest(extended, undefined, c.contextWindow, { sidePath: c.sidePath, onOrphanRepair })
  const div2 = engine.consumePrefixDivergence()

  const canonicalFirst = canonicalizeHostBytes(stableStringify(first))
  const canonicalSecond = canonicalizeHostBytes(stableStringify(second))
  const sideEffects = {
    div1: divergenceBreadcrumb(div1),
    div2: divergenceBreadcrumb(div2),
    frozenAnchors: engine.getFrozenAnchorCount(),
    secondRequest: sha256(canonicalSecond.bytes),
    repairs,
  }

  return {
    id: c.id,
    shape: c.shape,
    requestHash: sha256(canonicalFirst.bytes),
    messagesHash: sha256(canonicalizeHostBytes(stableStringify(first.messages)).bytes),
    // Canonicalized too: `repairs` carries message bytes that include the
    // host-derived <environment> element.
    sideEffectHash: sha256(canonicalizeHostBytes(stableStringify(sideEffects)).bytes),
    hostTags: canonicalFirst.hostTags,
    divergenceOffsetsBounded:
      offsetBounded(div1, stableStringify(first).length) &&
      offsetBounded(div2, stableStringify(second).length),
  }
}

export function runEquivalenceSuite(): CaseResult[] {
  return buildEquivalenceCorpus().map(runCase)
}

/**
 * Branch reachability signals for one case. A harness whose corpus quietly
 * stops hitting a pass is a false green; the test asserts every signal is
 * > 0 across the corpus, so a shape that no longer triggers its branch fails
 * loudly instead of degrading into a no-op comparison.
 */
export interface BranchSignals {
  diskTruncated: boolean
  deduped: boolean
  masked: boolean
  unicode: boolean
  visionParts: boolean
  systemReminder: boolean
  largeWindow: boolean
  sidePath: boolean
  orphanRepaired: boolean
  collapseStrippedReasoning: boolean
}

export function observeBranches(c: EquivalenceCase): BranchSignals {
  const engine = makeEngine()
  const repairs: number[] = []
  const req = engine.buildOaiRequest(c.messages, undefined, c.contextWindow, {
    sidePath: c.sidePath,
    onOrphanRepair: (_m, count) => { repairs.push(count) },
  })
  const body = stableStringify(req.messages)
  const inputAssistantWithReasoning = c.messages.filter(
    m => m.role === 'assistant' && 'reasoning_content' in m && Boolean((m as { reasoning_content?: string }).reasoning_content),
  ).length
  const outputAssistantWithReasoning = req.messages.filter(
    m => m.role === 'assistant' && 'reasoning_content' in m && Boolean((m as { reasoning_content?: string }).reasoning_content),
  ).length
  return {
    diskTruncated: body.includes('[output truncated:'),
    deduped: body.includes('[duplicate content, see later tool result]'),
    masked: body.includes('[observation masked,'),
    unicode: body.includes('\\u0000') || body.includes('𝕏𝕐') || body.includes('\\ud800'),
    visionParts: req.messages.some(m => Array.isArray(m.content)),
    systemReminder: body.includes('<system-reminder>'),
    largeWindow: (c.contextWindow ?? 0) >= 200_000,
    sidePath: c.sidePath === true,
    orphanRepaired: repairs.length > 0,
    collapseStrippedReasoning: inputAssistantWithReasoning > outputAssistantWithReasoning,
  }
}

// ---------------------------------------------------------------------------
// Latency benchmark — median-of-N per tier.
// ---------------------------------------------------------------------------

export interface BenchTier {
  turns: number
  payloadChars: number
  contextWindow?: number
}

export interface BenchRow {
  turns: number
  payloadChars: number
  contextWindow?: number
  samples: number
  ms: number[]
  p50: number
}

export function runLatencyBenchmark(tiers: readonly BenchTier[], samples = 7): BenchRow[] {
  const rows: BenchRow[] = []
  for (const tier of tiers) {
    const messages = buildBenchSession(tier.turns, tier.payloadChars)
    const times: number[] = []
    for (let i = 0; i < samples; i++) {
      const engine = makeEngine()
      const t0 = performance.now()
      engine.buildOaiRequest(messages, undefined, tier.contextWindow)
      times.push(performance.now() - t0)
    }
    const sorted = [...times].sort((a, b) => a - b)
    rows.push({
      ...tier,
      samples,
      ms: times,
      p50: sorted[Math.floor(sorted.length / 2)]!,
    })
  }
  return rows
}

function buildBenchSession(turns: number, payloadChars: number): OaiMessage[] {
  const messages: OaiMessage[] = []
  for (let t = 0; t < turns; t++) {
    messages.push({ role: 'user', content: `turn ${t}: inspect file ${t}` })
    messages.push({
      role: 'assistant',
      content: '',
      tool_calls: [{ id: `b${t}`, type: 'function', function: { name: 'read_file', arguments: JSON.stringify({ file_path: `src/file${t}.ts` }) } }],
    })
    messages.push({ role: 'tool', tool_call_id: `b${t}`, content: `# file${t}\n` + 'x'.repeat(Math.max(0, payloadChars - 10)) })
    messages.push({ role: 'assistant', content: `done ${t}` })
  }
  return messages
}
