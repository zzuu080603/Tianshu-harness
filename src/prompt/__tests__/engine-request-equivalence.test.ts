import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import {
  buildEquivalenceCorpus,
  canonicalizeHostBytes,
  observeBranches,
  runCase,
  runEquivalenceSuite,
  SHAPE_NAMES,
  type CaseResult,
} from './helpers/request-equivalence-harness.js'

/**
 * Cross-version sha256 equivalence gate for `PromptEngine.buildOaiRequest`.
 *
 * `engine-cache-stability.test.ts` guards same-version drift; this file guards
 * *across* versions: the 300 seeded cases are hashed against a fixture captured
 * from a frozen engine revision (see `generatedFrom` in the fixture; the factory
 * is byte-identical from the frozen base commit to this branch). Any optimization
 * of the five scans / token estimate / divergence probe that moves a single byte
 * fails here before it can break every session's prefix cache.
 *
 * The host-derived volatile regions are canonicalized out of the hash input by the
 * harness, so the fixture is host-independent: the `<environment platform=… os=… />`
 * attributes (the first CI run failed 300/300 cases on the runner's kernel release
 * while passing on the authoring host) and the Windows-only `<path-style-note>` /
 * `<shell-note>` / `<platform-note>` elements (absent on linux/darwin, present when
 * the host is win32 — a win32 simulation drifted 300/300 before the strip). The
 * host-independence of the canonicalizer itself is asserted below, so a regression
 * of that property fails on CI instead of only on a Windows machine.
 *
 * Known deviation from the work order (Issue #2 §四 双通道防假绿): the second,
 * real-session-log replay channel is not implemented — this environment has no
 * real conversation logs (only test-generated session dirs under `~/.rivet`), and
 * a test reading the user's session store would be non-hermetic. The explicit
 * shape matrix plus the branch-reachability assertions in this file cover the
 * same false-green concern hermetically; real-log replay stays a maintainer-side
 * follow-up (a fixture captured from a real log would slot into the same corpus).
 *
 * Regenerate the fixture deliberately (only when prompt bytes are intentionally
 * changed) with:
 *   npx tsx scripts/prompt-request-benchmark.ts --emit-golden
 */

interface GoldenCase {
  /** sha256(JSON.stringify(request)) — the hard gate, wire-identical bytes. */
  requestWire: string
  /** sha256(stableStringify(request)) — sorted-key diagnosis granularity. */
  requestStable: string
  messages: string
  sideEffects: string
}

interface Golden {
  note: string
  generatedFrom: string
  cases: Record<string, GoldenCase>
}

const GOLDEN_PATH = fileURLToPath(new URL('./fixtures/engine-request-golden.json', import.meta.url))

function loadGolden(): Golden {
  return JSON.parse(readFileSync(GOLDEN_PATH, 'utf8')) as Golden
}

function label(r: CaseResult): string {
  return `${r.id} (shape=${r.shape})`
}

describe('buildOaiRequest cross-version byte equivalence', () => {
  it('the corpus is the 300-case form matrix (15 shapes × 20 seeds)', () => {
    const corpus = buildEquivalenceCorpus()
    assert.equal(corpus.length, 300)
    assert.equal(new Set(corpus.map(c => c.shape)).size, SHAPE_NAMES.length)
    assert.equal(SHAPE_NAMES.length, 15)
    // ids are unique — a corpus that silently collapsed two shapes would still
    // be 300 results but would stop covering the matrix.
    assert.equal(new Set(corpus.map(c => c.id)).size, 300)
  })

  it('every case matches the frozen golden sha256 (hard gate)', () => {
    const golden = loadGolden()
    const results = runEquivalenceSuite()
    const mismatches: string[] = []
    for (const r of results) {
      const g = golden.cases[r.id]
      if (!g) { mismatches.push(`${r.id}: missing from golden`); continue }
      // The WIRE hash is the gate: JSON.stringify insertion order is what
      // `openai-client` sends, and what the prefix cache sees. A sorted-key
      // hash would stay green through an insertion-order regression.
      if (g.requestWire !== r.requestWireHash) mismatches.push(`${label(r)}: wire request hash drifted`)
      if (g.requestStable !== r.requestStableHash) mismatches.push(`${label(r)}: sorted-key request hash drifted`)
      if (g.messages !== r.messagesHash) mismatches.push(`${label(r)}: messages hash drifted`)
      if (g.sideEffects !== r.sideEffectHash) mismatches.push(`${label(r)}: side-effect hash drifted`)
    }
    assert.deepEqual(mismatches, [], `byte-equivalence gate failed:\n${mismatches.slice(0, 10).join('\n')}`)
    assert.equal(results.length, Object.keys(golden.cases).length, 'golden case count must match the corpus')
    // Host-byte canonicalization must actually fire: a stale regex would silently
    // re-bind the fixture to one machine's kernel/platform and fail in CI again.
    assert.deepEqual(
      results.filter(r => r.hostTags === 0).map(r => r.id),
      [],
      'every case must carry at least one canonicalized <environment> tag',
    )
    // The offset is the one breadcrumb field that cannot be hashed host-independently
    // (it sums message lengths that embed the <environment> element), so its
    // contract is checked relationally against live bytes instead of by hash.
    assert.deepEqual(
      results.filter(r => !r.divergenceOffsetsBounded).map(r => r.id),
      [],
      'divergence breadcrumb offsets must lie inside the request (0/negative/past-the-end = broken probe)',
    )
  })

  it('canonicalizes every host shape to one byte string (linux/darwin/win32)', () => {
    // Simulates the serialized request of the three host shapes the fixture is
    // hashed on. The win32 shape carries the two extra volatile elements the
    // engine emits only there, plus a foreign-kernel `os` attribute — the exact
    // bytes that drifted 300/300 cases on a win32 host before they were stripped.
    const shapes = [
      '<environment platform=\\"linux\\" cwd=\\"/x\\" os=\\"Linux 6.18.48\\" />\\n\\n<sober>s</sober>',
      '<environment platform=\\"darwin\\" cwd=\\"/x\\" os=\\"Darwin 23.6.0\\" />\\n\\n<sober>s</sober>',
      '<environment platform=\\"win32\\" cwd=\\"/x\\" os=\\"Windows_NT 10.0.22631\\" />\\n\\n' +
        '<path-style-note>backslash guidance</path-style-note>\\n\\n<shell-note>use git bash</shell-note>\\n\\n<sober>s</sober>',
      '<environment platform=\\"linux\\" host=\\"win32\\" cwd=\\"/x\\" os=\\"Windows_NT 10.0.19045\\" />\\n\\n' +
        '<platform-note>foreign target</platform-note>\\n\\n<sober>s</sober>',
    ]
    const canonical = shapes.map(s => canonicalizeHostBytes(s))
    const expected = '<environment platform=\\"<host>\\" cwd=\\"/x\\" os=\\"<host>\\" />\\n\\n<sober>s</sober>'
    assert.deepEqual(
      canonical.map(c => c.bytes),
      [expected, expected, expected, expected],
      'every host shape must canonicalize to the same host-independent bytes, with non-host content intact',
    )
    assert.deepEqual(canonical.map(c => c.hostTags), [1, 1, 1, 1], 'each shape must report its <environment> tag')
  })

  it('is deterministic across fresh engines within a process', () => {
    const corpus = buildEquivalenceCorpus()
    // Sample the corpus: the full suite already runs in the golden test; a
    // 30-case sample across shapes keeps this cheap while still catching
    // engine-global state (caches, registries) leaking between builds.
    const sample = corpus.filter((_, i) => i % 10 === 0)
    for (const c of sample) {
      const a = runCase(c)
      const b = runCase(c)
      assert.equal(a.requestWireHash, b.requestWireHash, `${c.id}: fresh-engine wire bytes must be identical`)
      assert.equal(a.sideEffectHash, b.sideEffectHash, `${c.id}: fresh-engine side effects must be identical`)
    }
  })

  it('sidePath builds stay hermetic on the same engine (main → side → main)', () => {
    // The poisoning accident this guards (engine.ts, 2026-07-05) is a
    // side-path build leaking a state write into the NEXT main rebuild on the
    // SAME engine — invisible across fresh engines by construction. runCase
    // runs the main → side → main sequence for every sidePath case: the
    // post-side main bytes are hashed into the golden (postSideMainRequest)
    // and the relational flag asserts byte-identity here.
    const sidePathCases = buildEquivalenceCorpus().filter(c => c.sidePath)
    assert.ok(sidePathCases.length > 0, 'corpus must include sidePath cases')
    const results = sidePathCases.map(runCase)
    assert.deepEqual(
      results.filter(r => r.sidePathHermetic !== true).map(r => r.id),
      [],
      'side-path build must not change the next main-turn request bytes',
    )
  })

  it('the corpus still reaches every guarded pass (no false green)', () => {
    const signals = buildEquivalenceCorpus().map(observeBranches)
    const total = (k: keyof ReturnType<typeof observeBranches>): number => signals.filter(s => s[k]).length
    for (const key of [
      'diskTruncated',
      'deduped',
      'masked',
      'unicode',
      'visionParts',
      'systemReminder',
      'dedupHint',
      'toolsSchema',
      'orphanRepaired',
      'collapseStrippedReasoning',
    ] as const) {
      assert.ok(total(key) > 0, `corpus no longer exercises "${key}" — shape coverage regressed`)
    }
  })
})
