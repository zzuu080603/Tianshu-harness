import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import {
  buildEquivalenceCorpus,
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
 * from the frozen base commit. Any optimization of the five scans / token
 * estimate / divergence probe that moves a single byte fails here before it can
 * break every session's prefix cache.
 *
 * Regenerate the fixture deliberately (only when prompt bytes are intentionally
 * changed) with:
 *   npx tsx scripts/prompt-request-benchmark.ts --emit-golden
 */

interface GoldenCase {
  request: string
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
      if (g.request !== r.requestHash) mismatches.push(`${label(r)}: request hash drifted`)
      if (g.messages !== r.messagesHash) mismatches.push(`${label(r)}: messages hash drifted`)
      if (g.sideEffects !== r.sideEffectHash) mismatches.push(`${label(r)}: side-effect hash drifted`)
    }
    assert.deepEqual(mismatches, [], `byte-equivalence gate failed:\n${mismatches.slice(0, 10).join('\n')}`)
    assert.equal(results.length, Object.keys(golden.cases).length, 'golden case count must match the corpus')
  })

  it('is deterministic across fresh engines within a process', () => {
    const corpus = buildEquivalenceCorpus()
    // Sample the corpus: two full suite runs already run in the golden test; a
    // 30-case sample across shapes keeps this cheap while still catching
    // engine-global state (caches, registries) leaking between builds.
    const sample = corpus.filter((_, i) => i % 10 === 0)
    for (const c of sample) {
      const a = runCase(c)
      const b = runCase(c)
      assert.equal(a.requestHash, b.requestHash, `${c.id}: fresh-engine bytes must be identical`)
      assert.equal(a.sideEffectHash, b.sideEffectHash, `${c.id}: fresh-engine side effects must be identical`)
    }
  })

  it('sidePath builds stay hermetic for every side-path case', () => {
    const sidePathCases = buildEquivalenceCorpus().filter(c => c.sidePath)
    assert.ok(sidePathCases.length > 0, 'corpus must include sidePath cases')
    for (const c of sidePathCases) {
      const r = runCase(c)
      assert.equal(
        r.sideEffectHash,
        runCase(c).sideEffectHash,
        `${c.id}: side-path build must not poison engine state`,
      )
    }
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
      'largeWindow',
      'sidePath',
      'orphanRepaired',
      'collapseStrippedReasoning',
    ] as const) {
      assert.ok(total(key) > 0, `corpus no longer exercises "${key}" — shape coverage regressed`)
    }
  })
})
