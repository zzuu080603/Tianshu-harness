import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { runLatencyBenchmark, type BenchTier } from './helpers/request-equivalence-harness.js'

/**
 * End-to-end latency sanity for `buildOaiRequest` (Issue #2 / upstream #139).
 *
 * `engine-perf.test.ts` pins the sub-passes' O(n) data structures, not the
 * assembled request. This test records median-of-7 wall-clock numbers for
 * growing histories so a catastrophic regression (an accidental O(n²) in any
 * of the five scans) shows up as either a failure or a PR-reviewable table.
 * Seven samples per the acceptance line in upstream #142.
 *
 * Wall-clock assertions are deliberately generous — CI machines vary, and a
 * flaky perf test is worse than no perf test. The real byte gate lives in
 * `engine-request-equivalence.test.ts`; the full tier table (up to 200 turns)
 * is `npm run prompt:bench`, kept out of CI.
 */
const TIERS: BenchTier[] = [
  { turns: 10, payloadChars: 200 },
  { turns: 50, payloadChars: 200 },
  { turns: 100, payloadChars: 200 },
  { turns: 100, payloadChars: 100_000, contextWindow: 200_000 },
]

// Measured p50 on CI-class hardware is single-digit ms for the heaviest tier;
// 200ms leaves two orders of magnitude of headroom while still catching the
// accidental-O(n²) class (2ms → 200ms+) that a 3000ms cap would wave through.
const CAP_MS = 200

describe('buildOaiRequest end-to-end latency (median of 7)', () => {
  it('builds growing histories within a generous cap', () => {
    const rows = runLatencyBenchmark(TIERS, 7)
    console.log('| turns | payload chars | window | P50 ms |')
    console.log('|---:|---:|---:|---:|')
    for (const r of rows) {
      console.log(`| ${r.turns} | ${r.payloadChars} | ${r.contextWindow ?? 'default'} | ${r.p50.toFixed(2)} |`)
    }
    for (const r of rows) {
      assert.ok(
        r.p50 < CAP_MS,
        `buildOaiRequest P50 ${r.p50.toFixed(1)}ms exceeds ${CAP_MS}ms cap at turns=${r.turns} payload=${r.payloadChars} window=${r.contextWindow ?? 'default'}`,
      )
    }
  })
})
