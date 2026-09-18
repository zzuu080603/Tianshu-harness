/**
 * buildOaiRequest performance/equivalence harness CLI (Issue #2 / upstream #139).
 *
 * Two modes:
 *   npx tsx scripts/prompt-request-benchmark.ts                 # median-of-7 latency table
 *   npx tsx scripts/prompt-request-benchmark.ts --emit-golden   # regenerate the byte-equivalence fixture
 *
 * The benchmark keeps large tiers here (not in `npm test`) so CI stays in the
 * seconds; the equivalence corpus is shared with the node:test gate.
 */
import { execSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  runEquivalenceSuite,
  runLatencyBenchmark,
  type BenchTier,
} from '../src/prompt/__tests__/helpers/request-equivalence-harness.js'

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const GOLDEN_PATH = `${repoRoot}/src/prompt/__tests__/fixtures/engine-request-golden.json`

function headSha(): string {
  try {
    return execSync('git rev-parse HEAD', { cwd: repoRoot, encoding: 'utf8' }).trim()
  } catch {
    return 'unknown'
  }
}

function emitGolden(): void {
  const results = runEquivalenceSuite()
  const cases: Record<string, { request: string; messages: string; sideEffects: string }> = {}
  for (const r of results) {
    cases[r.id] = { request: r.requestHash, messages: r.messagesHash, sideEffects: r.sideEffectHash }
  }
  const golden = {
    note:
      'buildOaiRequest cross-version byte-equivalence fixture. Generated from the ' +
      'engine at generatedFrom; host-derived bytes are canonicalized out before hashing ' +
      '(<environment platform/os> attributes plus the win32-only <path-style-note>/' +
      '<shell-note>/<platform-note> elements — see canonicalizeHostBytes), so the ' +
      'fixture is machine-independent across linux/darwin/win32. Regenerate only when a ' +
      'byte change is intentional: ' +
      'npx tsx scripts/prompt-request-benchmark.ts --emit-golden',
    generatedFrom: headSha(),
    cases,
  }
  mkdirSync(dirname(GOLDEN_PATH), { recursive: true })
  writeFileSync(GOLDEN_PATH, `${JSON.stringify(golden, null, 2)}\n`)
  console.log(`wrote ${Object.keys(cases).length} cases → ${GOLDEN_PATH}`)
}

function runBench(): void {
  const tiers: BenchTier[] = [
    { turns: 10, payloadChars: 200 },
    { turns: 50, payloadChars: 200 },
    { turns: 100, payloadChars: 200 },
    { turns: 200, payloadChars: 200 },
    { turns: 10, payloadChars: 100_000 },
    { turns: 50, payloadChars: 100_000 },
    { turns: 100, payloadChars: 100_000 },
    { turns: 200, payloadChars: 100_000 },
    { turns: 50, payloadChars: 100_000, contextWindow: 200_000 },
    { turns: 100, payloadChars: 100_000, contextWindow: 200_000 },
    { turns: 200, payloadChars: 100_000, contextWindow: 200_000 },
  ]
  const rows = runLatencyBenchmark(tiers, 7)
  console.log('| turns | payload chars | window | P50 ms | samples ms |')
  console.log('|---:|---:|---:|---:|---|')
  for (const r of rows) {
    console.log(
      `| ${r.turns} | ${r.payloadChars} | ${r.contextWindow ?? 'default(<1M 5-pass)'} | ${r.p50.toFixed(2)} | ${r.ms.map(v => v.toFixed(2)).join(', ')} |`,
    )
  }
}

if (process.argv.includes('--emit-golden')) emitGolden()
else runBench()
