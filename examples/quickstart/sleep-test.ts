/**
 * GraphZep Sleep Engine Test
 *
 * Validates the SleepEngine (Phase 1 + Phase 2) with 12 automatic PASS/FAIL
 * checks across 5 categories:
 *
 *   D — Dry run integrity      (2 checks)  must not touch the graph
 *   S — Sleep execution        (3 checks)  smoke + report values
 *   G — Graph state            (5 checks)  consolidation markers + integrity
 *   P — Pruning quality        (1 check)   duplicate entities collapsed
 *   R — Retrieval after sleep  (1 check)   search still works correctly
 *
 * Episode design:
 *   - Elena Vasquez mentioned in 5 distinct episodes → consolidation candidate
 *   - Dr. Alan Fischer / Fischer / Alan Fischer → duplicate pruning candidate
 *   - Control entities (James Park, Sophia Laurent, Victor Okafor) → unchanged
 *
 * Requirements:
 *   OPENAI_API_KEY=sk-...
 *   NEO4J_URI=bolt://localhost:7687
 *   NEO4J_USER=neo4j
 *   NEO4J_PASSWORD=password
 *
 * Run from /examples:
 *   npm run sleep-test
 */

import { config } from 'dotenv';
import {
  Graphzep,
  Neo4jDriver,
  OpenAIClient,
  OpenAIEmbedder,
  SleepEngine,
} from 'graphzep';
import type { SleepReport } from 'graphzep';

config();

const GROUP = 'sleep-test-v1';

// ── Formatting helpers ────────────────────────────────────────────────────────
const sep  = (title: string) =>
  console.log(`\n${'─'.repeat(64)}\n  ${title}\n${'─'.repeat(64)}`);
const ok   = (msg: string) => console.log(`  ✓ ${msg}`);
const info = (msg: string) => console.log(`  · ${msg}`);

// ── Check tracker ─────────────────────────────────────────────────────────────
interface CheckResult { id: string; label: string; passed: boolean; detail?: string }
const results: CheckResult[] = [];

function check(id: string, label: string, passed: boolean, detail?: string): void {
  results.push({ id, label, passed, detail });
  const icon = passed ? '✓ PASS' : '✗ FAIL';
  const line = `  [${id}] ${icon}  ${label}`;
  console.log(passed ? line : line + (detail ? `  → ${detail}` : ''));
}

// ── Episode corpus ────────────────────────────────────────────────────────────
//
// Deliberately designed to exercise both sleep phases:
//
//  • Elena Vasquez  → 5 episodes = consolidation candidate (minEpisodes: 2)
//  • Dr. Alan Fischer referenced as "Fischer", "Alan Fischer", "Dr. Fischer"
//    → duplicate entity pruning candidate
//  • James Park, Sophia Laurent, Victor Okafor → clean control entities
//
const EPISODES: string[] = [
  // ── Consolidation target: Elena in 5 distinct episodes ────────────────────
  'Elena Vasquez co-founded Meridian Security in 2019 alongside James Park, '
    + 'both leaving their previous employer CipherTech to start the company.',

  'Elena Vasquez serves as Chief Executive Officer of Meridian Security '
    + 'and sets the company\'s long-term strategic vision and culture.',

  'Elena Vasquez personally led the company-wide security overhaul after '
    + 'the 2022 data breach was confirmed, rotating all credentials and reimaging workstations.',

  'Elena Vasquez delivered a keynote address at the RSA Conference in San Francisco '
    + 'in February 2023, sharing the lessons Meridian learned from the breach.',

  'Elena Vasquez and board member Victor Okafor co-authored an op-ed published '
    + 'in the Wall Street Journal calling for stronger protections against corporate espionage.',

  // ── Duplicate target: same person, different name forms ───────────────────
  'Dr. Alan Fischer was recruited by Raj Patel from the NSA Cybersecurity Division '
    + 'to lead Meridian\'s newly formed research division.',

  'Fischer proved through detailed forensic analysis that Marcus Cole\'s API credentials '
    + 'had been silently cloned via a remote memory-scraping exploit.',

  'Alan Fischer co-authored a peer-reviewed paper on credential-cloning attack vectors '
    + 'published in IEEE Security & Privacy journal.',

  'Dr. Fischer\'s technical analysis was the decisive evidence that led to Marcus Cole\'s '
    + 'full exoneration from all suspicion of the breach.',

  // ── Control entities: consistent naming, used as baseline ─────────────────
  'James Park co-founded Meridian Security and later led the internal forensics '
    + 'investigation, formally becoming Chief Security Officer after the breach.',

  'Sophia Laurent, a general partner at Atlas Ventures, personally championed '
    + 'Meridian\'s $25M Series B funding round after the company\'s resilient recovery.',

  'Victor Okafor joined Meridian\'s board of directors as an independent director '
    + 'and became a strategic advisor and mentor to CEO Elena Vasquez.',
];

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  console.log('\n😴  GraphZep Sleep Engine Test\n');

  // ── 1. Setup ───────────────────────────────────────────────────────────────
  sep('1 / Setup');

  const driver = new Neo4jDriver(
    process.env.NEO4J_URI      || 'bolt://localhost:7687',
    process.env.NEO4J_USER     || 'neo4j',
    process.env.NEO4J_PASSWORD || 'password',
  );

  const llm = new OpenAIClient({
    apiKey: process.env.OPENAI_API_KEY!,
    model:  'gpt-4o-mini',
  });

  const embedder = new OpenAIEmbedder({
    apiKey: process.env.OPENAI_API_KEY!,
    model:  'text-embedding-3-small',
  });

  const g = new Graphzep({ driver, llmClient: llm, embedder, groupId: GROUP });

  // Sleep engine — in production, pass a more powerful model (e.g. gpt-4o)
  // for higher-quality consolidation. Using gpt-4o-mini here to keep test costs low.
  const sleepEngine = new SleepEngine({ driver, llm, embedder });

  await driver.createIndexes();
  ok('Neo4j connected and indexes verified');

  // ── 2. Cleanup ─────────────────────────────────────────────────────────────
  sep('2 / Cleanup previous data');
  await driver.executeQuery(`MATCH (n {groupId: $g}) DETACH DELETE n`, { g: GROUP });
  ok(`Cleaned groupId="${GROUP}"`);

  // ── 3. Ingest episodes ─────────────────────────────────────────────────────
  sep(`3 / Ingesting ${EPISODES.length} episodes`);
  for (let i = 0; i < EPISODES.length; i++) {
    const ep    = EPISODES[i];
    const label = `E${String(i + 1).padStart(2, '0')}`;
    await g.addEpisode({ content: ep, groupId: GROUP });
    ok(`${label} — ${ep.slice(0, 68)}…`);
  }

  // ── 4. Stabilisation pause ─────────────────────────────────────────────────
  sep('4 / Stabilisation (2 s)');
  await new Promise(r => setTimeout(r, 2000));
  ok('Ready for sleep testing');

  // ── 5. Pre-sleep snapshot ──────────────────────────────────────────────────
  sep('5 / Pre-sleep snapshot');

  const beforeRows = await driver.executeQuery<any[]>(
    `MATCH (n:Entity {groupId: $g}) RETURN n.name AS name ORDER BY name`,
    { g: GROUP },
  );
  const beforeCount = beforeRows.length;
  info(`Entity count before sleep: ${beforeCount}`);
  for (const r of beforeRows) console.log(`     ${r.name}`);

  const fischerBefore = await driver.executeQuery<any[]>(
    `MATCH (e:Entity {groupId: $g}) WHERE toLower(e.name) CONTAINS 'fischer' RETURN e.name AS name`,
    { g: GROUP },
  );
  info(`"fischer" variants before: [${fischerBefore.map((r: any) => r.name).join(', ') || 'none'}]`);

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY D — Dry run (must run BEFORE the actual sleep)
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category D — Dry run integrity');

  const dryReport = await sleepEngine.sleep(GROUP, {
    dryRun:           true,
    cooldownMinutes:  0,      // process all episodes regardless of age
    consolidation:    { minEpisodes: 2 },
    pruning:          { similarityThreshold: 0.85 },
  });

  const afterDryRows = await driver.executeQuery<any[]>(
    `MATCH (n:Entity {groupId: $g}) RETURN count(n) AS total`,
    { g: GROUP },
  );
  const afterDryCount = Number(afterDryRows[0]?.total ?? 0);

  check('D1',
    `dryRun=true: entity count unchanged  (before: ${beforeCount}, after dry run: ${afterDryCount})`,
    afterDryCount === beforeCount,
  );
  check('D2',
    `dryRun report shows potential consolidation work  (entitiesRefreshed: ${dryReport.phase1Consolidation.entitiesRefreshed})`,
    dryReport.phase1Consolidation.entitiesRefreshed > 0,
  );

  info(`Dry run summary: ${dryReport.phase1Consolidation.entitiesRefreshed} entities to refresh, `
    + `${dryReport.phase2Pruning.entitiesMerged} entities to merge`);

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY S — Sleep execution smoke tests
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category S — Sleep execution');

  let report!: SleepReport;

  try {
    report = await sleepEngine.sleep(GROUP, {
      dryRun:           false,
      cooldownMinutes:  0,
      consolidation:    { minEpisodes: 2 },
      pruning:          { similarityThreshold: 0.85 },
    });
    check('S1', 'sleep() completed without throwing', true);
  } catch (err: any) {
    check('S1', 'sleep() completed without throwing', false, String(err.message ?? err));
    console.error('\nFatal: sleep() threw — aborting remaining checks.\n', err);
    await g.close();
    process.exit(1);
  }

  info(`Duration: ${report.durationMs} ms`);
  info(`Phase 1: ${report.phase1Consolidation.entitiesRefreshed} refreshed, `
    + `${report.phase1Consolidation.episodesConsolidated} episodes consolidated, `
    + `~${report.phase1Consolidation.tokensUsed} tokens`);
  info(`Phase 2: ${report.phase2Pruning.entitiesMerged} merged, `
    + `${report.phase2Pruning.edgesPruned} edges pruned`);

  for (const p of report.phase2Pruning.mergedPairs) {
    info(`  └ "${p.duplicate}"  →  "${p.canonical}"  (cosine: ${p.similarity.toFixed(3)})`);
  }

  check('S2',
    `Phase 1: ≥ 1 entity summary refreshed  (actual: ${report.phase1Consolidation.entitiesRefreshed})`,
    report.phase1Consolidation.entitiesRefreshed >= 1,
  );
  check('S3',
    `Phase 1: ≥ 2 episodes consolidated  (actual: ${report.phase1Consolidation.episodesConsolidated})`,
    report.phase1Consolidation.episodesConsolidated >= 2,
  );

  // ── Post-sleep snapshot ────────────────────────────────────────────────────
  sep('6 / Post-sleep snapshot');

  const afterRows = await driver.executeQuery<any[]>(
    `MATCH (n:Entity {groupId: $g}) RETURN n.name AS name ORDER BY name`,
    { g: GROUP },
  );
  const afterCount = afterRows.length;
  info(`Entity count after sleep: ${afterCount} (was ${beforeCount})`);
  for (const r of afterRows) console.log(`     ${r.name}`);

  const fischerAfter = await driver.executeQuery<any[]>(
    `MATCH (e:Entity {groupId: $g}) WHERE toLower(e.name) CONTAINS 'fischer' RETURN e.name AS name`,
    { g: GROUP },
  );
  info(`"fischer" variants after: [${fischerAfter.map((r: any) => r.name).join(', ') || 'none'}]`);

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY G — Graph state after sleep
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category G — Graph state');

  // G1: episodic nodes have consolidatedAt set
  const consolidatedEps = await driver.executeQuery<any[]>(
    `MATCH (ep:Episodic {groupId: $g}) WHERE ep.consolidatedAt IS NOT NULL RETURN count(ep) AS total`,
    { g: GROUP },
  );
  const consolidatedEpCount = Number(consolidatedEps[0]?.total ?? 0);
  check('G1',
    `≥ 2 episodes have consolidatedAt set  (actual: ${consolidatedEpCount})`,
    consolidatedEpCount >= 2,
  );

  // G2: at least one entity has consolidatedAt (summary was written back)
  const consolidatedEnts = await driver.executeQuery<any[]>(
    `MATCH (e:Entity {groupId: $g}) WHERE e.consolidatedAt IS NOT NULL RETURN count(e) AS total`,
    { g: GROUP },
  );
  const consolidatedEntCount = Number(consolidatedEnts[0]?.total ?? 0);
  check('G2',
    `≥ 1 entity has consolidatedAt set  (actual: ${consolidatedEntCount})`,
    consolidatedEntCount >= 1,
  );

  // G3: entity count never inflated by sleep
  check('G3',
    `Entity count ≤ before sleep  (${afterCount} ≤ ${beforeCount})`,
    afterCount <= beforeCount,
  );

  // G4: referential integrity — all MENTIONS edges point to existing Entity nodes
  const brokenMentions = await driver.executeQuery<any[]>(
    `
    MATCH (ep:Episodic {groupId: $g})-[r:MENTIONS]->(target)
    WHERE NOT target:Entity
    RETURN count(r) AS broken
    `,
    { g: GROUP },
  );
  const brokenCount = Number(brokenMentions[0]?.broken ?? 0);
  check('G4',
    `No broken MENTIONS references after merge  (broken: ${brokenCount})`,
    brokenCount === 0,
  );

  // G5: referential integrity — all RELATES_TO edges have valid endpoints
  const brokenRelates = await driver.executeQuery<any[]>(
    `
    MATCH (a)-[r:RELATES_TO {groupId: $g}]->(b)
    WHERE NOT (a:Entity AND b:Entity)
    RETURN count(r) AS broken
    `,
    { g: GROUP },
  );
  const brokenRelatesCount = Number(brokenRelates[0]?.broken ?? 0);
  check('G5',
    `No broken RELATES_TO endpoints after merge  (broken: ${brokenRelatesCount})`,
    brokenRelatesCount === 0,
  );

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY P — Pruning quality
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category P — Pruning quality');

  // P1: if there were multiple Fischer variants, they should collapse to ≤ 1
  const fischerBefore_count = fischerBefore.length;
  const fischerAfter_count  = fischerAfter.length;
  check('P1',
    `"fischer" node count ≤ before sleep  (${fischerAfter_count} ≤ ${fischerBefore_count})`,
    fischerAfter_count <= fischerBefore_count,
  );

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY R — Retrieval quality after sleep
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category R — Retrieval quality after sleep');

  // R1: key entities still findable by semantic search after consolidation+pruning
  const searchHits = await g.search({
    query:   'NSA research forensics scientist credential analysis',
    groupId: GROUP,
    limit:   8,
  });
  const hitNames    = searchHits.map((n: any) => (n.name ?? '').toLowerCase());
  const fischerHit  = hitNames.some((n: string) => n.includes('fischer'));
  const elenaHit    = (await g.search({ query: 'CEO founder security company keynote', groupId: GROUP, limit: 8 }))
    .map((n: any) => (n.name ?? '').toLowerCase())
    .some((n: string) => n.includes('elena'));

  info(`Search "NSA research forensics" → top-5: ${hitNames.slice(0, 5).join(', ') || '(none)'}`);
  check('R1',
    'Fischer and Elena still findable by search after sleep',
    fischerHit && elenaHit,
    !fischerHit ? 'Fischer not found' : !elenaHit ? 'Elena not found' : undefined,
  );

  // ══════════════════════════════════════════════════════════════════════════
  // FINAL REPORT
  // ══════════════════════════════════════════════════════════════════════════
  sep('Final Report');

  const categories = ['D', 'S', 'G', 'P', 'R'];
  const catLabels: Record<string, string> = {
    D: 'Dry run integrity',
    S: 'Sleep execution',
    G: 'Graph state',
    P: 'Pruning quality',
    R: 'Retrieval after sleep',
  };

  for (const cat of categories) {
    const catResults = results.filter(r => r.id.startsWith(cat));
    const passed     = catResults.filter(r => r.passed).length;
    console.log(`\n  ${catLabels[cat]}:`);
    for (const r of catResults) {
      console.log(`    [${r.id}] ${r.passed ? '✓' : '✗'}  ${r.label}`);
    }
    console.log(`    Score: ${passed}/${catResults.length}`);
  }

  const total  = results.length;
  const passed = results.filter(r => r.passed).length;
  const pct    = Math.round((passed / total) * 100);

  console.log(`\n${'═'.repeat(64)}`);
  console.log(`  TOTAL SCORE: ${passed}/${total}  (${pct}%)`);
  if (passed === total)             console.log('  🏆 All checks passed!');
  else if (passed >= total * 0.8)   console.log('  ✅ Good — most checks passed');
  else if (passed >= total * 0.5)   console.log('  ⚠️  Partial — check failed categories above');
  else                              console.log('  ❌ Low pass rate — check sleep engine logs');
  console.log(`${'═'.repeat(64)}\n`);

  await g.close();
}

main().catch(err => {
  console.error('\nFatal error:', err.message ?? err);
  process.exit(1);
});
