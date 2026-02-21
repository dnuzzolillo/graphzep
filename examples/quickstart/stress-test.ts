/**
 * GraphZep Stress-Test — NovaTech Labs scenario
 *
 * Ingests ~20 interconnected episodes and evaluates graph quality
 * with 12 automatic PASS/FAIL checks across 4 categories:
 *
 *   A — Semantic search recall   (4 checks)
 *   B — Graph integrity          (3 checks)
 *   C — Temporality & negation   (3 checks)
 *   D — Traversal & graphExpand  (2 checks)
 *
 * Requirements:
 *   OPENAI_API_KEY=sk-...
 *   NEO4J_URI=bolt://localhost:7687
 *   NEO4J_USER=neo4j
 *   NEO4J_PASSWORD=password
 *
 * Run from /examples:
 *   npm run stress-test
 */

import { config } from 'dotenv';
import {
  Graphzep,
  Neo4jDriver,
  OpenAIClient,
  OpenAIEmbedder,
} from 'graphzep';

config();

const GROUP = 'stress-test-novatech';

// ── Formatting helpers ────────────────────────────────────────────────────────
const sep   = (title: string) =>
  console.log(`\n${'─'.repeat(64)}\n  ${title}\n${'─'.repeat(64)}`);
const ok    = (msg: string) => console.log(`  ✓ ${msg}`);
const info  = (msg: string) => console.log(`  · ${msg}`);
const fail  = (msg: string) => console.log(`  ✗ ${msg}`);

// ── Check result tracker ──────────────────────────────────────────────────────
interface CheckResult {
  id:      string;
  label:   string;
  passed:  boolean;
  detail?: string;
}

const results: CheckResult[] = [];

function check(id: string, label: string, passed: boolean, detail?: string): void {
  results.push({ id, label, passed, detail });
  const icon = passed ? '✓ PASS' : '✗ FAIL';
  const line = `  [${id}] ${icon}  ${label}`;
  if (passed) console.log(line);
  else        console.log(line + (detail ? `  → ${detail}` : ''));
}

// ── Episode corpus ─────────────────────────────────────────────────────────────
const EPISODES: string[] = [
  // E01-E05: People
  'Sarah Kim is the CEO and co-founder of NovaTech Labs. She founded the company in 2018.',
  'Marco Rossi is the CTO of NovaTech Labs. He leads all engineering efforts across product teams.',
  'Priya Patel is the Lead Engineer at NovaTech Labs. She reports directly to Marco Rossi.',
  'James Chen is a Data Scientist at NovaTech Labs. He is an expert in TensorFlow and Python.',
  'Lisa Wong is the Head of Design at NovaTech Labs. She works closely with Priya Patel on product UX.',

  // E06-E09: Projects & tech stack
  'NovaTech Labs develops Project Atlas, an AI-powered analytics platform built on TensorFlow and Python.',
  'Project Atlas uses Kubernetes for deployment orchestration and PostgreSQL as its primary database.',
  'CloudSync Inc is a strategic partner of NovaTech Labs, providing cloud infrastructure services.',
  'NovaTech Labs raised a Series B funding round of $30 million led by VentureX in 2023.',

  // E10-E13: More org relationships
  'Priya Patel serves as technical lead for Project Atlas since its inception.',
  'DataCore is a direct competitor of NovaTech Labs in the enterprise analytics market.',
  'NovaTech Labs launched Project Beacon in Q3 2023: a real-time data streaming service for enterprise clients.',
  'Project Beacon uses React for its frontend and integrates tightly with Project Atlas backend APIs.',

  // E14: Historical — James previously worked at DataCore
  'James Chen previously worked at DataCore as a senior analyst before joining NovaTech Labs in 2022. ' +
  'He left DataCore when he accepted the offer from NovaTech Labs.',

  // E15: Negated — Marco does NOT report to Sarah
  'Marco Rossi does NOT report directly to Sarah Kim on day-to-day technical decisions. ' +
  'The board of directors supervises overall technical strategy independently.',

  // E16: Historical — Lisa previously worked at CloudSync
  'Lisa Wong used to work at CloudSync Inc as a UX lead before joining NovaTech Labs. ' +
  'She transitioned to NovaTech Labs to build the design team from scratch.',

  // E17-E20: Additional relationships
  'Sarah Kim and David Park, a partner at VentureX, hold regular board meetings to review NovaTech Labs strategy.',
  'Priya Patel is actively mentoring junior engineers on Kubernetes best practices and container orchestration.',
  'NovaTech Labs plans to acquire a machine-learning startup called DeepMind Analytics to expand its AI portfolio.',
  'James Chen is leading the migration of Project Atlas from TensorFlow 1.x to TensorFlow 2.x to improve performance.',
];

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  console.log('\n🏋️  GraphZep Stress-Test — NovaTech Labs\n');

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

  await driver.createIndexes();
  ok('Neo4j connected and indexes verified');

  // ── 2. Cleanup previous run ────────────────────────────────────────────────
  sep('2 / Cleanup previous data');

  await driver.executeQuery(
    `MATCH (n {groupId: $g}) DETACH DELETE n`,
    { g: GROUP },
  );
  ok(`Cleaned all nodes with groupId="${GROUP}"`);

  // ── 3. Ingest 20 episodes ──────────────────────────────────────────────────
  sep(`3 / Ingesting ${EPISODES.length} episodes`);

  for (let i = 0; i < EPISODES.length; i++) {
    const episode = EPISODES[i];
    const label   = `E${String(i + 1).padStart(2, '0')}`;
    await g.addEpisode({ content: episode, groupId: GROUP });
    ok(`${label} ingested — ${episode.slice(0, 72)}…`);
  }

  // ── 4. Brief stabilisation pause ──────────────────────────────────────────
  sep('4 / Stabilisation pause (2 s)');
  await new Promise(r => setTimeout(r, 2000));
  ok('Ready for evaluation');

  // ── 5. Graph state snapshot ────────────────────────────────────────────────
  sep('5 / Graph state snapshot');

  const entities = await driver.executeQuery<any[]>(
    `MATCH (n:Entity {groupId: $g}) RETURN n.name AS name, n.entityType AS type ORDER BY name`,
    { g: GROUP },
  );
  info(`Entities in graph: ${entities.length}`);
  for (const e of entities) console.log(`     ${e.name} (${e.type})`);

  const edges = await driver.executeQuery<any[]>(
    `MATCH ()-[r:RELATES_TO {groupId: $g}]->() RETURN r.name AS rel, r.invalidAt AS inv, r.isNegated AS neg`,
    { g: GROUP },
  );
  info(`Edges in graph: ${edges.length}`);
  for (const r of edges) {
    const flags = [r.inv ? 'HISTORICAL' : '', r.neg ? 'NEGATED' : ''].filter(Boolean).join(', ');
    console.log(`     ${r.rel}${flags ? `  [${flags}]` : ''}`);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY A — Semantic search recall
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category A — Semantic search recall');

  type SearchQuery = { id: string; query: string; expected: string };
  const semanticChecks: SearchQuery[] = [
    { id: 'A1', query: 'machine learning data scientist',  expected: 'James Chen'    },
    { id: 'A2', query: 'cloud infrastructure partnership', expected: 'CloudSync Inc' },
    { id: 'A3', query: 'real-time data streaming service', expected: 'Project Beacon' },
    { id: 'A4', query: 'Series B venture funding round',   expected: 'VentureX'      },
  ];

  for (const sc of semanticChecks) {
    const hits = await g.search({ query: sc.query, groupId: GROUP, limit: 8 });
    const names = hits.map((n: any) => n.name ?? n.entityName ?? '');
    const found = names.some((n: string) => n.toLowerCase().includes(sc.expected.toLowerCase()));
    info(`[${sc.id}] query="${sc.query}" → top-8: ${names.slice(0, 5).join(', ') || '(none)'}`);
    check(sc.id, `"${sc.expected}" appears in top-8 results for: "${sc.query}"`, found);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY B — Graph integrity
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category B — Graph integrity');

  // B1: entity count ≥ 10
  const entityCount = entities.length;
  check('B1', `Entity count ≥ 10  (actual: ${entityCount})`, entityCount >= 10);

  // B2: edge count ≥ 6
  const edgeCount = edges.length;
  check('B2', `Edge count ≥ 6  (actual: ${edgeCount})`, edgeCount >= 6);

  // B3: at least one WORKS_AT-style relationship exists
  const worksAtEdges = await driver.executeQuery<any[]>(
    `MATCH ()-[r:RELATES_TO {groupId: $g}]->()
     WHERE toLower(r.name) CONTAINS 'work' OR toLower(r.name) CONTAINS 'employ'
        OR toLower(r.name) CONTAINS 'member' OR toLower(r.name) CONTAINS 'lead'
     RETURN count(r) AS total`,
    { g: GROUP },
  );
  const worksAtCount = Number(worksAtEdges[0]?.total ?? 0);
  check('B3', `At least one WORKS_AT/LEADS_AT-style edge exists  (actual: ${worksAtCount})`, worksAtCount >= 1);

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY C — Temporality & negation
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category C — Temporality & negation');

  // C1: at least one historical edge (invalidAt set)
  const historicalEdges = await driver.executeQuery<any[]>(
    `MATCH ()-[r:RELATES_TO {groupId: $g}]->()
     WHERE r.invalidAt IS NOT NULL
     RETURN count(r) AS total`,
    { g: GROUP },
  );
  const historicalCount = Number(historicalEdges[0]?.total ?? 0);
  check('C1', `Historical edges with invalidAt ≥ 1  (actual: ${historicalCount})`, historicalCount >= 1);

  // C2: no active (non-historical) direct edge from DataCore to James Chen
  const dataCoreChenActive = await driver.executeQuery<any[]>(
    `MATCH (a:Entity {groupId: $g})-[r:RELATES_TO]->(b:Entity {groupId: $g})
     WHERE (toLower(a.name) CONTAINS 'datacore' AND toLower(b.name) CONTAINS 'james chen')
        OR (toLower(a.name) CONTAINS 'james chen' AND toLower(b.name) CONTAINS 'datacore')
     AND r.invalidAt IS NULL
     RETURN count(r) AS total`,
    { g: GROUP },
  );
  const dataCoreChenCount = Number(dataCoreChenActive[0]?.total ?? 0);
  check(
    'C2',
    `No active DataCore↔James Chen edge (past employment correctly historicised)  (active: ${dataCoreChenCount})`,
    dataCoreChenCount === 0,
  );

  // C3: no active REPORTS_TO edge Marco → Sarah
  const marcoSarahReports = await driver.executeQuery<any[]>(
    `MATCH (a:Entity {groupId: $g})-[r:RELATES_TO]->(b:Entity {groupId: $g})
     WHERE (toLower(a.name) CONTAINS 'marco' AND toLower(b.name) CONTAINS 'sarah')
       AND (toLower(r.name) CONTAINS 'report' OR toLower(r.name) CONTAINS 'direct')
       AND r.invalidAt IS NULL
       AND (r.isNegated IS NULL OR r.isNegated = false)
     RETURN count(r) AS total`,
    { g: GROUP },
  );
  const marcoSarahCount = Number(marcoSarahReports[0]?.total ?? 0);
  check(
    'C3',
    `No active REPORTS_TO Marco→Sarah (negated relationship not stored)  (active: ${marcoSarahCount})`,
    marcoSarahCount === 0,
  );

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY D — Traversal & graphExpand
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category D — Traversal & graphExpand');

  // D1: traverse from Sarah Kim with maxHops=2 reaches ≥ 3 nodes
  const traversal = await g.traverse({
    startEntityName: 'Sarah Kim',
    maxHops: 2,
    direction: 'both',
    groupId: GROUP,
  });

  const traversedCount = traversal.nodes?.length ?? 0;
  info(`Traversal from "Sarah Kim" maxHops=2: ${traversedCount} nodes reached`);
  if (traversal.nodes?.length) {
    for (const n of traversal.nodes) console.log(`     ${(n as any).name}`);
  }
  check('D1', `Traversal from "Sarah Kim" (maxHops=2) reaches ≥ 3 nodes  (actual: ${traversedCount})`, traversedCount >= 3);

  // D2: graphExpand=true returns more nodes than plain search
  const plainHits    = await g.search({ query: 'analytics platform', groupId: GROUP, limit: 8 });
  const expandedHits = await g.search({
    query:       'analytics platform',
    groupId:     GROUP,
    limit:       8,
    graphExpand: true,
    expandHops:  1,
  });
  info(`Plain search "analytics platform":    ${plainHits.length} results`);
  info(`Expanded search "analytics platform": ${expandedHits.length} results`);
  check(
    'D2',
    `graphExpand=true returns more nodes than plain search  (${expandedHits.length} vs ${plainHits.length})`,
    expandedHits.length > plainHits.length,
  );

  // ══════════════════════════════════════════════════════════════════════════
  // FINAL REPORT
  // ══════════════════════════════════════════════════════════════════════════
  sep('Final Report');

  const categories = ['A', 'B', 'C', 'D'];
  const catLabels: Record<string, string> = {
    A: 'Semantic recall',
    B: 'Graph integrity',
    C: 'Temporality & negation',
    D: 'Traversal & graphExpand',
  };

  for (const cat of categories) {
    const catResults = results.filter(r => r.id.startsWith(cat));
    const passed     = catResults.filter(r => r.passed).length;
    console.log(`\n  ${catLabels[cat]}:`);
    for (const r of catResults) {
      const icon = r.passed ? '✓' : '✗';
      console.log(`    [${r.id}] ${icon}  ${r.label}`);
    }
    console.log(`    Score: ${passed}/${catResults.length}`);
  }

  const total  = results.length;
  const passed = results.filter(r => r.passed).length;
  const pct    = Math.round((passed / total) * 100);

  console.log(`\n${'═'.repeat(64)}`);
  console.log(`  TOTAL SCORE: ${passed}/${total}  (${pct}%)`);
  if (passed === total)        console.log('  🏆 All checks passed!');
  else if (passed >= total * 0.8) console.log('  ✅ Good — most checks passed');
  else if (passed >= total * 0.5) console.log('  ⚠️  Partial — review failed checks above');
  else                            console.log('  ❌ Low pass rate — check LLM extraction quality');
  console.log(`${'═'.repeat(64)}\n`);

  await g.close();
}

main().catch(err => {
  console.error('\nFatal error:', err.message ?? err);
  process.exit(1);
});
