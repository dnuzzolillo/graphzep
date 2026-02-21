/**
 * GraphZep Story QA — "The Meridian Protocol"
 *
 * Ingests a continuous 35-episode cybersecurity narrative and evaluates
 * graph quality with 21 automatic PASS/FAIL checks across 4 categories:
 *
 *   Q — Factual recall          (6 checks)
 *   R — Relationship retrieval  (6 checks)
 *   T — Temporal & negation     (6 checks)
 *   G — Graph traversal         (3 checks)
 *
 * Requirements:
 *   OPENAI_API_KEY=sk-...
 *   NEO4J_URI=bolt://localhost:7687
 *   NEO4J_USER=neo4j
 *   NEO4J_PASSWORD=password
 *
 * Run from /examples:
 *   npm run story-qa
 */

import { config } from 'dotenv';
import {
  Graphzep,
  Neo4jDriver,
  OpenAIClient,
  OpenAIEmbedder,
} from 'graphzep';

config();

const GROUP = 'story-qa-meridian';

// ── Formatting helpers ────────────────────────────────────────────────────────
const sep  = (title: string) =>
  console.log(`\n${'─'.repeat(64)}\n  ${title}\n${'─'.repeat(64)}`);
const ok   = (msg: string) => console.log(`  ✓ ${msg}`);
const info = (msg: string) => console.log(`  · ${msg}`);

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

// ── Episode corpus — "The Meridian Protocol" ──────────────────────────────────
//
// Cast (17 entities):
//   Elena Vasquez  — CEO, co-founder
//   James Park     — Co-founder, forensics lead → CSO
//   Raj Patel      — CTO, ex-NSA
//   Dr. Alan Fischer — Chief Scientist, Raj's ex-supervisor at NSA
//   Yuki Tanaka    — Head of Product, ex-Stripe
//   Marcus Cole    — Lead Engineer (wrongly suspected) → Principal Engineer
//   Tom Briggs     — Head of Sales, double agent
//   Nadia Hassan   — Junior Engineer → Senior Security Researcher
//   Victor Okafor  — Board member, Elena's mentor
//   Sophia Laurent — VC partner at Atlas Ventures
//   Derek Holt     — CEO of Nexus Systems (then resigned)
//   Meridian Security — the company
//   Nexus Systems  — competitor/antagonist
//   Atlas Ventures — VC firm
//   CipherTech     — prior employer of Elena & Derek
//   Meridian Shield — ML-powered endpoint security platform
//   Meridian Threat Intelligence Unit — internal red team
//
const EPISODES: string[] = [
  // ── Act I: Origins (E01–E07) ────────────────────────────────────────────────

  // E01
  'Elena Vasquez and James Park co-founded Meridian Security in 2019, both leaving CipherTech to start the company. ' +
  'Elena became CEO and James took the role of Chief Operating Officer, with a shared vision of building next-generation endpoint security.',

  // E02
  'Raj Patel joined Meridian as CTO in early 2020, recruited from the NSA Cybersecurity Division where he had served for 8 years ' +
  'as a senior cryptographic systems engineer. Elena Vasquez personally recruited him after seeing him speak at DEF CON.',

  // E03
  'Atlas Ventures led Meridian\'s $8M Series A in mid-2020. Sophia Laurent, a general partner at Atlas Ventures, ' +
  'joined Meridian\'s board of directors and chaired the compensation committee.',

  // E04
  'Victor Okafor joined Meridian\'s board as an independent director. A veteran of three successful cybersecurity exits, ' +
  'he became a trusted mentor and strategic advisor to CEO Elena Vasquez, meeting with her weekly.',

  // E05
  'Raj Patel recruited Dr. Alan Fischer from the NSA to lead Meridian\'s newly formed research division. ' +
  'Fischer had been Raj\'s direct supervisor at the NSA Cybersecurity Division and was widely regarded as one of ' +
  'the foremost authorities on adversarial machine learning.',

  // E06
  'Yuki Tanaka joined as Head of Product in late 2020, reporting directly to Elena Vasquez. ' +
  'Yuki had previously led product management at Stripe, where she shipped the Stripe Radar fraud detection system.',

  // E07
  'Marcus Cole was hired as Lead Engineer in early 2021, personally recommended by Meridian co-founder James Park. ' +
  'Cole had an exceptional reputation for secure systems design and had previously built zero-trust network architectures ' +
  'at a DARPA-affiliated research lab.',

  // ── Act II: Growth & launch (E08–E11) ───────────────────────────────────────

  // E08
  'Meridian launched "Meridian Shield" in Q2 2021 — an enterprise endpoint security platform powered by machine learning. ' +
  'The product used behavioural anomaly detection trained on petabytes of threat telemetry. ' +
  'Dr. Alan Fischer\'s research division contributed the core ML models at launch.',

  // E09
  'Tom Briggs joined Meridian as Head of Sales in 2021, hired to close Fortune 500 accounts. ' +
  'He was recruited by Elena Vasquez after a glowing referral from Victor Okafor, who had worked with Briggs at a previous company. ' +
  'Within six months Briggs had signed three enterprise contracts worth $4M combined.',

  // E10
  'Meridian\'s primary competitor is Nexus Systems, led by CEO Derek Holt. ' +
  'Elena Vasquez and Derek Holt had worked together at CipherTech for four years before each left to found their respective companies. ' +
  'Nexus Systems targeted the same Fortune 500 market as Meridian Shield.',

  // E11
  'Nadia Hassan joined Meridian as a junior engineer in late 2021, fresh from a computer science PhD at Carnegie Mellon. ' +
  'Marcus Cole volunteered to be her professional mentor and introduced her to the codebase over her first three months.',

  // ── Act III: The breach (E12–E18) ───────────────────────────────────────────

  // E12
  'In early 2022, Meridian\'s proprietary threat signatures — painstakingly compiled by Dr. Alan Fischer\'s team over two years — ' +
  'began appearing verbatim in Nexus Systems\' competing product. The discovery triggered an internal security review.',

  // E13
  'Nadia Hassan was the first to notice the anomaly. While auditing server performance metrics late one evening, ' +
  'she found a pattern of anomalous API calls in the logs, all originating from an internal service account with elevated privileges ' +
  'that had no legitimate business reason for those requests.',

  // E14
  'The initial breach investigation focused on Marcus Cole because the anomalous API calls were authenticated using ' +
  'credentials that were registered to his account. Cole was placed on administrative leave pending the investigation. ' +
  'He maintained his innocence throughout.',

  // E15
  'James Park formally took charge of the internal forensics investigation once the breach was confirmed. ' +
  'He assembled a small team including Nadia Hassan and brought in two outside incident-response consultants ' +
  'to ensure objectivity.',

  // E16
  'Dr. Alan Fischer led the technical analysis that proved Marcus Cole\'s API credentials had been silently cloned ' +
  'using a remote memory-scraping exploit targeting Meridian\'s developer workstations. ' +
  'Marcus Cole was fully and publicly exonerated. He returned from administrative leave the following Monday.',

  // E17
  'Digital forensics recovered encrypted communications showing that Tom Briggs had been secretly employed by Nexus Systems ' +
  'as a double agent from the day he joined Meridian in 2021. He had been systematically exfiltrating threat signature databases, ' +
  'customer lists, and product roadmaps — all delivered to Derek Holt through an encrypted dead-drop service.',

  // E18
  'Tom Briggs was immediately terminated and escorted from the building. Meridian Security filed a civil lawsuit against ' +
  'Nexus Systems in the Northern District of California, seeking $50M in damages for misappropriation of trade secrets. ' +
  'The complaint named Derek Holt personally as a co-conspirator.',

  // ── Act IV: Recovery (E19–E20) ──────────────────────────────────────────────

  // E19
  'In the immediate aftermath of the breach, Elena Vasquez personally led a company-wide security overhaul: ' +
  'all credentials were rotated, developer workstations were reimaged, and hardware security keys became mandatory for all staff. ' +
  'Raj Patel simultaneously conducted a full system-wide architectural audit, producing a 200-page remediation report.',

  // E20
  'Meridian closed a $25M Series B funding round led by Atlas Ventures in Q4 2022. ' +
  'Sophia Laurent personally championed the deal to the Atlas investment committee, arguing that Meridian\'s transparent ' +
  'handling of the breach and rapid recovery demonstrated exceptional operational maturity.',

  // ── Act V: Aftermath & consequences (E21–E28) ───────────────────────────────

  // E21
  'Nadia Hassan was promoted to Senior Security Researcher following her pivotal discovery of the breach indicators. ' +
  'She was formally transferred from the engineering organisation to Dr. Alan Fischer\'s research division, ' +
  'where she began specialising in insider-threat detection algorithms.',

  // E22
  'Marcus Cole, restored to full standing and promoted to Principal Engineer, led the complete rebuild of Meridian\'s ' +
  'internal API authentication layer. The new system used hardware-backed attestation tokens and mutual TLS, ' +
  'eliminating the credential-cloning attack vector that had been exploited against him.',

  // E23
  'James Park was formally appointed Chief Security Officer of Meridian Security, a new executive role created to reflect ' +
  'the expanded security mandate he had assumed during and after the breach. He continued to report directly to Elena Vasquez.',

  // E24
  'Derek Holt resigned as CEO of Nexus Systems in January 2023 under pressure from the Nexus board, ' +
  'who cited the mounting legal exposure from Meridian\'s lawsuit and Holt\'s named status as a co-conspirator. ' +
  'The board installed an interim CEO and hired an outside law firm to conduct an internal review.',

  // E25
  'Tom Briggs was indicted by a federal grand jury on charges of economic espionage and theft of trade secrets in Q1 2023. ' +
  'The Department of Justice cited the Meridian breach as one of the most damaging cases of corporate espionage ' +
  'in the cybersecurity industry in recent years.',

  // E26
  'Elena Vasquez delivered a keynote address at the RSA Conference in San Francisco in February 2023. ' +
  'Speaking to an audience of 4,000 security professionals, she outlined the lessons Meridian had learned from the breach — ' +
  'particularly the importance of insider-threat detection and zero-trust architecture — and received a standing ovation.',

  // E27
  'Victor Okafor played a central role in managing Meridian\'s investor and media relations throughout the federal prosecution, ' +
  'leveraging his personal relationships with financial journalists and his decades of experience in corporate crisis management. ' +
  'He helped Elena Vasquez frame the narrative around Meridian\'s strength rather than its victimhood.',

  // E28
  'Raj Patel formally established the Meridian Threat Intelligence Unit in mid-2023 — a dedicated internal red team ' +
  'tasked with continuous adversarial simulation, vulnerability hunting, and proactive threat intelligence gathering. ' +
  'The unit reported directly to Raj and operated under a separate security clearance framework.',

  // ── Act VI: Research, product & recognition (E29–E35) ───────────────────────

  // E29
  'Dr. Alan Fischer and Nadia Hassan co-authored a peer-reviewed paper entitled "Silent Cloning: Remote Credential ' +
  'Exfiltration via User-Space Memory Scraping" published in IEEE Security & Privacy. ' +
  'The paper detailed the forensic methodology used in the Meridian breach investigation and proposed new detection heuristics.',

  // E30
  'Yuki Tanaka led a complete ground-up redesign of the Meridian Shield user interface in early 2023. ' +
  'Working closely with Marcus Cole\'s engineering team, she incorporated eighteen months of customer feedback ' +
  'and reduced mean time to detection for analysts by 40 percent in beta trials.',

  // E31
  'Sophia Laurent personally introduced Elena Vasquez to the CISOs of three Fortune 100 companies at the Atlas Ventures ' +
  'annual portfolio summit in March 2023. All three initiated procurement discussions within sixty days, ' +
  'representing Meridian\'s largest potential enterprise contracts to date.',

  // E32
  'The Meridian Threat Intelligence Unit, led by Raj Patel, detected and neutralised a second intrusion attempt in ' +
  'June 2023. Forensic attribution linked the attacker to a former contractor who had worked at Nexus Systems, ' +
  'providing additional evidence for Meridian\'s civil lawsuit.',

  // E33
  'Meridian Security was named to the Gartner Magic Quadrant for Endpoint Protection Platforms in Q3 2023, ' +
  'positioned as a Visionary. Elena Vasquez credited the entire team\'s resilience during the breach recovery for ' +
  'demonstrating the operational maturity Gartner evaluators valued most.',

  // E34
  'Tom Briggs pleaded guilty to two counts of federal economic espionage in October 2023. ' +
  'As part of his plea agreement, he provided sworn testimony confirming that Derek Holt had personally directed ' +
  'the data theft campaign from its inception. Sentencing was scheduled for the following spring.',

  // E35
  'Elena Vasquez and Victor Okafor co-authored an op-ed published in the Wall Street Journal calling for ' +
  'stronger federal protections against corporate espionage targeting technology startups. ' +
  'They cited the Meridian breach as a case study and proposed a voluntary industry framework for ' +
  'insider-threat information sharing between peer companies.',
];

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  console.log('\n📖  GraphZep Story QA — "The Meridian Protocol"\n');

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

  // ── 3. Ingest 35 episodes ──────────────────────────────────────────────────
  sep(`3 / Ingesting ${EPISODES.length} episodes`);

  for (let i = 0; i < EPISODES.length; i++) {
    const episode = EPISODES[i];
    const label   = `E${String(i + 1).padStart(2, '0')}`;
    await g.addEpisode({ content: episode, groupId: GROUP });
    ok(`${label} ingested — ${episode.slice(0, 72)}…`);
  }

  // ── 4. Stabilisation pause ─────────────────────────────────────────────────
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

  // ── Helper: semantic search check ─────────────────────────────────────────
  async function searchCheck(
    id: string,
    query: string,
    expectedEntities: string[],
  ): Promise<void> {
    const hits  = await g.search({ query, groupId: GROUP, limit: 8 });
    const names = hits.map((n: any) => (n.name ?? n.entityName ?? '').toLowerCase());
    const passed = expectedEntities.some(e =>
      names.some((n: string) => n.includes(e.toLowerCase())),
    );
    const expectedLabel = expectedEntities.join(' or ');
    info(`[${id}] query="${query}" → top-8: ${names.slice(0, 5).join(', ') || '(none)'}`);
    check(id, `"${expectedLabel}" in top-8 for: "${query}"`, passed);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY Q — Factual recall
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category Q — Factual recall');

  await searchCheck('Q1', 'machine learning endpoint security platform product launch',
    ['Meridian Shield']);
  await searchCheck('Q2', 'NSA cybersecurity division CTO recruiter executive',
    ['Raj Patel']);
  await searchCheck('Q3', 'Series A venture capital board member investor funding',
    ['Sophia Laurent']);
  await searchCheck('Q4', 'junior engineer first job mentor 2021 joined company',
    ['Nadia Hassan']);
  await searchCheck('Q5', 'Gartner Magic Quadrant endpoint protection visionary recognition',
    ['Meridian Security', 'Meridian Shield']);
  await searchCheck('Q6', 'RSA Conference keynote speech security industry address audience',
    ['Elena Vasquez']);

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY R — Relationship retrieval
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category R — Relationship retrieval');

  await searchCheck('R1', 'supervisor mentor NSA research director scientist adversarial ML',
    ['Dr. Alan Fischer', 'Alan Fischer']);
  await searchCheck('R2', 'competitor CEO rival company former colleague same employer',
    ['Derek Holt']);
  await searchCheck('R3', 'independent board director advisor mentor founder CEO',
    ['Victor Okafor']);
  await searchCheck('R4', 'forensics investigation breach internal security lead',
    ['James Park']);
  await searchCheck('R5', 'chief security officer CSO appointed executive promoted',
    ['James Park']);
  await searchCheck('R6', 'red team threat intelligence unit internal adversarial proactive vulnerability',
    ['Raj Patel', 'Meridian Threat Intelligence Unit']);

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY T — Temporal & negation
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category T — Temporal & negation');

  await searchCheck('T1', 'wrongly accused credentials stolen cloned exonerated cleared innocent',
    ['Marcus Cole']);
  await searchCheck('T2', 'double agent spy corporate espionage secretly working competitor leaking',
    ['Tom Briggs']);
  await searchCheck('T3', 'security overhaul audit protocol restructure after breach incident',
    ['Elena Vasquez', 'Raj Patel']);
  await searchCheck('T4', 'Series B second funding round recovery investor champion',
    ['Atlas Ventures', 'Sophia Laurent']);
  await searchCheck('T5', 'competitor CEO resigned pressure legal board scandal named co-conspirator',
    ['Derek Holt']);
  await searchCheck('T6', 'federal indictment economic espionage guilty plea criminal charges sentenced',
    ['Tom Briggs']);

  // ══════════════════════════════════════════════════════════════════════════
  // CATEGORY G — Graph traversal
  // ══════════════════════════════════════════════════════════════════════════
  sep('Category G — Graph traversal');

  // G1: traverse from Nadia Hassan maxHops=2 → ≥ 3 reachable nodes
  const t1 = await g.traverse({
    startEntityName: 'Nadia Hassan',
    maxHops: 2,
    direction: 'both',
    groupId: GROUP,
  });
  const t1Count = t1.nodes?.length ?? 0;
  info(`Traversal from "Nadia Hassan" maxHops=2: ${t1Count} nodes reached`);
  if (t1.nodes?.length) {
    for (const n of t1.nodes) console.log(`     ${(n as any).name}`);
  }
  check('G1', `"Nadia Hassan" traversal (maxHops=2) reaches ≥ 3 nodes  (actual: ${t1Count})`, t1Count >= 3);

  // G2: traverse from Tom Briggs maxHops=2 → ≥ 3 reachable nodes
  const t2 = await g.traverse({
    startEntityName: 'Tom Briggs',
    maxHops: 2,
    direction: 'both',
    groupId: GROUP,
  });
  const t2Count = t2.nodes?.length ?? 0;
  info(`Traversal from "Tom Briggs" maxHops=2: ${t2Count} nodes reached`);
  if (t2.nodes?.length) {
    for (const n of t2.nodes) console.log(`     ${(n as any).name}`);
  }
  check('G2', `"Tom Briggs" traversal (maxHops=2) reaches ≥ 3 nodes  (actual: ${t2Count})`, t2Count >= 3);

  // G3: traverse from Dr. Alan Fischer maxHops=2 → ≥ 4 reachable nodes
  // (Fischer connects to: Raj Patel, Meridian Security, Nadia Hassan, Meridian Shield, NSA, etc.)
  const t3 = await g.traverse({
    startEntityName: 'Dr. Alan Fischer',
    maxHops: 2,
    direction: 'both',
    groupId: GROUP,
  });
  const t3Count = t3.nodes?.length ?? 0;
  info(`Traversal from "Dr. Alan Fischer" maxHops=2: ${t3Count} nodes reached`);
  if (t3.nodes?.length) {
    for (const n of t3.nodes) console.log(`     ${(n as any).name}`);
  }
  check('G3', `"Dr. Alan Fischer" traversal (maxHops=2) reaches ≥ 4 nodes  (actual: ${t3Count})`, t3Count >= 4);

  // ══════════════════════════════════════════════════════════════════════════
  // FINAL REPORT
  // ══════════════════════════════════════════════════════════════════════════
  sep('Final Report');

  const categories = ['Q', 'R', 'T', 'G'];
  const catLabels: Record<string, string> = {
    Q: 'Factual recall',
    R: 'Relationship retrieval',
    T: 'Temporal & negation',
    G: 'Graph traversal',
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
  if (passed === total)             console.log('  🏆 All checks passed!');
  else if (passed >= total * 0.8)   console.log('  ✅ Good — most checks passed');
  else if (passed >= total * 0.5)   console.log('  ⚠️  Partial — review failed checks above');
  else                              console.log('  ❌ Low pass rate — check LLM extraction quality');
  console.log(`${'═'.repeat(64)}\n`);

  await g.close();
}

main().catch(err => {
  console.error('\nFatal error:', err.message ?? err);
  process.exit(1);
});
