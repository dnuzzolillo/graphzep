/**
 * GraphZep Memory Demo — Neo4j + OpenAI
 *
 * Demuestra el pipeline de ingestión mejorado:
 *   - Filtrado por confianza
 *   - Relaciones negadas (isNegated)
 *   - Relaciones históricas (temporalValidity)
 *   - Búsqueda con expansión por grafo (graphExpand)
 *   - Traversal de grafo (traverse)
 *
 * Requisitos:
 *   OPENAI_API_KEY=sk-...
 *   NEO4J_URI=bolt://localhost:7687
 *   NEO4J_USER=neo4j
 *   NEO4J_PASSWORD=password
 *
 * Ejecutar desde /examples:
 *   npm run memory-demo
 */

import { config } from 'dotenv';
import {
  Graphzep,
  Neo4jDriver,
  OpenAIClient,
  OpenAIEmbedder,
} from 'graphzep';

config();

const GROUP = 'memory-demo';

const sep = (title: string) =>
  console.log(`\n${'─'.repeat(60)}\n  ${title}\n${'─'.repeat(60)}`);

const ok  = (msg: string) => console.log(`  ✓ ${msg}`);
const info = (msg: string) => console.log(`  · ${msg}`);

// ─────────────────────────────────────────────────────────────
async function main() {
  console.log('\n🧠  GraphZep Memory Demo\n');

  // ── Setup ──────────────────────────────────────────────────
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

  // Limpiar datos del demo anterior
  await driver.executeQuery(
    `MATCH (n {groupId: $g}) DETACH DELETE n`,
    { g: GROUP },
  );
  ok('Neo4j conectado y datos de demo limpiados');

  // ── Escenario 1: Ingestión normal ──────────────────────────
  sep('Escenario 1 — Ingestión con entidades claras');

  await g.addEpisode({
    content: 'Alice Chen is a senior machine learning engineer at TechCorp. She leads the computer vision team.',
    groupId: GROUP,
  });
  ok('Episodio 1 añadido: Alice en TechCorp');

  await g.addEpisode({
    content: 'Bob Martinez is the engineering manager at TechCorp. He oversees multiple product teams.',
    groupId: GROUP,
  });
  ok('Episodio 2 añadido: Bob en TechCorp');

  await g.addEpisode({
    content: 'Alice Chen and Bob Martinez work closely together on TechCorp product roadmap.',
    groupId: GROUP,
  });
  ok('Episodio 3 añadido: relación Alice–Bob');

  // ── Escenario 2: Relación histórica ────────────────────────
  sep('Escenario 2 — Relación histórica (temporalValidity: historical)');

  await g.addEpisode({
    content: 'Alice Chen used to work at StartupXYZ before joining TechCorp. She left StartupXYZ in 2022.',
    groupId: GROUP,
  });
  ok('Episodio 4 añadido: Alice solía trabajar en StartupXYZ');
  info('El LLM debería extraer WORKED_AT con temporalValidity: "historical"');
  info('→ La arista se guardará con invalidAt = ahora');

  // ── Escenario 3: Relación negada ───────────────────────────
  sep('Escenario 3 — Relación negada (isNegated: true)');

  await g.addEpisode({
    content: 'Alice Chen does NOT have access to the production database. Only DevOps team members have that access.',
    groupId: GROUP,
  });
  ok('Episodio 5 añadido: Alice NO tiene acceso a producción');
  info('El LLM debería extraer HAS_ACCESS con isNegated: true');
  info('→ No se creará ninguna arista para esa relación');

  // ── Verificar el grafo ─────────────────────────────────────
  sep('Estado del grafo tras ingestión');

  const entities = await driver.executeQuery<any[]>(
    `MATCH (n:Entity {groupId: $g}) RETURN n.name AS name, n.entityType AS type ORDER BY name`,
    { g: GROUP },
  );
  info(`Entidades guardadas: ${entities.length}`);
  for (const e of entities) console.log(`     ${e.name} (${e.type})`);

  const edges = await driver.executeQuery<any[]>(
    `MATCH ()-[r:RELATES_TO {groupId: $g}]->() RETURN r.name AS rel, r.invalidAt AS inv`,
    { g: GROUP },
  );
  info(`Aristas guardadas: ${edges.length}`);
  for (const r of edges) {
    const historical = r.inv ? ' [HISTÓRICA — invalidAt set]' : '';
    console.log(`     ${r.rel}${historical}`);
  }

  // ── Escenario 4: Búsqueda normal vs. con expansión ─────────
  sep('Escenario 4 — Search: normal vs. graphExpand');

  const plain = await g.search({ query: 'machine learning engineer', groupId: GROUP, limit: 5 });
  info(`Búsqueda normal → ${plain.length} resultado(s):`);
  for (const n of plain) console.log(`     ${n.name}`);

  const expanded = await g.search({
    query: 'machine learning engineer',
    groupId: GROUP,
    limit: 5,
    graphExpand: true,
    expandHops: 1,
  });
  info(`Búsqueda con graphExpand:true → ${expanded.length} resultado(s):`);
  for (const n of expanded) console.log(`     ${n.name}`);

  if (expanded.length > plain.length)
    ok('graphExpand retornó más nodos (vecinos incluidos)');

  // ── Escenario 5: Traversal ─────────────────────────────────
  sep('Escenario 5 — Traversal desde Alice');

  const traversal = await g.traverse({
    startEntityName: 'Alice Chen',
    maxHops: 2,
    direction: 'both',
    groupId: GROUP,
  });

  if (traversal.start) {
    ok(`Nodo inicio: ${traversal.start.name}`);
    info(`Nodos alcanzables: ${traversal.nodes.length}`);
    for (const n of traversal.nodes) console.log(`     ${n.name}`);
    info(`Aristas en el subgrafo: ${traversal.edges.length}`);
    for (const e of traversal.edges) console.log(`     ${e.name}`);
  } else {
    info('Alice Chen no encontrada en el grafo');
  }

  // ── Fin ────────────────────────────────────────────────────
  sep('Resumen');
  ok(`Entidades en grafo:      ${entities.length}`);
  ok(`Aristas totales:         ${edges.length}`);
  ok(`Aristas históricas:      ${edges.filter((r: any) => r.inv).length} (con invalidAt)`);
  ok(`graphExpand extra nodos: ${expanded.length - plain.length}`);
  ok(`Nodos desde traverse:    ${traversal.nodes.length}`);

  await g.close();
  console.log('\n✅ Demo completado\n');
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
