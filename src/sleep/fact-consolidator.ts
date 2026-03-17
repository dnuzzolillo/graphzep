/**
 * Sleep Engine — Fact Consolidator
 *
 * Inspired by Mem0's memory consolidation pipeline, this extracts discrete
 * facts from unconsolidated episodes and maintains them as first-class Fact
 * nodes in the knowledge graph.
 *
 * Pipeline per batch of episodes:
 *   1. Extract discrete facts from episode text (LLM call)
 *   2. For each fact, vector-search existing Fact nodes (top 5)
 *   3. Ask LLM to decide: ADD / UPDATE / DELETE / NONE
 *   4. Execute the action on the graph
 *
 * Fact nodes have their own embeddings and are included in search results,
 * providing consolidated, deduplicated knowledge alongside entities and episodes.
 */

import { v4 as uuidv4 } from 'uuid';
import { z } from 'zod';
import type { GraphDriver } from '../types/index.js';
import type { BaseLLMClient } from '../llm/client.js';
import type { BaseEmbedderClient } from '../embedders/client.js';
import type { SleepOptions } from './types.js';

// ── Schemas ──────────────────────────────────────────────────────────────────

const FactExtractionSchema = z.object({
  facts: z.array(z.object({
    content: z.string().describe('Self-contained atomic proposition with explicit subject'),
    temporalType: z.enum(['STATIC', 'DYNAMIC', 'ATEMPORAL']).describe(
      'STATIC: enduring truth unlikely to change (e.g. "María works at Transportes del Norte"). ' +
      'DYNAMIC: time-sensitive fact that may change (e.g. "The rate to Monterrey is $4,500 MXN"). ' +
      'ATEMPORAL: procedural/definitional knowledge (e.g. "ISC stands for Import Security Charge").',
    ),
  })).describe('Atomic facts with temporal classification'),
});

const FactDecisionSchema = z.object({
  decisions: z.array(z.object({
    factIndex: z.number(),
    action: z.enum(['ADD', 'UPDATE', 'DELETE', 'NONE']),
    existingFactId: z.string().optional().describe('UUID of existing fact to UPDATE or DELETE'),
    mergedContent: z.string().optional().describe('New content for ADD or UPDATE'),
    temporalType: z.enum(['STATIC', 'DYNAMIC', 'ATEMPORAL']).optional().describe('Temporal type for ADD/UPDATE'),
  })),
});

// ── Report ───────────────────────────────────────────────────────────────────

export interface FactConsolidationReport {
  factsAdded: number;
  factsUpdated: number;
  factsDeleted: number;
  factsSkipped: number;
  episodesProcessed: number;
  tokensUsed: number;
}

// ── Consolidator ─────────────────────────────────────────────────────────────

export class FactConsolidator {
  constructor(
    private readonly driver: GraphDriver,
    private readonly llm: BaseLLMClient,
    private readonly embedder: BaseEmbedderClient,
  ) {}

  /**
   * Inline fact extraction for a single episode — called from addEpisode().
   * Extracts atomic facts, deduplicates against existing Fact nodes, and stores them.
   * Marks the episode as fact-consolidated when done.
   */
  async extractForEpisode(
    episodeUuid: string,
    content: string,
    groupId: string,
  ): Promise<{ added: number; updated: number; deleted: number; skipped: number }> {
    const result = { added: 0, updated: 0, deleted: 0, skipped: 0 };

    // Step 1: Extract atomic facts with temporal classification
    let extractedFacts: Array<{ content: string; temporalType: string }>;
    try {
      const extraction = await this.llm.generateStructuredResponse(
        this.buildExtractionPrompt(content),
        FactExtractionSchema,
      );
      extractedFacts = extraction.facts
        .filter(f => f.content && f.content.trim().length > 10)
        .map(f => ({ content: f.content!, temporalType: f.temporalType ?? 'DYNAMIC' }));
    } catch {
      return result;
    }
    if (extractedFacts.length === 0) {
      await this.markFactConsolidated([episodeUuid]);
      return result;
    }
    const factTexts = extractedFacts.map(f => f.content);

    // Step 2: Find similar existing facts (top-10 per fact)
    const existingFacts = await this.findSimilarFacts(factTexts, groupId);

    // Step 3: Decide ADD/UPDATE/DELETE/NONE
    let decisions: z.infer<typeof FactDecisionSchema>['decisions'];
    try {
      const decisionResult = await this.llm.generateStructuredResponse(
        this.buildDecisionPrompt(extractedFacts, existingFacts),
        FactDecisionSchema,
      );
      decisions = decisionResult.decisions;
    } catch {
      return result;
    }

    // Step 4: Execute — DYNAMIC facts auto-UPDATE existing same-topic facts
    for (const decision of decisions) {
      if (decision.factIndex < 0 || decision.factIndex >= extractedFacts.length) continue;
      const temporalType = decision.temporalType ?? extractedFacts[decision.factIndex].temporalType;
      switch (decision.action) {
        case 'ADD': {
          await this.addFact(decision.mergedContent ?? factTexts[decision.factIndex], groupId, [episodeUuid], temporalType);
          result.added++;
          break;
        }
        case 'UPDATE': {
          if (decision.existingFactId && decision.mergedContent) {
            await this.updateFact(decision.existingFactId, decision.mergedContent, temporalType);
            result.updated++;
          }
          break;
        }
        case 'DELETE': {
          if (decision.existingFactId) {
            await this.deleteFact(decision.existingFactId);
            result.deleted++;
          }
          break;
        }
        case 'NONE':
          result.skipped++;
          break;
      }
    }

    await this.markFactConsolidated([episodeUuid]);
    return result;
  }

  async run(groupId: string, options: SleepOptions): Promise<FactConsolidationReport> {
    const cooldown = options.cooldownMinutes ?? 5;
    const maxEpisodes = options.factConsolidation?.maxEpisodes ?? 50;
    const batchSize = options.factConsolidation?.batchSize ?? 10;
    const dryRun = options.dryRun ?? false;

    const report: FactConsolidationReport = {
      factsAdded: 0, factsUpdated: 0, factsDeleted: 0, factsSkipped: 0,
      episodesProcessed: 0, tokensUsed: 0,
    };

    // Fetch unconsolidated episodes (those without factConsolidatedAt)
    const episodes = await this.fetchUnconsolidatedEpisodes(groupId, cooldown, maxEpisodes);
    if (episodes.length === 0) return report;

    // Process in batches
    for (let i = 0; i < episodes.length; i += batchSize) {
      const batch = episodes.slice(i, i + batchSize);
      const batchText = batch
        .map((ep, j) => `[${j + 1}] ${ep.content}`)
        .join('\n\n');

      // Step 1: Extract facts with temporal classification
      let extractedFacts: Array<{ content: string; temporalType: string }>;
      try {
        const extraction = await this.llm.generateStructuredResponse(
          this.buildExtractionPrompt(batchText),
          FactExtractionSchema,
        );
        extractedFacts = extraction.facts
          .filter(f => f.content && f.content.trim().length > 10)
          .map(f => ({ content: f.content!, temporalType: f.temporalType ?? 'DYNAMIC' }));
      } catch {
        continue;
      }
      report.tokensUsed += estimateTokens(batchText) + 500;

      if (extractedFacts.length === 0) {
        if (!dryRun) await this.markFactConsolidated(batch.map(e => e.uuid));
        report.episodesProcessed += batch.length;
        continue;
      }
      const factTexts = extractedFacts.map(f => f.content);

      // Step 2: Find similar existing facts (top-10 per fact)
      const existingFacts = await this.findSimilarFacts(factTexts, groupId);

      // Step 3: Ask LLM to decide ADD/UPDATE/DELETE/NONE
      let decisions: z.infer<typeof FactDecisionSchema>['decisions'];
      try {
        const decisionResult = await this.llm.generateStructuredResponse(
          this.buildDecisionPrompt(extractedFacts, existingFacts),
          FactDecisionSchema,
        );
        decisions = decisionResult.decisions;
      } catch {
        continue;
      }
      report.tokensUsed += estimateTokens(JSON.stringify(existingFacts)) + 800;

      // Step 4: Execute decisions
      if (!dryRun) {
        for (const decision of decisions) {
          if (decision.factIndex < 0 || decision.factIndex >= extractedFacts.length) continue;
          const temporalType = decision.temporalType ?? extractedFacts[decision.factIndex].temporalType;

          switch (decision.action) {
            case 'ADD': {
              const content = decision.mergedContent ?? factTexts[decision.factIndex];
              await this.addFact(content, groupId, batch.map(e => e.uuid), temporalType);
              report.factsAdded++;
              break;
            }
            case 'UPDATE': {
              if (decision.existingFactId && decision.mergedContent) {
                await this.updateFact(decision.existingFactId, decision.mergedContent, temporalType);
                report.factsUpdated++;
              }
              break;
            }
            case 'DELETE': {
              if (decision.existingFactId) {
                await this.deleteFact(decision.existingFactId);
                report.factsDeleted++;
              }
              break;
            }
            case 'NONE':
              report.factsSkipped++;
              break;
          }
        }

        await this.markFactConsolidated(batch.map(e => e.uuid));
      } else {
        for (const d of decisions) {
          switch (d.action) {
            case 'ADD': report.factsAdded++; break;
            case 'UPDATE': report.factsUpdated++; break;
            case 'DELETE': report.factsDeleted++; break;
            case 'NONE': report.factsSkipped++; break;
          }
        }
      }

      report.episodesProcessed += batch.length;
    }

    return report;
  }

  // ── Prompts ──────────────────────────────────────────────────────────────────

  private buildExtractionPrompt(episodeText: string): string {
    return `You are a Personal Information Organizer for a logistics/shipping operations system.

Your task is to decompose the following episodes into ATOMIC PROPOSITIONS — minimal, self-contained statements that each express exactly ONE piece of information.

## Proposition rules (Dense-X style)

1. Every proposition MUST include an explicit subject (no pronouns, no implicit references). Bad: "Ya salió de bodega" → Good: "El pedido #8842 ya salió de bodega de Querétaro"
2. Each proposition captures ONE datum — if it has "and", "also", "además", split it.
3. Under 20 words. Short, specific propositions produce the best vector search matches.
4. Preserve ALL identifiers: order numbers, AWBs, dates, amounts, emails, names.
5. Write in the same language as the input.
6. Skip trivial interactions (greetings, "ok", "gracias", acknowledgments).

## Temporal classification

Classify each fact as one of:
- **STATIC**: Enduring truth unlikely to change (identity, roles, relationships). Example: "María López trabaja en Transportes del Norte"
- **DYNAMIC**: Time-sensitive fact that may be superseded by newer info (prices, statuses, ETAs). Example: "La tarifa de consolidados a Monterrey es $4,500 MXN por tarima"
- **ATEMPORAL**: Procedural, definitional, or general knowledge (how things work, what acronyms mean). Example: "ISC stands for Import Security Charge"

## Few-shot examples

Input: "Hablé con María López de Transportes del Norte. Dice que la tarifa para consolidados a Monterrey subió a $4,500 MXN por tarima a partir del 15 de marzo. También me confirmó que el pedido #8842 ya salió de bodega y llega mañana."
Output: {"facts": [
  {"content": "María López trabaja en Transportes del Norte", "temporalType": "STATIC"},
  {"content": "La tarifa de consolidados a Monterrey de Transportes del Norte es $4,500 MXN por tarima", "temporalType": "DYNAMIC"},
  {"content": "La tarifa de consolidados a Monterrey aplica desde el 15 de marzo", "temporalType": "DYNAMIC"},
  {"content": "El pedido #8842 ya salió de bodega", "temporalType": "DYNAMIC"},
  {"content": "El pedido #8842 tiene ETA de llegada para mañana", "temporalType": "DYNAMIC"}
]}

Input: "El AWB 176-4892-7731 de DHL tiene un retraso en aduana. Carlos Méndez del despacho aduanal dice que falta el certificado fitosanitario. El cliente es Agroinsumos del Bajío."
Output: {"facts": [
  {"content": "El AWB 176-4892-7731 es un envío por DHL", "temporalType": "STATIC"},
  {"content": "El AWB 176-4892-7731 está retenido en aduana por falta de documentos", "temporalType": "DYNAMIC"},
  {"content": "El AWB 176-4892-7731 requiere certificado fitosanitario para liberación", "temporalType": "DYNAMIC"},
  {"content": "Carlos Méndez es agente del despacho aduanal", "temporalType": "STATIC"},
  {"content": "El cliente del AWB 176-4892-7731 es Agroinsumos del Bajío", "temporalType": "STATIC"}
]}

Input: "Email sent from apabot@apaair.com to alvaro@apaair.com, accounting@apaair.com — subject 'Status Orders DHL today'. SMTP via smtp.fatcow.com. Email checker PASS."
Output: {"facts": [
  {"content": "apabot@apaair.com es la dirección de envío de emails de APA Air", "temporalType": "STATIC"},
  {"content": "Los reportes de estado se envían a alvaro@apaair.com y accounting@apaair.com", "temporalType": "STATIC"},
  {"content": "El servidor SMTP de APA Air es smtp.fatcow.com", "temporalType": "STATIC"},
  {"content": "El email checker verificó exitosamente el envío con resultado PASS", "temporalType": "DYNAMIC"}
]}

Input: "Customs release BFV-0943641-1 on March 14. Paid $124 ISC via PayCargo ID 25528567. Cargo ready for pickup at NAS MIA."
Output: {"facts": [
  {"content": "Customs release BFV-0943641-1 fue emitido el 14 de marzo", "temporalType": "DYNAMIC"},
  {"content": "Se pagó $124 de ISC vía PayCargo con ID 25528567", "temporalType": "DYNAMIC"},
  {"content": "La carga está lista para pickup en NAS MIA después del customs release", "temporalType": "DYNAMIC"},
  {"content": "ISC es el cargo de seguridad de importación (Import Security Charge)", "temporalType": "ATEMPORAL"},
  {"content": "PayCargo es la plataforma usada para pago de ISC", "temporalType": "STATIC"}
]}

Episodes:
${episodeText}

Return JSON with the extracted facts.`;
  }

  private buildDecisionPrompt(
    newFacts: Array<{ content: string; temporalType: string }>,
    existingFacts: Array<{ uuid: string; content: string; similarity: number; temporalType?: string }>,
  ): string {
    const newFactsList = newFacts
      .map((f, i) => `  [${i}] (${f.temporalType}) ${f.content}`)
      .join('\n');

    const existingList = existingFacts.length > 0
      ? existingFacts.map(f => `  [${f.uuid}]${f.temporalType ? ` (${f.temporalType})` : ''} ${f.content}`).join('\n')
      : '  (no existing facts)';

    return `You are maintaining a knowledge base of atomic facts. For each new fact, decide what to do by comparing it against existing facts.

## Temporal types and their consolidation behavior

- **DYNAMIC** facts (prices, statuses, ETAs) REPLACE older DYNAMIC facts on the same topic. When you see a new DYNAMIC fact that covers the same entity/topic as an existing DYNAMIC fact, prefer UPDATE (or DELETE+ADD if the old one is now false).
- **STATIC** facts (identities, roles, relationships) only change rarely. UPDATE only if the new info genuinely adds detail. Use NONE if the info is already captured.
- **ATEMPORAL** facts (definitions, procedures) are almost never superseded. Use NONE unless the definition itself has changed.

EXISTING FACTS in the knowledge base:
${existingList}

NEW FACTS extracted from recent activity:
${newFactsList}

For each new fact (by index), choose one action:
- **ADD**: Genuinely new information not covered by any existing fact. Set mergedContent to the fact text.
- **UPDATE**: An existing fact covers the same topic but the new info adds detail, corrects it, or supersedes it (especially for DYNAMIC facts). Set existingFactId and mergedContent.
- **DELETE**: The new fact directly contradicts an existing fact, making the old one false. Set existingFactId. Also ADD the new fact separately.
- **NONE**: The new fact is already captured by an existing fact. Skip it.

Worked examples:

Example 1 — UPDATE (DYNAMIC fact supersedes older DYNAMIC):
  Existing: [uuid-a] (DYNAMIC) "La tarifa a Monterrey es $4,000 MXN por tarima"
  New: [0] (DYNAMIC) "La tarifa a Monterrey es $4,500 MXN por tarima desde el 15 de marzo"
  → action: UPDATE, existingFactId: "uuid-a", mergedContent: "La tarifa a Monterrey es $4,500 MXN por tarima desde el 15 de marzo", temporalType: "DYNAMIC"

Example 2 — DELETE + ADD (contradiction):
  Existing: [uuid-b] (DYNAMIC) "El carrier para Guadalajara-León es Fletes Rápidos"
  New: [0] (DYNAMIC) "El corredor Guadalajara-León cambió de Fletes Rápidos a LogiExpress"
  → Decision 1: action: DELETE, existingFactId: "uuid-b"
  → Decision 2: action: ADD, mergedContent: "El corredor Guadalajara-León usa LogiExpress como carrier", temporalType: "DYNAMIC"

Example 3 — NONE (duplicate STATIC):
  Existing: [uuid-c] (STATIC) "Carlos Méndez es agente del despacho aduanal"
  New: [0] (STATIC) "Carlos Méndez trabaja en el despacho aduanal"
  → action: NONE

Example 4 — ADD (completely new):
  New: [0] (STATIC) "El almacén de Querétaro tiene capacidad máxima de 500 tarimas"
  → action: ADD, mergedContent: "El almacén de Querétaro tiene capacidad máxima de 500 tarimas", temporalType: "STATIC"

Rules:
- For DYNAMIC facts, always prefer UPDATE when an existing fact covers the same topic — the newest value wins
- mergedContent must be a single atomic fact with explicit subject — never merge unrelated info
- Preserve the temporalType from the new fact (or reclassify if the merge changes the nature)

Return JSON with your decisions.`;
  }

  // ── Data access ──────────────────────────────────────────────────────────────

  private async fetchUnconsolidatedEpisodes(
    groupId: string,
    cooldownMinutes: number,
    maxEpisodes: number,
  ): Promise<Array<{ uuid: string; content: string }>> {
    const rows = await this.driver.executeQuery<any[]>(
      `MATCH (ep:Episodic {groupId: $groupId})
       WHERE ep.factConsolidatedAt IS NULL
         AND ep.createdAt <= datetime() - duration({minutes: $cooldown})
       RETURN ep.uuid AS uuid, ep.content AS content
       ORDER BY ep.createdAt ASC
       LIMIT $max`,
      { groupId, cooldown: cooldownMinutes, max: maxEpisodes },
    );
    return rows.map(r => ({ uuid: r.uuid, content: r.content ?? '' }));
  }

  private async findSimilarFacts(
    newFacts: string[],
    groupId: string,
  ): Promise<Array<{ uuid: string; content: string; similarity: number; temporalType?: string }>> {
    // Embed all new facts
    const embeddings = await Promise.all(
      newFacts.map(f => this.embedder.embed(f)),
    );

    // For each fact embedding, search existing Fact nodes
    const allResults = new Map<string, { uuid: string; content: string; similarity: number; temporalType?: string }>();

    for (const embedding of embeddings) {
      try {
        const rows = await this.driver.executeQuery<any[]>(
          `CALL db.index.vector.queryNodes('fact_embedding', 10, $embedding)
           YIELD node, score
           WHERE node.groupId = $groupId
           RETURN node.uuid AS uuid, node.content AS content, score AS similarity, node.temporalType AS temporalType`,
          { embedding, groupId },
        );
        for (const r of rows) {
          if (r.uuid && !allResults.has(r.uuid)) {
            allResults.set(r.uuid, {
              uuid: r.uuid,
              content: r.content ?? '',
              similarity: r.similarity ?? 0,
              temporalType: r.temporalType,
            });
          }
        }
      } catch {
        // Vector index may not exist yet — fall back to brute force
        const rows = await this.driver.executeQuery<any[]>(
          `MATCH (f:Fact {groupId: $groupId})
           RETURN f.uuid AS uuid, f.content AS content, f.embedding AS embedding
           LIMIT 20`,
          { groupId },
        );
        for (const r of rows) {
          if (r.uuid && r.embedding && !allResults.has(r.uuid)) {
            const sim = cosineSimilarity(embedding, r.embedding);
            if (sim > 0.5) {
              allResults.set(r.uuid, {
                uuid: r.uuid,
                content: r.content ?? '',
                similarity: sim,
              });
            }
          }
        }
      }
    }

    // Keyword-based matching: find existing facts that share identifiers
    // (ORDER numbers, AWBs, etc.) with any new fact.
    // This catches contradictions that vector search misses due to different phrasing.
    const identifiers = new Set<string>();
    for (const f of newFacts) {
      for (const m of f.matchAll(/ORDER-\d{5,}|AWB[\s-]*\d{3}-?\d{5,}|PO\d{5,}|\d{3}-\d{5,}/gi)) {
        identifiers.add(m[0]);
      }
    }
    if (identifiers.size > 0) {
      const tokens = [...identifiers];
      try {
        const rows = await this.driver.executeQuery<any[]>(
          `MATCH (f:Fact {groupId: $groupId})
           WHERE ANY(t IN $tokens WHERE f.content CONTAINS t)
           RETURN f.uuid AS uuid, f.content AS content, f.temporalType AS temporalType
           LIMIT 20`,
          { groupId, tokens },
        );
        for (const r of rows) {
          if (r.uuid && !allResults.has(r.uuid)) {
            allResults.set(r.uuid, {
              uuid: r.uuid,
              content: r.content ?? '',
              similarity: 0.7, // High enough to be considered by the decision LLM
              temporalType: r.temporalType,
            });
          }
        }
      } catch {
        // Non-critical — keyword matching is a bonus
      }
    }

    return Array.from(allResults.values()).sort((a, b) => b.similarity - a.similarity);
  }

  private async addFact(content: string, groupId: string, sourceEpisodeIds: string[], temporalType?: string): Promise<void> {
    const uuid = uuidv4();
    const embedding = await this.embedder.embed(content);
    await this.driver.executeQuery(
      `CREATE (f:Fact {
         uuid: $uuid,
         groupId: $groupId,
         content: $content,
         embedding: $embedding,
         sourceEpisodeIds: $sourceEpisodeIds,
         temporalType: $temporalType,
         createdAt: datetime(),
         updatedAt: datetime()
       })`,
      { uuid, groupId, content, embedding, sourceEpisodeIds, temporalType: temporalType ?? 'DYNAMIC' },
    );
  }

  private async updateFact(uuid: string, newContent: string, temporalType?: string): Promise<void> {
    const embedding = await this.embedder.embed(newContent);
    await this.driver.executeQuery(
      `MATCH (f:Fact {uuid: $uuid})
       SET f.content = $content,
           f.embedding = $embedding,
           f.updatedAt = datetime()` +
      (temporalType ? `, f.temporalType = $temporalType` : ''),
      { uuid, content: newContent, embedding, ...(temporalType ? { temporalType } : {}) },
    );
  }

  private async deleteFact(uuid: string): Promise<void> {
    await this.driver.executeQuery(
      `MATCH (f:Fact {uuid: $uuid}) DETACH DELETE f`,
      { uuid },
    );
  }

  private async markFactConsolidated(episodeUuids: string[]): Promise<void> {
    if (episodeUuids.length === 0) return;
    await this.driver.executeQuery(
      `UNWIND $uuids AS uuid
       MATCH (ep:Episodic {uuid: uuid})
       SET ep.factConsolidatedAt = datetime()`,
      { uuids: episodeUuids },
    );
  }
}

// ── Utility ──────────────────────────────────────────────────────────────────

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const mag = Math.sqrt(magA) * Math.sqrt(magB);
  return mag === 0 ? 0 : dot / mag;
}
