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
  facts: z.array(z.string()).describe('Discrete facts extracted from the text'),
});

const FactDecisionSchema = z.object({
  decisions: z.array(z.object({
    factIndex: z.number(),
    action: z.enum(['ADD', 'UPDATE', 'DELETE', 'NONE']),
    existingFactId: z.string().optional().describe('UUID of existing fact to UPDATE or DELETE'),
    mergedContent: z.string().optional().describe('New content for ADD or UPDATE'),
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

    // Step 1: Extract atomic facts
    let facts: string[];
    try {
      const extraction = await this.llm.generateStructuredResponse(
        this.buildExtractionPrompt(content),
        FactExtractionSchema,
      );
      facts = extraction.facts.filter(f => f.trim().length > 10);
    } catch {
      return result;
    }
    if (facts.length === 0) {
      await this.markFactConsolidated([episodeUuid]);
      return result;
    }

    // Step 2: Find similar existing facts
    const existingFacts = await this.findSimilarFacts(facts, groupId);

    // Step 3: Decide ADD/UPDATE/DELETE/NONE
    let decisions: z.infer<typeof FactDecisionSchema>['decisions'];
    try {
      const decisionResult = await this.llm.generateStructuredResponse(
        this.buildDecisionPrompt(facts, existingFacts),
        FactDecisionSchema,
      );
      decisions = decisionResult.decisions;
    } catch {
      return result;
    }

    // Step 4: Execute
    for (const decision of decisions) {
      if (decision.factIndex < 0 || decision.factIndex >= facts.length) continue;
      switch (decision.action) {
        case 'ADD': {
          await this.addFact(decision.mergedContent ?? facts[decision.factIndex], groupId, [episodeUuid]);
          result.added++;
          break;
        }
        case 'UPDATE': {
          if (decision.existingFactId && decision.mergedContent) {
            await this.updateFact(decision.existingFactId, decision.mergedContent);
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

      // Step 1: Extract facts from this batch
      let facts: string[];
      try {
        const extraction = await this.llm.generateStructuredResponse(
          this.buildExtractionPrompt(batchText),
          FactExtractionSchema,
        );
        facts = extraction.facts.filter(f => f.trim().length > 10);
      } catch {
        continue;
      }
      report.tokensUsed += estimateTokens(batchText) + 500;

      if (facts.length === 0) {
        // Mark episodes as processed even if no facts extracted
        if (!dryRun) await this.markFactConsolidated(batch.map(e => e.uuid));
        report.episodesProcessed += batch.length;
        continue;
      }

      // Step 2: For each fact, find similar existing facts
      const existingFacts = await this.findSimilarFacts(facts, groupId);

      // Step 3: Ask LLM to decide ADD/UPDATE/DELETE/NONE
      let decisions: z.infer<typeof FactDecisionSchema>['decisions'];
      try {
        const decisionResult = await this.llm.generateStructuredResponse(
          this.buildDecisionPrompt(facts, existingFacts),
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
          if (decision.factIndex < 0 || decision.factIndex >= facts.length) continue;

          switch (decision.action) {
            case 'ADD': {
              const content = decision.mergedContent ?? facts[decision.factIndex];
              await this.addFact(content, groupId, batch.map(e => e.uuid));
              report.factsAdded++;
              break;
            }
            case 'UPDATE': {
              if (decision.existingFactId && decision.mergedContent) {
                await this.updateFact(decision.existingFactId, decision.mergedContent);
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
        // Count what would happen
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

Your task is to extract ATOMIC facts from the following episodes. Each fact must capture exactly ONE piece of information — never combine multiple pieces of info into a single fact. Short, specific facts produce better search results.

Focus areas:
- Entity details (who someone is, their role, contact info)
- Events and actions (what happened, when, to whom)
- Domain knowledge (rates, procedures, requirements)
- Relationships (who works with whom, what connects to what)
- Status updates (order states, delivery progress, payments)

Here are some few-shot examples:

Input: "Hablé con María López de Transportes del Norte. Dice que la tarifa para consolidados a Monterrey subió a $4,500 MXN por tarima a partir del 15 de marzo. También me confirmó que el pedido #8842 ya salió de bodega y llega mañana."
Output: {"facts": ["María López trabaja en Transportes del Norte", "La tarifa de consolidados a Monterrey es $4,500 MXN por tarima", "La tarifa a Monterrey aplica a partir del 15 de marzo", "El pedido #8842 ya salió de bodega", "El pedido #8842 llega mañana"]}

Input: "El AWB 176-4892-7731 de DHL tiene un retraso en aduana. Carlos Méndez del despacho aduanal dice que falta el certificado fitosanitario. El cliente es Agroinsumos del Bajío y ya le avisé que se retrasa 2 días."
Output: {"facts": ["El AWB 176-4892-7731 es un envío por DHL", "El AWB 176-4892-7731 está retenido en aduana", "Falta el certificado fitosanitario para el AWB 176-4892-7731", "Carlos Méndez es del despacho aduanal", "El cliente del AWB 176-4892-7731 es Agroinsumos del Bajío", "Se notificó a Agroinsumos del Bajío del retraso de 2 días"]}

Input: "Update on route optimization: we switched the Guadalajara-León corridor from carrier Fletes Rápidos to LogiExpress because they offer next-day delivery and their damage rate is under 0.5%. Contract signed on March 3rd, valid for 6 months."
Output: {"facts": ["The Guadalajara-León corridor switched from Fletes Rápidos to LogiExpress", "LogiExpress offers next-day delivery on Guadalajara-León", "LogiExpress damage rate is under 0.5%", "LogiExpress contract was signed on March 3rd", "LogiExpress contract is valid for 6 months"]}

Input: "Revisé el inventario del almacén de Querétaro. Hay 340 tarimas disponibles, pero 52 están apartadas para el pedido de Grupo Bimbo que sale el viernes. La capacidad máxima es 500 tarimas."
Output: {"facts": ["El almacén de Querétaro tiene 340 tarimas disponibles", "52 tarimas en Querétaro están apartadas para Grupo Bimbo", "El pedido de Grupo Bimbo sale el viernes", "La capacidad máxima del almacén de Querétaro es 500 tarimas"]}

Rules:
- Extract ONE piece of information per fact — break compound statements apart
- Each fact must be self-contained and understandable without the original context
- Keep facts short: one sentence, ideally under 20 words
- Preserve specific identifiers (order numbers, AWBs, dates, amounts, emails)
- Detect the language of the input and write facts in the same language
- Skip trivial interactions (greetings, acknowledgments, "ok", "gracias")

Episodes:
${episodeText}

Return JSON with the extracted facts.`;
  }

  private buildDecisionPrompt(
    newFacts: string[],
    existingFacts: Array<{ uuid: string; content: string; similarity: number }>,
  ): string {
    const newFactsList = newFacts
      .map((f, i) => `  [${i}] ${f}`)
      .join('\n');

    const existingList = existingFacts.length > 0
      ? existingFacts.map(f => `  [${f.uuid}] ${f.content}`).join('\n')
      : '  (no existing facts)';

    return `You are maintaining a knowledge base of atomic facts. For each new fact, decide what to do by comparing it against existing facts.

EXISTING FACTS in the knowledge base:
${existingList}

NEW FACTS extracted from recent activity:
${newFactsList}

For each new fact (by index), choose one action:
- ADD: Genuinely new information not covered by any existing fact. Set mergedContent to the fact text.
- UPDATE: An existing fact covers the same topic but the new info adds detail or corrects it. Set existingFactId to the UUID and mergedContent to the improved merged version.
- DELETE: The new fact directly contradicts an existing fact, making the old one obsolete. Set existingFactId to the UUID of the fact to remove. Also ADD the new fact separately.
- NONE: The new fact is already captured by an existing fact (same info, possibly different wording). Skip it.

Here are worked examples:

Example 1 — UPDATE (more specific info):
  Existing: [uuid-a] "La tarifa a Monterrey es $4,000 MXN por tarima"
  New: "La tarifa a Monterrey subió a $4,500 MXN por tarima a partir del 15 de marzo"
  → action: UPDATE, existingFactId: "uuid-a", mergedContent: "La tarifa a Monterrey es $4,500 MXN por tarima desde el 15 de marzo"
  Why: Same topic (tarifa a Monterrey), new fact has updated price AND a date — keep the more specific version.

Example 2 — DELETE (direct contradiction):
  Existing: [uuid-b] "El carrier para Guadalajara-León es Fletes Rápidos"
  New: "The Guadalajara-León corridor switched from Fletes Rápidos to LogiExpress"
  → action: DELETE, existingFactId: "uuid-b"
  (The new fact itself should be ADDed as a separate decision.)
  Why: The old fact is now false — the carrier changed. Delete the stale fact.

Example 3 — NONE (duplicate, different wording):
  Existing: [uuid-c] "Carlos Méndez trabaja en el despacho aduanal"
  New: "Carlos Méndez es del despacho aduanal"
  → action: NONE
  Why: Same information expressed differently. No need to add or update.

Example 4 — ADD (completely new):
  Existing: (no related facts)
  New: "El almacén de Querétaro tiene capacidad máxima de 500 tarimas"
  → action: ADD, mergedContent: "El almacén de Querétaro tiene capacidad máxima de 500 tarimas"
  Why: No existing fact covers warehouse capacity in Querétaro.

Rules:
- Prefer UPDATE over ADD when an existing fact covers the same topic — merge them
- When updating, always keep the version with MORE specific information (dates, numbers, names)
- Only DELETE when there is a clear contradiction (the old fact is now false), not just an update
- NONE for duplicates or trivial information already captured
- mergedContent must be a single atomic fact — do not merge unrelated info together

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
  ): Promise<Array<{ uuid: string; content: string; similarity: number }>> {
    // Embed all new facts
    const embeddings = await Promise.all(
      newFacts.map(f => this.embedder.embed(f)),
    );

    // For each fact embedding, search existing Fact nodes
    const allResults = new Map<string, { uuid: string; content: string; similarity: number }>();

    for (const embedding of embeddings) {
      try {
        const rows = await this.driver.executeQuery<any[]>(
          `CALL db.index.vector.queryNodes('fact_embedding', 5, $embedding)
           YIELD node, score
           WHERE node.groupId = $groupId
           RETURN node.uuid AS uuid, node.content AS content, score AS similarity`,
          { embedding, groupId },
        );
        for (const r of rows) {
          if (r.uuid && !allResults.has(r.uuid)) {
            allResults.set(r.uuid, {
              uuid: r.uuid,
              content: r.content ?? '',
              similarity: r.similarity ?? 0,
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

    return [...allResults.values()].sort((a, b) => b.similarity - a.similarity);
  }

  private async addFact(content: string, groupId: string, sourceEpisodeIds: string[]): Promise<void> {
    const uuid = uuidv4();
    const embedding = await this.embedder.embed(content);
    await this.driver.executeQuery(
      `CREATE (f:Fact {
         uuid: $uuid,
         groupId: $groupId,
         content: $content,
         embedding: $embedding,
         sourceEpisodeIds: $sourceEpisodeIds,
         createdAt: datetime(),
         updatedAt: datetime()
       })`,
      { uuid, groupId, content, embedding, sourceEpisodeIds },
    );
  }

  private async updateFact(uuid: string, newContent: string): Promise<void> {
    const embedding = await this.embedder.embed(newContent);
    await this.driver.executeQuery(
      `MATCH (f:Fact {uuid: $uuid})
       SET f.content = $content,
           f.embedding = $embedding,
           f.updatedAt = datetime()`,
      { uuid, content: newContent, embedding },
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
