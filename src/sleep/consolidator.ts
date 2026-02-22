/**
 * Sleep Engine — Phase 1: Episodic → Semantic Consolidation
 *
 * Neuroscience analogy: slow-wave NREM sleep.
 * The hippocampus replays recent experiences and transfers the gist to
 * long-term cortical storage, compressing many episodes into durable facts.
 *
 * What this does:
 *   1. Find entities that have been mentioned in ≥ N unconsolidated episodes.
 *   2. For each entity, gather the full text of every episode that mentions it.
 *   3. Call the LLM (ideally a more powerful model) to synthesise an improved
 *      summary that incorporates all accumulated evidence.
 *   4. Re-embed the new summary and write it back to the entity node.
 *   5. Mark processed episodic nodes with `consolidatedAt` so they are not
 *      reprocessed in future sleep cycles.
 */

import { z } from 'zod';
import type { GraphDriver } from '../types/index.js';
import type { BaseLLMClient } from '../llm/client.js';
import type { BaseEmbedderClient } from '../embedders/client.js';
import type { ConsolidationReport, SleepOptions } from './types.js';

// ── Zod schema for the LLM response ──────────────────────────────────────────

const EntitySynthesisSchema = z.object({
  summary: z
    .string()
    .describe(
      'Updated comprehensive summary of the entity based on all evidence (2-4 sentences).',
    ),
  confidence: z
    .number()
    .min(0)
    .max(1)
    .describe('Your confidence in the synthesis (0 = uncertain, 1 = certain).'),
});

// ── Internal types ────────────────────────────────────────────────────────────

interface EntityCluster {
  uuid: string;
  name: string;
  entityType: string;
  currentSummary: string;
  episodeUuids: string[];
  episodeContents: string[];
}

// ── Consolidator ──────────────────────────────────────────────────────────────

export class Consolidator {
  constructor(
    private readonly driver: GraphDriver,
    private readonly llm: BaseLLMClient,
    private readonly embedder: BaseEmbedderClient,
  ) {}

  async run(groupId: string, options: SleepOptions): Promise<ConsolidationReport> {
    const cooldown = options.cooldownMinutes ?? 5;
    const minEpisodes = options.consolidation?.minEpisodes ?? 2;
    const maxEntities = options.consolidation?.maxEntities ?? 50;
    const dryRun = options.dryRun ?? false;

    const report: ConsolidationReport = {
      entitiesRefreshed: 0,
      episodesConsolidated: 0,
      tokensUsed: 0,
      entitiesProcessed: [],
    };

    // ── Step 1: Find entities with enough unconsolidated episodes ─────────────
    const clusters = await this.fetchEntityClusters(groupId, cooldown, minEpisodes, maxEntities);

    if (clusters.length === 0) return report;

    // ── Step 2: For each cluster, synthesise an improved summary ──────────────
    for (const cluster of clusters) {
      const prompt = this.buildPrompt(cluster);

      let synthesis: z.infer<typeof EntitySynthesisSchema>;
      try {
        synthesis = await this.llm.generateStructuredResponse(prompt, EntitySynthesisSchema);
      } catch {
        // If structured response fails, skip this entity rather than crashing
        continue;
      }

      report.tokensUsed += estimateTokens(prompt) + estimateTokens(synthesis.summary);

      if (dryRun) {
        report.entitiesProcessed.push(cluster.name);
        report.entitiesRefreshed++;
        report.episodesConsolidated += cluster.episodeUuids.length;
        continue;
      }

      // ── Step 3: Re-embed the new summary ──────────────────────────────────
      let embedding: number[];
      try {
        embedding = await this.embedder.embed(synthesis.summary);
      } catch {
        continue;
      }

      // ── Step 4: Write updated entity back to graph ─────────────────────────
      await this.updateEntity(cluster.uuid, synthesis.summary, embedding);

      // ── Step 5: Mark episodes as consolidated ──────────────────────────────
      await this.markConsolidated(cluster.episodeUuids);

      report.entitiesRefreshed++;
      report.episodesConsolidated += cluster.episodeUuids.length;
      report.entitiesProcessed.push(cluster.name);
    }

    return report;
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  private async fetchEntityClusters(
    groupId: string,
    cooldownMinutes: number,
    minEpisodes: number,
    maxEntities: number,
  ): Promise<EntityCluster[]> {
    const rows = await this.driver.executeQuery<any[]>(
      `
      MATCH (ep:Episodic {groupId: $groupId})-[:MENTIONS]->(e:Entity {groupId: $groupId})
      WHERE ep.consolidatedAt IS NULL
        AND ep.createdAt <= datetime() - duration({minutes: $cooldown})
      WITH e,
           collect(DISTINCT ep.uuid)    AS episodeUuids,
           collect(DISTINCT ep.content) AS episodeContents
      WHERE size(episodeUuids) >= $minEpisodes
      RETURN
        e.uuid        AS uuid,
        e.name        AS name,
        e.entityType  AS entityType,
        e.summary     AS currentSummary,
        episodeUuids,
        episodeContents
      ORDER BY size(episodeUuids) DESC
      LIMIT $maxEntities
      `,
      { groupId, cooldown: cooldownMinutes, minEpisodes, maxEntities },
    );

    return rows.map((r: any) => ({
      uuid: r.uuid,
      name: r.name,
      entityType: r.entityType ?? 'Unknown',
      currentSummary: r.currentSummary ?? '',
      episodeUuids: Array.isArray(r.episodeUuids) ? r.episodeUuids : [],
      episodeContents: Array.isArray(r.episodeContents) ? r.episodeContents : [],
    }));
  }

  private buildPrompt(cluster: EntityCluster): string {
    const episodesText = cluster.episodeContents
      .map((content, i) => `[Episode ${i + 1}]: ${content}`)
      .join('\n\n');

    return `You are maintaining a knowledge graph. Update the summary for a graph entity.

Entity: "${cluster.name}" (type: ${cluster.entityType})
Current summary: "${cluster.currentSummary || 'No summary yet.'}"

New episodes that mention this entity:
${episodesText}

Write an updated comprehensive summary of "${cluster.name}" that integrates the current \
summary with the new evidence from the episodes above.

Rules:
- Be factual — only include information directly supported by the evidence
- Be concise — 2 to 4 sentences
- Preserve important facts from the current summary unless contradicted
- Include key roles, relationships, and notable events
- Do not speculate beyond the provided text`;
  }

  private async updateEntity(
    uuid: string,
    newSummary: string,
    embedding: number[],
  ): Promise<void> {
    await this.driver.executeQuery(
      `
      MATCH (e:Entity {uuid: $uuid})
      SET e.summary           = $summary,
          e.summaryEmbedding  = $embedding,
          e.embedding         = $embedding,
          e.consolidatedAt    = datetime()
      `,
      { uuid, summary: newSummary, embedding },
    );
  }

  private async markConsolidated(episodeUuids: string[]): Promise<void> {
    if (episodeUuids.length === 0) return;
    await this.driver.executeQuery(
      `
      UNWIND $uuids AS uuid
      MATCH (ep:Episodic {uuid: uuid})
      SET ep.consolidatedAt = datetime()
      `,
      { uuids: episodeUuids },
    );
  }
}

// ── Utility ───────────────────────────────────────────────────────────────────

/** Rough token estimate (4 chars ≈ 1 token) for usage tracking */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}
