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

import { v4 as uuidv4 } from 'uuid';
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

interface LTMNeighbor {
  relation: string;
  neighborName: string;
  direction: 'out' | 'in';
}

interface LTMCounterpart {
  uuid: string;
  name: string;
  summary: string;
}

interface STMRelationEdge {
  relName: string;
  relUuid: string;
  episodes: string[];
  validAt: string;
  peerName: string;
}

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
- Preserve ALL attribution facts (named after, dedicated to, founded by, described by, discovered by) — these are irreversible if lost and must never be omitted even for brevity
- Do not speculate beyond the provided text

Return JSON with the updated summary and your confidence score.`;
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

  // ── Cross-graph (STM → LTM) consolidation ─────────────────────────────────

  /**
   * Tiered consolidation: reads episodic clusters from `stmGroupId` and
   * upserts consolidated entities + relations into `ltmGroupId`.
   *
   * Three traversals per entity cluster:
   *   T1 — Lookup LTM counterpart (exact name → vector similarity fallback)
   *   T2 — Fetch 1-hop LTM neighbourhood for merge context
   *   T3 — Migrate STM relations to LTM (exact-name lookup only; defers if
   *         target not yet consolidated)
   */
  async runTiered(
    stmGroupId: string,
    ltmGroupId: string,
    options: SleepOptions,
  ): Promise<ConsolidationReport> {
    const cooldown    = options.cooldownMinutes              ?? 5;
    const minEpisodes = options.consolidation?.minEpisodes   ?? 2;
    const maxEntities = options.consolidation?.maxEntities   ?? 50;
    const dryRun      = options.dryRun                       ?? false;

    const report: ConsolidationReport = {
      entitiesRefreshed: 0,
      episodesConsolidated: 0,
      tokensUsed: 0,
      entitiesProcessed: [],
    };

    const clusters = await this.fetchEntityClusters(stmGroupId, cooldown, minEpisodes, maxEntities);
    if (clusters.length === 0) return report;

    for (const cluster of clusters) {
      // ── Synthesise new summary from STM episodes ──────────────────────────
      const stmPrompt = this.buildPrompt(cluster);
      let stmSynthesis: { summary: string; confidence: number };
      try {
        stmSynthesis = await this.llm.generateStructuredResponse(stmPrompt, EntitySynthesisSchema);
      } catch {
        continue;
      }
      report.tokensUsed += estimateTokens(stmPrompt) + estimateTokens(stmSynthesis.summary);

      if (dryRun) {
        report.entitiesProcessed.push(cluster.name);
        report.entitiesRefreshed++;
        report.episodesConsolidated += cluster.episodeUuids.length;
        continue;
      }

      // ── Embed the synthesis (used for T1 vector search + LTM write) ───────
      let stmEmbedding: number[];
      try {
        stmEmbedding = await this.embedder.embed(stmSynthesis.summary);
      } catch {
        continue;
      }

      // ── T1: find counterpart in LTM ───────────────────────────────────────
      const ltmCounterpart = await this.lookupLTMCounterpart(
        cluster.name, stmEmbedding, ltmGroupId,
      );

      let ltmEntityUuid: string;

      if (ltmCounterpart) {
        // ── T2: fetch 1-hop neighbourhood for richer merge context ───────────
        const neighborhood = await this.fetchLTMNeighborhood(ltmCounterpart.uuid, ltmGroupId);

        const mergePrompt = this.buildMergePrompt(
          cluster.name, ltmCounterpart.summary, neighborhood, stmSynthesis.summary,
        );
        let merged: { summary: string; confidence: number };
        try {
          merged = await this.llm.generateStructuredResponse(mergePrompt, EntitySynthesisSchema);
        } catch {
          merged = stmSynthesis;
        }
        report.tokensUsed += estimateTokens(mergePrompt) + estimateTokens(merged.summary);

        let mergedEmbedding: number[];
        try {
          mergedEmbedding = await this.embedder.embed(merged.summary);
        } catch {
          mergedEmbedding = stmEmbedding;
        }

        await this.updateEntity(ltmCounterpart.uuid, merged.summary, mergedEmbedding);
        ltmEntityUuid = ltmCounterpart.uuid;
      } else {
        // No counterpart → create new entity in LTM
        ltmEntityUuid = await this.createLTMEntity(
          cluster, stmSynthesis.summary, stmEmbedding, ltmGroupId,
        );
      }

      // ── T3: migrate STM relations into LTM ───────────────────────────────
      await this.migrateRelations(cluster.uuid, ltmEntityUuid, stmGroupId, ltmGroupId);

      // ── Mark STM episodes as consolidated ────────────────────────────────
      await this.markConsolidated(cluster.episodeUuids);

      report.entitiesRefreshed++;
      report.episodesConsolidated += cluster.episodeUuids.length;
      report.entitiesProcessed.push(cluster.name);
    }

    return report;
  }

  // ── T1: LTM counterpart lookup ─────────────────────────────────────────────

  /**
   * Look up an existing LTM entity by exact name only.
   *
   * NOTE: Vector similarity was removed from this lookup because short entity
   * summaries in the same knowledge domain (e.g. all Chinese geography) have
   * cosine similarity well above 0.92, causing false-positive merges that
   * collapse all entities into one.  Phase 2 Pruning already handles true
   * near-duplicates (e.g. "UK" vs "United Kingdom") via its own similarity
   * pass over the full LTM graph, so adding it here would double-count and
   * produce incorrectly merged entities.
   */
  private async lookupLTMCounterpart(
    name: string,
    _embedding: number[],
    ltmGroupId: string,
  ): Promise<LTMCounterpart | null> {
    const exact = await this.driver.executeQuery<any[]>(
      `MATCH (n:Entity {name: $name, groupId: $ltmGroupId})
       RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary
       LIMIT 1`,
      { name, ltmGroupId },
    );
    if (exact.length > 0 && exact[0].uuid) {
      return { uuid: exact[0].uuid, name: exact[0].name ?? name, summary: exact[0].summary ?? '' };
    }
    return null;
  }

  // ── T2: LTM neighbourhood fetch ────────────────────────────────────────────

  private async fetchLTMNeighborhood(
    ltmEntityUuid: string,
    ltmGroupId: string,
  ): Promise<LTMNeighbor[]> {
    const [outRows, inRows] = await Promise.all([
      this.driver.executeQuery<any[]>(
        `MATCH (e:Entity {uuid: $uuid})-[r:RELATES_TO]->(n:Entity {groupId: $group})
         WHERE r.invalidAt IS NULL
         RETURN r.name AS relation, n.name AS neighborName
         LIMIT 6`,
        { uuid: ltmEntityUuid, group: ltmGroupId },
      ),
      this.driver.executeQuery<any[]>(
        `MATCH (e:Entity {uuid: $uuid})<-[r:RELATES_TO]-(n:Entity {groupId: $group})
         WHERE r.invalidAt IS NULL
         RETURN r.name AS relation, n.name AS neighborName
         LIMIT 4`,
        { uuid: ltmEntityUuid, group: ltmGroupId },
      ),
    ]);
    return [
      ...outRows.map((r: any) => ({
        relation: r.relation ?? '',
        neighborName: r.neighborName ?? '',
        direction: 'out' as const,
      })),
      ...inRows.map((r: any) => ({
        relation: r.relation ?? '',
        neighborName: r.neighborName ?? '',
        direction: 'in' as const,
      })),
    ];
  }

  // ── Merge prompt ───────────────────────────────────────────────────────────

  private buildMergePrompt(
    entityName: string,
    ltmSummary: string,
    neighborhood: LTMNeighbor[],
    stmSynthesis: string,
  ): string {
    const relContext =
      neighborhood.length > 0
        ? `\nKnown relationships in long-term memory:\n` +
          neighborhood
            .map(n =>
              n.direction === 'out'
                ? `  ${entityName} –[${n.relation}]→ ${n.neighborName}`
                : `  ${n.neighborName} –[${n.relation}]→ ${entityName}`,
            )
            .join('\n')
        : '';

    return `You are maintaining a long-term knowledge graph. Update the entity summary \
for "${entityName}".

Current long-term summary: "${ltmSummary}"${relContext}

New information synthesised from recent short-term memory:
"${stmSynthesis}"

Write an updated summary that:
1. Preserves accurate, durable facts from the long-term summary
2. Integrates genuinely new information from recent activity
3. Stays concise and abstract — 2 to 4 sentences
4. Omits ephemeral details (specific meeting dates, one-off tasks)
5. Preserves ALL attribution facts (named after, dedicated to, founded by, described by) — never drop these even if they seem minor
6. Does not speculate beyond the evidence provided

Return JSON with the updated summary and your confidence.`;
  }

  // ── Create new LTM entity ──────────────────────────────────────────────────

  private async createLTMEntity(
    cluster: EntityCluster,
    summary: string,
    embedding: number[],
    ltmGroupId: string,
  ): Promise<string> {
    const uuid = uuidv4();
    await this.driver.executeQuery(
      `MERGE (n:Entity {name: $name, groupId: $groupId})
       ON CREATE SET
         n.uuid             = $uuid,
         n.entityType       = $entityType,
         n.summary          = $summary,
         n.summaryEmbedding = $embedding,
         n.embedding        = $embedding,
         n.factIds          = [],
         n.createdAt        = datetime(),
         n.consolidatedAt   = datetime()
       ON MATCH SET
         n.summary          = $summary,
         n.summaryEmbedding = $embedding,
         n.embedding        = $embedding,
         n.consolidatedAt   = datetime()`,
      { uuid, name: cluster.name, entityType: cluster.entityType, summary, embedding, groupId: ltmGroupId },
    );
    // MERGE may have matched an existing node with a different uuid — refetch
    const result = await this.driver.executeQuery<any[]>(
      `MATCH (n:Entity {name: $name, groupId: $groupId}) RETURN n.uuid AS uuid LIMIT 1`,
      { name: cluster.name, groupId: ltmGroupId },
    );
    return result[0]?.uuid ?? uuid;
  }

  // ── T3: relation migration ─────────────────────────────────────────────────

  /**
   * For each active RELATES_TO edge on the STM entity, attempt to find the
   * peer in LTM via exact name match and upsert the relation there.
   * Edges whose peer is not yet in LTM are silently deferred — they will be
   * migrated in a future sleep cycle once the peer is consolidated.
   */
  private async migrateRelations(
    stmEntityUuid: string,
    ltmEntityUuid: string,
    stmGroupId: string,
    ltmGroupId: string,
  ): Promise<void> {
    const outgoing = await this.driver.executeQuery<any[]>(
      `MATCH (src:Entity {uuid: $uuid})-[r:RELATES_TO]->(tgt:Entity {groupId: $group})
       WHERE r.invalidAt IS NULL
       RETURN r.name AS relName, r.uuid AS relUuid,
              r.episodes AS episodes, toString(r.validAt) AS validAt,
              tgt.name AS peerName`,
      { uuid: stmEntityUuid, group: stmGroupId },
    );

    const incoming = await this.driver.executeQuery<any[]>(
      `MATCH (src:Entity {groupId: $group})-[r:RELATES_TO]->(tgt:Entity {uuid: $uuid})
       WHERE r.invalidAt IS NULL
       RETURN r.name AS relName, r.uuid AS relUuid,
              r.episodes AS episodes, toString(r.validAt) AS validAt,
              src.name AS peerName`,
      { uuid: stmEntityUuid, group: stmGroupId },
    );

    await Promise.all([
      ...outgoing.map((edge: STMRelationEdge) =>
        this.migrateOneRelation(ltmEntityUuid, 'out', edge, ltmGroupId),
      ),
      ...incoming.map((edge: STMRelationEdge) =>
        this.migrateOneRelation(ltmEntityUuid, 'in', edge, ltmGroupId),
      ),
    ]);
  }

  private async migrateOneRelation(
    ltmEntityUuid: string,
    direction: 'out' | 'in',
    edge: STMRelationEdge,
    ltmGroupId: string,
  ): Promise<void> {
    // Exact-name lookup only for relation peers — no vector search to avoid mismatches
    const peer = await this.driver.executeQuery<any[]>(
      `MATCH (n:Entity {name: $name, groupId: $groupId}) RETURN n.uuid AS uuid LIMIT 1`,
      { name: edge.peerName, groupId: ltmGroupId },
    );
    if (peer.length === 0 || !peer[0].uuid) return; // peer not yet in LTM — defer

    const [srcUuid, tgtUuid] =
      direction === 'out'
        ? [ltmEntityUuid, peer[0].uuid]
        : [peer[0].uuid, ltmEntityUuid];

    await this.upsertLTMRelation(
      srcUuid, tgtUuid, edge.relName,
      edge.relUuid, edge.episodes ?? [], edge.validAt, ltmGroupId,
    );
  }

  private async upsertLTMRelation(
    sourceUuid: string,
    targetUuid: string,
    relName: string,
    stmRelUuid: string,
    episodes: string[],
    validAt: string,
    ltmGroupId: string,
  ): Promise<void> {
    await this.driver.executeQuery(
      `MATCH (src:Entity {uuid: $sourceUuid})
       MATCH (tgt:Entity {uuid: $targetUuid})
       MERGE (src)-[r:RELATES_TO {name: $relName, groupId: $groupId}]->(tgt)
       ON CREATE SET
         r.uuid      = $uuid,
         r.factIds   = [],
         r.episodes  = $episodes,
         r.validAt   = datetime($validAt),
         r.createdAt = datetime()
       ON MATCH SET
         r.episodes  = r.episodes + $episodes,
         r.validAt   = datetime($validAt)`,
      {
        sourceUuid,
        targetUuid,
        relName,
        groupId: ltmGroupId,
        uuid: stmRelUuid + ':ltm',
        episodes,
        validAt: validAt ?? new Date().toISOString(),
      },
    );
  }
}

// ── Utility ───────────────────────────────────────────────────────────────────

/** Rough token estimate (4 chars ≈ 1 token) for usage tracking */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}
