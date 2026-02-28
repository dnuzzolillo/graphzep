/**
 * GraphZep Community Builder — Phase 3 of the Sleep Engine
 *
 * Detects semantic communities (clusters) in the entity graph using a
 * pure-TypeScript implementation of the Louvain modularity algorithm.
 * No GDS or external graph library is required.
 *
 * For each detected community:
 *   1. Stability matching: if ≥ 70% of members (Jaccard) overlap with an
 *      existing Community node, reuse its UUID (prevents churn across cycles).
 *   2. LLM summarisation: synthesise name, summary, and domain hints from
 *      the member entity summaries.
 *   3. Persistence: write Community nodes + HAS_MEMBER edges to the graph.
 *
 * Rebuild trigger: runs only when (current entity count − entity count at
 * the last rebuild) ≥ rebuildThreshold.  This avoids expensive LLM calls
 * after every sleep cycle.
 *
 * Reference: Blondel et al. 2008 "Fast unfolding of communities in large
 * networks" — https://doi.org/10.1088/1742-5468/2008/10/P10008
 */

import { v4 as uuidv4 } from 'uuid';
import { z } from 'zod';
import type { GraphDriver } from '../types/index.js';
import type { BaseLLMClient } from '../llm/client.js';
import type { BaseEmbedderClient } from '../embedders/client.js';
import type { SleepOptions, CommunityReport } from './types.js';

// ── Internal graph representation ─────────────────────────────────────────────

interface EntityRecord {
  uuid: string;
  name: string;
  entityType: string;
  summary: string;
}

interface EdgeRecord {
  source: string;
  target: string;
}

// ── Pure-TypeScript Louvain ───────────────────────────────────────────────────

/**
 * Single-pass Louvain modularity optimisation (local phase only).
 * Returns a Map from entity UUID → community label (an arbitrary UUID string).
 *
 * Runs until no single-node greedy move improves modularity.
 * Typical convergence for < 10 000 nodes: < 50 ms.
 */
function runLouvain(entities: EntityRecord[], edges: EdgeRecord[]): Map<string, string> {
  // Build undirected adjacency with uniform weight 1
  const adj = new Map<string, Map<string, number>>();
  const degree = new Map<string, number>();

  for (const e of entities) {
    adj.set(e.uuid, new Map());
    degree.set(e.uuid, 0);
  }

  let m = 0; // total edge weight (sum, not doubled)
  for (const edge of edges) {
    if (!adj.has(edge.source) || !adj.has(edge.target)) continue;
    if (edge.source === edge.target) continue;
    const w = 1;
    adj.get(edge.source)!.set(edge.target, (adj.get(edge.source)!.get(edge.target) ?? 0) + w);
    adj.get(edge.target)!.set(edge.source, (adj.get(edge.target)!.get(edge.source) ?? 0) + w);
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + w);
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + w);
    m += w;
  }

  // Initialise: every entity is its own community
  const assignments = new Map<string, string>();
  const commDeg = new Map<string, number>(); // Σ_tot per community

  for (const e of entities) {
    assignments.set(e.uuid, e.uuid);
    commDeg.set(e.uuid, degree.get(e.uuid) ?? 0);
  }

  if (m === 0) return assignments; // No edges → leave as singletons

  // Local optimisation: iterate until stable
  let improved = true;
  while (improved) {
    improved = false;
    for (const entity of entities) {
      const nodeId = entity.uuid;
      const ki = degree.get(nodeId) ?? 0;
      if (ki === 0) continue; // Isolated node — skip

      const currentComm = assignments.get(nodeId)!;
      const neighbors = adj.get(nodeId)!;

      // k_{i, in(C_old \ {i})}: edges from i to OTHER members of its current community
      let kOld = 0;
      for (const [neighborId, w] of neighbors) {
        if (assignments.get(neighborId) === currentComm) kOld += w;
      }
      const sigmaTotOld = commDeg.get(currentComm) ?? 0;

      // Gather candidate communities from the neighbourhood
      const neighborComms = new Map<string, number>(); // community → k_{i, in(C)}
      for (const [neighborId, w] of neighbors) {
        const nc = assignments.get(neighborId)!;
        if (nc !== currentComm) neighborComms.set(nc, (neighborComms.get(nc) ?? 0) + w);
      }

      // Pick the best community move by ΔQ
      // ΔQ = (k_new − k_old) / m  −  ki × (Σ_tot_new − Σ_tot_old + ki) / (2m²)
      let bestDelta = 0;
      let bestComm = currentComm;

      for (const [targetComm, kNew] of neighborComms) {
        const sigmaTotNew = commDeg.get(targetComm) ?? 0;
        const delta =
          (kNew - kOld) / m - (ki * (sigmaTotNew - sigmaTotOld + ki)) / (2 * m * m);
        if (delta > bestDelta) {
          bestDelta = delta;
          bestComm = targetComm;
        }
      }

      if (bestComm !== currentComm) {
        commDeg.set(currentComm, (commDeg.get(currentComm) ?? 0) - ki);
        commDeg.set(bestComm, (commDeg.get(bestComm) ?? 0) + ki);
        assignments.set(nodeId, bestComm);
        improved = true;
      }
    }
  }

  return assignments;
}

// ── LLM schema for community summarisation ───────────────────────────────────

const CommunitySummarySchema = z.object({
  name: z.string(),
  summary: z.string(),
  // The LLM sometimes returns a comma-separated string instead of an array
  domainHints: z.preprocess(
    val => (typeof val === 'string' ? val.split(/[,;\s]+/).map(s => s.trim()).filter(Boolean) : val),
    z.array(z.string()).default([]),
  ),
  importanceScore: z.coerce.number().min(0).max(1).default(0.5),
});

// ── CommunityBuilder ──────────────────────────────────────────────────────────

export class CommunityBuilder {
  constructor(
    private readonly driver: GraphDriver,
    private readonly llm: BaseLLMClient,
    private readonly embedder: BaseEmbedderClient,
  ) {}

  async run(groupId: string, sleepOptions: SleepOptions = {}): Promise<CommunityReport> {
    const opts = sleepOptions.communities ?? {};
    const minGraphSize = opts.minGraphSize ?? 15;
    const minCommunitySize = opts.minCommunitySize ?? 3;
    const rebuildThreshold = opts.rebuildThreshold ?? 10;
    const dryRun = sleepOptions.dryRun ?? false;

    // ── 1. Fetch all Entity nodes ─────────────────────────────────────────────
    const entityRows = await this.driver.executeQuery<any[]>(
      `MATCH (e:Entity {groupId: $groupId})
       RETURN e.uuid AS uuid, e.name AS name, e.entityType AS entityType,
              e.summary AS summary
       ORDER BY e.createdAt DESC`,
      { groupId },
    );

    const entities: EntityRecord[] = entityRows
      .map(r => ({
        uuid: String(r.uuid ?? ''),
        name: String(r.name ?? ''),
        entityType: String(r.entityType ?? 'Other'),
        summary: String(r.summary ?? ''),
      }))
      .filter(e => e.uuid && e.name);

    if (entities.length < minGraphSize) {
      return {
        skipped: true,
        reason: `Only ${entities.length} entities (need ≥ ${minGraphSize})`,
        communitiesBuilt: 0,
        communitiesRemoved: 0,
        entityCount: entities.length,
      };
    }

    // ── 2. Rebuild threshold check ────────────────────────────────────────────
    // We store the entity count at the time of the last rebuild directly on
    // Community nodes so we avoid datetime parsing across drivers.
    const lastCountRow = await this.driver.executeQuery<any[]>(
      `MATCH (c:Community {groupId: $groupId})
       WHERE c.entityCountAtLastRebuild IS NOT NULL
       RETURN max(c.entityCountAtLastRebuild) AS lastCount`,
      { groupId },
    );

    const lastCount = Number(lastCountRow[0]?.lastCount ?? 0);
    const delta = entities.length - lastCount;

    if (delta < rebuildThreshold) {
      return {
        skipped: true,
        reason: `Only ${delta} new entities since last rebuild (need ≥ ${rebuildThreshold})`,
        communitiesBuilt: 0,
        communitiesRemoved: 0,
        entityCount: entities.length,
      };
    }

    // ── 3. Fetch entity-entity edges ──────────────────────────────────────────
    const edgeRows = await this.driver.executeQuery<any[]>(
      `MATCH (a:Entity {groupId: $groupId})-[:RELATES_TO]->(b:Entity {groupId: $groupId})
       RETURN a.uuid AS source, b.uuid AS target`,
      { groupId },
    );

    const edges: EdgeRecord[] = edgeRows
      .map(r => ({ source: String(r.source ?? ''), target: String(r.target ?? '') }))
      .filter(e => e.source && e.target);

    // ── 4. Run Louvain ────────────────────────────────────────────────────────
    const assignments = runLouvain(entities, edges);

    // Group entities by community label
    const communityGroups = new Map<string, EntityRecord[]>();
    for (const entity of entities) {
      const label = assignments.get(entity.uuid)!;
      if (!communityGroups.has(label)) communityGroups.set(label, []);
      communityGroups.get(label)!.push(entity);
    }

    // Keep only communities that meet the minimum size
    const validCommunities = [...communityGroups.values()].filter(
      members => members.length >= minCommunitySize,
    );

    // ── 5. Load existing Community nodes (UUID stability) ─────────────────────
    const existingRows = await this.driver.executeQuery<any[]>(
      `MATCH (c:Community {groupId: $groupId})
       OPTIONAL MATCH (c)-[:HAS_MEMBER]->(e:Entity)
       RETURN c.uuid AS uuid, c.name AS name,
              collect(e.uuid) AS memberUuids`,
      { groupId },
    );

    const existingCommunities = existingRows.map(r => ({
      uuid: String(r.uuid),
      name: String(r.name ?? ''),
      memberUuids: new Set<string>((r.memberUuids as string[]) ?? []),
    }));

    // ── 6. Build each valid community ─────────────────────────────────────────
    const nowIso = new Date().toISOString();
    const usedExistingUuids = new Set<string>();
    let communitiesBuilt = 0;

    for (const members of validCommunities) {
      const memberUuids = new Set(members.map(e => e.uuid));

      // UUID stability: Jaccard ≥ 0.7 with an existing community → reuse UUID
      let reuseUuid: string | undefined;
      let bestJaccard = 0;
      for (const existing of existingCommunities) {
        if (usedExistingUuids.has(existing.uuid)) continue;
        const intersection = [...memberUuids].filter(u => existing.memberUuids.has(u)).length;
        const union = new Set([...memberUuids, ...existing.memberUuids]).size;
        const jaccard = union > 0 ? intersection / union : 0;
        if (jaccard >= 0.7 && jaccard > bestJaccard) {
          bestJaccard = jaccard;
          reuseUuid = existing.uuid;
        }
      }

      const communityUuid = reuseUuid ?? uuidv4();
      if (reuseUuid) usedExistingUuids.add(reuseUuid);

      communitiesBuilt++;

      if (dryRun) continue;

      // LLM: synthesise community name, summary, domain hints
      const memberSummaries = members
        .slice(0, 20) // cap at 20 to keep the prompt compact
        .map((e, i) => `${i + 1}. "${e.name}" (${e.entityType}): ${e.summary}`)
        .join('\n');

      const prompt = `You are analysing a semantic community (cluster) of entities in a knowledge graph.

Entities in this cluster:
${memberSummaries}

Write a JSON response that characterises this cluster:
{
  "name": "short descriptive name for this community (3-7 words)",
  "summary": "2-3 sentence summary of what this cluster represents and how the entities relate",
  "domainHints": ["topic-tag-1", "topic-tag-2"],
  "importanceScore": 0.7
}

domainHints: 2-4 lowercase kebab-case tags (e.g. "software-engineering", "world-history", "sports").
importanceScore: 0.0-1.0 reflecting how densely connected and factually rich this cluster is.`;

      const summaryData = await this.llm.generateStructuredResponse(prompt, CommunitySummarySchema);
      const summaryEmbedding = await this.embedder.embed(summaryData.summary);

      // Upsert Community node
      await this.driver.executeQuery(
        `MERGE (c:Community {uuid: $uuid})
         SET c.name = $name,
             c.communityLevel = 0,
             c.summary = $summary,
             c.summaryEmbedding = $summaryEmbedding,
             c.embedding = $summaryEmbedding,
             c.groupId = $groupId,
             c.createdAt = COALESCE(c.createdAt, datetime($now)),
             c.memberEntityIds = $memberEntityIds,
             c.memberCount = $memberCount,
             c.domainHints = $domainHints,
             c.importanceScore = $importanceScore,
             c.entityCountAtLastRebuild = $entityCountAtLastRebuild,
             c.lastFullRebuild = datetime($now)`,
        {
          uuid: communityUuid,
          name: summaryData.name,
          summary: summaryData.summary,
          summaryEmbedding,
          groupId,
          now: nowIso,
          memberEntityIds: [...memberUuids],
          memberCount: members.length,
          domainHints: summaryData.domainHints,
          importanceScore: summaryData.importanceScore,
          entityCountAtLastRebuild: entities.length,
        },
      );

      // Rebuild HAS_MEMBER edges
      await this.driver.executeQuery(
        `MATCH (c:Community {uuid: $uuid})-[r:HAS_MEMBER]->() DELETE r`,
        { uuid: communityUuid },
      );

      for (const memberUuid of memberUuids) {
        await this.driver.executeQuery(
          `MATCH (c:Community {uuid: $cUuid}), (e:Entity {uuid: $eUuid})
           MERGE (c)-[:HAS_MEMBER]->(e)`,
          { cUuid: communityUuid, eUuid: memberUuid },
        );
      }
    }

    // ── 7. Remove stale Community nodes ──────────────────────────────────────
    const staleUuids = existingCommunities
      .filter(ec => !usedExistingUuids.has(ec.uuid))
      .map(ec => ec.uuid);

    if (!dryRun && staleUuids.length > 0) {
      await this.driver.executeQuery(
        `MATCH (c:Community {groupId: $groupId})
         WHERE c.uuid IN $staleUuids
         DETACH DELETE c`,
        { groupId, staleUuids },
      );
    }

    return {
      skipped: false,
      communitiesBuilt,
      communitiesRemoved: staleUuids.length,
      entityCount: entities.length,
    };
  }
}
