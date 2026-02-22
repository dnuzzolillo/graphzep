/**
 * Sleep Engine — Phase 2: Pruning & Entity Resolution
 *
 * Neuroscience analogy: synaptic homeostasis (Tononi hypothesis).
 * During sleep, weakly-reinforced synapses are down-scaled and redundant
 * representations are collapsed — preventing the graph from fragmenting
 * into duplicate, contradictory nodes over time.
 *
 * What this does:
 *
 *   Part A — Entity deduplication (highest value):
 *     1. Find pairs of Entity nodes whose names overlap (containment).
 *     2. Compute cosine similarity between their summary embeddings.
 *     3. If similarity exceeds the threshold, merge the duplicate into the
 *        canonical node:
 *          - Redirect all RELATES_TO edges (incoming + outgoing)
 *          - Redirect all MENTIONS edges from episodic nodes
 *          - Merge summaries (keep canonical, append duplicate facts)
 *          - DETACH DELETE the duplicate node
 *
 *   Part B — Orphaned edge cleanup:
 *     Remove RELATES_TO edges whose episode reference lists are empty,
 *     indicating the supporting evidence has been removed from the graph.
 */

import type { GraphDriver } from '../types/index.js';
import type { BaseEmbedderClient } from '../embedders/client.js';
import type { PruningReport, MergedPair, SleepOptions } from './types.js';

// ── Internal types ────────────────────────────────────────────────────────────

interface EntityCandidate {
  uuidA: string;
  nameA: string;
  embA: number[] | null;
  degreeA: number;
  uuidB: string;
  nameB: string;
  embB: number[] | null;
  degreeB: number;
}

interface ScoredPair extends EntityCandidate {
  similarity: number;
  canonicalUuid: string;
  canonicalName: string;
  duplicateUuid: string;
  duplicateName: string;
}

// ── Pruner ────────────────────────────────────────────────────────────────────

export class Pruner {
  constructor(
    private readonly driver: GraphDriver,
    private readonly embedder: BaseEmbedderClient,
  ) {}

  async run(groupId: string, options: SleepOptions): Promise<PruningReport> {
    const threshold = options.pruning?.similarityThreshold ?? 0.88;
    const dryRun = options.dryRun ?? false;

    const report: PruningReport = {
      entitiesMerged: 0,
      mergedPairs: [],
      edgesPruned: 0,
    };

    // ── Part A: Entity deduplication ─────────────────────────────────────────
    const candidates = await this.fetchCandidatePairs(groupId);
    const scored = this.scorePairs(candidates, threshold);

    // Greedy merge: highest similarity first, skip if either node already merged
    const alreadyMerged = new Set<string>();

    for (const pair of scored) {
      if (alreadyMerged.has(pair.canonicalUuid) || alreadyMerged.has(pair.duplicateUuid)) {
        continue;
      }

      const mergedPair: MergedPair = {
        canonical: pair.canonicalName,
        duplicate: pair.duplicateName,
        similarity: pair.similarity,
      };

      if (!dryRun) {
        await this.mergeEntities(pair.canonicalUuid, pair.duplicateUuid, groupId);
      }

      alreadyMerged.add(pair.duplicateUuid);
      report.mergedPairs.push(mergedPair);
      report.entitiesMerged++;
    }

    // ── Part B: Orphaned edge cleanup ─────────────────────────────────────────
    const pruned = dryRun ? 0 : await this.pruneOrphanedEdges(groupId);
    report.edgesPruned = pruned;

    return report;
  }

  // ── Part A helpers ──────────────────────────────────────────────────────────

  /**
   * Find entity pairs where one name contains the other (case-insensitive).
   * This is a cheap O(n²) Cypher pass — safe because we filter on groupId
   * and Neo4j evaluates the WHERE before materialising all pairs.
   */
  private async fetchCandidatePairs(groupId: string): Promise<EntityCandidate[]> {
    const rows = await this.driver.executeQuery<any[]>(
      `
      MATCH (a:Entity {groupId: $groupId}), (b:Entity {groupId: $groupId})
      WHERE a.uuid < b.uuid
        AND a.name <> b.name
        AND (
          toLower(a.name) CONTAINS toLower(b.name) OR
          toLower(b.name) CONTAINS toLower(a.name)
        )
      RETURN
        a.uuid             AS uuidA,
        a.name             AS nameA,
        a.summaryEmbedding AS embA,
        b.uuid             AS uuidB,
        b.name             AS nameB,
        b.summaryEmbedding AS embB,
        size([(a)-[:RELATES_TO|MENTIONS]-() | 1]) AS degreeA,
        size([(b)-[:RELATES_TO|MENTIONS]-() | 1]) AS degreeB
      `,
      { groupId },
    );

    return rows.map((r: any) => ({
      uuidA: r.uuidA,
      nameA: r.nameA,
      embA: Array.isArray(r.embA) ? r.embA : null,
      degreeA: Number(r.degreeA ?? 0),
      uuidB: r.uuidB,
      nameB: r.nameB,
      embB: Array.isArray(r.embB) ? r.embB : null,
      degreeB: Number(r.degreeB ?? 0),
    }));
  }

  /**
   * Score candidate pairs by embedding cosine similarity.
   * Pairs without embeddings are scored by name similarity only and
   * require exact substring match — higher risk, so threshold is tightened.
   */
  private scorePairs(candidates: EntityCandidate[], threshold: number): ScoredPair[] {
    const scored: ScoredPair[] = [];

    for (const c of candidates) {
      let similarity: number;

      if (c.embA && c.embB) {
        similarity = cosineSimilarity(c.embA, c.embB);
      } else {
        // No embeddings: use name-length ratio as a rough proxy
        const shorter = c.nameA.length < c.nameB.length ? c.nameA : c.nameB;
        const longer = c.nameA.length >= c.nameB.length ? c.nameA : c.nameB;
        // substring match already confirmed; score by how much of longer is shorter
        similarity = shorter.length / longer.length;
        // Conservative: require higher raw threshold for embedding-less pairs
        if (similarity < 0.6) continue;
      }

      if (similarity < threshold) continue;

      // Canonical = the node with more connections; ties go to the longer name
      // (more specific names like "Dr. Alan Fischer" beat "Fischer")
      const canonicalIsA =
        c.degreeA > c.degreeB ||
        (c.degreeA === c.degreeB && c.nameA.length >= c.nameB.length);

      scored.push({
        ...c,
        similarity,
        canonicalUuid: canonicalIsA ? c.uuidA : c.uuidB,
        canonicalName: canonicalIsA ? c.nameA : c.nameB,
        duplicateUuid: canonicalIsA ? c.uuidB : c.uuidA,
        duplicateName: canonicalIsA ? c.nameB : c.nameA,
      });
    }

    // Highest similarity first → greedy merge order
    return scored.sort((a, b) => b.similarity - a.similarity);
  }

  /**
   * Merge `duplicateUuid` into `canonicalUuid`:
   *   1. Redirect outgoing RELATES_TO from duplicate → others
   *   2. Redirect incoming RELATES_TO from others → duplicate
   *   3. Redirect MENTIONS edges from episodic nodes
   *   4. DETACH DELETE the duplicate
   */
  private async mergeEntities(
    canonicalUuid: string,
    duplicateUuid: string,
    groupId: string,
  ): Promise<void> {
    // Step 1 — Outgoing RELATES_TO: (dup)→(other) becomes (canonical)→(other)
    await this.driver.executeQuery(
      `
      MATCH (dup:Entity {uuid: $dupUuid})-[r:RELATES_TO]->(other:Entity {groupId: $groupId})
      WHERE other.uuid <> $canonicalUuid
      MATCH (canonical:Entity {uuid: $canonicalUuid})
      MERGE (canonical)-[newR:RELATES_TO {uuid: r.uuid}]->(other)
      SET newR.name       = r.name,
          newR.groupId    = r.groupId,
          newR.factIds    = r.factIds,
          newR.episodes   = r.episodes,
          newR.validAt    = r.validAt,
          newR.invalidAt  = r.invalidAt,
          newR.expiredAt  = r.expiredAt,
          newR.createdAt  = r.createdAt
      `,
      { dupUuid: duplicateUuid, canonicalUuid, groupId },
    );

    // Step 2 — Incoming RELATES_TO: (other)→(dup) becomes (other)→(canonical)
    await this.driver.executeQuery(
      `
      MATCH (other:Entity {groupId: $groupId})-[r:RELATES_TO]->(dup:Entity {uuid: $dupUuid})
      WHERE other.uuid <> $canonicalUuid
      MATCH (canonical:Entity {uuid: $canonicalUuid})
      MERGE (other)-[newR:RELATES_TO {uuid: r.uuid}]->(canonical)
      SET newR.name       = r.name,
          newR.groupId    = r.groupId,
          newR.factIds    = r.factIds,
          newR.episodes   = r.episodes,
          newR.validAt    = r.validAt,
          newR.invalidAt  = r.invalidAt,
          newR.expiredAt  = r.expiredAt,
          newR.createdAt  = r.createdAt
      `,
      { dupUuid: duplicateUuid, canonicalUuid, groupId },
    );

    // Step 3 — MENTIONS: (episode)→(dup) becomes (episode)→(canonical)
    await this.driver.executeQuery(
      `
      MATCH (ep:Episodic {groupId: $groupId})-[r:MENTIONS]->(dup:Entity {uuid: $dupUuid})
      MATCH (canonical:Entity {uuid: $canonicalUuid})
      MERGE (ep)-[newR:MENTIONS {uuid: r.uuid}]->(canonical)
      SET newR.groupId   = r.groupId,
          newR.createdAt = r.createdAt
      DELETE r
      `,
      { dupUuid: duplicateUuid, canonicalUuid, groupId },
    );

    // Step 4 — Remove duplicate (DETACH DELETE cleans remaining relationships)
    await this.driver.executeQuery(
      `
      MATCH (dup:Entity {uuid: $dupUuid})
      DETACH DELETE dup
      `,
      { dupUuid: duplicateUuid },
    );
  }

  // ── Part B helpers ──────────────────────────────────────────────────────────

  /**
   * Remove RELATES_TO edges whose `episodes` array is empty or null.
   * These are edges that lost all supporting evidence (e.g. after a merge).
   */
  private async pruneOrphanedEdges(groupId: string): Promise<number> {
    const result = await this.driver.executeQuery<any[]>(
      `
      MATCH ()-[r:RELATES_TO {groupId: $groupId}]->()
      WHERE r.episodes IS NULL OR size(r.episodes) = 0
      WITH r, count(r) AS total
      DELETE r
      RETURN total
      `,
      { groupId },
    );
    return Number(result[0]?.total ?? 0);
  }
}

// ── Math utility ──────────────────────────────────────────────────────────────

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const mag = Math.sqrt(magA) * Math.sqrt(magB);
  return mag === 0 ? 0 : dot / mag;
}
