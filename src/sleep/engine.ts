/**
 * GraphZep Sleep Engine
 *
 * Inspired by the role of sleep in biological memory consolidation,
 * this engine runs a background maintenance cycle over the knowledge graph.
 * Like the brain during sleep, it does work that is too expensive or risky
 * to do in real-time — improving quality holistically across all stored memories.
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  WAKE MODE (addEpisode)          SLEEP MODE (SleepEngine.sleep())   │
 * │  ─────────────────────────────   ─────────────────────────────────  │
 * │  Fast, per-episode extraction    Holistic, cross-episode synthesis  │
 * │  Small LLM (gpt-4o-mini)         Powerful LLM (gpt-4o)             │
 * │  Optimised for latency           Optimised for quality              │
 * │  Episodic memory                 Semantic consolidation             │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * Implemented phases
 * ──────────────────
 *   Phase 1 — Consolidation (NREM slow-wave analogy)
 *     For each entity that has been mentioned in ≥ N new episodes, refresh
 *     its summary by synthesising all accumulated evidence with an LLM.
 *     Re-embed the new summary so semantic search improves automatically.
 *
 *   Phase 2 — Pruning & entity resolution (synaptic homeostasis analogy)
 *     Detect duplicate entity nodes (same real-world entity, different name
 *     strings). Merge duplicates, redirect all edges to the canonical node,
 *     and remove orphaned edges left behind by prior merges.
 *
 * Planned phases (annotated for future implementation)
 * ─────────────────────────────────────────────────────
 *   Phase 3 — Association (REM analogy)
 *   Phase 4 — Ontology refinement (schema evolution)
 */

import { Consolidator } from './consolidator.js';
import { Pruner } from './pruner.js';
import type { SleepEngineConfig, SleepOptions, SleepReport } from './types.js';

export class SleepEngine {
  private readonly consolidator: Consolidator;
  private readonly pruner: Pruner;

  constructor(private readonly config: SleepEngineConfig) {
    this.consolidator = new Consolidator(config.driver, config.llm, config.embedder);
    this.pruner = new Pruner(config.driver, config.embedder);
  }

  /**
   * Run a full sleep cycle for the given groupId.
   *
   * @param groupId  The workspace/session to process (same groupId used with addEpisode)
   * @param options  Fine-grained control over each phase; safe defaults apply if omitted
   * @returns        A detailed report of every change made (or that would be made in dryRun mode)
   *
   * @example
   * const engine = new SleepEngine({ driver, llm, embedder });
   *
   * // Preview without writing
   * const preview = await engine.sleep('my-group', { dryRun: true });
   * console.log(preview);
   *
   * // Full cycle
   * const report = await engine.sleep('my-group');
   */
  async sleep(groupId: string, options: SleepOptions = {}): Promise<SleepReport> {
    const startedAt = new Date();
    const dryRun = options.dryRun ?? false;

    // ── Phase 1 — Consolidation ───────────────────────────────────────────────
    const phase1Enabled = options.consolidation?.enabled !== false;
    const phase1 = phase1Enabled
      ? await this.consolidator.run(groupId, options)
      : emptyConsolidationReport();

    // ── Phase 2 — Pruning & entity resolution ─────────────────────────────────
    const phase2Enabled = options.pruning?.enabled !== false;
    const phase2 = phase2Enabled
      ? await this.pruner.run(groupId, options)
      : emptyPruningReport();

    // ── Phase 3 — Association (REM) ───────────────────────────────────────────
    // TODO: implement src/sleep/associator.ts
    //
    // Idea: sample pairs of episodic nodes that are NOT directly connected but
    // whose entity sets overlap in embedding space. For each candidate pair,
    // ask the LLM: "Is there an implicit relationship between these two events
    // that is not already captured in the graph?" If yes and confidence is high,
    // add an inferred RELATES_TO edge with `isInferred: true` and a lower weight.
    //
    // Key challenge: O(n²) pairs is infeasible at scale. Smart sampling strategies:
    //   - Random walk from high-degree nodes
    //   - Embedding-cluster proximity (episodes in same cluster but different episodes)
    //   - Surprise metric: pairs that embedding says should be related but aren't connected
    //
    // Reference: REM sleep consolidates associative/insight memory in humans.
    // See: Walker 2009 "The Role of Sleep in Cognition and Emotion"

    // ── Phase 4 — Ontology refinement ─────────────────────────────────────────
    // TODO: implement src/sleep/ontology-refiner.ts
    //
    // Idea: after each sleep cycle, sample the most recent extractions and ask
    // the LLM: "What entity types and relationship patterns appear repeatedly
    // that are not yet in the ontology schema?" Produce OntologyProposal objects
    // with confidence scores. Auto-accept proposals above a high threshold
    // (e.g. 0.92), queue the rest for human review in a proposals log.
    //
    // Critical constraint: schema changes must be versioned. A new entity type
    // added at cycle N should not retroactively invalidate queries from cycle 0.
    // Store ontology versions with a validFrom timestamp, similar to how the
    // graph stores temporal validity for edges.
    //
    // Reference: the brain updates its "schemas" (mental models) during sleep,
    // integrating new experiences into existing knowledge frameworks.
    // See: Tamminen et al. 2010 "Sleep spindle activity is associated with
    //      the integration of new memories and existing knowledge"

    const completedAt = new Date();

    return {
      groupId,
      dryRun,
      startedAt,
      completedAt,
      durationMs: completedAt.getTime() - startedAt.getTime(),
      phase1Consolidation: phase1,
      phase2Pruning: phase2,
    };
  }
}

// ── Empty reports (used when a phase is disabled) ─────────────────────────────

function emptyConsolidationReport() {
  return {
    entitiesRefreshed: 0,
    episodesConsolidated: 0,
    tokensUsed: 0,
    entitiesProcessed: [] as string[],
  };
}

function emptyPruningReport() {
  return {
    entitiesMerged: 0,
    mergedPairs: [] as import('./types.js').MergedPair[],
    edgesPruned: 0,
  };
}
