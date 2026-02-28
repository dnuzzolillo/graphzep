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
import { CommunityBuilder } from './community-builder.js';
import type { AutoSleepConfig, SleepEngineConfig, SleepOptions, SleepReport, TierConfig } from './types.js';

/** Returns the number of milliseconds until the next occurrence of hour:minute (local time). */
function msUntilNext(hour: number, minute: number): number {
  const now  = new Date();
  const next = new Date(now);
  next.setHours(hour, minute, 0, 0);
  if (next <= now) next.setDate(next.getDate() + 1);
  return next.getTime() - now.getTime();
}

export class SleepEngine {
  private readonly consolidator: Consolidator;
  private readonly pruner: Pruner;
  private readonly communityBuilder: CommunityBuilder;
  private _autoSleepTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly config: SleepEngineConfig) {
    this.consolidator = new Consolidator(config.driver, config.llm, config.embedder);
    this.pruner = new Pruner(config.driver, config.embedder);
    this.communityBuilder = new CommunityBuilder(config.driver, config.llm, config.embedder);
  }

  /**
   * Start an automatic nightly sleep cycle.
   *
   * By default runs every day at 03:00 local time.  Call `stopAutoSleep()`
   * to cancel.  If a scheduled run throws, `onError` is called and the
   * scheduler continues — it will retry the next day.
   *
   * @example — default (3 am every night)
   * engine.startAutoSleep({ target: { stmGroupId: 'stm-alice', ltmGroupId: 'ltm-alice' } });
   *
   * @example — custom time + callbacks
   * engine.startAutoSleep({
   *   target: 'my-group',
   *   hour: 2, minute: 30,
   *   onComplete: report => console.log('sleep done', report.durationMs),
   *   onError:    err    => console.error('sleep failed', err),
   * });
   */
  startAutoSleep(config: AutoSleepConfig): void {
    this.stopAutoSleep(); // cancel any existing schedule

    const hour   = config.hour   ?? 3;
    const minute = config.minute ?? 0;

    const schedule = () => {
      const ms = msUntilNext(hour, minute);
      this._autoSleepTimer = setTimeout(async () => {
        try {
          const report = await this.sleep(config.target, config.options);
          config.onComplete?.(report);
        } catch (err) {
          config.onError?.(err);
        }
        schedule(); // reschedule for the following day
      }, ms);
    };

    schedule();
  }

  /** Cancel the automatic sleep scheduler. Safe to call even if not started. */
  stopAutoSleep(): void {
    if (this._autoSleepTimer !== null) {
      clearTimeout(this._autoSleepTimer);
      this._autoSleepTimer = null;
    }
  }

  /** `true` while an automatic schedule is active. */
  get isAutoSleepActive(): boolean {
    return this._autoSleepTimer !== null;
  }

  /**
   * Run a full sleep cycle.
   *
   * @param target   Either a plain `groupId` string (legacy mode — consolidates
   *                 within a single graph) or a `TierConfig` object with
   *                 `{ stmGroupId, ltmGroupId }` (tiered mode — reads from STM,
   *                 writes consolidated entities and relations to LTM, and runs
   *                 pruning on LTM only).
   * @param options  Fine-grained control over each phase; safe defaults apply if omitted.
   * @returns        A detailed report of every change made (or that would be made in dryRun mode).
   *
   * @example — legacy mode (single graph)
   * const report = await engine.sleep('my-group');
   *
   * @example — tiered mode (STM → LTM)
   * const report = await engine.sleep({
   *   stmGroupId: 'stm-alice',
   *   ltmGroupId: 'ltm-alice',
   * });
   */
  async sleep(target: string | TierConfig, options: SleepOptions = {}): Promise<SleepReport> {
    const startedAt  = new Date();
    const dryRun     = options.dryRun ?? false;
    const isTiered   = typeof target !== 'string';
    const groupId    = isTiered ? target.stmGroupId : target;
    const ltmGroupId = isTiered ? target.ltmGroupId : undefined;

    // ── Phase 1 — Consolidation ───────────────────────────────────────────────
    const phase1Enabled = options.consolidation?.enabled !== false;
    const phase1 = phase1Enabled
      ? isTiered
        ? await this.consolidator.runTiered(target.stmGroupId, target.ltmGroupId, options)
        : await this.consolidator.run(groupId, options)
      : emptyConsolidationReport();

    // ── Phase 2 — Pruning & entity resolution ─────────────────────────────────
    // In tiered mode: prune only the LTM graph (STM entities are ephemeral).
    const phase2Enabled = options.pruning?.enabled !== false;
    const pruningTarget = isTiered ? target.ltmGroupId : groupId;
    const phase2 = phase2Enabled
      ? await this.pruner.run(pruningTarget, options)
      : emptyPruningReport();

    // ── Phase 3 — Community detection (REM analogy) ───────────────────────────
    // Group entity nodes into semantic communities using the Louvain modularity
    // algorithm.  Community nodes act as a routing tier in search(): when a
    // query matches a Community, its member entities are included automatically.
    //
    // Rebuild is gated by a delta threshold so it only runs when enough new
    // entities have been added since the previous community build.
    const phase3Enabled = options.communities?.enabled !== false;
    const communityTarget = isTiered ? target.ltmGroupId : groupId;
    const phase3 = phase3Enabled
      ? await this.communityBuilder.run(communityTarget, options)
      : emptyCommunityReport();

    // ── Phase 4 — Ontology refinement (planned) ───────────────────────────────
    // TODO: implement src/sleep/ontology-refiner.ts
    // Detect recurring entity types / relationship patterns not yet in the
    // schema and propose versioned ontology updates.

    const completedAt = new Date();

    return {
      groupId,
      ltmGroupId,
      dryRun,
      startedAt,
      completedAt,
      durationMs: completedAt.getTime() - startedAt.getTime(),
      phase1Consolidation: phase1,
      phase2Pruning: phase2,
      phase3Communities: phase3,
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

function emptyCommunityReport() {
  return {
    skipped: true,
    reason: 'Phase 3 disabled',
    communitiesBuilt: 0,
    communitiesRemoved: 0,
    entityCount: 0,
  };
}
