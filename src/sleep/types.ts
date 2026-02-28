import type { BaseLLMClient } from '../llm/client.js';
import type { BaseEmbedderClient } from '../embedders/client.js';
import type { GraphDriver } from '../types/index.js';

// ── Tiered memory config ───────────────────────────────────────────────────────

/**
 * Pass this instead of a plain groupId string to run the sleep cycle in
 * tiered mode: episodes are read from `stmGroupId` (short-term memory) and
 * consolidated entities / relations are written to `ltmGroupId` (long-term
 * memory).  Pruning runs on `ltmGroupId` only.
 */
export interface TierConfig {
  stmGroupId: string;
  ltmGroupId: string;
}

// ── Engine config ─────────────────────────────────────────────────────────────

export interface SleepEngineConfig {
  /** The database driver (same one used with Graphzep) */
  driver: GraphDriver;
  /**
   * LLM client for consolidation synthesis.
   * You may pass a more powerful model here than the one used for addEpisode
   * (e.g. gpt-4o instead of gpt-4o-mini) since sleep is not time-sensitive.
   */
  llm: BaseLLMClient;
  /** Embedder to re-embed refreshed entity summaries */
  embedder: BaseEmbedderClient;
}

export interface SleepOptions {
  /**
   * If true, compute and return the report without writing anything to the graph.
   * Useful for previewing what the sleep cycle would do.
   * Default: false
   */
  dryRun?: boolean;

  /**
   * Only process episodes older than this many minutes.
   * Prevents touching data that was just ingested.
   * Default: 5
   */
  cooldownMinutes?: number;

  consolidation?: {
    /** Run Phase 1. Default: true */
    enabled?: boolean;
    /**
     * Minimum number of unconsolidated episodes mentioning an entity
     * before its summary is refreshed.
     * Default: 2
     */
    minEpisodes?: number;
    /**
     * Maximum entities to process per sleep cycle.
     * Default: 50
     */
    maxEntities?: number;
  };

  pruning?: {
    /** Run Phase 2. Default: true */
    enabled?: boolean;
    /**
     * Cosine similarity threshold above which two entities are considered
     * the same and merged. Combined with name-containment check.
     * Default: 0.88
     */
    similarityThreshold?: number;
  };

  communities?: {
    /** Run Phase 3. Default: true */
    enabled?: boolean;
    /**
     * Minimum total entities in the graph before community detection runs.
     * Default: 15
     */
    minGraphSize?: number;
    /**
     * Minimum entities required to form a community (smaller clusters discarded).
     * Default: 3
     */
    minCommunitySize?: number;
    /**
     * How many new entities must exist since the last rebuild before we re-run.
     * Set to 0 to always rebuild.
     * Default: 10
     */
    rebuildThreshold?: number;
  };
}

// ── Per-phase reports ─────────────────────────────────────────────────────────

export interface ConsolidationReport {
  /** Number of entity summaries refreshed */
  entitiesRefreshed: number;
  /** Number of episodic nodes marked as consolidated */
  episodesConsolidated: number;
  /** Total LLM tokens consumed */
  tokensUsed: number;
  /** Names of entities that were refreshed */
  entitiesProcessed: string[];
}

export interface MergedPair {
  canonical: string;   // entity name kept
  duplicate: string;   // entity name removed
  similarity: number;  // cosine similarity score
}

export interface PruningReport {
  /** Number of entity nodes merged away */
  entitiesMerged: number;
  /** Details of each merge */
  mergedPairs: MergedPair[];
  /** Number of orphaned edges removed */
  edgesPruned: number;
}

export interface CommunityReport {
  /** true when phase 3 was skipped (not enough entities, below threshold, etc.) */
  skipped: boolean;
  /** Human-readable reason when skipped */
  reason?: string;
  /** Number of Community nodes written or updated */
  communitiesBuilt: number;
  /** Number of stale Community nodes removed */
  communitiesRemoved: number;
  /** Total entity count in the graph at the time of the run */
  entityCount: number;
}

// ── Auto-sleep scheduler ──────────────────────────────────────────────────────

export interface AutoSleepConfig {
  /**
   * The graph target — same value you would pass to `sleep()`.
   * Either a plain groupId string or a `{ stmGroupId, ltmGroupId }` TierConfig.
   */
  target: string | TierConfig;
  /** Sleep-cycle options forwarded to every automatic run. */
  options?: SleepOptions;
  /**
   * Hour of day (0–23, local time) at which to run.
   * Default: 3  (3 am)
   */
  hour?: number;
  /**
   * Minute (0–59, local time) at which to run.
   * Default: 0
   */
  minute?: number;
  /** Called after every successful automatic run. */
  onComplete?: (report: SleepReport) => void;
  /** Called when an automatic run throws. Does NOT stop the scheduler. */
  onError?: (err: unknown) => void;
}

// ── Final report ──────────────────────────────────────────────────────────────

export interface SleepReport {
  /** STM groupId (or the single groupId in legacy mode) */
  groupId: string;
  /** LTM groupId — defined only when running in tiered mode */
  ltmGroupId?: string;
  dryRun: boolean;
  startedAt: Date;
  completedAt: Date;
  durationMs: number;
  phase1Consolidation: ConsolidationReport;
  phase2Pruning: PruningReport;
  phase3Communities: CommunityReport;
}
