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
}
