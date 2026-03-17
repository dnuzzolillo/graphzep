import { z } from 'zod';
import {
  GraphDriver,
  EpisodicNode,
  EpisodeType,
} from './types/index.js';
import { Node, EntityNodeImpl, EpisodicNodeImpl, CommunityNodeImpl, FactNodeImpl } from './core/nodes.js';
import { Edge, EntityEdgeImpl, EpisodicEdgeImpl, CommunityEdgeImpl } from './core/edges.js';
import { BaseLLMClient } from './llm/client.js';
import { BaseEmbedderClient } from './embedders/client.js';
import { FactConsolidator } from './sleep/fact-consolidator.js';
import { utcNow } from './utils/datetime.js';

export interface GraphzepConfig {
  driver: GraphDriver;
  llmClient: BaseLLMClient;
  embedder: BaseEmbedderClient;
  groupId?: string;
  ensureAscii?: boolean;
  /** Extract atomic facts inline during addEpisode(). Requires an extra LLM call per episode. */
  inlineFacts?: boolean;
  /** Optional separate LLM for inline fact extraction (defaults to llmClient). */
  factLlm?: BaseLLMClient;
}

export interface AddEpisodeParams {
  content: string;
  episodeType?: EpisodeType;
  referenceId?: string;
  groupId?: string;
  /**
   * When the event actually happened.
   * Defaults to the ingestion time (utcNow()) if not provided.
   * Use this to back-date episodes: e.g. an email sent on Feb 11 should
   * have validAt = new Date('2026-02-11').
   */
  validAt?: Date;
  metadata?: Record<string, any>;
}

export interface SearchParams {
  query: string;
  groupId?: string;
  limit?: number;
  searchType?: 'semantic' | 'keyword' | 'hybrid';
  nodeTypes?: ('entity' | 'episodic' | 'community')[];
  graphExpand?: boolean; // expand results by traversing edges from seed entities
  expandHops?: number;   // hops for expansion (default 1)
  /**
   * Only return Episodic nodes whose `validAt` is on or after this date.
   * Entity and Community nodes are unaffected by this filter.
   */
  validFrom?: Date;
  /**
   * Only return Episodic nodes whose `validAt` is on or before this date.
   * Entity and Community nodes are unaffected by this filter.
   */
  validTo?: Date;
  /**
   * Reference time used for temporal scoring. When provided, Episodic nodes
   * whose validAt is close to this time receive a higher score.
   */
  queryTime?: Date;
  /**
   * Weight of the temporal proximity bonus (0–1). Default 0.3.
   * Set to 0 to disable temporal weighting.
   */
  temporalAlpha?: number;
  /**
   * Half-life in days for temporal proximity decay. Default 30.
   * Smaller values emphasise very recent events more strongly.
   */
  halfLifeDays?: number;
}

export interface TraverseParams {
  startEntityName?: string;
  startEntityUuid?: string;
  maxHops?: number;
  direction?: 'outgoing' | 'incoming' | 'both';
  groupId?: string;
  limit?: number;
}

export interface TraverseResult {
  start: EntityNodeImpl | null;
  nodes: EntityNodeImpl[];
  edges: EntityEdgeImpl[];
}

const ENTITY_TYPES = [
  'Person', 'Organization', 'Location', 'Product',
  'Event', 'Concept', 'Technology', 'Other',
] as const;

const ENTITY_CONFIDENCE_THRESHOLD = 0.5;
const RELATION_CONFIDENCE_THRESHOLD = 0.5;

export interface ExtractedEntity {
  name: string;
  entityType: string;
  summary: string;
  confidence?: number;
  metadata?: Record<string, any>;
}

export interface ExtractedRelation {
  sourceName: string;
  targetName: string;
  relationName: string;
  confidence?: number;
  isNegated?: boolean;
  temporalValidity?: 'current' | 'historical';
  metadata?: Record<string, any>;
}

const ExtractedEntitySchema = z.object({
  name: z.string(),
  entityType: z.string(),
  summary: z.string(),
  confidence: z.coerce.number().min(0).max(1).default(1.0),
  metadata: z.record(z.any()).optional(),
});

const ExtractedRelationSchema = z.object({
  sourceName: z.string(),
  targetName: z.string(),
  relationName: z.string(),
  confidence: z.coerce.number().min(0).max(1).default(1.0),
  isNegated: z
    .any()
    .transform((val): boolean => {
      if (typeof val === 'boolean') return val;
      if (val === 'true') return true;
      return false;
    })
    .default(false),
  temporalValidity: z.enum(['current', 'historical']).default('current'),
  metadata: z.record(z.any()).optional(),
});

const ExtractionResultSchema = z.object({
  entities: z.array(ExtractedEntitySchema),
  relations: z.array(ExtractedRelationSchema),
});

const MergedEntitySchema = z.object({
  mergedSummary: z.string(),
});

/** Converts a Neo4j DateTime object, ISO string, or Date to a JS Date. */
function coerceDate(v: any): Date {
  if (v instanceof Date) return v;
  // Neo4j DateTime objects expose toStandardDate() in the driver
  if (v && typeof v.toStandardDate === 'function') return v.toStandardDate() as Date;
  const str = typeof v === 'string' ? v : (v && typeof v.toString === 'function' ? v.toString() : '');
  if (str) {
    // Neo4j datetime strings have nanoseconds: "...T00:00:00.000000000Z" — strip to ms
    const normalized = str.replace(/(\.\d{3})\d+(Z|[+-]\d{2}:\d{2}|$)/, '$1$2');
    const d = new Date(normalized);
    if (!isNaN(d.getTime())) return d;
  }
  return new Date();
}

export class Graphzep {
  private driver: GraphDriver;
  private llmClient: BaseLLMClient;
  private embedder: BaseEmbedderClient;
  private defaultGroupId: string;
  private ensureAscii: boolean;
  private factConsolidator: FactConsolidator | null;

  constructor(config: GraphzepConfig) {
    this.driver = config.driver;
    this.llmClient = config.llmClient;
    this.embedder = config.embedder;
    this.defaultGroupId = config.groupId || 'default';
    this.ensureAscii = config.ensureAscii ?? false;
    this.factConsolidator = config.inlineFacts
      ? new FactConsolidator(config.driver, config.factLlm ?? config.llmClient, config.embedder)
      : null;
  }

  async addEpisode(params: AddEpisodeParams): Promise<EpisodicNode> {
    const groupId = params.groupId || this.defaultGroupId;
    const episodeType = params.episodeType || EpisodeType.TEXT;

    const embedding = await this.embedder.embed(params.content);

    const createdAt = utcNow();
    const validAt = params.validAt ?? createdAt;
    // retroactiveDays: how many days before ingestion did the event occur?
    // 0 = contemporaneous recording; positive = back-dated episode.
    const retroactiveDays = Math.max(
      0,
      Math.round((createdAt.getTime() - validAt.getTime()) / 86_400_000),
    );

    // Create episodic node
    const episodicNode = new EpisodicNodeImpl({
      uuid: '',
      name: params.content.substring(0, 50),
      groupId,
      episodeType,
      content: params.content,
      embedding,
      validAt,
      referenceId: params.referenceId,
      labels: [],
      createdAt,
      retroactiveDays,
      disputedBy: [],
    });

    await episodicNode.save(this.driver);

    const anchorTokens = this.extractAnchorTokens(params.content);
    const existingEntities = await this.fetchExistingEntities(groupId, embedding, anchorTokens);
    const extractedData = await this.extractEntitiesAndRelations(params.content, existingEntities);

    const entityNodes = await this.processExtractedEntities(extractedData.entities, groupId);

    await this.linkEpisodeToEntities(episodicNode, entityNodes);

    await this.processExtractedRelations(extractedData.relations, entityNodes, groupId, episodicNode.uuid);

    // Inline fact extraction: extract atomic facts and deduplicate against existing Fact nodes.
    // Runs in background (fire-and-forget) to avoid slowing down addEpisode().
    if (this.factConsolidator) {
      this.factConsolidator
        .extractForEpisode(episodicNode.uuid, params.content, groupId)
        .catch(() => {}); // Swallow errors — fact extraction is best-effort
    }

    return episodicNode;
  }

  private async fetchExistingEntities(
    groupId: string,
    episodeEmbedding?: number[],
    anchorTokens?: string[],
  ): Promise<Array<{ uuid: string; name: string; entityType: string }>> {
    let primary: Array<{ uuid: string; name: string; entityType: string }>;

    if (!episodeEmbedding) {
      const result = await this.driver.executeQuery<any[]>(
        `MATCH (n:Entity {groupId: $groupId})
         RETURN n.uuid AS uuid, n.name AS name, n.entityType AS entityType
         ORDER BY n.createdAt DESC
         LIMIT 20`,
        { groupId },
      );
      primary = result
        .map(r => ({ uuid: r.uuid ?? '', name: r.name ?? '', entityType: r.entityType ?? '' }))
        .filter(e => e.name);
    } else {
      // Fetch semantic candidates from DB — wider pool to re-rank afterwards
      const candidates = await this.driver.executeQuery<any[]>(
        `MATCH (n:Entity {groupId: $groupId})
         WHERE n.summaryEmbedding IS NOT NULL
         WITH n,
           reduce(sim = 0.0, i IN range(0, size(n.summaryEmbedding)-1) |
             sim + (n.summaryEmbedding[i] * $embedding[i])
           ) AS semanticScore
         WHERE semanticScore > $threshold
         RETURN n.uuid AS uuid, n.name AS name, n.entityType AS entityType,
                semanticScore, n.createdAt AS createdAt
         ORDER BY semanticScore DESC
         LIMIT $pool`,
        { groupId, embedding: episodeEmbedding, threshold: 0.65, pool: 50 },
      );

      if (candidates.length === 0) {
        primary = [];
      } else {
        // Re-rank: semantic relevance + recency (exponential decay, half-life ~7 days)
        const now = Date.now();
        const ALPHA = 0.7;   // weight for semantic score
        const LAMBDA = 0.1;  // decay rate: score = e^(-lambda * ageDays)

        primary = candidates
          .map(r => {
            const ageDays = (now - new Date(r.createdAt).getTime()) / 86_400_000;
            const recencyScore = Math.exp(-LAMBDA * ageDays);
            const finalScore = ALPHA * r.semanticScore + (1 - ALPHA) * recencyScore;
            return { uuid: r.uuid ?? '', name: r.name ?? '', entityType: r.entityType ?? '', finalScore };
          })
          .filter(e => e.name)
          .sort((a, b) => b.finalScore - a.finalScore)
          .slice(0, 20)
          .map(({ uuid, name, entityType }) => ({ uuid, name, entityType }));
      }
    }

    // Supplement with token-based matches so cross-language references to the
    // same entity (e.g. "ticket 20258006" vs "Order 20258006") are surfaced in
    // the LLM extraction context, enabling the model to reuse the canonical name.
    if (anchorTokens && anchorTokens.length > 0) {
      const seen = new Set(primary.map(e => e.uuid));
      const tokenRows = await this.driver.executeQuery<any[]>(
        `MATCH (n:Entity {groupId: $groupId})
         WHERE ANY(t IN $tokens WHERE n.name CONTAINS t)
         RETURN n.uuid AS uuid, n.name AS name, n.entityType AS entityType
         LIMIT 10`,
        { groupId, tokens: anchorTokens },
      );
      for (const r of tokenRows) {
        if (r.uuid && r.name && !seen.has(r.uuid)) {
          primary.push({ uuid: r.uuid, name: r.name, entityType: r.entityType ?? '' });
          seen.add(r.uuid);
        }
      }
    }

    return primary;
  }

  private async extractEntitiesAndRelations(
    content: string,
    existingEntities: Array<{ uuid: string; name: string; entityType: string }>,
  ): Promise<{
    entities: ExtractedEntity[];
    relations: ExtractedRelation[];
  }> {
    const entityTypesStr = ENTITY_TYPES.join(' | ');

    const existingCtx = existingEntities.length > 0
      ? `\nKnown entities already in the graph (if the text refers to one, reuse its exact name):\n${existingEntities.map(e => `  - "${e.name}" (${e.entityType})`).join('\n')}\n`
      : '';

    const prompt = `Extract entities and relationships from the text below.

Text: ${content}
${existingCtx}
Rules:
1. Extract only entities CENTRAL to the text — skip generic nouns, time references, and filler words.
2. Use the canonical full name (prefer "Alice Smith" over "Alice" or "Ms. Smith").
3. If the text refers to a known entity listed above, reuse its exact name.
4. entityType must be one of: ${entityTypesStr}.
5. relationName must be UPPER_SNAKE_CASE (e.g. WORKS_AT, KNOWS, LOCATED_IN, NAMED_AFTER, DEDICATED_TO, FOUNDED_BY, DESCRIBED_BY, DISCOVERED_BY, COMPETES_IN, PART_OF).
6. CRITICAL — always extract attribution and dedication relations: if the text says "named after X", "dedicated to X", "founded by X", "described by X", "discovered by X", "honors X" — you MUST create a RELATES_TO edge for it. These are the highest-value facts in the graph and must never be omitted.
7. confidence: 0.0–1.0 — how clearly stated the entity or relation is in the text.
8. isNegated: true if the text explicitly denies the relation (e.g. "Alice does NOT work at ACME").
9. temporalValidity: "historical" if the relation is past or explicitly ended (e.g. "used to", "formerly", "left", "was").
10. If an entity is a Person and the text mentions their phone number or email, include it in the metadata field: {"phone": "+34...", "email": "user@example.com"}.

Respond with valid JSON:
{
  "entities": [
    {
      "name": "string",
      "entityType": "${entityTypesStr.split(' | ')[0]} | ...",
      "summary": "string",
      "confidence": 0.0,
      "metadata": {}
    }
  ],
  "relations": [
    {
      "sourceName": "string",
      "targetName": "string",
      "relationName": "UPPER_SNAKE_CASE",
      "confidence": 0.0,
      "isNegated": false,
      "temporalValidity": "current"
    }
  ]
}`;

    return this.llmClient.generateStructuredResponse(prompt, ExtractionResultSchema);
  }

  private async processExtractedEntities(
    entities: ExtractedEntity[],
    groupId: string,
  ): Promise<EntityNodeImpl[]> {
    const processedEntities: EntityNodeImpl[] = [];
    entities = entities.filter(e => (e.confidence ?? 1.0) >= ENTITY_CONFIDENCE_THRESHOLD);

    for (const entity of entities) {
      const existing = await this.findExistingEntity(entity.name, groupId);

      if (existing) {
        // Graphiti-style: merge summaries via LLM and re-embed
        const mergedSummary = await this.mergeEntitySummary(
          entity.name,
          existing.summary,
          entity.summary,
        );
        const mergedEmbedding = await this.embedder.embed(mergedSummary);

        existing.summary = mergedSummary;
        existing.summaryEmbedding = mergedEmbedding;
        // Only update entityType if the existing one is generic/unknown
        if (!existing.entityType || existing.entityType === 'Unknown') {
          existing.entityType = entity.entityType;
        }
        // Shallow-merge metadata from extraction into existing entity
        if (entity.metadata && Object.keys(entity.metadata).length > 0) {
          existing.metadata = { ...(existing.metadata ?? {}), ...entity.metadata };
        }

        await existing.save(this.driver); // MERGE on uuid → updates in place
        processedEntities.push(existing);
      } else {
        const embedding = await this.embedder.embed(entity.summary);

        const entityNode = new EntityNodeImpl({
          uuid: '',
          name: entity.name,
          groupId,
          entityType: entity.entityType,
          summary: entity.summary,
          summaryEmbedding: embedding,
          labels: [],
          createdAt: utcNow(),
          metadata: entity.metadata,
        });

        await entityNode.save(this.driver);
        processedEntities.push(entityNode);
      }
    }

    return processedEntities;
  }

  /**
   * Extract numeric anchor tokens (5+ digit sequences) from a text string.
   * These are likely unique identifiers: order IDs, ticket IDs, etc.
   * Used to resolve cross-language / cross-terminology entity references
   * where the same real-world entity appears with different surface names
   * (e.g. "ticket 20258006" in Spanish vs "Order 20258006" in English).
   */
  private extractAnchorTokens(text: string): string[] {
    const matches = text.match(/\d{5,}/g);
    return matches ? [...new Set(matches)] : [];
  }

  private async findExistingEntity(name: string, groupId: string): Promise<EntityNodeImpl | null> {
    // 1. Exact name match (primary path — zero overhead for known entities)
    const result = await this.driver.executeQuery<any[]>(
      `
      MATCH (n:Entity {name: $name, groupId: $groupId})
      RETURN n
      LIMIT 1
      `,
      { name, groupId },
    );

    if (result.length > 0) {
      const raw = result[0].n;
      const props = (raw?.properties ?? raw) as Record<string, any>;
      return new EntityNodeImpl({
        uuid: props.uuid ?? '',
        name: props.name ?? '',
        groupId: props.groupId ?? groupId,
        entityType: props.entityType ?? '',
        summary: props.summary ?? '',
        summaryEmbedding: props.summaryEmbedding ?? undefined,
        factIds: props.factIds ?? [],
        labels: [],
        createdAt: props.createdAt ? new Date(props.createdAt) : new Date(),
        metadata: Graphzep._parseMetadataJson(props.metadataJson),
        pinned: props.pinned ?? false,
      });
    }

    // 2. Anchor-token fallback: match by shared numeric ID (e.g. order/ticket numbers).
    //    This resolves cross-language duplicates where the entity name differs but the
    //    embedded identifier is the same ("ticket 20258006" === "Order 20258006").
    //    Returns the most-connected candidate (highest degree = canonical entity).
    const tokens = this.extractAnchorTokens(name);
    if (tokens.length === 0) return null;

    const tokenMatches = await this.driver.executeQuery<any[]>(
      `MATCH (n:Entity {groupId: $groupId})
       WHERE ANY(t IN $tokens WHERE n.name CONTAINS t)
       WITH n, size([(n)-[:RELATES_TO|MENTIONS]-() | 1]) AS degree
       RETURN n, degree
       ORDER BY degree DESC
       LIMIT 5`,
      { groupId, tokens },
    );

    if (tokenMatches.length === 0) return null;

    const raw = tokenMatches[0].n;
    const props = (raw?.properties ?? raw) as Record<string, any>;
    return new EntityNodeImpl({
      uuid: props.uuid ?? '',
      name: props.name ?? '',
      groupId: props.groupId ?? groupId,
      entityType: props.entityType ?? '',
      summary: props.summary ?? '',
      summaryEmbedding: props.summaryEmbedding ?? undefined,
      factIds: props.factIds ?? [],
      labels: [],
      createdAt: props.createdAt ? new Date(props.createdAt) : new Date(),
      metadata: Graphzep._parseMetadataJson(props.metadataJson),
      pinned: props.pinned ?? false,
    });
  }

  private async mergeEntitySummary(
    entityName: string,
    existingSummary: string,
    newContext: string,
  ): Promise<string> {
    const prompt = `
You are maintaining a knowledge graph. Update the entity summary for "${entityName}".

Existing summary: "${existingSummary}"
New context from a recent interaction: "${newContext}"

Create an updated summary that:
1. Preserves accurate information from the existing summary
2. Integrates new information from the new context
3. Resolves contradictions by preferring the newer information
4. Remains concise and factual (2-3 sentences max)

Return JSON with the merged summary.`;

    const response = await this.llmClient.generateStructuredResponse(prompt, MergedEntitySchema);
    return response.mergedSummary;
  }

  private async linkEpisodeToEntities(
    episode: EpisodicNodeImpl,
    entities: EntityNodeImpl[],
  ): Promise<void> {
    for (const entity of entities) {
      const edge = new EpisodicEdgeImpl({
        uuid: '',
        groupId: episode.groupId,
        sourceNodeUuid: episode.uuid,
        targetNodeUuid: entity.uuid,
        createdAt: utcNow(),
      });

      await edge.save(this.driver);
    }
  }

  private async processExtractedRelations(
    relations: ExtractedRelation[],
    entities: EntityNodeImpl[],
    groupId: string,
    episodeUuid: string,
  ): Promise<void> {
    const entityMap = new Map(entities.map((e) => [e.name, e]));

    for (const relation of relations) {
      // Skip low-confidence relations
      if ((relation.confidence ?? 1.0) < RELATION_CONFIDENCE_THRESHOLD) continue;

      // Negated relations: detect conflict, mark both sides, but don't create an edge
      if (relation.isNegated ?? false) {
        const source = entityMap.get(relation.sourceName);
        const target = entityMap.get(relation.targetName);
        if (source && target) {
          await this._resolveConflict(
            source,
            target,
            relation.relationName,
            episodeUuid,
            groupId,
          );
        }
        continue;
      }

      const source = entityMap.get(relation.sourceName);
      const target = entityMap.get(relation.targetName);

      if (source && target) {
        const existingEdge = await this.findExistingRelation(
          source.uuid,
          target.uuid,
          relation.relationName,
        );

        // Historical relations that already exist: mark as invalidated and skip
        if (existingEdge && (relation.temporalValidity ?? 'current') === 'historical') {
          if (!existingEdge.invalidAt) {
            existingEdge.invalidAt = utcNow();
            await existingEdge.save(this.driver);
          }
          continue;
        }

        if (existingEdge) {
          // Graphiti-style: accumulate episode reference and refresh validAt
          if (!existingEdge.episodes.includes(episodeUuid)) {
            existingEdge.episodes.push(episodeUuid);
          }
          existingEdge.validAt = utcNow();
          // Keep the highest confidence seen across all supporting episodes
          const newConf = relation.confidence ?? 1.0;
          existingEdge.confidence = Math.max(existingEdge.confidence ?? 0, newConf);
          await existingEdge.save(this.driver);
        } else {
          const edge = new EntityEdgeImpl({
            uuid: '',
            groupId,
            sourceNodeUuid: source.uuid,
            targetNodeUuid: target.uuid,
            name: relation.relationName,
            factIds: [],
            episodes: [episodeUuid],
            validAt: utcNow(),
            // Historical relations are stored but immediately marked as invalid
            invalidAt: (relation.temporalValidity ?? 'current') === 'historical' ? utcNow() : undefined,
            createdAt: utcNow(),
            confidence: relation.confidence ?? 1.0,
          });

          await edge.save(this.driver);
        }
      }
    }
  }

  /**
   * When a negated relation is extracted, find the positive counterpart in the
   * graph and cross-mark both the edge and the new episode as disputed.
   *
   * Returns true if a conflict was detected and marked.
   */
  private async _resolveConflict(
    source: EntityNodeImpl,
    target: EntityNodeImpl,
    relationName: string,
    newEpisodeUuid: string,
    groupId: string,
  ): Promise<boolean> {
    // Find active positive edge between the same pair
    const result = await this.driver.executeQuery<any[]>(
      `
      MATCH (src:Entity {uuid: $sourceUuid, groupId: $groupId})
            -[e:RELATES_TO {name: $relationName}]->
            (tgt:Entity {uuid: $targetUuid, groupId: $groupId})
      WHERE e.invalidAt IS NULL
      RETURN e
      LIMIT 1
      `,
      { sourceUuid: source.uuid, targetUuid: target.uuid, relationName, groupId },
    );

    if (result.length === 0) return false;

    const raw = result[0].e;
    const props = (raw?.properties ?? raw) as Record<string, any>;
    const edgeUuid: string = props.uuid ?? '';
    const supportingEpisodes: string[] = props.episodes ?? [];

    // Mark the edge as disputed by the new episode
    const edgeDisputedBy: string[] = [...(props.disputedBy ?? [])];
    if (!edgeDisputedBy.includes(newEpisodeUuid)) edgeDisputedBy.push(newEpisodeUuid);
    await this.driver.executeQuery(
      `MATCH ()-[e:RELATES_TO {uuid: $edgeUuid}]->()
       SET e.disputedBy = $disputedBy`,
      { edgeUuid, disputedBy: edgeDisputedBy },
    );

    // Mark the new episode as disputed by the episodes that support the positive relation
    if (supportingEpisodes.length > 0) {
      await this.driver.executeQuery(
        `MATCH (n:Episodic {uuid: $episodeUuid})
         SET n.disputedBy = $disputedBy`,
        { episodeUuid: newEpisodeUuid, disputedBy: supportingEpisodes },
      );
    }

    console.warn(
      '[conflict] Episode %s disputes relation %s (%s → %s); edge %s marked as disputed',
      newEpisodeUuid,
      relationName,
      source.name,
      target.name,
      edgeUuid,
    );

    return true;
  }

  private async findExistingRelation(
    sourceUuid: string,
    targetUuid: string,
    relationName: string,
  ): Promise<EntityEdgeImpl | null> {
    const result = await this.driver.executeQuery<any[]>(
      `
      MATCH (s:Entity {uuid: $sourceUuid})-[r:RELATES_TO {name: $relationName}]->(t:Entity {uuid: $targetUuid})
      RETURN r
      LIMIT 1
      `,
      { sourceUuid, targetUuid, relationName },
    );

    if (result.length === 0) {
      return null;
    }

    const raw = result[0].r;
    const props = (raw?.properties ?? raw) as Record<string, any>;
    return new EntityEdgeImpl({
      uuid: props.uuid ?? '',
      groupId: props.groupId ?? '',
      sourceNodeUuid: sourceUuid,
      targetNodeUuid: targetUuid,
      name: props.name ?? relationName,
      factIds: props.factIds ?? [],
      episodes: props.episodes ?? [],
      validAt: props.validAt ? new Date(props.validAt) : new Date(),
      invalidAt: props.invalidAt ? new Date(props.invalidAt) : undefined,
      expiredAt: props.expiredAt ? new Date(props.expiredAt) : undefined,
      createdAt: props.createdAt ? new Date(props.createdAt) : new Date(),
      confidence: props.confidence != null ? Number(props.confidence) : undefined,
    });
  }

  /**
   * Search with scores — returns nodes with their RRF/temporal scores attached.
   */
  async searchWithScores(params: SearchParams): Promise<Array<{ node: Node; score: number }>> {
    const embedding = await this.embedder.embed(params.query);
    const groupId = params.groupId || this.defaultGroupId;
    const limit = Math.floor(params.limit || 10);

    const [entityScored, episodicScored, factScored, keywordFacts] = await Promise.all([
      this._searchEntityCommunity(embedding, groupId, limit),
      this._searchEpisodic(embedding, groupId, limit, params),
      this._searchFacts(embedding, groupId, limit),
      this._searchFactsByKeyword(params.query, groupId, limit),
    ]);

    // Merge keyword-matched facts into vector facts (keyword matches fill gaps)
    const mergedFacts = this._mergeFactResults(factScored, keywordFacts);

    // Facts-first architecture: facts are primary, entity/episodic fill remaining slots
    let merged = this._factsFirstMerge(mergedFacts, entityScored, episodicScored, limit, params.query);

    const scoreMap = new Map<string, number>();
    for (const { node, score } of merged) scoreMap.set(node.uuid, score);

    let seedNodes: Node[] = merged.map((m) => m.node);

    const communityNodes = seedNodes.filter(
      (n): n is CommunityNodeImpl => n instanceof CommunityNodeImpl,
    );
    if (communityNodes.length > 0) {
      seedNodes = await this._expandCommunityMembers(seedNodes, communityNodes, groupId);
    }

    if (params.graphExpand) {
      seedNodes = await this._expandByGraph(seedNodes, groupId, params.expandHops ?? 1, limit);
    }

    if (params.queryTime) {
      seedNodes = this._applyTemporalWeighting(
        seedNodes,
        scoreMap,
        params.queryTime,
        params.temporalAlpha ?? 0.3,
        params.halfLifeDays ?? 30,
      );
    }

    return seedNodes.map((node) => ({
      node,
      score: scoreMap.get(node.uuid) ?? 0,
    }));
  }

  async search(params: SearchParams): Promise<Node[]> {
    const embedding = await this.embedder.embed(params.query);
    const groupId = params.groupId || this.defaultGroupId;
    const limit = Math.floor(params.limit || 10);

    // Run entity+community, episodic, and fact searches in parallel.
    const [entityScored, episodicScored, factScored, keywordFacts] = await Promise.all([
      this._searchEntityCommunity(embedding, groupId, limit),
      this._searchEpisodic(embedding, groupId, limit, params),
      this._searchFacts(embedding, groupId, limit),
      this._searchFactsByKeyword(params.query, groupId, limit),
    ]);

    // Merge keyword-matched facts into vector facts (keyword matches fill gaps)
    const mergedFacts = this._mergeFactResults(factScored, keywordFacts);

    // Facts-first architecture: facts are primary, entity/episodic fill remaining slots
    let merged = this._factsFirstMerge(mergedFacts, entityScored, episodicScored, limit, params.query);

    // Build scoreMap (RRF scores) for temporal weighting downstream.
    const scoreMap = new Map<string, number>();
    for (const { node, score } of merged) scoreMap.set(node.uuid, score);

    let seedNodes: Node[] = merged.map((m) => m.node);

    // Community-guided expansion: include member Entity nodes automatically.
    const communityNodes = seedNodes.filter(
      (n): n is CommunityNodeImpl => n instanceof CommunityNodeImpl,
    );
    if (communityNodes.length > 0) {
      seedNodes = await this._expandCommunityMembers(seedNodes, communityNodes, groupId);
    }

    if (params.graphExpand) {
      seedNodes = await this._expandByGraph(seedNodes, groupId, params.expandHops ?? 1, limit);
    }

    // Temporal weighting: re-rank Episodic nodes by proximity to queryTime.
    if (params.queryTime) {
      seedNodes = this._applyTemporalWeighting(
        seedNodes,
        scoreMap,
        params.queryTime,
        params.temporalAlpha ?? 0.3,
        params.halfLifeDays ?? 30,
      );
    }

    return seedNodes;
  }

  /**
   * ANN search over Entity and Community nodes using vector indexes.
   * Runs two parallel index queries (one per label) and merges by score.
   * Falls back to brute-force cosine if the indexes don't exist yet.
   */
  private async _searchEntityCommunity(
    embedding: number[],
    groupId: string,
    limit: number,
  ): Promise<Array<{ node: Node; score: number }>> {
    // Oversample to absorb groupId filtering loss (same ratio as episodic search).
    const vecK = limit * 3;

    const toScored = (r: any): { node: Node; score: number } => {
      const d = r.n?.properties ?? r.n;
      const labels: string[] = r.labels ?? [];
      const node = labels.includes('Community')
        ? new CommunityNodeImpl({ ...d, labels })
        : new EntityNodeImpl({ ...d, labels });
      return { node, score: r.score ?? 0 };
    };

    try {
      // Primary: two parallel ANN index queries (Entity + Community)
      const [entityRows, communityRows] = await Promise.all([
        this.driver.executeQuery<any[]>(
          `CALL db.index.vector.queryNodes('entity_embedding', $vecK, $embedding)
           YIELD node AS n, score
           WHERE n.groupId = $groupId
           RETURN n, labels(n) AS labels, score
           LIMIT $limit`,
          { groupId, embedding, vecK, limit },
        ),
        this.driver.executeQuery<any[]>(
          `CALL db.index.vector.queryNodes('community_embedding', $vecK, $embedding)
           YIELD node AS n, score
           WHERE n.groupId = $groupId
           RETURN n, labels(n) AS labels, score
           LIMIT $limit`,
          { groupId, embedding, vecK, limit },
        ),
      ]);

      return [...entityRows, ...communityRows]
        .map(toScored)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
    } catch {
      // Fallback: brute-force cosine (indexes not yet created)
      const results = await this.driver.executeQuery<any[]>(
        `MATCH (n)
         WHERE n.groupId = $groupId
           AND (n:Entity OR n:Community)
           AND n.embedding IS NOT NULL
         WITH n,
           reduce(s = 0.0, i IN range(0, size(n.embedding)-1) |
             s + (n.embedding[i] * $embedding[i])
           ) AS score
         ORDER BY score DESC
         LIMIT $limit
         RETURN n, labels(n) AS labels, score`,
        { groupId, embedding, limit },
      );
      return results.map(toScored);
    }
  }

  /**
   * ANN search over Episodic nodes using the `episodic_content` vector index.
   * Falls back to brute-force cosine if the index doesn't exist yet.
   * Date range filters (validFrom / validTo) are applied inline.
   */
  private async _searchEpisodic(
    embedding: number[],
    groupId: string,
    limit: number,
    params: SearchParams,
  ): Promise<Array<{ node: Node; score: number }>> {
    const dateParts: string[] = [];
    if (params.validFrom) dateParts.push('n.validAt >= datetime($validFrom)');
    if (params.validTo) dateParts.push('n.validAt <= datetime($validTo)');
    const dateFilter = dateParts.length > 0 ? `AND (${dateParts.join(' AND ')})` : '';

    // Oversample from the index to absorb groupId filtering loss.
    const vecK = limit * 3;
    const qp: Record<string, any> = { groupId, embedding, limit, vecK };
    if (params.validFrom) qp.validFrom = params.validFrom.toISOString();
    if (params.validTo) qp.validTo = params.validTo.toISOString();

    const toScored = (r: any): { node: Node; score: number } => {
      const d = r.n?.properties ?? r.n;
      return {
        node: new EpisodicNodeImpl({
          ...d,
          labels: r.labels ?? [],
          validAt: coerceDate(d.validAt),
          invalidAt: d.invalidAt ? coerceDate(d.invalidAt) : undefined,
        }),
        score: r.score ?? 0,
      };
    };

    try {
      // Primary: vector index (fast ANN)
      const results = await this.driver.executeQuery<any[]>(
        `CALL db.index.vector.queryNodes('episodic_content', $vecK, $embedding)
         YIELD node AS n, score
         WHERE n.groupId = $groupId
           ${dateFilter}
         RETURN n, labels(n) AS labels, score
         LIMIT $limit`,
        qp,
      );
      return results.map(toScored);
    } catch {
      // Fallback: brute-force on n.embedding (exists on all episodic nodes,
      // including those inserted before contentEmbedding was added).
      const results = await this.driver.executeQuery<any[]>(
        `MATCH (n:Episodic {groupId: $groupId})
         WHERE n.embedding IS NOT NULL
           ${dateFilter}
         WITH n,
           reduce(s = 0.0, i IN range(0, size(n.embedding)-1) |
             s + (n.embedding[i] * $embedding[i])
           ) AS score
         ORDER BY score DESC
         LIMIT $limit
         RETURN n, labels(n) AS labels, score`,
        qp,
      );
      return results.map(toScored);
    }
  }

  /**
   * ANN search over Fact nodes (consolidated knowledge from the Sleep Engine).
   * Falls back to brute-force cosine if the vector index doesn't exist yet.
   */
  private async _searchFacts(
    embedding: number[],
    groupId: string,
    limit: number,
  ): Promise<Array<{ node: Node; score: number }>> {
    const vecK = limit * 3;

    const toScored = (r: any): { node: Node; score: number } => {
      const d = r.n?.properties ?? r.n;
      return {
        node: new FactNodeImpl({ ...d, labels: r.labels ?? ['Fact'] }),
        score: r.score ?? 0,
      };
    };

    try {
      const results = await this.driver.executeQuery<any[]>(
        `CALL db.index.vector.queryNodes('fact_embedding', $vecK, $embedding)
         YIELD node AS n, score
         WHERE n.groupId = $groupId
         RETURN n, labels(n) AS labels, score
         LIMIT $limit`,
        { groupId, embedding, vecK, limit },
      );
      return results.map(toScored);
    } catch {
      // Fallback: brute-force cosine (index not yet created or no Fact nodes)
      try {
        const results = await this.driver.executeQuery<any[]>(
          `MATCH (n:Fact {groupId: $groupId})
           WHERE n.embedding IS NOT NULL
           WITH n,
             reduce(s = 0.0, i IN range(0, size(n.embedding)-1) |
               s + (n.embedding[i] * $embedding[i])
             ) AS score
           ORDER BY score DESC
           LIMIT $limit
           RETURN n, labels(n) AS labels, score`,
          { groupId, embedding, limit },
        );
        return results.map(toScored);
      } catch {
        // No Fact nodes exist yet — return empty list
        return [];
      }
    }
  }

  /**
   * Keyword-based fact search: find facts that contain specific identifiers
   * (ORDER numbers, AWB numbers, etc.) from the query.
   * This supplements vector search which can't distinguish similar IDs.
   */
  private async _searchFactsByKeyword(
    query: string,
    groupId: string,
    limit: number,
  ): Promise<Array<{ node: Node; score: number }>> {
    const ids = this._extractTypedIdentifiers(query);
    if (ids.length === 0) return [];

    // Build CONTAINS tokens from the identifiers
    const tokens = ids.map(id => {
      if (id.type === 'ORDER') return `ORDER-${id.value}`;
      if (id.type === 'PO') return `PO${id.value}`;
      if (id.type === 'AWB') return id.value;
      return id.value;
    });

    try {
      const results = await this.driver.executeQuery<any[]>(
        `MATCH (n:Fact {groupId: $groupId})
         WHERE ANY(t IN $tokens WHERE n.content CONTAINS t)
         RETURN n, labels(n) AS labels, 0.85 AS score
         LIMIT $limit`,
        { groupId, tokens, limit: limit * 2 },
      );
      return results.map((r: any) => {
        const d = r.n?.properties ?? r.n;
        return {
          node: new FactNodeImpl({ ...d, labels: r.labels ?? ['Fact'] }),
          score: r.score ?? 0.85,
        };
      });
    } catch {
      return [];
    }
  }

  /**
   * Merge keyword-matched facts into vector facts.
   * Keyword matches that aren't already in the vector results get added
   * with their keyword score. Deduplicates by UUID.
   */
  private _mergeFactResults(
    vectorFacts: Array<{ node: Node; score: number }>,
    keywordFacts: Array<{ node: Node; score: number }>,
  ): Array<{ node: Node; score: number }> {
    if (keywordFacts.length === 0) return vectorFacts;

    const seen = new Set(vectorFacts.map(f => f.node.uuid));
    const merged = [...vectorFacts];
    for (const kf of keywordFacts) {
      if (!seen.has(kf.node.uuid)) {
        seen.add(kf.node.uuid);
        merged.push(kf);
      }
    }
    return merged;
  }

  /**
   * Facts-first merge: facts are the primary search source (like Mem0).
   * Entity/episodic results fill remaining slots after facts.
   *
   * Strategy:
   *   1. Take all facts sorted by vector score as the primary results
   *   2. Apply identifier-aware filtering (penalize facts about wrong ORDER/AWB)
   *   3. RRF-merge entity + episodic as supplementary results
   *   4. Interleave: facts get priority slots, supplementary fills the rest
   */
  private _factsFirstMerge(
    factScored: Array<{ node: Node; score: number }>,
    entityScored: Array<{ node: Node; score: number }>,
    episodicScored: Array<{ node: Node; score: number }>,
    limit: number,
    query?: string,
  ): Array<{ node: Node; score: number }> {
    // Identifier-aware filtering: extract specific IDs from the query
    // and penalize facts that mention DIFFERENT IDs of the same type
    const queryIds = query ? this._extractTypedIdentifiers(query) : [];
    const applyIdFilter = queryIds.length > 0;

    // Facts sorted by vector similarity — these are our primary results
    let facts = factScored
      .filter(f => f.score > 0.5) // Minimum relevance threshold
      .slice(0, limit * 2); // Take more candidates for filtering

    if (applyIdFilter) {
      facts = facts.map(({ node, score }) => {
        const content = ((node as any).content ?? '').toString();
        const factIds = this._extractTypedIdentifiers(content);
        if (factIds.length === 0) return { node, score }; // Neutral — no IDs in fact

        // Check if this fact mentions any of the query's identifiers
        let hasMatch = false;
        let hasWrongId = false;
        for (const qId of queryIds) {
          for (const fId of factIds) {
            if (fId.type !== qId.type) continue; // Different ID types — ignore
            if (fId.value === qId.value) {
              hasMatch = true; // Exact match — boost
            } else {
              hasWrongId = true; // Same type, different value — penalize
            }
          }
        }

        if (hasMatch) return { node, score: score * 1.2 }; // Boost exact matches
        if (hasWrongId) return { node, score: score * 0.1 }; // Heavy penalty for wrong ID
        return { node, score };
      }).sort((a, b) => b.score - a.score);
    }

    facts = facts.slice(0, limit);

    // RRF-merge entity + episodic as supplementary
    const supplementary = this._rrfMerge([entityScored, episodicScored], limit);

    // Assign scores: facts get higher base scores than supplementary
    // This ensures facts rank above supplementary when interleaved
    const seen = new Set<string>();
    const merged: Array<{ node: Node; score: number }> = [];

    // Facts first — score normalized to [0.5, 1.0] range
    for (const { node, score } of facts) {
      if (!seen.has(node.uuid)) {
        seen.add(node.uuid);
        merged.push({ node, score: 0.5 + score * 0.5 });
      }
    }

    // Fill remaining slots with supplementary — score in [0.0, 0.5) range
    for (const { node, score } of supplementary) {
      if (!seen.has(node.uuid) && merged.length < limit) {
        seen.add(node.uuid);
        merged.push({ node, score: score * 0.5 });
      }
    }

    return merged.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  /**
   * Extract typed identifiers from text — ORDER numbers, AWB numbers, PO numbers.
   * Returns type + value pairs so we can compare only within the same ID type.
   * e.g. ORDER-20260128 in query should penalize ORDER-20260120 facts but NOT AWB facts.
   */
  private _extractTypedIdentifiers(text: string): Array<{ type: string; value: string }> {
    const ids: Array<{ type: string; value: string }> = [];
    // ORDER-XXXXXXXX
    for (const m of text.matchAll(/ORDER-(\d{5,})/gi)) {
      ids.push({ type: 'ORDER', value: m[1] });
    }
    // PO numbers (POXXXXXXXX)
    for (const m of text.matchAll(/PO(\d{5,})/gi)) {
      ids.push({ type: 'PO', value: m[1] });
    }
    // AWB numbers (947-XXXXXXXX or standalone 10+ digit sequences after AWB)
    for (const m of text.matchAll(/AWB[\s-]*(\d{3}-?\d{5,})/gi)) {
      ids.push({ type: 'AWB', value: m[1].replace(/-/g, '') });
    }
    return ids;
  }

  /**
   * Reciprocal Rank Fusion: merge multiple ranked lists into one.
   * k=60 is the standard constant from the original RRF paper.
   */
  private _rrfMerge(
    lists: Array<Array<{ node: Node; score: number }>>,
    limit: number,
    k: number = 60,
  ): Array<{ node: Node; score: number }> {
    const scores = new Map<string, { node: Node; score: number }>();
    for (const list of lists) {
      list.forEach(({ node }, rank) => {
        const rrf = 1 / (k + rank + 1);
        const existing = scores.get(node.uuid);
        if (existing) {
          existing.score += rrf;
        } else {
          scores.set(node.uuid, { node, score: rrf });
        }
      });
    }
    return [...scores.values()].sort((a, b) => b.score - a.score).slice(0, limit);
  }



  /**
   * Re-rank nodes by blending the original semantic similarity score with a
   * temporal proximity bonus for Episodic nodes.
   *
   * Formula:
   *   adjustedScore = baseScore * (1 + alpha * proximityScore * contemporaneity)
   *
   * where:
   *   proximityScore  = exp(-|validAt - queryTime| / halfLifeDays)
   *   contemporaneity = exp(-retroactiveDays / 30)   [1.0 for real-time recording]
   *
   * Entity and Community nodes are returned at their original semantic score.
   */
  private _applyTemporalWeighting(
    nodes: Node[],
    scoreMap: Map<string, number>,
    queryTime: Date,
    alpha: number,
    halfLifeDays: number,
  ): Node[] {
    const scored = nodes.map((node) => {
      const baseScore = scoreMap.get(node.uuid) ?? 0;

      if (!(node instanceof EpisodicNodeImpl)) {
        return { node, score: baseScore };
      }

      const episodic = node as EpisodicNodeImpl;
      const daysDelta =
        Math.abs(episodic.validAt.getTime() - queryTime.getTime()) / 86_400_000;
      const proximityScore = Math.exp(-daysDelta / halfLifeDays);
      const contemporaneity = Math.exp(-Number(episodic.retroactiveDays ?? 0) / 30);
      const adjustedScore = baseScore * (1 + alpha * proximityScore * contemporaneity);

      return { node, score: adjustedScore };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.map((s) => s.node);
  }

  private async _expandCommunityMembers(
    seedNodes: Node[],
    communityNodes: CommunityNodeImpl[],
    groupId: string,
  ): Promise<Node[]> {
    const communityUuids = communityNodes.map(n => n.uuid).filter(Boolean);
    const memberResults = await this.driver.executeQuery<any[]>(
      `MATCH (c:Community)-[:HAS_MEMBER]->(e:Entity {groupId: $groupId})
       WHERE c.uuid IN $communityUuids
       RETURN e AS n, labels(e) AS labels`,
      { communityUuids, groupId },
    );
    const expanded = new Map<string, Node>();
    for (const node of seedNodes) expanded.set(node.uuid, node);
    for (const row of memberResults) {
      const d = row.n?.properties ?? row.n;
      if (d?.uuid && !expanded.has(d.uuid)) {
        expanded.set(d.uuid, new EntityNodeImpl({ ...d, labels: row.labels ?? [] }));
      }
    }
    return [...expanded.values()];
  }

  private async _expandByGraph(
    seedNodes: Node[],
    groupId: string,
    hops: number,
    originalLimit: number,
  ): Promise<Node[]> {
    const seedEntities = seedNodes.filter(
      (n): n is EntityNodeImpl => n instanceof EntityNodeImpl,
    );
    if (seedEntities.length === 0) return seedNodes;

    const seedUuids = seedEntities.map(e => e.uuid).filter(Boolean);
    const expandLimit = Math.floor(originalLimit * 2);

    const neighborResults = await this.driver.executeQuery<any[]>(
      `MATCH (seed:Entity {groupId: $groupId})-[:RELATES_TO*1..${hops}]-(neighbor:Entity {groupId: $groupId})
       WHERE seed.uuid IN $seedUuids AND NOT neighbor.uuid IN $seedUuids
       WITH neighbor, labels(neighbor) AS nodeLabels, count(*) AS pathCount
       ORDER BY pathCount DESC
       RETURN neighbor AS n, nodeLabels
       LIMIT $expandLimit`,
      { groupId, seedUuids, expandLimit },
    );

    const expanded = new Map<string, Node>();
    for (const node of seedNodes) expanded.set(node.uuid, node);
    for (const row of neighborResults) {
      const d = row.n?.properties ?? row.n;
      if (d?.uuid && !expanded.has(d.uuid)) {
        expanded.set(
          d.uuid,
          new EntityNodeImpl({ ...d, labels: row.nodeLabels ?? d?.labels ?? [] }),
        );
      }
    }
    return [...expanded.values()];
  }

  async getNode(uuid: string): Promise<Node | null> {
    return Node.getByUuid(this.driver, uuid);
  }

  async getEdge(uuid: string): Promise<Edge | null> {
    return Edge.getByUuid(this.driver, uuid);
  }

  async deleteNode(uuid: string): Promise<void> {
    const node = await this.getNode(uuid);
    if (node) {
      if (node instanceof EntityNodeImpl && (node as EntityNodeImpl).pinned) {
        throw new Error(`Cannot delete pinned entity "${node.name}" (uuid: ${uuid}). Unpin it first.`);
      }
      await node.delete(this.driver);
    }
  }

  async deleteEdge(uuid: string): Promise<void> {
    const edge = await this.getEdge(uuid);
    if (edge) {
      await edge.delete(this.driver);
    }
  }

  /**
   * Create (or merge) an entity directly, bypassing LLM extraction.
   * Useful for explicit contact creation via tools.
   */
  async createEntity(params: {
    name: string;
    entityType: string;
    summary: string;
    groupId?: string;
    metadata?: Record<string, unknown>;
    pinned?: boolean;
    labels?: string[];
  }): Promise<EntityNodeImpl> {
    const groupId = params.groupId || this.defaultGroupId;
    const existing = await this.findExistingEntity(params.name, groupId);

    if (existing) {
      // Merge: update summary, metadata, pinned
      if (params.summary && params.summary !== existing.summary) {
        const mergedSummary = await this.mergeEntitySummary(
          params.name,
          existing.summary,
          params.summary,
        );
        existing.summary = mergedSummary;
        existing.summaryEmbedding = await this.embedder.embed(mergedSummary);
      }
      if (params.metadata && Object.keys(params.metadata).length > 0) {
        existing.metadata = { ...(existing.metadata ?? {}), ...params.metadata };
      }
      if (params.pinned !== undefined) {
        existing.pinned = params.pinned;
      }
      if (!existing.entityType || existing.entityType === 'Unknown') {
        existing.entityType = params.entityType;
      }
      await existing.save(this.driver);
      return existing;
    }

    const embedding = await this.embedder.embed(params.summary);
    const entityNode = new EntityNodeImpl({
      uuid: '',
      name: params.name,
      groupId,
      entityType: params.entityType,
      summary: params.summary,
      summaryEmbedding: embedding,
      labels: params.labels ?? [],
      createdAt: utcNow(),
      metadata: params.metadata,
      pinned: params.pinned,
    });

    await entityNode.save(this.driver);
    return entityNode;
  }

  /**
   * Create (or update) a RELATES_TO edge between two entities resolved by name.
   */
  async createRelation(params: {
    sourceName: string;
    targetName: string;
    relationName: string;
    groupId?: string;
    confidence?: number;
  }): Promise<EntityEdgeImpl | null> {
    const groupId = params.groupId || this.defaultGroupId;
    const source = await this.findExistingEntity(params.sourceName, groupId);
    const target = await this.findExistingEntity(params.targetName, groupId);

    if (!source || !target) return null;

    const existing = await this.findExistingRelation(
      source.uuid,
      target.uuid,
      params.relationName,
    );

    if (existing) {
      existing.validAt = utcNow();
      existing.confidence = Math.max(existing.confidence ?? 0, params.confidence ?? 1.0);
      await existing.save(this.driver);
      return existing;
    }

    const edge = new EntityEdgeImpl({
      uuid: '',
      groupId,
      sourceNodeUuid: source.uuid,
      targetNodeUuid: target.uuid,
      name: params.relationName,
      factIds: [],
      episodes: [],
      validAt: utcNow(),
      createdAt: utcNow(),
      confidence: params.confidence ?? 1.0,
    });

    await edge.save(this.driver);
    return edge;
  }

  /** Parse a JSON string stored in Neo4j back into an object. */
  static _parseMetadataJson(raw: any): Record<string, unknown> | undefined {
    if (!raw || raw === 'null') return undefined;
    try {
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
      return parsed && typeof parsed === 'object' ? parsed : undefined;
    } catch {
      return undefined;
    }
  }

  async close(): Promise<void> {
    await this.driver.close();
  }

  async executeQuery<T = any>(query: string, params?: Record<string, any>): Promise<T> {
    return this.driver.executeQuery(query, params);
  }

  async createIndexes(): Promise<void> {
    return this.driver.createIndexes();
  }

  async clearDatabase(): Promise<void> {
    await this.driver.executeQuery('MATCH (n) DETACH DELETE n');
  }

  async testConnection(): Promise<void> {
    await this.driver.executeQuery('RETURN 1');
  }

  /**
   * Traverse the graph starting from an entity, collecting reachable nodes and edges.
   */
  async traverse(params: TraverseParams): Promise<TraverseResult> {
    if (!params.startEntityName && !params.startEntityUuid) {
      throw new Error('traverse() requires either startEntityName or startEntityUuid');
    }

    const maxHops   = params.maxHops   ?? 2;
    const direction = params.direction ?? 'both';
    const limit     = params.limit     ?? 50;
    const groupId   = params.groupId   || this.defaultGroupId;

    return this._traverseCypher(params, maxHops, direction, limit, groupId);
  }

  private async _traverseCypher(
    params: TraverseParams,
    maxHops: number,
    direction: 'outgoing' | 'incoming' | 'both',
    limit: number,
    groupId: string,
  ): Promise<TraverseResult> {
    // Query 1: resolve start node
    const [startQ, startId] = params.startEntityUuid
      ? [
          'MATCH (n:Entity {uuid: $id, groupId: $groupId}) RETURN n, labels(n) AS labels LIMIT 1',
          params.startEntityUuid,
        ]
      : [
          'MATCH (n:Entity {name: $id, groupId: $groupId}) RETURN n, labels(n) AS labels LIMIT 1',
          params.startEntityName!,
        ];

    const startResult = await this.driver.executeQuery<any[]>(startQ, { id: startId, groupId });
    if (startResult.length === 0) return { start: null, nodes: [], edges: [] };
    const sd = startResult[0].n?.properties ?? startResult[0].n;
    const startNode = new EntityNodeImpl({ ...sd, labels: startResult[0].labels ?? sd?.labels ?? [] });

    // Query 2: variable-length path neighbours
    // maxHops is interpolated directly — Cypher parameter syntax doesn't support *$n..$m
    const relPat =
      direction === 'outgoing'
        ? `-[:RELATES_TO*1..${maxHops}]->`
        : direction === 'incoming'
          ? `<-[:RELATES_TO*1..${maxHops}]-`
          : `-[:RELATES_TO*1..${maxHops}]-`;

    const neighborResults = await this.driver.executeQuery<any[]>(
      `MATCH (start:Entity {uuid: $startUuid, groupId: $groupId})
       MATCH (start)${relPat}(neighbor:Entity {groupId: $groupId})
       WHERE neighbor.uuid <> $startUuid
       RETURN DISTINCT neighbor AS n, labels(neighbor) AS nodeLabels
       LIMIT $limit`,
      { startUuid: startNode.uuid, groupId, limit },
    );

    const nodeMap = new Map<string, EntityNodeImpl>();
    for (const row of neighborResults) {
      const d = row.n?.properties ?? row.n;
      if (d?.uuid && !nodeMap.has(d.uuid))
        nodeMap.set(d.uuid, new EntityNodeImpl({ ...d, labels: row.nodeLabels ?? d?.labels ?? [] }));
    }

    if (nodeMap.size === 0) return { start: startNode, nodes: [], edges: [] };

    // Query 3: edges within the discovered subgraph
    const allUuids = [startNode.uuid, ...nodeMap.keys()];
    const edgeResults = await this.driver.executeQuery<any[]>(
      `MATCH (a:Entity {groupId: $groupId})-[e:RELATES_TO]->(b:Entity {groupId: $groupId})
       WHERE a.uuid IN $uuids AND b.uuid IN $uuids
       RETURN e`,
      { groupId, uuids: allUuids },
    );

    const edgeMap = new Map<string, EntityEdgeImpl>();
    for (const row of edgeResults) {
      const ed = row.e?.properties ?? row.e;
      if (!ed?.uuid || edgeMap.has(ed.uuid)) continue;
      edgeMap.set(
        ed.uuid,
        new EntityEdgeImpl({
          uuid: ed.uuid,
          groupId: ed.groupId || groupId,
          sourceNodeUuid: ed.sourceNodeUuid,
          targetNodeUuid: ed.targetNodeUuid,
          name: ed.name || '',
          factIds: ed.factIds || [],
          episodes: ed.episodes || [],
          validAt: coerceDate(ed.validAt),
          invalidAt: ed.invalidAt ? coerceDate(ed.invalidAt) : undefined,
          expiredAt: ed.expiredAt ? coerceDate(ed.expiredAt) : undefined,
          createdAt: coerceDate(ed.createdAt),
        }),
      );
    }

    return { start: startNode, nodes: [...nodeMap.values()], edges: [...edgeMap.values()] };
  }

}
