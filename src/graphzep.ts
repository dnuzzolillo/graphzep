import { z } from 'zod';
import {
  GraphDriver,
  EpisodicNode,
  EpisodeType,
} from './types/index.js';
import { Node, EntityNodeImpl, EpisodicNodeImpl, CommunityNodeImpl } from './core/nodes.js';
import { Edge, EntityEdgeImpl, EpisodicEdgeImpl, CommunityEdgeImpl } from './core/edges.js';
import { BaseLLMClient } from './llm/client.js';
import { BaseEmbedderClient } from './embedders/client.js';
import { utcNow } from './utils/datetime.js';

export interface GraphzepConfig {
  driver: GraphDriver;
  llmClient: BaseLLMClient;
  embedder: BaseEmbedderClient;
  groupId?: string;
  ensureAscii?: boolean;
}

export interface AddEpisodeParams {
  content: string;
  episodeType?: EpisodeType;
  referenceId?: string;
  groupId?: string;
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

export class Graphzep {
  private driver: GraphDriver;
  private llmClient: BaseLLMClient;
  private embedder: BaseEmbedderClient;
  private defaultGroupId: string;
  private ensureAscii: boolean;

  constructor(config: GraphzepConfig) {
    this.driver = config.driver;
    this.llmClient = config.llmClient;
    this.embedder = config.embedder;
    this.defaultGroupId = config.groupId || 'default';
    this.ensureAscii = config.ensureAscii ?? false;
  }

  async addEpisode(params: AddEpisodeParams): Promise<EpisodicNode> {
    const groupId = params.groupId || this.defaultGroupId;
    const episodeType = params.episodeType || EpisodeType.TEXT;

    const embedding = await this.embedder.embed(params.content);

    // Create episodic node
    const episodicNode = new EpisodicNodeImpl({
      uuid: '',
      name: params.content.substring(0, 50),
      groupId,
      episodeType,
      content: params.content,
      embedding,
      validAt: utcNow(),
      referenceId: params.referenceId,
      labels: [],
      createdAt: utcNow(),
    });

    await episodicNode.save(this.driver);

    const existingEntities = await this.fetchExistingEntities(groupId, embedding);
    const extractedData = await this.extractEntitiesAndRelations(params.content, existingEntities);

    const entityNodes = await this.processExtractedEntities(extractedData.entities, groupId);

    await this.linkEpisodeToEntities(episodicNode, entityNodes);

    await this.processExtractedRelations(extractedData.relations, entityNodes, groupId, episodicNode.uuid);

    return episodicNode;
  }

  private async fetchExistingEntities(
    groupId: string,
    episodeEmbedding?: number[],
  ): Promise<Array<{ uuid: string; name: string; entityType: string }>> {
    if (!episodeEmbedding) {
      const result = await this.driver.executeQuery<any[]>(
        `MATCH (n:Entity {groupId: $groupId})
         RETURN n.uuid AS uuid, n.name AS name, n.entityType AS entityType
         ORDER BY n.createdAt DESC
         LIMIT 20`,
        { groupId },
      );
      return result
        .map(r => ({ uuid: r.uuid ?? '', name: r.name ?? '', entityType: r.entityType ?? '' }))
        .filter(e => e.name);
    }

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

    if (candidates.length === 0) return [];

    // Re-rank: semantic relevance + recency (exponential decay, half-life ~7 days)
    const now = Date.now();
    const ALPHA = 0.7;   // weight for semantic score
    const LAMBDA = 0.1;  // decay rate: score = e^(-lambda * ageDays)

    return candidates
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

Respond with valid JSON:
{
  "entities": [
    {
      "name": "string",
      "entityType": "${entityTypesStr.split(' | ')[0]} | ...",
      "summary": "string",
      "confidence": 0.0
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
        });

        await entityNode.save(this.driver);
        processedEntities.push(entityNode);
      }
    }

    return processedEntities;
  }

  private async findExistingEntity(name: string, groupId: string): Promise<EntityNodeImpl | null> {
    const result = await this.driver.executeQuery<any[]>(
      `
      MATCH (n:Entity {name: $name, groupId: $groupId})
      RETURN n
      LIMIT 1
      `,
      { name, groupId },
    );

    if (result.length === 0) {
      return null;
    }

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
      // Skip negated relations — "Alice does NOT work at ACME" should not create an edge
      if (relation.isNegated ?? false) continue;

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
          });

          await edge.save(this.driver);
        }
      }
    }
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
    });
  }

  async search(params: SearchParams): Promise<Node[]> {
    const embedding = await this.embedder.embed(params.query);
    const groupId = params.groupId || this.defaultGroupId;
    const limit = Math.floor(params.limit || 10);

    const query = `
      MATCH (n)
      WHERE n.groupId = $groupId
        AND (n:Entity OR n:Episodic OR n:Community)
        AND n.embedding IS NOT NULL
      WITH n,
        reduce(similarity = 0.0, i IN range(0, size(n.embedding)-1) |
          similarity + (n.embedding[i] * $embedding[i])
        ) AS similarity
      ORDER BY similarity DESC
      LIMIT $limit
      RETURN n, labels(n) as labels
    `;

    const results = await this.driver.executeQuery<any[]>(query, {
      groupId,
      embedding,
      limit,
    });

    const seedNodes: Node[] = results.map((result) => {
      const nodeData = result.n.properties || result.n;
      const labels = result.labels || [];

      if (labels.includes('Entity')) {
        return new EntityNodeImpl({ ...nodeData, labels });
      } else if (labels.includes('Episodic')) {
        return new EpisodicNodeImpl({ ...nodeData, labels });
      } else if (labels.includes('Community')) {
        return new CommunityNodeImpl({ ...nodeData, labels });
      }

      throw new Error(`Unknown node type for labels: ${labels}`);
    });

    if (!params.graphExpand) return seedNodes;

    return this._expandByGraph(seedNodes, groupId, params.expandHops ?? 1, limit);
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
       RETURN DISTINCT neighbor AS n, labels(neighbor) AS nodeLabels
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
      await node.delete(this.driver);
    }
  }

  async deleteEdge(uuid: string): Promise<void> {
    const edge = await this.getEdge(uuid);
    if (edge) {
      await edge.delete(this.driver);
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

    const coerceDate = (v: any): Date => {
      if (v instanceof Date) return v;
      if (v && typeof v === 'object' && typeof v.toString === 'function') {
        const d = new Date(v.toString());
        if (!isNaN(d.getTime())) return d;
      }
      if (typeof v === 'string') return new Date(v);
      return new Date();
    };

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
