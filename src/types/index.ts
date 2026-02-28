export interface GraphzepClients {
  driver: GraphDriver;
  llmClient: LLMClient;
  embedder: EmbedderClient;
  crossEncoder: CrossEncoderClient;
  ensureAscii?: boolean;
}

export interface GraphDriver {
  provider: GraphProvider;
  executeQuery<T = any>(query: string, params?: Record<string, any>): Promise<T>;
  close(): Promise<void>;
  createIndexes(): Promise<void>;
}

export enum GraphProvider {
  NEO4J = 'neo4j',
  FALKORDB = 'falkordb',
  NEPTUNE = 'neptune',
  RDF = 'rdf',
}

export interface LLMClient {
  generateResponse<T = any>(prompt: string, schema?: any): Promise<T>;
  generateStructuredResponse<T = any>(prompt: string, schema: any): Promise<T>;
}

export interface EmbedderClient {
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}

export interface CrossEncoderClient {
  rerank(query: string, documents: string[]): Promise<number[]>;
}

export enum EpisodeType {
  MESSAGE = 'message',
  JSON = 'json',
  TEXT = 'text',
}

export interface BaseNode {
  uuid: string;
  name: string;
  groupId: string;
  labels: string[];
  createdAt: Date;
}

export interface BaseEdge {
  uuid: string;
  groupId: string;
  sourceNodeUuid: string;
  targetNodeUuid: string;
  createdAt: Date;
}

export interface EntityNode extends BaseNode {
  entityType: string;
  summary: string;
  summaryEmbedding?: number[];
  factIds?: string[];
}

export interface EpisodicNode extends BaseNode {
  episodeType: EpisodeType;
  content: string;
  embedding?: number[];
  validAt: Date;
  invalidAt?: Date;
  createdAt: Date;
  referenceId?: string;
  /** Days between validAt (when event occurred) and createdAt (ingestion time). 0 = contemporaneous. */
  retroactiveDays?: number;
  /** UUIDs of episodes that contradict this episode. */
  disputedBy?: string[];
}

export interface CommunityNode extends BaseNode {
  communityLevel: number;
  summary: string;
  summaryEmbedding?: number[];
  factIds?: string[];
  /** UUIDs of the Entity nodes that belong to this community */
  memberEntityIds?: string[];
  /** Number of member entities */
  memberCount?: number;
  /** Lowercase kebab-case domain tags, e.g. ["world-history", "sports"] */
  domainHints?: string[];
  /** 0–1 score reflecting how dense / factually rich this community is */
  importanceScore?: number;
  /** Timestamp of the most recent full community rebuild */
  lastFullRebuild?: Date;
}

export interface EntityEdge extends BaseEdge {
  name: string;
  factIds: string[];
  episodes: string[];
  expiredAt?: Date;
  validAt: Date;
  invalidAt?: Date;
  /** UUIDs of episodes that dispute this relation. */
  disputedBy?: string[];
}

export interface EpisodicEdge extends BaseEdge {}

export interface CommunityEdge extends BaseEdge {
  name: string;
  description?: string;
  factIds?: string[];
}
