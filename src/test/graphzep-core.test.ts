import { describe, it, before, after, mock } from 'node:test';
import assert from 'node:assert';
import { Graphzep } from '../graphzep.js';
import { GraphProvider, EpisodeType } from '../types/index.js';
import { EntityNodeImpl, EpisodicNodeImpl } from '../core/nodes.js';
import { EntityEdgeImpl, EpisodicEdgeImpl } from '../core/edges.js';

describe('Graphzep Core', () => {
  const mockDriver = {
    provider: GraphProvider.NEO4J,
    executeQuery: mock.fn(async (...args: any[]): Promise<any[]> => []),
    close: mock.fn(async () => {}),
  };

  const mockLLMClient = {
    generateResponse: mock.fn(async () => ({ content: 'test response' })),
    generateStructuredResponse: mock.fn(async (prompt: string) => {
      if (prompt.includes('maintaining a knowledge graph')) {
        return { mergedSummary: 'Updated summary with merged context' };
      }
      return {
        entities: [
          { name: 'Alice', entityType: 'Person', summary: 'A person named Alice' },
          { name: 'Bob', entityType: 'Person', summary: 'A person named Bob' },
        ],
        relations: [{ sourceName: 'Alice', targetName: 'Bob', relationName: 'KNOWS' }],
      };
    }),
  };

  const mockEmbedder = {
    embed: mock.fn(async () => new Array(384).fill(0.1)),
    embedBatch: mock.fn(async (texts: string[]) => texts.map(() => new Array(384).fill(0.1))),
  };

  let graphzep: Graphzep;

  before(() => {
    graphzep = new Graphzep({
      driver: mockDriver as any,
      llmClient: mockLLMClient as any,
      embedder: mockEmbedder as any,
      groupId: 'test-group',
    });
  });

  after(async () => {
    await graphzep.close();
  });

  describe('addEpisode', () => {
    it('should add an episode and extract entities', async () => {
      const episodeContent = 'Alice met Bob at the conference.';

      mockDriver.executeQuery.mock.mockImplementation(async (query: string, params?: any) => {
        if (query.includes('MATCH (n:Entity')) {
          return [];
        }
        if (query.includes('MATCH (s:Entity')) {
          return [];
        }
        return [];
      });

      const episode = await graphzep.addEpisode({
        content: episodeContent,
        episodeType: EpisodeType.TEXT,
      });

      assert(episode instanceof EpisodicNodeImpl);
      assert.strictEqual(episode.content, episodeContent);
      assert.strictEqual(episode.episodeType, EpisodeType.TEXT);
      assert.strictEqual(episode.groupId, 'test-group');

      assert(mockLLMClient.generateStructuredResponse.mock.calls.length > 0);

      assert(mockEmbedder.embed.mock.calls.length > 0);
      const embedCall = mockEmbedder.embed.mock.calls[0] as any;
      assert.strictEqual(embedCall.arguments[0], episodeContent);
    });

    it('should link episode to existing entities', async () => {
      const episodeContent = 'Alice and Bob had lunch together.';

      const existingAlice = {
        uuid: 'alice-uuid',
        name: 'Alice',
        entityType: 'Person',
        summary: 'A person named Alice',
        groupId: 'test-group',
        labels: ['Entity'],
        createdAt: new Date(),
      };

      (mockDriver.executeQuery as any).mock.mockImplementation(
        async (query: string, params: any) => {
          if (query.includes('MATCH (n:Entity') && params?.name === 'Alice') {
            return [{ n: existingAlice }];
          }
          return [];
        },
      );

      const episode = await graphzep.addEpisode({
        content: episodeContent,
        referenceId: 'ref-123',
      });

      assert.strictEqual(episode.referenceId, 'ref-123');
    });

    it('should handle custom group ID', async () => {
      const customGroupId = 'custom-group';

      (mockDriver.executeQuery as any).mock.mockImplementation(async () => []);

      const episode = await graphzep.addEpisode({
        content: 'Test content',
        groupId: customGroupId,
      });

      assert.strictEqual(episode.groupId, customGroupId);
    });

    it('should skip entities below confidence threshold', async () => {
      mockLLMClient.generateStructuredResponse.mock.resetCalls();
      (mockDriver.executeQuery as any).mock.resetCalls();

      (mockDriver.executeQuery as any).mock.mockImplementation(async () => []);

      mockLLMClient.generateStructuredResponse.mock.mockImplementationOnce(async () => ({
        entities: [
          { name: 'Alice', entityType: 'Person', summary: 'A person', confidence: 0.9 },
          { name: 'SomeVagueRef', entityType: 'Concept', summary: 'Vague', confidence: 0.2 },
        ],
        relations: [],
      }));

      await graphzep.addEpisode({ content: 'Alice mentioned something vague.' });

      const saveCalls = (mockDriver.executeQuery as any).mock.calls.filter((c: any) =>
        c.arguments[0].includes('MERGE') && c.arguments[0].includes('Entity'),
      );
      // Only Alice (confidence 0.9) should be saved — SomeVagueRef (0.2) filtered out
      // Filter to EntityNode saves specifically (MERGE (n:Entity), not edge MATCHes)
      const entityNodeSaves = saveCalls.filter((c: any) =>
        c.arguments[0].includes('MERGE (n:Entity'),
      );
      assert.strictEqual(entityNodeSaves.length, 1);
    });

    it('should not create an edge for negated relations (but may query for conflicts)', async () => {
      (mockDriver.executeQuery as any).mock.resetCalls();
      (mockDriver.executeQuery as any).mock.mockImplementation(async () => []);

      mockLLMClient.generateStructuredResponse.mock.mockImplementationOnce(async () => ({
        entities: [
          { name: 'Alice', entityType: 'Person', summary: 'Alice', confidence: 0.9 },
          { name: 'ACME', entityType: 'Organization', summary: 'ACME Corp', confidence: 0.9 },
        ],
        relations: [
          {
            sourceName: 'Alice',
            targetName: 'ACME',
            relationName: 'WORKS_AT',
            confidence: 0.9,
            isNegated: true,
            temporalValidity: 'current',
          },
        ],
      }));

      await graphzep.addEpisode({ content: 'Alice does not work at ACME.' });

      // No MERGE of a RELATES_TO edge should happen — conflict detection
      // uses MATCH-only queries which are allowed.
      const edgeCreates = (mockDriver.executeQuery as any).mock.calls.filter((c: any) =>
        c.arguments[0].includes('MERGE') && c.arguments[0].includes('RELATES_TO'),
      );
      assert.strictEqual(edgeCreates.length, 0, 'Negated relation should not create a RELATES_TO edge');
    });

    it('should mark historical relations with invalidAt', async () => {
      (mockDriver.executeQuery as any).mock.resetCalls();
      (mockDriver.executeQuery as any).mock.mockImplementation(async (query: string) => {
        if (query.includes('MATCH (s:Entity')) return []; // no existing edge
        return [];
      });

      mockLLMClient.generateStructuredResponse.mock.mockImplementationOnce(async () => ({
        entities: [
          { name: 'Alice', entityType: 'Person', summary: 'Alice', confidence: 0.9 },
          { name: 'OldCo', entityType: 'Organization', summary: 'Old company', confidence: 0.9 },
        ],
        relations: [
          {
            sourceName: 'Alice',
            targetName: 'OldCo',
            relationName: 'WORKED_AT',
            confidence: 0.9,
            isNegated: false,
            temporalValidity: 'historical',
          },
        ],
      }));

      await graphzep.addEpisode({ content: 'Alice used to work at OldCo.' });

      const edgeSaves = (mockDriver.executeQuery as any).mock.calls.filter((c: any) =>
        c.arguments[0].includes('RELATES_TO') && c.arguments[0].includes('MERGE'),
      );
      assert(edgeSaves.length > 0, 'Historical relation should still be saved');
      // invalidAt param should be set (non-null)
      const edgeParams = edgeSaves[0].arguments[1];
      assert(edgeParams?.invalidAt != null, 'Historical edge should have invalidAt set');
    });

    it('should merge summary when entity already exists', async () => {
      const existingAlice = {
        uuid: 'alice-uuid',
        name: 'Alice',
        entityType: 'Person',
        summary: 'A junior developer at ACME',
        groupId: 'test-group',
        labels: ['Entity'],
        createdAt: new Date(),
      };

      mockLLMClient.generateStructuredResponse.mock.resetCalls();
      mockEmbedder.embed.mock.resetCalls();

      (mockDriver.executeQuery as any).mock.mockImplementation(
        async (query: string, params: any) => {
          if (query.includes('MATCH (n:Entity') && params?.name === 'Alice') {
            return [{ n: existingAlice }];
          }
          return [];
        },
      );

      await graphzep.addEpisode({ content: 'Alice was promoted to tech lead at ACME.' });

      // LLM should have been called for extraction AND merge
      const llmCalls = mockLLMClient.generateStructuredResponse.mock.calls;
      const mergeCalls = llmCalls.filter((c: any) =>
        c.arguments[0].includes('maintaining a knowledge graph'),
      );
      assert(mergeCalls.length > 0, 'Should call LLM to merge entity summaries');

      // Embedder should have been called for the merged summary (at least twice: episode + merged entity)
      assert(mockEmbedder.embed.mock.calls.length >= 2);
    });

    it('should accumulate episodes on existing relation', async () => {
      const existingEdge = {
        uuid: 'edge-uuid',
        groupId: 'test-group',
        sourceNodeUuid: 'alice-uuid',
        targetNodeUuid: 'bob-uuid',
        name: 'KNOWS',
        factIds: [],
        episodes: ['old-episode-uuid'],
        validAt: new Date(Date.now() - 10000),
        createdAt: new Date(),
      };

      (mockDriver.executeQuery as any).mock.mockImplementation(
        async (query: string, params: any) => {
          if (
            query.includes('MATCH (s:Entity') &&
            params?.relationName === 'KNOWS'
          ) {
            return [{ r: existingEdge }];
          }
          return [];
        },
      );

      await graphzep.addEpisode({ content: 'Alice and Bob met again at the office.' });

      // Should have saved the edge with updated episodes (MERGE query)
      const saveCalls = (mockDriver.executeQuery as any).mock.calls.filter((c: any) =>
        c.arguments[0].includes('MERGE') && c.arguments[0].includes('RELATES_TO'),
      );
      assert(saveCalls.length > 0, 'Should save updated edge with new episode');
    });
  });

  describe('search', () => {
    it('should search for nodes by similarity', async () => {
      const searchQuery = 'Find information about Alice';

      const mockSearchResults = [
        {
          n: {
            uuid: 'alice-uuid',
            name: 'Alice',
            entityType: 'Person',
            summary: 'A person named Alice',
            groupId: 'test-group',
            createdAt: new Date(),
            embedding: new Array(384).fill(0.1),
          },
          labels: ['Entity'],
        },
      ];

      (mockDriver.executeQuery as any).mock.mockImplementation(async (query: string) => {
        if (query.includes('similarity')) {
          return mockSearchResults;
        }
        return [];
      });

      const results = await graphzep.search({
        query: searchQuery,
        limit: 5,
      });

      assert.strictEqual(results.length, 1);
      assert(results[0] instanceof EntityNodeImpl);
      assert.strictEqual(results[0].name, 'Alice');

      assert(mockEmbedder.embed.mock.calls.length > 0);
    });

    it('should handle empty search results', async () => {
      (mockDriver.executeQuery as any).mock.mockImplementation(async () => []);

      const results = await graphzep.search({
        query: 'Non-existent entity',
      });

      assert.strictEqual(results.length, 0);
    });

    it('should use custom group ID in search', async () => {
      const customGroupId = 'search-group';

      (mockDriver.executeQuery as any).mock.mockImplementation(
        async (query: string, params: any) => {
          assert.strictEqual(params?.groupId, customGroupId);
          return [];
        },
      );

      await graphzep.search({
        query: 'Test search',
        groupId: customGroupId,
      });
    });

    it('should expand results via graph when graphExpand is true', async () => {
      const alice = {
        uuid: 'alice-uuid',
        name: 'Alice',
        entityType: 'Person',
        summary: 'A person named Alice',
        groupId: 'test-group',
        createdAt: new Date(),
        embedding: new Array(384).fill(0.1),
      };
      const bob = {
        uuid: 'bob-uuid',
        name: 'Bob',
        entityType: 'Person',
        summary: 'A person named Bob',
        groupId: 'test-group',
        createdAt: new Date(),
      };

      (mockDriver.executeQuery as any).mock.mockImplementation(async (query: string) => {
        if (query.includes('similarity')) return [{ n: alice, labels: ['Entity'] }];
        if (query.includes('RELATES_TO*1..1')) return [{ n: bob, nodeLabels: ['Entity'] }];
        return [];
      });

      const results = await graphzep.search({
        query: 'Find Alice',
        graphExpand: true,
        expandHops: 1,
      });

      assert.strictEqual(results.length, 2);
      const names = results.map(r => r.name);
      assert(names.includes('Alice'));
      assert(names.includes('Bob'));
    });

    it('should return only seed nodes when graphExpand is false (default)', async () => {
      const alice = {
        uuid: 'alice-uuid',
        name: 'Alice',
        entityType: 'Person',
        summary: '',
        groupId: 'test-group',
        createdAt: new Date(),
        embedding: new Array(384).fill(0.1),
      };

      (mockDriver.executeQuery as any).mock.mockImplementation(async (query: string) => {
        if (query.includes('similarity')) return [{ n: alice, labels: ['Entity'] }];
        return [];
      });

      const results = await graphzep.search({ query: 'Find Alice' });
      assert.strictEqual(results.length, 1);
      assert.strictEqual(results[0].name, 'Alice');
    });
  });

  describe('node operations', () => {
    it('should get node by UUID', async () => {
      const nodeData = {
        uuid: 'test-uuid',
        name: 'Test Entity',
        entityType: 'Test',
        summary: 'A test entity',
        groupId: 'test-group',
        labels: ['Entity'],
        createdAt: new Date(),
      };

      (mockDriver.executeQuery as any).mock.mockImplementation(async () => [{ n: nodeData }]);

      const node = await graphzep.getNode('test-uuid');

      assert(node instanceof EntityNodeImpl);
      assert.strictEqual(node?.uuid, 'test-uuid');
      assert.strictEqual(node?.name, 'Test Entity');
    });

    it('should return null for non-existent node', async () => {
      (mockDriver.executeQuery as any).mock.mockImplementation(async () => []);

      const node = await graphzep.getNode('non-existent');

      assert.strictEqual(node, null);
    });

    it('should delete node by UUID', async () => {
      const nodeData = {
        uuid: 'delete-uuid',
        name: 'To Delete',
        entityType: 'Test',
        summary: 'Entity to delete',
        groupId: 'test-group',
        labels: ['Entity'],
        createdAt: new Date(),
      };

      (mockDriver.executeQuery as any).mock.mockImplementation(async (query: string) => {
        if (query.includes('RETURN n')) {
          return [{ n: nodeData }];
        }
        return [];
      });

      await graphzep.deleteNode('delete-uuid');

      const deleteCalls = (mockDriver.executeQuery as any).mock.calls.filter((call: any) =>
        call.arguments[0].includes('DELETE'),
      );
      assert(deleteCalls.length > 0);
    });
  });

  describe('edge operations', () => {
    it('should get edge by UUID', async () => {
      const edgeData = {
        uuid: 'edge-uuid',
        groupId: 'test-group',
        sourceNodeUuid: 'source-uuid',
        targetNodeUuid: 'target-uuid',
        name: 'RELATES_TO',
        factIds: [],
        episodes: [],
        validAt: new Date(),
        createdAt: new Date(),
      };

      (mockDriver.executeQuery as any).mock.mockImplementation(async () => [
        { e: edgeData, relType: 'RELATES_TO' },
      ]);

      const edge = await graphzep.getEdge('edge-uuid');

      assert(edge instanceof EntityEdgeImpl);
      assert.strictEqual(edge?.uuid, 'edge-uuid');
    });

    it('should return null for non-existent edge', async () => {
      (mockDriver.executeQuery as any).mock.mockImplementation(async () => []);

      const edge = await graphzep.getEdge('non-existent');

      assert.strictEqual(edge, null);
    });

    it('should delete edge by UUID', async () => {
      const edgeData = {
        uuid: 'delete-edge-uuid',
        groupId: 'test-group',
        sourceNodeUuid: 'source-uuid',
        targetNodeUuid: 'target-uuid',
        createdAt: new Date(),
      };

      (mockDriver.executeQuery as any).mock.mockImplementation(async (query: string) => {
        if (query.includes('RETURN e')) {
          return [{ e: edgeData, relType: 'MENTIONS' }];
        }
        return [];
      });

      await graphzep.deleteEdge('delete-edge-uuid');

      const deleteCalls = (mockDriver.executeQuery as any).mock.calls.filter((call: any) =>
        call.arguments[0].includes('DELETE'),
      );
      assert(deleteCalls.length > 0);
    });
  });

  describe('traverse', () => {
    it('should return start node and neighbors', async () => {
      const alice = {
        uuid: 'alice-uuid',
        name: 'Alice',
        entityType: 'Person',
        summary: '',
        groupId: 'test-group',
        labels: ['Entity'],
        createdAt: new Date(),
      };
      const bob = {
        uuid: 'bob-uuid',
        name: 'Bob',
        entityType: 'Person',
        summary: '',
        groupId: 'test-group',
        labels: ['Entity'],
        createdAt: new Date(),
      };
      const edge = {
        uuid: 'edge-uuid',
        groupId: 'test-group',
        sourceNodeUuid: 'alice-uuid',
        targetNodeUuid: 'bob-uuid',
        name: 'KNOWS',
        factIds: [],
        episodes: [],
        validAt: new Date(),
        createdAt: new Date(),
      };

      (mockDriver.executeQuery as any).mock.mockImplementation(async (query: string, params: any) => {
        if (query.includes('LIMIT 1') && params?.id === 'Alice') return [{ n: alice, labels: ['Entity'] }];
        if (query.includes('RELATES_TO*1..')) return [{ n: bob, nodeLabels: ['Entity'] }];
        if (query.includes('a.uuid IN')) return [{ e: edge }];
        return [];
      });

      const result = await graphzep.traverse({ startEntityName: 'Alice', maxHops: 1 });
      assert(result.start instanceof EntityNodeImpl);
      assert.strictEqual(result.start.name, 'Alice');
      assert.strictEqual(result.nodes.length, 1);
      assert.strictEqual(result.nodes[0].name, 'Bob');
      assert.strictEqual(result.edges.length, 1);
      assert.strictEqual(result.edges[0].name, 'KNOWS');
    });

    it('should return null start when entity not found', async () => {
      (mockDriver.executeQuery as any).mock.mockImplementation(async () => []);

      const result = await graphzep.traverse({ startEntityName: 'Ghost' });
      assert.strictEqual(result.start, null);
      assert.strictEqual(result.nodes.length, 0);
      assert.strictEqual(result.edges.length, 0);
    });

    it('should throw when no start identifier provided', async () => {
      await assert.rejects(
        () => graphzep.traverse({}),
        /startEntityName or startEntityUuid/,
      );
    });
  });

  describe('close', () => {
    it('should close the driver connection', async () => {
      await graphzep.close();

      assert(mockDriver.close.mock.calls.length > 0);
    });
  });
});
