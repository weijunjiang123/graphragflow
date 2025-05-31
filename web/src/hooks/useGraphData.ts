import { useState, useCallback } from 'react';

interface Node {
  id: string;
  label: string;
  type: 'entity' | 'document' | 'concept';
  x: number;
  y: number;
  size: number;
  color: string;
  data?: any;
}

interface Edge {
  id: string;
  source: string;
  target: string;
  label?: string;
  weight: number;
  type?: string;
}

interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

interface Neo4jPath {
  p: [any, string, any];
}

interface Neo4jResponse {
  cypher_query: string;
  results: Neo4jPath[];
  count: number;
}

export function useGraphData(apiEndpoint: string) {
  const [data, setData] = useState<GraphData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Force-directed layout algorithm
  const calculateLayout = useCallback((nodes: Node[], edges: Edge[]): Node[] => {
    const width = 600;
    const height = 400;
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Initialize positions if not set
    const positionedNodes = nodes.map((node, index) => ({
      ...node,
      x: node.x || centerX + Math.cos(index * 2 * Math.PI / nodes.length) * 100,
      y: node.y || centerY + Math.sin(index * 2 * Math.PI / nodes.length) * 100,
      vx: 0,
      vy: 0
    }));

    // Simple force-directed layout simulation
    for (let iteration = 0; iteration < 50; iteration++) {
      // Repulsive forces between nodes
      for (let i = 0; i < positionedNodes.length; i++) {
        for (let j = i + 1; j < positionedNodes.length; j++) {
          const nodeA = positionedNodes[i];
          const nodeB = positionedNodes[j];
          const dx = nodeB.x - nodeA.x;
          const dy = nodeB.y - nodeA.y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = 1000 / (distance * distance);
          
          nodeA.vx -= (dx / distance) * force;
          nodeA.vy -= (dy / distance) * force;
          nodeB.vx += (dx / distance) * force;
          nodeB.vy += (dy / distance) * force;
        }
      }

      // Attractive forces for connected nodes
      edges.forEach(edge => {
        const source = positionedNodes.find(n => n.id === edge.source);
        const target = positionedNodes.find(n => n.id === edge.target);
        if (source && target) {
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = distance * 0.01;
          
          source.vx += (dx / distance) * force;
          source.vy += (dy / distance) * force;
          target.vx -= (dx / distance) * force;
          target.vy -= (dy / distance) * force;
        }
      });

      // Apply velocities with damping
      positionedNodes.forEach(node => {
        node.x += node.vx * 0.1;
        node.y += node.vy * 0.1;
        node.vx *= 0.9;
        node.vy *= 0.9;
        
        // Keep nodes within bounds
        node.x = Math.max(50, Math.min(width - 50, node.x));
        node.y = Math.max(50, Math.min(height - 50, node.y));
      });
    }

    return positionedNodes.map(({ vx, vy, ...node }) => node);
  }, []);

  // Convert Neo4j API response to graph data
  const convertNeo4jToGraphData = useCallback((response: Neo4jResponse): GraphData => {
    const nodesMap = new Map<string, Node>();
    const edges: Edge[] = [];

    response.results.forEach((result, index) => {
      const [sourceNode, relationship, targetNode] = result.p;
      
      // Process source node (document)
      const sourceId = sourceNode.id || `doc_${index}`;
      if (!nodesMap.has(sourceId)) {
        const label = sourceNode.page_label ? 
          `Page ${sourceNode.page_label}` : 
          sourceNode.title || sourceNode.source?.split('/').pop()?.split('.')[0] || 'Document';
        
        nodesMap.set(sourceId, {
          id: sourceId,
          label: label.length > 20 ? label.substring(0, 20) + '...' : label,
          type: 'document',
          x: 0,
          y: 0,
          size: 15,
          color: '#8b5cf6',
          data: sourceNode
        });
      }

      // Process target node (entity)
      const targetId = targetNode.id || targetNode.name || `entity_${index}`;
      if (!nodesMap.has(targetId)) {
        const label = targetNode.name || targetNode.id || 'Entity';
        nodesMap.set(targetId, {
          id: targetId,
          label: label.length > 20 ? label.substring(0, 20) + '...' : label,
          type: 'entity',
          x: 0,
          y: 0,
          size: 12,
          color: '#10b981',
          data: targetNode
        });
      }

      // Create edge
      edges.push({
        id: `edge_${index}`,
        source: sourceId,
        target: targetId,
        label: relationship,
        weight: 1,
        type: relationship
      });
    });

    const nodes = Array.from(nodesMap.values());
    const positionedNodes = calculateLayout(nodes, edges);

    return {
      nodes: positionedNodes,
      edges
    };
  }, [calculateLayout]);

  // Fetch data from API
  const fetchData = useCallback(async (query: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const encodedQuery = encodeURIComponent(query);
      const url = `${apiEndpoint}?cypher_query=${encodedQuery}`;
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'accept': 'application/json',
        },
        body: ''
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const apiData: Neo4jResponse = await response.json();
      const convertedData = convertNeo4jToGraphData(apiData);
      setData(convertedData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch graph data');
    } finally {
      setIsLoading(false);
    }
  }, [apiEndpoint, convertNeo4jToGraphData]);

  return {
    data,
    isLoading,
    error,
    fetchData,
    setData,
    setError
  };
}
