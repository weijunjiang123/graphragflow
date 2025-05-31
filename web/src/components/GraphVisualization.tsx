import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Network, Search, Filter, Download, Maximize, ZoomIn, ZoomOut, RotateCcw, Loader2, RefreshCw, Settings } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useGraphData } from '@/hooks/useGraphData';

interface Node {
  id: string;
  label: string;
  type: 'entity' | 'document' | 'concept';
  x: number;
  y: number;
  size: number;
  color: string;
  data?: any; // Additional node data
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

// Neo4j API response interfaces
interface Neo4jPath {
  p: [any, string, any]; // [source_node, relationship, target_node]
}

interface Neo4jResponse {
  cypher_query: string;
  results: Neo4jPath[];
  count: number;
}

interface GraphVisualizationProps {
  className?: string;
  data?: GraphData;
  apiEndpoint?: string;
  defaultQuery?: string;
}

export function GraphVisualization({ 
  className, 
  data, 
  apiEndpoint = 'http://localhost:8000/api/graph-query/execute-cypher',
  defaultQuery = "MATCH p=()-[:MENTIONS]->() RETURN p LIMIT 25;"
}: GraphVisualizationProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cypherQuery, setCypherQuery] = useState(defaultQuery);
  const [showSettings, setShowSettings] = useState(false);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  // 新增节点拖拽相关状态
  const [draggedNodeId, setDraggedNodeId] = useState<string | null>(null);
  const [nodeDragStart, setNodeDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [nodeDragOffset, setNodeDragOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  // 节点编辑面板相关状态
  const [editingNode, setEditingNode] = useState<Node | null>(null);
  const [editLabel, setEditLabel] = useState('');
  const [editColor, setEditColor] = useState('');
  
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Force-directed layout algorithm
  const calculateLayout = useCallback((nodes: Node[], edges: Edge[]): Node[] => {
    // 优化：加大节点间距，提升美观和文字显示
    const width = 1200;
    const height = 800;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.max(250, Math.min(width, height) / 2 - 100);
    // 环形初始分布，间距更大
    const positionedNodes = nodes.map((node, index) => ({
      ...node,
      x: node.x || centerX + Math.cos(index * 2 * Math.PI / nodes.length) * radius,
      y: node.y || centerY + Math.sin(index * 2 * Math.PI / nodes.length) * radius,
      vx: 0,
      vy: 0
    }));
    // 力导向模拟，增大斥力和弹簧长度
    for (let iteration = 0; iteration < 60; iteration++) {
      // 斥力
      for (let i = 0; i < positionedNodes.length; i++) {
        for (let j = i + 1; j < positionedNodes.length; j++) {
          const nodeA = positionedNodes[i];
          const nodeB = positionedNodes[j];
          const dx = nodeB.x - nodeA.x;
          const dy = nodeB.y - nodeA.y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = 3000 / (distance * distance); // 斥力更大
          nodeA.vx -= (dx / distance) * force;
          nodeA.vy -= (dy / distance) * force;
          nodeB.vx += (dx / distance) * force;
          nodeB.vy += (dy / distance) * force;
        }
      }
      // 引力
      edges.forEach(edge => {
        const source = positionedNodes.find(n => n.id === edge.source);
        const target = positionedNodes.find(n => n.id === edge.target);
        if (source && target) {
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 1;
          const ideal = 220; // 理想边长更大
          const force = (distance - ideal) * 0.02;
          source.vx += (dx / distance) * force;
          source.vy += (dy / distance) * force;
          target.vx -= (dx / distance) * force;
          target.vy -= (dy / distance) * force;
        }
      });
      // 应用速度
      positionedNodes.forEach(node => {
        node.x += node.vx * 0.12;
        node.y += node.vy * 0.12;
        node.vx *= 0.85;
        node.vy *= 0.85;
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
          color: '#3b82f6',
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
  const fetchGraphData = useCallback(async (query: string) => {
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

      const data: Neo4jResponse = await response.json();
      const convertedData = convertNeo4jToGraphData(data);
      setGraphData(convertedData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch graph data');
    } finally {
      setIsLoading(false);
    }
  }, [apiEndpoint, convertNeo4jToGraphData]);

  // Load data on component mount
  useEffect(() => {
    if (!data) {
      fetchGraphData(cypherQuery);
    }
  }, [data, cypherQuery, fetchGraphData]);

  // Sample data when no data is provided and no API data
  const sampleData: GraphData = {
    nodes: [
      { id: '1', label: '人工智能', type: 'concept', x: 200, y: 150, size: 20, color: '#3b82f6' },
      { id: '2', label: '机器学习', type: 'concept', x: 350, y: 100, size: 18, color: '#3b82f6' },
      { id: '3', label: '深度学习', type: 'concept', x: 500, y: 150, size: 16, color: '#3b82f6' },
      { id: '4', label: '神经网络', type: 'entity', x: 350, y: 250, size: 15, color: '#10b981' },
      { id: '5', label: '算法', type: 'entity', x: 150, y: 300, size: 14, color: '#10b981' },
      { id: '6', label: '数据处理', type: 'document', x: 450, y: 300, size: 12, color: '#8b5cf6' },
    ],
    edges: [
      { id: 'e1', source: '1', target: '2', label: '包含', weight: 1 },
      { id: 'e2', source: '2', target: '3', label: '发展为', weight: 0.8 },
      { id: 'e3', source: '3', target: '4', label: '基于', weight: 0.9 },
      { id: 'e4', source: '1', target: '5', label: '使用', weight: 0.7 },
      { id: 'e5', source: '4', target: '6', label: '需要', weight: 0.6 },
    ]
  };
  const currentData = data || graphData || sampleData;

  const handleRefresh = () => {
    fetchGraphData(cypherQuery);
  };

  const handleQueryChange = (newQuery: string) => {
    setCypherQuery(newQuery);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPanOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.2, 3));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.2, 0.3));
  };

  const handleReset = () => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
    setSelectedNode(null);
  };

  const handleExport = () => {
    if (!svgRef.current) return;
    const svgData = new XMLSerializer().serializeToString(svgRef.current);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx?.drawImage(img, 0, 0);
      
      const link = document.createElement('a');
      link.download = 'knowledge-graph.png';
      link.href = canvas.toDataURL();
      link.click();
    };
    
    img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
  };
  const filteredNodes = currentData.nodes.filter(node =>
    node.label.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredEdges = currentData.edges.filter(edge =>
    filteredNodes.some(node => node.id === edge.source) &&
    filteredNodes.some(node => node.id === edge.target)
  );

  const getNodeColor = (node: Node) => {
    if (selectedNode === node.id) return '#fbbf24';
    if (hoveredNode === node.id) return '#f59e0b';
    return node.color;
  };

  // 节点拖拽事件
  const handleNodeMouseDown = (e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    setDraggedNodeId(nodeId);
    setNodeDragStart({ x: e.clientX, y: e.clientY });
    const node = currentData.nodes.find(n => n.id === nodeId);
    if (node) {
      setNodeDragOffset({ x: node.x, y: node.y });
      setSelectedNode(nodeId);
    }
  };

  // 拖拽节流，提升性能
  const lastDragTime = useRef(0);
  // 滚轮缩放
  const handleWheel = (e: React.WheelEvent) => {
    // 最佳实践：彻底阻止页面滚动，无论事件冒泡来源
    e.preventDefault();
    e.stopPropagation();
    if (e.deltaY < 0) {
      setZoomLevel(z => Math.min(z * 1.1, 5));
    } else {
      setZoomLevel(z => Math.max(z / 1.1, 0.2));
    }
  };

  // 优化拖动动画效果，节点间不能重叠
  // 在拖动节点时，检测与其他节点的距离，避免重叠
  const MIN_DIST = 30;
  const handleNodeMouseMove = (e: React.MouseEvent) => {
    if (draggedNodeId) {
      const now = Date.now();
      if (now - lastDragTime.current < 16) return;
      lastDragTime.current = now;
      const dx = (e.clientX - nodeDragStart.x) / zoomLevel;
      const dy = (e.clientY - nodeDragStart.y) / zoomLevel;
      if (data) return;
      setGraphData(prev => {
        if (!prev) return prev;
        // 计算新位置
        const newNodes = prev.nodes.map(n => {
          if (n.id === draggedNodeId) {
            let newX = nodeDragOffset.x + dx;
            let newY = nodeDragOffset.y + dy;
            // 检查与其他节点的距离，避免重叠
            prev.nodes.forEach(other => {
              if (other.id !== n.id) {
                const dist = Math.sqrt((newX - other.x) ** 2 + (newY - other.y) ** 2);
                if (dist < MIN_DIST) {
                  const angle = Math.atan2(newY - other.y, newX - other.x);
                  newX = other.x + Math.cos(angle) * MIN_DIST;
                  newY = other.y + Math.sin(angle) * MIN_DIST;
                }
              }
            });
            return { ...n, x: newX, y: newY };
          }
          return n;
        });
        return { ...prev, nodes: newNodes };
      });
    }
  };

  // 鼠标全局松开时自动结束拖拽
  useEffect(() => {
    const onUp = () => setDraggedNodeId(null);
    window.addEventListener('mouseup', onUp);
    return () => window.removeEventListener('mouseup', onUp);
  }, []);

  // 双击节点弹出编辑面板
  const handleNodeDoubleClick = (node: Node) => {
    setEditingNode(node);
    setEditLabel(node.label);
    setEditColor(node.color);
  };

  const handleEditSave = () => {
    if (!editingNode) return;
    if (data) return; // 外部 data 不可编辑
    setGraphData(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        nodes: prev.nodes.map(n =>
          n.id === editingNode.id
            ? { ...n, label: editLabel, color: editColor }
            : n
        )
      };
    });
    setEditingNode(null);
  };

  const handleEditCancel = () => {
    setEditingNode(null);
  };

  // 取消画布限制，SVG无限大自适应节点分布
  // 计算节点边界
  const nodeXs = currentData.nodes.map(n => n.x);
  const nodeYs = currentData.nodes.map(n => n.y);
  const minX = Math.min(...nodeXs, 0) - 100;
  const maxX = Math.max(...nodeXs, 600) + 100;
  const minY = Math.min(...nodeYs, 0) - 100;
  const maxY = Math.max(...nodeYs, 400) + 100;
  const svgWidth = maxX - minX;
  const svgHeight = maxY - minY;

  // 节点hover计时器
  const hoverTimer = useRef<NodeJS.Timeout | null>(null);
  const [hoverDetailNode, setHoverDetailNode] = useState<string | null>(null);

  // 鼠标移入节点时2s后显示详情
  const handleNodeMouseEnter = (nodeId: string) => {
    setHoveredNode(nodeId);
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    hoverTimer.current = setTimeout(() => {
      setHoverDetailNode(nodeId);
    }, 2000);
  };
  const handleNodeMouseLeave = (nodeId: string) => {
    setHoveredNode(null);
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    setHoverDetailNode(prev => (prev === nodeId ? null : prev));
  };

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader>        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Network className="w-5 h-5" />
            知识图谱可视化
            {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" onClick={handleRefresh} disabled={isLoading}>
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => setShowSettings(!showSettings)}>
              <Settings className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={handleZoomIn}>
              <ZoomIn className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={handleZoomOut}>
              <ZoomOut className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={handleReset}>
              <RotateCcw className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={handleExport}>
              <Download className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => setIsFullscreen(!isFullscreen)}>
              <Maximize className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>      <CardContent className="space-y-4">
        {/* Settings Panel */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="p-4 bg-gray-800/50 rounded-lg border border-white/10"
            >
              <div className="space-y-2">
                <label className="text-sm font-medium text-white">Cypher 查询:</label>
                <Input
                  value={cypherQuery}
                  onChange={(e) => handleQueryChange(e.target.value)}
                  placeholder="输入 Cypher 查询语句..."
                  className="font-mono text-sm"
                />
                <Button 
                  onClick={() => fetchGraphData(cypherQuery)} 
                  disabled={isLoading}
                  className="w-full"
                >
                  {isLoading ? '执行中...' : '执行查询'}
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error Display */}
        {error && (
          <div className="p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
            <p className="text-sm text-red-400">错误: {error}</p>
          </div>
        )}

        {/* Controls */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-white/50" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="搜索节点或关系..."
              className="pl-10"
            />
          </div>
          <Button variant="ghost" size="icon">
            <Filter className="w-4 h-4" />
          </Button>
        </div>        {/* Graph Container */}
        <div
          ref={containerRef}
          className="relative bg-gradient-to-br from-gray-900/50 to-black/50 rounded-xl border border-white/10 overflow-auto"
          onMouseDown={e => {
            if (!draggedNodeId) handleMouseDown(e);
            if (selectedNode) setSelectedNode(null);
          }}
          onMouseMove={e => { handleMouseMove(e); handleNodeMouseMove(e); }}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          // 优化：自适应高度，最大高度限制，防止页面过长
          style={{ cursor: draggedNodeId ? 'grabbing' : isDragging ? 'grabbing' : 'grab', height: '40vh', minHeight: 400, maxHeight: '80vh' }}
        >
          {/* Loading Overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
              <div className="flex items-center gap-2 text-white">
                <Loader2 className="w-6 h-6 animate-spin" />
                <span>加载中...</span>
              </div>
            </div>
          )}

          {/* SVG Graph */}
          <svg
            ref={svgRef}
            width={svgWidth}
            height={svgHeight}
            viewBox={`${minX} ${minY} ${svgWidth} ${svgHeight}`}
            style={{
              display: 'block',
              maxWidth: '100%',
              maxHeight: '75vh',
              height: '100%',
              width: '100%',
              transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomLevel})`,
              transformOrigin: 'center',
              transition: draggedNodeId ? 'none' : 'transform 0.2s cubic-bezier(0.4,0,0.2,1)',
              background: 'none',
            }}
            onWheel={handleWheel}
          >
            {/* Edges */}
            {filteredEdges.map((edge) => {
              const sourceNode = filteredNodes.find(n => n.id === edge.source);
              const targetNode = filteredNodes.find(n => n.id === edge.target);
              
              if (!sourceNode || !targetNode) return null;
              
              return (
                <g key={edge.id}>
                  <line
                    x1={sourceNode.x}
                    y1={sourceNode.y}
                    x2={targetNode.x}
                    y2={targetNode.y}
                    stroke="rgba(255,255,255,0.3)"
                    strokeWidth={edge.weight * 2}
                    strokeDasharray={edge.label ? "none" : "5,5"}
                  />
                  {edge.label && (
                    <text
                      x={(sourceNode.x + targetNode.x) / 2}
                      y={(sourceNode.y + targetNode.y) / 2}
                      fill="rgba(255,255,255,0.6)"
                      fontSize="10"
                      textAnchor="middle"
                      dy="0.3em"
                    >
                      {edge.label}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Nodes */}
            {filteredNodes.map((node, idx) => {
              // label竖直偏移避免重叠
              let labelYOffset = node.size + 20;
              const overlap = filteredNodes.some((other, jdx) =>
                jdx !== idx &&
                Math.abs(node.x - other.x) < 40 &&
                Math.abs(node.y + labelYOffset - (other.y + other.size + 20)) < 18
              );
              if (overlap) labelYOffset += 18;
              // 拖动节点时平滑动画
              const isDraggingThis = draggedNodeId === node.id;
              const isSelected = selectedNode === node.id;
              const isDetail = hoverDetailNode === node.id;
              return (
                <g key={node.id} style={{ transition: isDraggingThis ? 'none' : 'transform 0.3s cubic-bezier(0.4,0,0.2,1)' }}>
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={node.size}
                    fill={getNodeColor(node)}
                    stroke="rgba(255,255,255,0.5)"
                    strokeWidth={isSelected ? 3 : 1}
                    className="cursor-move transition-all duration-200"
                    onClick={e => { e.stopPropagation(); setSelectedNode(isSelected ? null : node.id); }}
                    onMouseEnter={() => handleNodeMouseEnter(node.id)}
                    onMouseLeave={() => handleNodeMouseLeave(node.id)}
                    onMouseDown={e => handleNodeMouseDown(e, node.id)}
                    onDoubleClick={() => handleNodeDoubleClick(node)}
                    style={{ filter: isDraggingThis ? 'drop-shadow(0 0 8px #fbbf24)' : undefined, transition: isDraggingThis ? 'none' : 'filter 0.3s' }}
                  />
                  <text
                    x={node.x}
                    y={node.y + labelYOffset}
                    fill="white"
                    fontSize="12"
                    textAnchor="middle"
                    alignmentBaseline="middle"
                    className="pointer-events-none select-none"
                    style={{ fontWeight: 500, paintOrder: 'stroke', stroke: 'rgba(0,0,0,0.7)', strokeWidth: 2 }}
                  >
                    {node.label}
                  </text>
                  {/* 悬停2s后显示详细卡片 */}
                  {isDetail && (
                    <foreignObject
                      x={node.x + 30}
                      y={node.y - 30}
                      width={260}
                      height={180}
                      style={{ pointerEvents: 'auto', zIndex: 100 }}
                    >
                      <div
                        style={{ background: 'rgba(24,24,32,0.98)', borderRadius: 12, boxShadow: '0 4px 24px #0008', border: '1px solid #fff2', padding: 18, color: '#fff', fontSize: 14, minHeight: 120, position: 'relative', animation: 'fadeIn .3s' }}
                        tabIndex={-1}
                        onClick={e => e.stopPropagation()}
                      >
                        <button
                          style={{ position: 'absolute', top: 8, right: 8, background: 'none', border: 'none', color: '#fff', fontSize: 18, cursor: 'pointer', lineHeight: 1 }}
                          onClick={() => setHoverDetailNode(null)}
                          aria-label="关闭"
                        >×</button>
                        <div style={{ fontWeight: 600, fontSize: 18, marginBottom: 6 }}>{node.label}</div>
                        <div style={{ color: '#fbbf24', fontSize: 12, marginBottom: 4 }}>类型: {node.type}</div>
                        <div style={{ color: '#aaa', fontSize: 12, marginBottom: 4 }}>ID: {node.id}</div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, marginBottom: 4 }}>
                          <div style={{ width: 12, height: 12, borderRadius: 6, background: node.color, border: '1px solid #fff3' }} />
                          <span>权重: {node.size}</span>
                        </div>
                        {node.data && (
                          <div style={{ marginTop: 8, color: '#eee', fontSize: 13 }}>
                            {node.type === 'document' && node.data.source && (
                              <div style={{ marginBottom: 4 }}>来源: {node.data.source.split('/').pop()}</div>
                            )}
                            {node.data.text && (
                              <div style={{ marginTop: 4, color: '#ccc' }}>预览: {node.data.text.substring(0, 120)}...</div>
                            )}
                          </div>
                        )}
                      </div>
                    </foreignObject>
                  )}
                </g>
              );
            })}
            {/* 节点详细信息卡片，浮动在节点右上，点击空白关闭 */}
            {selectedNode && (() => {
              const node = filteredNodes.find(n => n.id === selectedNode);
              if (!node) return null;
              return (
                <foreignObject
                  x={node.x + 30}
                  y={node.y - 30}
                  width={260}
                  height={180}
                  style={{ pointerEvents: 'auto', zIndex: 100 }}
                >
                  <div
                    style={{ background: 'rgba(24,24,32,0.98)', borderRadius: 12, boxShadow: '0 4px 24px #0008', border: '1px solid #fff2', padding: 18, color: '#fff', fontSize: 14, minHeight: 120, position: 'relative', animation: 'fadeIn .3s' }}
                    tabIndex={-1}
                    onClick={e => e.stopPropagation()}
                  >
                    <button
                      style={{ position: 'absolute', top: 8, right: 8, background: 'none', border: 'none', color: '#fff', fontSize: 18, cursor: 'pointer', lineHeight: 1 }}
                      onClick={() => setSelectedNode(null)}
                      aria-label="关闭"
                    >×</button>
                    <div style={{ fontWeight: 600, fontSize: 18, marginBottom: 6 }}>{node.label}</div>
                    <div style={{ color: '#fbbf24', fontSize: 12, marginBottom: 4 }}>类型: {node.type}</div>
                    <div style={{ color: '#aaa', fontSize: 12, marginBottom: 4 }}>ID: {node.id}</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, marginBottom: 4 }}>
                      <div style={{ width: 12, height: 12, borderRadius: 6, background: node.color, border: '1px solid #fff3' }} />
                      <span>权重: {node.size}</span>
                    </div>
                    {node.data && (
                      <div style={{ marginTop: 8, color: '#eee', fontSize: 13 }}>
                        {node.type === 'document' && node.data.source && (
                          <div style={{ marginBottom: 4 }}>来源: {node.data.source.split('/').pop()}</div>
                        )}
                        {node.data.text && (
                          <div style={{ marginTop: 4, color: '#ccc' }}>预览: {node.data.text.substring(0, 120)}...</div>
                        )}
                      </div>
                    )}
                  </div>
                </foreignObject>
              );
            })()}
          </svg>
          {/* 节点编辑面板 */}
          {editingNode && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute top-10 left-1/2 -translate-x-1/2 bg-gray-900/95 p-6 rounded-lg border border-white/20 z-50 w-80"
            >
              <h4 className="text-white font-bold mb-2">编辑节点</h4>
              <div className="mb-2">
                <label className="block text-xs text-white/70 mb-1">标签</label>
                <input
                  className="w-full rounded px-2 py-1 bg-gray-800 text-white border border-white/10"
                  value={editLabel}
                  onChange={e => setEditLabel(e.target.value)}
                />
              </div>
              <div className="mb-2">
                <label className="block text-xs text-white/70 mb-1">颜色</label>
                <input
                  type="color"
                  className="w-8 h-8 p-0 border-none bg-transparent"
                  value={editColor}
                  onChange={e => setEditColor(e.target.value)}
                />
              </div>
              <div className="flex gap-2 justify-end mt-4">
                <button className="px-3 py-1 rounded bg-gray-700 text-white" onClick={handleEditCancel}>取消</button>
                <button className="px-3 py-1 rounded bg-blue-600 text-white" onClick={handleEditSave}>保存</button>
              </div>
            </motion.div>
          )}
        </div>        {/* Legend */}
        <div className="flex justify-between items-center text-xs text-white/50">
          <div className="flex gap-4">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span>概念</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span>实体</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-purple-500" />
              <span>文档</span>
            </div>
          </div>
          <span>缩放: {Math.round(zoomLevel * 100)}%</span>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 pt-2 border-t border-white/10">
          <div className="text-center">
            <p className="text-lg font-bold text-white">{filteredNodes.length}</p>
            <p className="text-xs text-white/60">节点</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-white">{filteredEdges.length}</p>
            <p className="text-xs text-white/60">边</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-white">{new Set(filteredNodes.map(n => n.type)).size}</p>
            <p className="text-xs text-white/60">类型</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
