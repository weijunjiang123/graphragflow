import { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Database, Code, Eye, EyeOff } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Textarea } from '@/components/ui/Textarea';
import { api } from '@/lib/api';

export function GraphQuery() {
  const [query, setQuery] = useState('');
  const [cypherQuery, setCypherQuery] = useState('');
  const [result, setResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showCypher, setShowCypher] = useState(false);
  const [mode, setMode] = useState<'natural' | 'cypher'>('natural');

  const handleNaturalQuery = async () => {
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    try {
      const response = await api.text2Cypher({
        query: query.trim(),
        return_direct: false,
        top_k: 10,
      });
      
      setResult(response);
      if (response.generated_cypher) {
        setCypherQuery(response.generated_cypher);
      }
    } catch (error) {
      console.error('查询失败:', error);
      setResult({ error: '查询失败，请稍后重试' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCypherQuery = async () => {
    if (!cypherQuery.trim() || isLoading) return;

    setIsLoading(true);
    try {
      const response = await api.executeCypher(cypherQuery.trim());
      setResult(response);
    } catch (error) {
      console.error('Cypher查询失败:', error);
      setResult({ error: 'Cypher查询失败，请检查语法' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = () => {
    if (mode === 'natural') {
      handleNaturalQuery();
    } else {
      handleCypherQuery();
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Database className="w-5 h-5" />
            图数据查询
          </CardTitle>
          <div className="flex gap-2">
            <Button
              variant={mode === 'natural' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setMode('natural')}
            >
              自然语言
            </Button>
            <Button
              variant={mode === 'cypher' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setMode('cypher')}
            >
              Cypher
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {mode === 'natural' ? (
          <div className="space-y-3">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="例如: 查找所有Person节点的名字"
              onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
            />
            <Button
              onClick={handleSubmit}
              disabled={!query.trim() || isLoading}
              className="w-full"
            >
              {isLoading ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
              ) : (
                <Search className="w-4 h-4 mr-2" />
              )}
              查询
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            <Textarea
              value={cypherQuery}
              onChange={(e) => setCypherQuery(e.target.value)}
              placeholder="输入Cypher查询语句..."
              className="font-mono text-sm"
              rows={4}
            />
            <Button
              onClick={handleSubmit}
              disabled={!cypherQuery.trim() || isLoading}
              className="w-full"
            >
              {isLoading ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
              ) : (
                <Code className="w-4 h-4 mr-2" />
              )}
              执行查询
            </Button>
          </div>
        )}

        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            {result.error ? (
              <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                <p className="text-red-400">{result.error}</p>
              </div>
            ) : (
              <>
                {result.generated_cypher && mode === 'natural' && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="text-white font-medium">生成的Cypher查询:</h4>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowCypher(!showCypher)}
                      >
                        {showCypher ? (
                          <EyeOff className="w-4 h-4" />
                        ) : (
                          <Eye className="w-4 h-4" />
                        )}
                      </Button>
                    </div>
                    {showCypher && (
                      <div className="p-3 bg-gray-900 rounded-lg border border-white/10">
                        <code className="text-green-400 text-sm font-mono">
                          {result.generated_cypher}
                        </code>
                      </div>
                    )}
                  </div>
                )}

                <div className="space-y-2">
                  <h4 className="text-white font-medium">查询结果:</h4>
                  <div className="p-4 bg-white/5 rounded-lg border border-white/10 max-h-64 overflow-y-auto">
                    <pre className="text-white/90 text-sm whitespace-pre-wrap">
                      {JSON.stringify(result.results || result.raw_results || result, null, 2)}
                    </pre>
                  </div>
                </div>

                {result.result && (
                  <div className="space-y-2">
                    <h4 className="text-white font-medium">AI解释:</h4>
                    <div className="p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
                      <p className="text-white">{result.result}</p>
                    </div>
                  </div>
                )}
              </>
            )}
          </motion.div>
        )}
      </CardContent>
    </Card>
  );
}
