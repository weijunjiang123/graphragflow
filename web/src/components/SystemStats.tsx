import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { Database, FileText, Network, RefreshCw, Activity } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { useAppStore } from '@/store/appStore';
import { useSystemStatus } from '@/hooks';
import { api } from '@/lib/api';
import { formatDate } from '@/lib/utils';

export function SystemStats() {
  const { stats, systemStatus, setStats } = useAppStore();
  const { checkStatus } = useSystemStatus();

  const refreshStats = async () => {
    try {
      const newStats = await api.getKnowledgeGraphStats();
      setStats(newStats);
    } catch (error) {
      console.error('刷新统计信息失败:', error);
    }
  };

  useEffect(() => {
    refreshStats();
  }, []);

  const getStatusColor = () => {
    switch (systemStatus) {
      case 'online':
        return 'text-green-400';
      case 'offline':
        return 'text-red-400';
      default:
        return 'text-yellow-400';
    }
  };

  const getStatusText = () => {
    switch (systemStatus) {
      case 'online':
        return '在线';
      case 'offline':
        return '离线';
      default:
        return '未知';
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            系统状态
          </CardTitle>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
            <span className={`text-sm ${getStatusColor()}`}>
              {getStatusText()}
            </span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                checkStatus();
                refreshStats();
              }}
              className="w-8 h-8 ml-2"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white/5 rounded-xl p-4 border border-white/10"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <FileText className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <p className="text-white/70 text-sm">文档数量</p>
                <p className="text-white text-2xl font-bold">
                  {stats?.document_count ?? 0}
                </p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white/5 rounded-xl p-4 border border-white/10"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                <Database className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <p className="text-white/70 text-sm">实体数量</p>
                <p className="text-white text-2xl font-bold">
                  {stats?.entity_count ?? 0}
                </p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white/5 rounded-xl p-4 border border-white/10"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                <Network className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <p className="text-white/70 text-sm">关系数量</p>
                <p className="text-white text-2xl font-bold">
                  {stats?.relationship_count ?? 0}
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        {stats?.last_updated && (
          <div className="mt-6 text-center">
            <p className="text-white/50 text-sm">
              最后更新: {formatDate(stats.last_updated)}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
