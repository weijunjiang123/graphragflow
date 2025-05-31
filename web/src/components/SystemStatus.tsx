import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, Database, FileText, Users, Clock, Server } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { useAppStore } from '@/store/appStore';
import { useSystemStatus } from '@/hooks';
import { formatDate } from '@/lib/utils';

interface StatusIndicatorProps {
  status: 'online' | 'offline' | 'unknown';
  label: string;
}

function StatusIndicator({ status, label }: StatusIndicatorProps) {
  const colors = {
    online: 'bg-green-500',
    offline: 'bg-red-500',
    unknown: 'bg-yellow-500',
  };

  return (
    <div className="flex items-center gap-2">
      <motion.div
        className={`w-2 h-2 rounded-full ${colors[status]}`}
        animate={{
          opacity: status === 'online' ? [1, 0.5, 1] : 1,
        }}
        transition={{
          duration: 2,
          repeat: status === 'online' ? Infinity : 0,
        }}
      />
      <span className="text-sm text-white/70">{label}</span>
    </div>
  );
}

interface StatCardProps {
  icon: React.ReactNode;
  title: string;
  value: string | number;
  subtitle?: string;
  color?: string;
}

function StatCard({ icon, title, value, subtitle, color = 'blue' }: StatCardProps) {
  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/20 border-blue-500/30',
    green: 'from-green-500/20 to-green-600/20 border-green-500/30',
    purple: 'from-purple-500/20 to-purple-600/20 border-purple-500/30',
    orange: 'from-orange-500/20 to-orange-600/20 border-orange-500/30',
  };

  return (
    <motion.div
      className={`p-4 rounded-xl bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} border backdrop-blur-sm`}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-white/10">
          {icon}
        </div>
        <div className="flex-1">
          <p className="text-xs text-white/60 uppercase tracking-wider font-medium">
            {title}
          </p>
          <p className="text-xl font-bold text-white">
            {typeof value === 'number' ? value.toLocaleString() : value}
          </p>
          {subtitle && (
            <p className="text-xs text-white/50 mt-1">{subtitle}</p>
          )}
        </div>
      </div>
    </motion.div>
  );
}

export function SystemStatus() {
  const { stats, systemStatus } = useAppStore();
  const { checkStatus } = useSystemStatus();

  useEffect(() => {
    checkStatus();
  }, []);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="w-5 h-5" />
          系统状态
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* System Status */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-white/80">服务状态</h3>
          <div className="grid grid-cols-1 gap-2">
            <StatusIndicator
              status={systemStatus}
              label={`API 服务 ${systemStatus === 'online' ? '运行正常' : systemStatus === 'offline' ? '离线' : '未知'}`}
            />
            <StatusIndicator
              status={stats ? 'online' : 'unknown'}
              label={`知识图谱 ${stats ? '已连接' : '连接中'}`}
            />
          </div>
        </div>

        {/* Statistics */}
        {stats && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-white/80">知识图谱统计</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <StatCard
                icon={<FileText className="w-4 h-4 text-blue-400" />}
                title="文档数量"
                value={stats.document_count}
                color="blue"
              />
              <StatCard
                icon={<Database className="w-4 h-4 text-green-400" />}
                title="实体数量"
                value={stats.entity_count}
                color="green"
              />
              <StatCard
                icon={<Users className="w-4 h-4 text-purple-400" />}
                title="关系数量"
                value={stats.relationship_count}
                color="purple"
              />
              <StatCard
                icon={<Clock className="w-4 h-4 text-orange-400" />}
                title="最后更新"
                value={formatDate(stats.last_updated)}
                color="orange"
              />
            </div>
          </div>
        )}

        {/* Performance Metrics */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-white/80">性能指标</h3>
          <div className="grid grid-cols-1 gap-3">
            <motion.div
              className="p-3 rounded-lg bg-white/5 border border-white/10"
              whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm text-white/70">响应时间</span>
                <span className="text-sm font-mono text-green-400">~200ms</span>
              </div>
            </motion.div>
            
            <motion.div
              className="p-3 rounded-lg bg-white/5 border border-white/10"
              whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm text-white/70">内存使用</span>
                <span className="text-sm font-mono text-blue-400">~45%</span>
              </div>
            </motion.div>
            
            <motion.div
              className="p-3 rounded-lg bg-white/5 border border-white/10"
              whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm text-white/70">图数据库连接</span>
                <span className="text-sm font-mono text-green-400">活跃</span>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-white/80">快速操作</h3>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={checkStatus}
              className="text-xs"
            >
              <Server className="w-3 h-3 mr-1" />
              刷新状态
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
