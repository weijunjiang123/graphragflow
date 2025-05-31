import { motion } from 'framer-motion';
import { Network, Zap, Database } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';

export function KnowledgeGraphVisualization() {
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Network className="w-5 h-5" />
          知识图谱可视化
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative h-64 bg-gradient-to-br from-blue-900/20 to-purple-900/20 rounded-xl border border-white/10 overflow-hidden">
          {/* Grid pattern background */}
          <div 
            className="absolute inset-0 opacity-20"
            style={{
              backgroundImage: `
                linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
              `,
              backgroundSize: '20px 20px'
            }}
          />
          
          {/* Animated nodes */}
          <motion.div
            className="absolute top-12 left-12 w-4 h-4 bg-blue-400 rounded-full"
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.8, 1, 0.8],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              repeatType: 'reverse',
            }}
          />
          
          <motion.div
            className="absolute top-20 right-16 w-3 h-3 bg-green-400 rounded-full"
            animate={{
              scale: [1, 1.3, 1],
              opacity: [0.7, 1, 0.7],
            }}
            transition={{
              duration: 2.5,
              repeat: Infinity,
              repeatType: 'reverse',
              delay: 0.5,
            }}
          />
          
          <motion.div
            className="absolute bottom-16 left-20 w-3 h-3 bg-purple-400 rounded-full"
            animate={{
              scale: [1, 1.1, 1],
              opacity: [0.9, 1, 0.9],
            }}
            transition={{
              duration: 1.8,
              repeat: Infinity,
              repeatType: 'reverse',
              delay: 1,
            }}
          />
          
          <motion.div
            className="absolute bottom-12 right-12 w-4 h-4 bg-yellow-400 rounded-full"
            animate={{
              scale: [1, 1.4, 1],
              opacity: [0.6, 1, 0.6],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              repeatType: 'reverse',
              delay: 1.5,
            }}
          />
          
          {/* Connecting lines */}
          <svg className="absolute inset-0 w-full h-full">
            <defs>
              <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="rgba(59, 130, 246, 0.6)" />
                <stop offset="100%" stopColor="rgba(147, 51, 234, 0.6)" />
              </linearGradient>
            </defs>
            
            <motion.line
              x1="56" y1="56" x2="240" y2="96"
              stroke="url(#lineGradient)"
              strokeWidth="2"
              strokeDasharray="5,5"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            
            <motion.line
              x1="56" y1="56" x2="92" y2="192"
              stroke="url(#lineGradient)"
              strokeWidth="2"
              strokeDasharray="5,5"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 2.5, repeat: Infinity, delay: 0.5 }}
            />
            
            <motion.line
              x1="240" y1="96" x2="240" y2="160"
              stroke="url(#lineGradient)"
              strokeWidth="2"
              strokeDasharray="5,5"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1.8, repeat: Infinity, delay: 1 }}
            />
          </svg>
          
          {/* Center content */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 2, type: 'spring', stiffness: 200 }}
                className="w-16 h-16 mx-auto bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mb-4"
              >
                <Database className="w-8 h-8 text-white" />
              </motion.div>
              
              <motion.h3
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 2.5 }}
                className="text-white font-medium mb-2"
              >
                知识图谱可视化
              </motion.h3>
              
              <motion.p
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 3 }}
                className="text-white/70 text-sm"
              >
                实体关系动态展示
              </motion.p>
            </div>
          </div>
          
          {/* Feature indicators */}
          <div className="absolute top-4 right-4 flex gap-2">
            <motion.div
              className="w-2 h-2 bg-green-400 rounded-full"
              animate={{
                opacity: [0.3, 1, 0.3],
              }}
              transition={{
                duration: 1,
                repeat: Infinity,
                repeatType: 'reverse',
              }}
            />
            <span className="text-white/50 text-xs">实时更新</span>
          </div>
          
          <div className="absolute bottom-4 left-4 flex items-center gap-2">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span className="text-white/50 text-xs">交互式探索</span>
          </div>
        </div>
        
        <div className="mt-4 text-center">
          <p className="text-white/70 text-sm">
            图谱数据加载后将在此处显示实体关系网络
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
