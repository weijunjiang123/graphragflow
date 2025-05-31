'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Database, Upload, Activity } from 'lucide-react';
import { ChatInterface } from '@/components/ChatInterface';
import { MessageList } from '@/components/MessageList';
import { FileUpload } from '@/components/FileUpload';
import { SystemStatus } from '@/components/SystemStatus';
import { GraphVisualization } from '@/components/GraphVisualization';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { useAppStore } from '@/store/appStore';
import { useSystemStatus } from '@/hooks';

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<'chat' | 'upload' | 'graph' | 'status'>('chat');
  const { messages, stats } = useAppStore();
  useSystemStatus();

  const tabs = [
    { id: 'chat', label: '智能问答', icon: MessageSquare },
    { id: 'upload', label: '文档上传', icon: Upload },
    { id: 'graph', label: '知识图谱', icon: Database },
    { id: 'status', label: '系统状态', icon: Activity },
  ] as const;

  return (
    <div className="min-h-screen p-4 md:p-6 lg:p-8 max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        className="text-center mb-6 lg:mb-8"
        variants={fadeInUp}
        initial="initial"
        animate="animate"
      >
        <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold gradient-text mb-3">
          GraphRAG Q&A
        </h1>
        <p className="text-lg md:text-xl text-white/70 max-w-3xl mx-auto">
          基于图增强检索生成技术的智能问答系统
        </p>
        <div className="mt-3 flex flex-wrap justify-center gap-2 md:gap-4 text-sm text-white/50">
          <span>📚 文档处理</span>
          <span>🧠 知识图谱</span>
          <span>💬 智能对话</span>
          <span>📊 可视化分析</span>
        </div>
      </motion.div>

      {/* Stats Bar */}
      {stats && (
        <motion.div
          className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4 mb-6"
          variants={staggerContainer}
          initial="initial"
          animate="animate"
        >
          { [
            {
              label: '文档数量',
              value: stats.document_count,
              color: 'blue',
            },
            {
              label: '实体数量',
              value: stats.entity_count,
              color: 'green',
            },
            {
              label: '关系数量',
              value: stats.relationship_count,
              color: 'purple',
            },
            {
              label: '对话数量',
              value: messages.length,
              color: 'orange',
            },
          ].map((stat, index) => (
            <motion.div
              key={stat.label}
              variants={fadeInUp}
              className="glass rounded-xl p-3 md:p-4 text-center"
            >
              <p className="text-xl md:text-2xl font-bold text-white">
                {stat.value.toLocaleString()}
              </p>
              <p className="text-xs md:text-sm text-white/60">{stat.label}</p>
            </motion.div>
          ))}
        </motion.div>
      )}

      {/* Navigation Tabs */}
      <motion.div
        className="flex flex-wrap gap-2 mb-6"
        variants={fadeInUp}
        initial="initial"
        animate="animate"
        transition={{ delay: 0.2 }}
      >
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <motion.button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3 md:px-4 py-2 rounded-xl transition-all duration-200 ${
                activeTab === tab.id
                  ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                  : 'glass text-white/70 hover:text-white hover:bg-white/10'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Icon className="w-4 h-4" />
              <span className="hidden sm:inline">{tab.label}</span>
            </motion.button>
          );
        })}
      </motion.div>

      {/* Main Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="w-full"
      >
        {activeTab === 'chat' && (
          <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 lg:gap-6">
            {/* Chat Messages - Takes more space on larger screens */}
            <div className="xl:col-span-3 order-2 xl:order-1">
              <Card className="h-[500px] md:h-[600px] flex flex-col">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <MessageSquare className="w-5 h-5" />
                    对话记录
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-1 p-0">
                  <MessageList className="h-full" />
                </CardContent>
              </Card>
            </div>

            {/* Right Sidebar */}
            <div className="xl:col-span-1 order-1 xl:order-2 space-y-4">
              <ChatInterface />
              <div className="hidden md:block">
                {/* <SystemStatus /> */}
              </div>
            </div>

            {/* Mobile SystemStatus */}
            <div className="md:hidden order-3 xl:order-3">
              <SystemStatus />
            </div>
          </div>
        )}

        {activeTab === 'upload' && (
          <div className="grid grid-cols-1 gap-4 lg:gap-6">
            <div className="lg:col-span-2">
              <FileUpload
                onUploadComplete={(result) => {
                  console.log('Upload completed:', result);
                }}
                onUploadError={(error) => {
                  console.error('Upload error:', error);
                }}
              />
            </div>
          </div>
        )}

        {activeTab === 'graph' && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 lg:gap-6">
            <div className="lg:col-span-3 order-2 lg:order-1">
              <GraphVisualization />
            </div>
            <div className="lg:col-span-1 order-1 lg:order-2 space-y-4">
              {/* <SystemStatus /> */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">图谱操作</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm text-white/70">
                    <p>🔍 使用搜索框查找特定节点</p>
                    <p>🎯 点击节点查看详细信息</p>
                    <p>🔗 双击展开节点关系</p>
                    <p>📊 右键菜单查看更多操作</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'status' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6">
            <div className="order-1">
              <SystemStatus />
            </div>
            <div className="order-2">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">系统信息</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-white/70">版本</span>
                      <span className="text-white font-medium">v1.0.0</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-white/70">运行环境</span>
                      <span className="text-white font-medium">Next.js + FastAPI</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-white/70">图数据库</span>
                      <span className="text-white font-medium">Neo4j</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-white/70">AI 模型</span>
                      <span className="text-white font-medium">OpenAI GPT</span>
                    </div>
                  </div>
                  
                  <div className="border-t border-white/10 pt-4">
                    <h4 className="font-medium text-white mb-3">功能特性</h4>
                    <ul className="space-y-2 text-sm text-white/70">
                      <li className="flex items-center gap-2">
                        <span className="text-green-400">✅</span>
                        文档自动解析与知识抽取
                      </li>
                      <li className="flex items-center gap-2">
                        <span className="text-green-400">✅</span>
                        动态知识图谱构建
                      </li>
                      <li className="flex items-center gap-2">
                        <span className="text-green-400">✅</span>
                        图增强检索生成
                      </li>
                      <li className="flex items-center gap-2">
                        <span className="text-green-400">✅</span>
                        实时可视化展示
                      </li>
                      <li className="flex items-center gap-2">
                        <span className="text-green-400">✅</span>
                        多轮对话支持
                      </li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}