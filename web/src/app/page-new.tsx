'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Brain, Sparkles, Database, MessageSquare } from 'lucide-react';
import { ChatInterface } from '@/components/ChatInterface';
import { ChatHistory } from '@/components/ChatHistory';
import { FileUpload } from '@/components/FileUpload';
import { SystemStats } from '@/components/SystemStats';
import { GraphQuery } from '@/components/GraphQuery';
import { KnowledgeGraphVisualization } from '@/components/KnowledgeGraphVisualization';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function HomePage() {
  const [uploadKey, setUploadKey] = useState(0);

  const handleUploadComplete = (result: any) => {
    console.log('文件上传完成:', result);
    // Force refresh the upload component
    setUploadKey(prev => prev + 1);
  };

  const handleUploadError = (error: Error) => {
    console.error('文件上传失败:', error);
    // TODO: Show error toast
  };

  return (
    <div className="min-h-screen p-4 md:p-6 lg:p-8">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 text-center"
      >
        <div className="flex items-center justify-center gap-3 mb-4">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
            className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center"
          >
            <Brain className="w-6 h-6 text-white" />
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="text-4xl md:text-5xl font-bold text-white"
          >
            GraphRAG
          </motion.h1>
          <motion.div
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
          >
            <Sparkles className="w-8 h-8 text-yellow-400" />
          </motion.div>
        </div>
        
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="text-white/70 text-lg max-w-2xl mx-auto"
        >
          基于图增强检索生成技术的智能知识问答系统
        </motion.p>
        
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="flex items-center justify-center gap-6 mt-4 text-white/50 text-sm"
        >
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4" />
            <span>知识图谱</span>
          </div>
          <div className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4" />
            <span>智能问答</span>
          </div>
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4" />
            <span>AI驱动</span>
          </div>
        </motion.div>
      </motion.header>

      {/* Main Content Grid */}
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Top Row - System Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
        >
          <SystemStats />
        </motion.div>

        {/* Second Row - Chat Interface and History */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          <ChatInterface />
          <ChatHistory />
        </motion.div>

        {/* Third Row - File Upload and Graph Query */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          <FileUpload
            key={uploadKey}
            onUploadComplete={handleUploadComplete}
            onUploadError={handleUploadError}
          />
          <GraphQuery />
        </motion.div>

        {/* Fourth Row - Knowledge Graph Visualization */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0 }}
        >
          <KnowledgeGraphVisualization />
        </motion.div>

        {/* Features Showcase */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.1 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6"
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Brain className="w-5 h-5 text-blue-400" />
                智能理解
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-white/70">
                深度理解文档内容，自动构建知识图谱，提供精准的语义检索能力。
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Database className="w-5 h-5 text-green-400" />
                图谱增强
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-white/70">
                利用图数据库存储实体关系，通过图推理增强检索结果的准确性。
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Sparkles className="w-5 h-5 text-purple-400" />
                生成增强
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-white/70">
                结合检索到的知识和大语言模型，生成准确、相关的回答内容。
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Footer */}
      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="mt-16 text-center text-white/50 text-sm"
      >
        <p>
          © 2025 GraphRAG Q&A System. 
          基于图增强检索生成技术构建的智能知识问答平台。
        </p>
      </motion.footer>
    </div>
  );
}
