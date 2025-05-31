import { motion, AnimatePresence } from 'framer-motion';
import { Download, Copy, User, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { useAppStore } from '@/store/appStore';
import { copyToClipboard, downloadAsFile } from '@/lib/utils';
import { Message } from '@/types/api';

interface ChatHistoryProps {
  onExportPDF?: () => void;
  onExportMarkdown?: () => void;
}

export function ChatHistory({ onExportPDF, onExportMarkdown }: ChatHistoryProps) {
  const { messages } = useAppStore();

  const handleCopyMessage = async (content: string) => {
    try {
      await copyToClipboard(content);
      // TODO: Show toast notification
    } catch (error) {
      console.error('复制失败:', error);
    }
  };

  const handleExportMarkdown = () => {
    const markdown = messages
      .map((msg) => {
        const role = msg.role === 'user' ? '用户' : '助手';
        return `## ${role}\n\n${msg.content}\n\n`;
      })
      .join('');

    downloadAsFile(
      markdown,
      `chat-history-${new Date().toISOString().split('T')[0]}.md`,
      'text/markdown'
    );
  };

  if (messages.length === 0) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>对话历史</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <Bot className="w-16 h-16 mx-auto text-white/30 mb-4" />
            <p className="text-white/70">暂无对话记录</p>
            <p className="text-white/50 text-sm mt-2">
              开始提问来体验智能问答功能
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>对话历史</CardTitle>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleExportMarkdown}
              className="text-white/70 hover:text-white"
            >
              <Download className="w-4 h-4 mr-2" />
              导出 MD
            </Button>
            {onExportPDF && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onExportPDF}
                className="text-white/70 hover:text-white"
              >
                <Download className="w-4 h-4 mr-2" />
                导出 PDF
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 max-h-96 overflow-y-auto">
        <AnimatePresence>
          {messages.map((message, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`flex gap-3 ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`flex gap-3 max-w-[80%] ${
                  message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    message.role === 'user'
                      ? 'bg-blue-500'
                      : 'bg-green-500'
                  }`}
                >
                  {message.role === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>

                <div
                  className={`group relative px-4 py-3 rounded-2xl ${
                    message.role === 'user'
                      ? 'bg-blue-500/20 text-white'
                      : 'bg-white/5 text-white'
                  }`}
                >
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown
                      components={{
                        code({ node, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || '');
                          return match ? (
                            <SyntaxHighlighter
                              style={oneDark as any}
                              language={match[1]}
                              PreTag="div"
                              className="rounded-lg"
                              {...props}
                              // 移除ref属性，防止类型冲突
                              ref={undefined}
                            >
                              {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                          ) : (
                            <code
                              className="bg-white/10 px-1 py-0.5 rounded text-sm"
                              {...props}
                            >
                              {children}
                            </code>
                          );
                        },
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>

                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleCopyMessage(message.content)}
                    className="absolute top-2 right-2 w-6 h-6 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Copy className="w-3 h-3" />
                  </Button>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}
