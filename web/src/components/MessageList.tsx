import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Bot, Copy, Download } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { useAppStore } from '@/store/appStore';
import { copyToClipboard, downloadAsFile, formatDate } from '@/lib/utils';
import { Message } from '@/types/api';

interface MessageListProps {
  className?: string;
}

interface MessageItemProps {
  message: Message;
  index: number;
}

function MessageItem({ message, index }: MessageItemProps) {
  const isUser = message.role === 'user';

  const handleCopy = () => {
    copyToClipboard(message.content);
  };

  const handleDownload = () => {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
    const filename = `message-${timestamp}.md`;
    downloadAsFile(message.content, filename, 'text/markdown');
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`flex gap-3 max-w-4xl ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
          isUser 
            ? 'bg-gradient-to-r from-blue-500 to-blue-600' 
            : 'bg-gradient-to-r from-purple-500 to-purple-600'
        }`}>
          {isUser ? (
            <User className="w-5 h-5 text-white" />
          ) : (
            <Bot className="w-5 h-5 text-white" />
          )}
        </div>

        {/* Message Content */}
        <Card className={`flex-1 ${isUser ? 'bg-blue-500/10' : 'bg-white/5'}`}>
          <CardContent className="p-4">
            <div className="flex items-start justify-between gap-2 mb-2">
              <span className="text-sm font-medium text-white/90">
                {isUser ? '您' : 'GraphRAG Assistant'}
              </span>
              <div className="flex gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleCopy}
                  className="w-6 h-6 opacity-70 hover:opacity-100"
                >
                  <Copy className="w-3 h-3" />
                </Button>
                {!isUser && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleDownload}
                    className="w-6 h-6 opacity-70 hover:opacity-100"
                  >
                    <Download className="w-3 h-3" />
                  </Button>
                )}
              </div>
            </div>

            <div className="prose prose-invert prose-sm max-w-none">
              {isUser ? (
                <p className="text-white/90 whitespace-pre-wrap">{message.content}</p>
              ) : (
                <ReactMarkdown
                  components={{
                    code: ({ node, className, children, ...props }) => {
                      const match = /language-(\w+)/.exec(className || '');
                      return match ? (
                        <SyntaxHighlighter
                          style={oneDark as any}
                          language={match[1]}
                          PreTag="div"
                          className="rounded-lg my-2"
                          {...props}
                          ref={undefined}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code
                          className="bg-white/10 px-1 py-0.5 rounded text-blue-200"
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    },
                    p: ({ children }) => (
                      <p className="text-white/90 mb-2 last:mb-0">{children}</p>
                    ),
                    h1: ({ children }) => (
                      <h1 className="text-xl font-bold text-white mb-3">{children}</h1>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-lg font-semibold text-white mb-2">{children}</h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-base font-medium text-white mb-2">{children}</h3>
                    ),
                    ul: ({ children }) => (
                      <ul className="text-white/90 list-disc pl-4 mb-2">{children}</ul>
                    ),
                    ol: ({ children }) => (
                      <ol className="text-white/90 list-decimal pl-4 mb-2">{children}</ol>
                    ),
                    li: ({ children }) => (
                      <li className="mb-1">{children}</li>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-blue-500 pl-4 text-white/80 italic">
                        {children}
                      </blockquote>
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              )}
            </div>

            <div className="mt-2 text-xs text-white/50">
              {formatDate(new Date())}
            </div>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
}

export function MessageList({ className }: MessageListProps) {
  const { messages } = useAppStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <Bot className="w-16 h-16 text-white/30 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white/70 mb-2">
            开始对话
          </h3>
          <p className="text-white/50 max-w-md">
            向 GraphRAG 助手提问，获得基于知识图谱的智能回答
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex-1 overflow-y-auto px-6 py-4 space-y-6 ${className}`}>
      <AnimatePresence>
        {messages.map((message, index) => (
          <MessageItem
            key={index}
            message={message}
            index={index}
          />
        ))}
      </AnimatePresence>
      <div ref={messagesEndRef} />
    </div>
  );
}
