import { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Textarea } from '@/components/ui/Textarea';
import { useAppStore } from '@/store/appStore';
import { api } from '@/lib/api';
import { Message } from '@/types/api';

interface ChatInterfaceProps {
  onMessageSent?: (message: Message) => void;
  onResponseReceived?: (response: Message) => void;
}

export function ChatInterface({ onMessageSent, onResponseReceived }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const { messages, addMessage, setLoading, isLoading } = useAppStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input.trim(),
    };

    addMessage(userMessage);
    onMessageSent?.(userMessage);
    setInput('');
    setLoading(true);

    try {
      const response = await api.createChatResponse({
        messages: [...messages, userMessage],
      });

      addMessage(response.message);
      onResponseReceived?.(response.message);
    } catch (error) {
      console.error('发送消息失败:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: '抱歉，发生了错误，请稍后重试。',
      };
      addMessage(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          智能问答
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="请输入您的问题..."
              className="min-h-[120px] pr-12 resize-none"
              disabled={isLoading}
            />
            
            <motion.div
              className="absolute bottom-3 right-3"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || isLoading}
                className="w-10 h-10 rounded-full"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </motion.div>
          </div>

          <div className="flex items-center justify-between text-sm text-white/70">
            <span>支持 Shift + Enter 换行</span>
            <span>{input.length} 字符</span>
          </div>
        </form>

        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 flex items-center gap-2 text-white/70"
          >
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>正在生成回答...</span>
          </motion.div>
        )}
      </CardContent>
    </Card>
  );
}
