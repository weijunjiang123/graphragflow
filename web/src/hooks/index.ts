import { useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';
import { useAppStore } from '@/store/appStore';
import { TaskStatus } from '@/types/api';

export function usePolling<T>(
  callback: () => Promise<T>,
  interval: number,
  dependencies: any[] = []
) {
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  const startPolling = () => {
    if (intervalRef.current) return;
    
    setIsPolling(true);
    intervalRef.current = setInterval(callback, interval);
  };

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      setIsPolling(false);
    }
  };

  useEffect(() => {
    return () => stopPolling();
  }, []);

  useEffect(() => {
    if (isPolling) {
      stopPolling();
      startPolling();
    }
  }, dependencies);

  return { startPolling, stopPolling, isPolling };
}

export function useTaskPolling(taskId: string | null, onComplete?: (result: any) => void) {
  const { updateTaskStatus, removeTask } = useAppStore();
  
  const pollTask = async () => {
    if (!taskId) return;
    
    try {
      const result = await api.getAsyncChatResult(taskId);
      
      if (result.status === 'completed') {
        updateTaskStatus(taskId, {
          status: 'completed',
          result: result.result,
        });
        onComplete?.(result.result);
        stopPolling();
      } else if (result.status === 'failed') {
        updateTaskStatus(taskId, {
          status: 'failed',
          error: result.error || '任务失败',
        });
        stopPolling();
      } else {
        updateTaskStatus(taskId, {
          status: 'processing',
          progress: result.progress,
        });
      }
    } catch (error) {
      console.error('任务轮询失败:', error);
      updateTaskStatus(taskId, {
        status: 'failed',
        error: '轮询失败',
      });
      stopPolling();
    }
  };

  const { startPolling, stopPolling, isPolling } = usePolling(pollTask, 2000, [taskId]);

  useEffect(() => {
    if (taskId) {
      startPolling();
    } else {
      stopPolling();
    }
    
    return () => stopPolling();
  }, [taskId]);

  return { isPolling };
}

export function useSystemStatus() {
  const { setSystemStatus, setStats } = useAppStore();
  
  const checkStatus = async () => {
    try {
      await api.healthCheck();
      setSystemStatus('online');
      
      // Also fetch stats while we're at it
      try {
        const stats = await api.getKnowledgeGraphStats();
        setStats(stats);
      } catch (error) {
        console.warn('获取统计信息失败:', error);
      }
    } catch (error) {
      setSystemStatus('offline');
    }
  };

  const { startPolling, stopPolling } = usePolling(checkStatus, 30000); // Check every 30 seconds

  useEffect(() => {
    checkStatus(); // Check immediately
    startPolling();
    
    return () => stopPolling();
  }, []);

  return { checkStatus };
}

export function useFileUpload() {
  const { addUploadingFile, removeUploadingFile, setUploadProgress } = useAppStore();
  const [isUploading, setIsUploading] = useState(false);

  const uploadFile = async (file: File, settings?: string) => {
    setIsUploading(true);
    addUploadingFile(file.name);
    setUploadProgress(0);

    try {
      // Simulate upload progress (since the API doesn't provide real-time progress)
      const progressInterval = setInterval(() => {
        setUploadProgress((prev: number) => Math.min(prev + 10, 90));
      }, 200);

      const result = await api.uploadDocument(file, settings);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      // Wait a bit to show 100% progress
      setTimeout(() => {
        removeUploadingFile(file.name);
        setUploadProgress(0);
        setIsUploading(false);
      }, 500);

      return result;
    } catch (error) {
      removeUploadingFile(file.name);
      setUploadProgress(0);
      setIsUploading(false);
      throw error;
    }
  };

  return { uploadFile, isUploading };
}

export function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === 'undefined') {
      return initialValue;
    }
    
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.warn(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = (value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      console.warn(`Error setting localStorage key "${key}":`, error);
    }
  };

  return [storedValue, setValue] as const;
}
