import { create } from 'zustand';
import { Message, DocumentStats, TaskStatus } from '@/types/api';

interface AppState {
  // Chat state
  messages: Message[];
  isLoading: boolean;
  currentTaskId: string | null;
  
  // Document state
  uploadProgress: number;
  uploadingFiles: string[];
  processingTasks: Map<string, TaskStatus>;
  
  // Knowledge graph state
  stats: DocumentStats | null;
  lastStatsUpdate: number;
  
  // System state
  systemStatus: 'online' | 'offline' | 'unknown';
  
  // Actions
  addMessage: (message: Message) => void;
  setMessages: (messages: Message[]) => void;
  setLoading: (loading: boolean) => void;
  setCurrentTaskId: (taskId: string | null) => void;
  
  setUploadProgress: (progress: number) => void;
  addUploadingFile: (filename: string) => void;
  removeUploadingFile: (filename: string) => void;
  
  updateTaskStatus: (taskId: string, status: TaskStatus) => void;
  removeTask: (taskId: string) => void;
  
  setStats: (stats: DocumentStats) => void;
  setSystemStatus: (status: 'online' | 'offline' | 'unknown') => void;
  
  reset: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  // Initial state
  messages: [],
  isLoading: false,
  currentTaskId: null,
  
  uploadProgress: 0,
  uploadingFiles: [],
  processingTasks: new Map(),
  
  stats: null,
  lastStatsUpdate: 0,
  
  systemStatus: 'unknown',
  
  // Actions
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),
    
  setMessages: (messages) => set({ messages }),
  
  setLoading: (isLoading) => set({ isLoading }),
  
  setCurrentTaskId: (currentTaskId) => set({ currentTaskId }),
  
  setUploadProgress: (uploadProgress) => set({ uploadProgress }),
  
  addUploadingFile: (filename) =>
    set((state) => ({
      uploadingFiles: [...state.uploadingFiles, filename],
    })),
    
  removeUploadingFile: (filename) =>
    set((state) => ({
      uploadingFiles: state.uploadingFiles.filter((f) => f !== filename),
    })),
    
  updateTaskStatus: (taskId, status) =>
    set((state) => {
      const newTasks = new Map(state.processingTasks);
      newTasks.set(taskId, status);
      return { processingTasks: newTasks };
    }),
    
  removeTask: (taskId) =>
    set((state) => {
      const newTasks = new Map(state.processingTasks);
      newTasks.delete(taskId);
      return { processingTasks: newTasks };
    }),
    
  setStats: (stats) =>
    set({
      stats,
      lastStatsUpdate: Date.now(),
    }),
    
  setSystemStatus: (systemStatus) => set({ systemStatus }),
  
  reset: () =>
    set({
      messages: [],
      isLoading: false,
      currentTaskId: null,
      uploadProgress: 0,
      uploadingFiles: [],
      processingTasks: new Map(),
      stats: null,
      lastStatsUpdate: 0,
      systemStatus: 'unknown',
    }),
}));
