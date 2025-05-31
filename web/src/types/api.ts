// API types based on OpenAPI specification

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatRequest {
  messages: Message[];
  settings?: Record<string, any>;
}

export interface ChatResponse {
  message: Message;
  sources?: Record<string, any>[];
  stats?: Record<string, any>;
}

export interface AsyncTaskResponse {
  task_id: string;
  status: string;
}

export interface GraphTaskResponse {
  task_id: string;
  status: string;
  filename: string;
}

export interface DocumentStats {
  document_count: number;
  entity_count: number;
  relationship_count: number;
  last_updated: string;
}

export interface Text2CypherRequest {
  query: string;
  return_direct?: boolean;
  top_k?: number;
}

export interface ValidationError {
  loc: (string | number)[];
  msg: string;
  type: string;
}

export interface HTTPValidationError {
  detail: ValidationError[];
}

export interface TaskStatus {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: any;
  error?: string;
  progress?: number;
}
