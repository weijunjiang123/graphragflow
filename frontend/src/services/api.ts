// Define the structure of API responses based on backend Pydantic models
// These should align with the models defined in `src/api/models.py` in the backend.

// For Document Processing
interface DocumentProcessRequestBody {
  texts: string[];
}
export interface DocumentProcessResponseData { // Exporting for use in page component
  message: string;
  document_ids: string[];
}

// For Querying
interface QueryRequestBody {
  query_text: string;
  search_type: string; // 'Hybrid', 'Vector', 'Graph', 'Fulltext'
  top_k: number;
}
export interface QueryResultItemData { // Exporting for use in page component
  source_document_id?: string | null; // Adjusted to match API model
  text_snippet: string;
  score?: number | null;
  graph_context?: any | null;
  node_id?: string | null;
  node_type?: string | null;
}
export interface QueryResponseData { // Exporting for use in page component
  query_text: string;
  search_type: string;
  results: QueryResultItemData[];
}

// For Entity Extraction
interface EntityExtractRequestBody {
  text: string;
}
export interface ApiEntityData { // Exporting for use in page component
  text: string;
  label_: string; // Matches the `Entity` model in `src/api/models.py`
}
export interface EntityExtractResponseData { // Exporting for use in page component
  text: string;
  entities: ApiEntityData[];
}

// --- API Service Implementation ---
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `API Error: ${response.status} ${response.statusText}`;
    try {
      const errorBody = await response.json();
      errorMessage = errorBody.detail || errorMessage; // FastAPI often returns errors in 'detail'
      if (Array.isArray(errorBody.detail)) { // Handle Pydantic validation errors
        errorMessage = errorBody.detail.map((err: any) => `${err.loc.join('.')} - ${err.msg}`).join('; ');
      }
    } catch (e) {
      // Ignore if error body is not JSON or empty
    }
    throw new Error(errorMessage);
  }
  return response.json() as Promise<T>;
}

export async function processDocuments(texts: string[]): Promise<DocumentProcessResponseData> {
  const body: DocumentProcessRequestBody = { texts };
  const response = await fetch(`${API_BASE_URL}/documents/process`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  return handleResponse<DocumentProcessResponseData>(response);
}

export async function submitQuery(queryText: string, searchType: string, topK: number): Promise<QueryResponseData> {
  const body: QueryRequestBody = {
    query_text: queryText,
    search_type: searchType.toLowerCase(), // Ensure backend receives lowercase search_type
    top_k: topK,
  };
  const response = await fetch(`${API_BASE_URL}/query`, { // Endpoint is /query not /query/
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  return handleResponse<QueryResponseData>(response);
}

export async function extractEntities(text: string): Promise<EntityExtractResponseData> {
  const body: EntityExtractRequestBody = { text };
  // Corrected endpoint from `/entities/extract/` to `/entities/extract`
  const response = await fetch(`${API_BASE_URL}/entities/extract`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  return handleResponse<EntityExtractResponseData>(response);
}
