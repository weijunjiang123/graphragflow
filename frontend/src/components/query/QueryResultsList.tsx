"use client";

import QueryResultItem from './QueryResultItem';

// Define the structure of a single result item based on API model
interface ResultItemData {
  source_document_id?: string;
  text_snippet: string;
  score?: number;
  graph_context?: any;
  node_id?: string;
  node_type?: string;
}

interface QueryResultsListProps {
  results: ResultItemData[];
}

export default function QueryResultsList({ results }: QueryResultsListProps) {
  if (!results || results.length === 0) {
    return (
      <div className="card mt-8">
        <h2 className="text-xl font-semibold mb-3 text-neutral-700">Query Results</h2>
        <p className="text-neutral-500">
          Enter a query using the form above to see search results here.
        </p>
      </div>
    );
  }

  return (
    <div className="mt-8">
      <h2 className="text-2xl font-semibold mb-6 text-neutral-800">Query Results</h2>
      <div className="space-y-6">
        {results.map((result, index) => (
          <QueryResultItem
            key={result.node_id || result.source_document_id || index} // Prefer stable key
            source_document_id={result.source_document_id}
            text_snippet={result.text_snippet}
            score={result.score}
            graph_context={result.graph_context}
            node_id={result.node_id}
            node_type={result.node_type}
          />
        ))}
      </div>
    </div>
  );
}
