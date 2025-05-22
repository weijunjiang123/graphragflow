"use client";

import React from 'react';

interface QueryResultItemProps {
  source_document_id?: string; // From API model
  text_snippet: string;
  score?: number;
  graph_context?: any; // Could be node properties or other graph data
  node_id?: string;
  node_type?: string;
}

export default function QueryResultItem({ 
  source_document_id, 
  text_snippet, 
  score, 
  graph_context,
  node_id,
  node_type 
}: QueryResultItemProps) {
  return (
    <div className="card mb-6"> {/* Each item is a card */}
      {source_document_id && (
        <p className="text-xs text-neutral-500 mb-1">
          Source Document: {source_document_id}
        </p>
      )}
      {node_id && (
         <p className="text-xs text-neutral-500 mb-1">
          Node ID: {node_id} {node_type && <span className="ml-1 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full">{node_type}</span>}
        </p>
      )}
      
      <p className="text-neutral-700 mb-3 leading-relaxed">
        {text_snippet}
      </p>
      
      {score !== undefined && (
        <p className="text-sm font-medium text-primary-DEFAULT mb-2">
          Relevance Score: {score.toFixed(4)}
        </p>
      )}

      {graph_context && (
        <div className="mt-3 pt-3 border-t border-neutral-200">
          <h4 className="text-sm font-semibold text-neutral-600 mb-1">Graph Context:</h4>
          <pre className="bg-neutral-100 p-3 rounded-md text-xs text-neutral-700 overflow-x-auto">
            {JSON.stringify(graph_context, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
