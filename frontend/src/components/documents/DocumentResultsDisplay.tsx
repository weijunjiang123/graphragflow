"use client";

import { useState } from 'react';

interface DocumentResult {
  // Define what a result item looks like, e.g.:
  id: string;
  status: 'Processing' | 'Completed' | 'Failed';
  message?: string;
}

interface DocumentResultsDisplayProps {
  // Props will be defined when API integration happens
  // For now, it's a placeholder
  results?: DocumentResult[]; // Example
}

export default function DocumentResultsDisplay({ results }: DocumentResultsDisplayProps) {
  // Example state, replace with actual data flow
  const [currentResults, setCurrentResults] = useState<DocumentResult[]>(results || []);

  if (!currentResults.length) {
    return (
      <div className="card mt-8">
        <h2 className="text-xl font-semibold mb-3 text-neutral-700">Processing Results</h2>
        <p className="text-neutral-500">
          Submit documents using the form above to see processing status and results here.
        </p>
      </div>
    );
  }

  return (
    <div className="card mt-8">
      <h2 className="text-xl font-semibold mb-4 text-neutral-700">Processing Results</h2>
      <div className="space-y-4">
        {currentResults.map((result) => (
          <div key={result.id} className="p-4 border border-neutral-200 rounded-md">
            <p className="font-medium text-neutral-800">Document ID: {result.id}</p>
            <p className={`text-sm ${
              result.status === 'Completed' ? 'text-green-600' :
              result.status === 'Processing' ? 'text-blue-600' :
              'text-red-600'
            }`}>
              Status: {result.status}
            </p>
            {result.message && <p className="text-xs text-neutral-500 mt-1">{result.message}</p>}
          </div>
        ))}
      </div>
    </div>
  );
}
