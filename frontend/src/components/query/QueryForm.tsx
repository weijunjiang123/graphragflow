"use client";

import { useState } from 'react';

interface QueryFormData {
  queryText: string;
  searchType: 'Hybrid' | 'Vector' | 'Graph' | 'Fulltext';
  topK: number;
}

interface QueryFormProps {
  onSubmit: (data: QueryFormData) => void;
  isLoading: boolean;
}

export default function QueryForm({ onSubmit, isLoading }: QueryFormProps) {
  const [queryText, setQueryText] = useState<string>('');
  const [searchType, setSearchType] = useState<QueryFormData['searchType']>('Hybrid');
  const [topK, setTopK] = useState<number>(5);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!queryText.trim()) {
      alert("Please enter a query.");
      return;
    }
    onSubmit({ queryText, searchType, topK });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label htmlFor="queryText" className="block text-sm font-medium text-neutral-700 mb-1">
          Enter Your Query
        </label>
        <textarea
          id="queryText"
          name="queryText"
          rows={5}
          className="block w-full"
          placeholder="e.g., 'What are the main products of Apple Inc.?' or 'Connections between Company X and Person Y'"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          disabled={isLoading}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label htmlFor="searchType" className="block text-sm font-medium text-neutral-700 mb-1">
            Search Type
          </label>
          <select
            id="searchType"
            name="searchType"
            className="block w-full"
            value={searchType}
            onChange={(e) => setSearchType(e.target.value as QueryFormData['searchType'])}
            disabled={isLoading}
          >
            <option value="Hybrid">Hybrid</option>
            <option value="Vector">Vector Search</option>
            <option value="Graph">Graph Traversal</option>
            <option value="Fulltext">Full-text Search</option>
          </select>
        </div>
        <div>
          <label htmlFor="topK" className="block text-sm font-medium text-neutral-700 mb-1">
            Number of Results (Top K)
          </label>
          <input
            type="number"
            id="topK"
            name="topK"
            className="block w-full"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value, 10))}
            min="1"
            max="50" // Reasonable max for UI
            disabled={isLoading}
          />
        </div>
      </div>

      <div>
        <button
          type="submit"
          className="btn-primary w-full md:w-auto"
          disabled={isLoading}
        >
          {isLoading ? 'Searching...' : 'Submit Query'}
        </button>
      </div>
    </form>
  );
}
