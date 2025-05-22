"use client";

import { useState } from 'react';
import QueryForm from '@/components/query/QueryForm';
import QueryResultsList from '@/components/query/QueryResultsList';
import { submitQuery, QueryResponseData, QueryResultItemData } from '@/services/api'; // Import API function and types

// Define the structure for QueryForm data (already matches component)
interface QueryFormData {
  queryText: string;
  searchType: 'Hybrid' | 'Vector' | 'Graph' | 'Fulltext'; // Matches QueryForm internal state
  topK: number;
}

export default function QueryPage() {
  const [results, setResults] = useState<QueryResultItemData[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [currentQuery, setCurrentQuery] = useState<string>("");
  const [currentSearchType, setCurrentSearchType] = useState<string>("");


  const handleQuerySubmit = async (data: QueryFormData) => {
    setIsLoading(true);
    setError(null);
    setCurrentQuery(data.queryText);
    setCurrentSearchType(data.searchType);
    setResults([]); // Clear previous results

    try {
      // The searchType from QueryForm is 'Hybrid', 'Vector', etc.
      // The API's submitQuery function expects a string and converts to lowercase.
      const apiResponse = await submitQuery(data.queryText, data.searchType, data.topK);
      setResults(apiResponse.results);
    } catch (err: any) {
      const errorMessage = err.message || "An unknown error occurred while querying.";
      setError(errorMessage);
      setResults([]); // Clear results on error
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h1 className="mb-8">Query Interface</h1>
      
      <div className="card mb-8">
        <h2 className="text-xl font-semibold mb-4 text-neutral-700">Search the Knowledge Graph</h2>
        <QueryForm onSubmit={handleQuerySubmit} isLoading={isLoading} />
      </div>

      {isLoading && (
        <div className="text-center py-4 card">
          <p className="text-lg text-primary-DEFAULT">Searching for: "{currentQuery}" (Type: {currentSearchType})...</p>
          {/* You can add a spinner here: e.g. <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-DEFAULT mx-auto mt-2"></div> */}
        </div>
      )}

      {error && (
        <div className="card bg-red-50 border-red-200 mt-8">
          <h3 className="text-lg font-semibold text-red-700 mb-2">Query Error</h3>
          <p className="text-red-600">{error}</p>
        </div>
      )}
      
      {!isLoading && !error && results.length > 0 && (
         <p className="text-lg text-neutral-700 my-6">
           Showing results for: "{currentQuery}" (Type: {currentSearchType})
         </p>
      )}
      
      {/* QueryResultsList expects 'results' prop of type QueryResultItemData[] which matches API response */}
      <QueryResultsList results={results} /> 
      
      {!isLoading && !error && results.length === 0 && currentQuery && (
        <div className="card mt-8">
          <p className="text-neutral-500">No results found for "{currentQuery}" (Type: {currentSearchType}). Try a different query or search type.</p>
        </div>
      )}
    </div>
  );
}
