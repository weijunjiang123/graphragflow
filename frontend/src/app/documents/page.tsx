"use client";

import { useState } from 'react';
import DocumentInputForm from '@/components/documents/DocumentInputForm';
import DocumentResultsDisplay from '@/components/documents/DocumentResultsDisplay';
import { processDocuments, DocumentProcessResponseData } from '@/services/api'; // Import the API function and response type

// Define the structure for displaying results in the UI
interface DisplayResult {
  id: string; // Can be one of the document_ids from API or a generated one
  status: 'Processing' | 'Completed' | 'Failed' | 'Submitted';
  message?: string;
  rawApiResponse?: DocumentProcessResponseData; // Store the full response if needed
}

export default function DocumentsPage() {
  const [results, setResults] = useState<DisplayResult[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleProcessDocuments = async (texts: string[]) => {
    setIsLoading(true);
    setError(null);

    // Add a preliminary "Submitted" status for immediate feedback
    const submittedResult: DisplayResult = {
      id: `submission-${Date.now()}`,
      status: 'Submitted',
      message: `Processing ${texts.length} text block(s)...`
    };
    setResults([submittedResult]); // Replace previous results with this new submission status

    try {
      const apiResponse = await processDocuments(texts);
      // Update results based on API response
      const processedResults: DisplayResult[] = apiResponse.document_ids.map((docId, index) => ({
        id: docId,
        status: 'Completed',
        message: index === 0 ? apiResponse.message : `Document ID: ${docId} processed.`, // Show main message for the first, specific for others
      }));
      setResults(processedResults);

    } catch (err: any) {
      const errorMessage = err.message || "An unknown error occurred during document processing.";
      setError(errorMessage);
      setResults([{ 
        id: submittedResult.id, // Keep the submission ID for context
        status: 'Failed', 
        message: errorMessage 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h1 className="mb-8">Document Processing</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-1 gap-8"> {/* Changed to 1 column for simplicity, results below form */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4 text-neutral-700">Input Document Text</h2>
          <DocumentInputForm onSubmit={handleProcessDocuments} isLoading={isLoading} />
        </div>
        
        {error && (
          <div className="card bg-red-50 border-red-200 mt-8">
            <h3 className="text-lg font-semibold text-red-700 mb-2">Processing Error</h3>
            <p className="text-red-600">{error}</p>
          </div>
        )}

        {/* Pass the UI-specific 'results' to DocumentResultsDisplay */}
        {/* The DocumentResultsDisplay component needs to be adapted to accept DisplayResult[] */}
        <DocumentResultsDisplay results={results.map(r => ({ // Adapt to DocumentResultsDisplay's expected props
            id: r.id, 
            status: r.status, 
            message: r.message
        }))} />
      </div>
    </div>
  );
}
