"use client";

import { useState } from 'react';
import EntityExtractionForm from '@/components/extract/EntityExtractionForm';
import EntityResultsDisplay from '@/components/extract/EntityResultsDisplay';
import { extractEntities, EntityExtractResponseData, ApiEntityData } from '@/services/api'; // Import API function and types

// EntityData for UI component (EntityResultsDisplay expects 'label' not 'label_')
interface DisplayEntityData {
  text: string;
  label: string; 
}

export default function ExtractPage() {
  // State for entities to be displayed, matching the structure expected by EntityResultsDisplay
  const [displayEntities, setDisplayEntities] = useState<DisplayEntityData[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [submittedText, setSubmittedText] = useState<string>("");

  const handleExtractEntities = async (text: string) => {
    setIsLoading(true);
    setError(null);
    setSubmittedText(text);
    setDisplayEntities([]); // Clear previous entities

    try {
      const apiResponse: EntityExtractResponseData = await extractEntities(text);
      // Transform ApiEntityData (with label_) to DisplayEntityData (with label)
      const transformedEntities: DisplayEntityData[] = apiResponse.entities.map(apiEntity => ({
        text: apiEntity.text,
        label: apiEntity.label_ // Map label_ from API to label for UI component
      }));
      setDisplayEntities(transformedEntities);

    } catch (err: any) {
      const errorMessage = err.message || "An unknown error occurred during entity extraction.";
      setError(errorMessage);
      setDisplayEntities([]); // Clear entities on error
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h1 className="mb-8">Entity Extraction</h1>
      
      <div className="card mb-8">
        <h2 className="text-xl font-semibold mb-4 text-neutral-700">Extract Entities from Text</h2>
        <EntityExtractionForm onSubmit={handleExtractEntities} isLoading={isLoading} />
      </div>

      {isLoading && (
        <div className="text-center py-4 card">
          <p className="text-lg text-primary-DEFAULT">Extracting entities from: "{submittedText.substring(0, 50)}..."</p>
          {/* Spinner: <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-DEFAULT mx-auto mt-2"></div> */}
        </div>
      )}

      {error && (
        <div className="card bg-red-50 border-red-200 mt-8">
          <h3 className="text-lg font-semibold text-red-700 mb-2">Extraction Error</h3>
          <p className="text-red-600">{error}</p>
        </div>
      )}
      
      {/* EntityResultsDisplay expects 'entities' prop of type DisplayEntityData[] */}
      <EntityResultsDisplay 
        entities={displayEntities} 
        originalText={!isLoading && !error && displayEntities.length > 0 ? submittedText : undefined} 
      />
      
      {!isLoading && !error && displayEntities.length === 0 && submittedText && (
         <div className="card mt-8">
          <p className="text-neutral-500">No entities were extracted from "{submittedText.substring(0,100)}...", or the extraction returned empty.</p>
        </div>
      )}
    </div>
  );
}
