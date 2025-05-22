"use client";

import EntityDisplayItem from './EntityDisplayItem';

// Define the structure of a single entity based on API model or expected data
interface EntityData {
  text: string;
  label: string; // In API model, this is label_
}

interface EntityResultsDisplayProps {
  entities: EntityData[];
  originalText?: string; // Optional: to show the text from which entities were extracted
}

export default function EntityResultsDisplay({ entities, originalText }: EntityResultsDisplayProps) {
  if (!entities || entities.length === 0) {
    return (
      <div className="card mt-8">
        <h2 className="text-xl font-semibold mb-3 text-neutral-700">Extracted Entities</h2>
        <p className="text-neutral-500">
          Enter text in the form above and click "Extract Entities" to see the results here.
        </p>
      </div>
    );
  }

  return (
    <div className="card mt-8">
      <h2 className="text-xl font-semibold mb-4 text-neutral-700">Extracted Entities</h2>
      {originalText && (
        <div className="mb-6 p-4 bg-neutral-50 border border-neutral-200 rounded-md">
          <h3 className="text-sm font-semibold text-neutral-600 mb-1">Original Text:</h3>
          <p className="text-sm text-neutral-700 whitespace-pre-wrap">{originalText}</p>
        </div>
      )}
      <div className="space-y-2">
        {entities.map((entity, index) => (
          <EntityDisplayItem
            key={`${entity.text}-${index}`} // Simple key, consider more robust if entities can repeat
            text={entity.text}
            label={entity.label}
          />
        ))}
      </div>
    </div>
  );
}
