"use client";

import { useState } from 'react';

interface DocumentInputFormProps {
  onSubmit: (texts: string[]) => void;
  isLoading: boolean;
}

export default function DocumentInputForm({ onSubmit, isLoading }: DocumentInputFormProps) {
  const [textInput, setTextInput] = useState<string>('');

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!textInput.trim()) {
      // Basic validation: prevent empty submissions
      alert("Please enter some text to process.");
      return;
    }
    // For simplicity, we'll treat each line as a separate document/text item for now.
    // Or, the backend API might expect a list of strings, and this component provides one string.
    // Adjust based on how the API's DocumentProcessRequest expects `texts: List[str]`.
    // Current API model `DocumentProcessRequest` expects `texts: List[str]`.
    // So we'll split by newlines, or send as a single-item list.
    // Let's assume the backend is fine with one large text block to be chunked there,
    // or the user is expected to paste multiple pre-segmented texts.
    // For now, sending as a list containing one string.
    onSubmit([textInput]);
    // setTextInput(''); // Optionally clear input after submission
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label htmlFor="documentText" className="block text-sm font-medium text-neutral-700 mb-1">
          Enter Text for Processing
        </label>
        <textarea
          id="documentText"
          name="documentText"
          rows={10}
          className="block w-full" // Base styles from globals.css
          placeholder="Paste your document text here. You can input multiple paragraphs or sections. The backend will process this content into graph structures."
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          disabled={isLoading}
        />
        <p className="mt-2 text-sm text-neutral-500">
          The entered text will be processed to extract entities and relationships, and then stored in the knowledge graph.
        </p>
      </div>
      <div>
        <button
          type="submit"
          className="btn-primary w-full md:w-auto"
          disabled={isLoading}
        >
          {isLoading ? 'Processing...' : 'Process Document Text'}
        </button>
      </div>
    </form>
  );
}
