"use client";

import { useState } from 'react';

interface EntityExtractionFormProps {
  onSubmit: (text: string) => void;
  isLoading: boolean;
}

export default function EntityExtractionForm({ onSubmit, isLoading }: EntityExtractionFormProps) {
  const [textInput, setTextInput] = useState<string>('');

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!textInput.trim()) {
      alert("Please enter some text to extract entities from.");
      return;
    }
    onSubmit(textInput);
    // setTextInput(''); // Optionally clear input
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label htmlFor="extractText" className="block text-sm font-medium text-neutral-700 mb-1">
          Enter Text for Entity Extraction
        </label>
        <textarea
          id="extractText"
          name="extractText"
          rows={8}
          className="block w-full"
          placeholder="Paste text here to identify and extract entities like people, organizations, and locations..."
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          disabled={isLoading}
        />
      </div>
      <div>
        <button
          type="submit"
          className="btn-primary w-full md:w-auto"
          disabled={isLoading}
        >
          {isLoading ? 'Extracting...' : 'Extract Entities'}
        </button>
      </div>
    </form>
  );
}
