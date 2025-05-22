import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import EntityExtractionForm from './EntityExtractionForm'; // Adjust path as necessary

describe('EntityExtractionForm', () => {
  const mockOnSubmit = jest.fn();

  beforeEach(() => {
    mockOnSubmit.mockClear();
  });

  test('renders the form with textarea and submit button', () => {
    render(<EntityExtractionForm onSubmit={mockOnSubmit} isLoading={false} />);
    
    expect(screen.getByLabelText(/Enter Text for Entity Extraction/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Paste text here to identify and extract entities/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Extract Entities/i })).toBeInTheDocument();
  });

  test('allows text input in the textarea', () => {
    render(<EntityExtractionForm onSubmit={mockOnSubmit} isLoading={false} />);
    const textarea = screen.getByLabelText(/Enter Text for Entity Extraction/i) as HTMLTextAreaElement;
    
    fireEvent.change(textarea, { target: { value: 'Apple Inc. is a company.' } });
    expect(textarea.value).toBe('Apple Inc. is a company.');
  });

  test('calls onSubmit with the input text when form is submitted', () => {
    render(<EntityExtractionForm onSubmit={mockOnSubmit} isLoading={false} />);
    const textarea = screen.getByLabelText(/Enter Text for Entity Extraction/i);
    const submitButton = screen.getByRole('button', { name: /Extract Entities/i });

    const testInput = 'Google is based in Mountain View.';
    fireEvent.change(textarea, { target: { value: testInput } });
    fireEvent.click(submitButton);

    expect(mockOnSubmit).toHaveBeenCalledTimes(1);
    expect(mockOnSubmit).toHaveBeenCalledWith(testInput);
  });

  test('does not call onSubmit if text input is empty or only whitespace', () => {
    const mockAlert = jest.spyOn(window, 'alert').mockImplementation(() => {});
    render(<EntityExtractionForm onSubmit={mockOnSubmit} isLoading={false} />);
    const submitButton = screen.getByRole('button', { name: /Extract Entities/i });

    fireEvent.click(submitButton); // Submit with empty input
    expect(mockOnSubmit).not.toHaveBeenCalled();
    expect(mockAlert).toHaveBeenCalledWith("Please enter some text to extract entities from.");

    const textarea = screen.getByLabelText(/Enter Text for Entity Extraction/i);
    fireEvent.change(textarea, { target: { value: '   ' } }); // Submit with whitespace
    fireEvent.click(submitButton);
    expect(mockOnSubmit).not.toHaveBeenCalled();
    expect(mockAlert).toHaveBeenCalledWith("Please enter some text to extract entities from.");
    
    mockAlert.mockRestore();
  });

  test('disables textarea and button when isLoading is true', () => {
    render(<EntityExtractionForm onSubmit={mockOnSubmit} isLoading={true} />);
    
    const textarea = screen.getByLabelText(/Enter Text for Entity Extraction/i);
    const submitButton = screen.getByRole('button', { name: /Extracting.../i }); // Text changes when loading

    expect(textarea).toBeDisabled();
    expect(submitButton).toBeDisabled();
    expect(screen.getByText('Extracting...')).toBeInTheDocument();
  });
});
