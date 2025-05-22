import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import DocumentInputForm from './DocumentInputForm'; // Adjust path as necessary

describe('DocumentInputForm', () => {
  const mockOnSubmit = jest.fn();

  beforeEach(() => {
    // Clear any previous mock calls before each test
    mockOnSubmit.mockClear();
  });

  test('renders the form with textarea and submit button', () => {
    render(<DocumentInputForm onSubmit={mockOnSubmit} isLoading={false} />);
    
    expect(screen.getByLabelText(/Enter Text for Processing/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Paste your document text here/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Process Document Text/i })).toBeInTheDocument();
  });

  test('allows text input in the textarea', () => {
    render(<DocumentInputForm onSubmit={mockOnSubmit} isLoading={false} />);
    const textarea = screen.getByLabelText(/Enter Text for Processing/i) as HTMLTextAreaElement;
    
    fireEvent.change(textarea, { target: { value: 'This is a test document.' } });
    expect(textarea.value).toBe('This is a test document.');
  });

  test('calls onSubmit with the input text when form is submitted', () => {
    render(<DocumentInputForm onSubmit={mockOnSubmit} isLoading={false} />);
    const textarea = screen.getByLabelText(/Enter Text for Processing/i);
    const submitButton = screen.getByRole('button', { name: /Process Document Text/i });

    const testInput = 'Test document content.';
    fireEvent.change(textarea, { target: { value: testInput } });
    fireEvent.click(submitButton);

    expect(mockOnSubmit).toHaveBeenCalledTimes(1);
    // The component wraps the text in an array: onSubmit([textInput])
    expect(mockOnSubmit).toHaveBeenCalledWith([testInput]);
  });

  test('does not call onSubmit if text input is empty or only whitespace', () => {
    // Mock window.alert as it's called by the component
    const mockAlert = jest.spyOn(window, 'alert').mockImplementation(() => {});

    render(<DocumentInputForm onSubmit={mockOnSubmit} isLoading={false} />);
    const submitButton = screen.getByRole('button', { name: /Process Document Text/i });

    fireEvent.click(submitButton); // Submit with empty input
    expect(mockOnSubmit).not.toHaveBeenCalled();
    expect(mockAlert).toHaveBeenCalledWith("Please enter some text to process.");

    const textarea = screen.getByLabelText(/Enter Text for Processing/i);
    fireEvent.change(textarea, { target: { value: '   ' } }); // Submit with whitespace
    fireEvent.click(submitButton);
    expect(mockOnSubmit).not.toHaveBeenCalled();
    expect(mockAlert).toHaveBeenCalledWith("Please enter some text to process.");
    
    mockAlert.mockRestore(); // Clean up the mock
  });

  test('disables textarea and button when isLoading is true', () => {
    render(<DocumentInputForm onSubmit={mockOnSubmit} isLoading={true} />);
    
    const textarea = screen.getByLabelText(/Enter Text for Processing/i);
    const submitButton = screen.getByRole('button', { name: /Processing.../i }); // Text changes when loading

    expect(textarea).toBeDisabled();
    expect(submitButton).toBeDisabled();
    expect(screen.getByText('Processing...')).toBeInTheDocument();
  });
});
