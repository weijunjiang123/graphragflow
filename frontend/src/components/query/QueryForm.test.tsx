import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import QueryForm from './QueryForm'; // Adjust path as necessary

describe('QueryForm', () => {
  const mockOnSubmit = jest.fn();

  beforeEach(() => {
    mockOnSubmit.mockClear();
  });

  test('renders the form with all fields and submit button', () => {
    render(<QueryForm onSubmit={mockOnSubmit} isLoading={false} />);

    expect(screen.getByLabelText(/Enter Your Query/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Search Type/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Number of Results \(Top K\)/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Submit Query/i })).toBeInTheDocument();
  });

  test('allows input in textarea, select, and number input', () => {
    render(<QueryForm onSubmit={mockOnSubmit} isLoading={false} />);

    const textarea = screen.getByLabelText(/Enter Your Query/i) as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: 'Test query' } });
    expect(textarea.value).toBe('Test query');

    const select = screen.getByLabelText(/Search Type/i) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'Vector' } });
    expect(select.value).toBe('Vector');

    const numberInput = screen.getByLabelText(/Number of Results \(Top K\)/i) as HTMLInputElement;
    fireEvent.change(numberInput, { target: { value: '10' } });
    expect(numberInput.value).toBe('10');
  });

  test('calls onSubmit with form data when submitted', () => {
    render(<QueryForm onSubmit={mockOnSubmit} isLoading={false} />);

    const textarea = screen.getByLabelText(/Enter Your Query/i);
    const select = screen.getByLabelText(/Search Type/i);
    const numberInput = screen.getByLabelText(/Number of Results \(Top K\)/i);
    const submitButton = screen.getByRole('button', { name: /Submit Query/i });

    fireEvent.change(textarea, { target: { value: 'Advanced search' } });
    fireEvent.change(select, { target: { value: 'Graph' } });
    fireEvent.change(numberInput, { target: { value: '7' } });
    fireEvent.click(submitButton);

    expect(mockOnSubmit).toHaveBeenCalledTimes(1);
    expect(mockOnSubmit).toHaveBeenCalledWith({
      queryText: 'Advanced search',
      searchType: 'Graph',
      topK: 7,
    });
  });

  test('does not call onSubmit if query text is empty', () => {
    const mockAlert = jest.spyOn(window, 'alert').mockImplementation(() => {});
    render(<QueryForm onSubmit={mockOnSubmit} isLoading={false} />);
    const submitButton = screen.getByRole('button', { name: /Submit Query/i });

    fireEvent.click(submitButton);
    expect(mockOnSubmit).not.toHaveBeenCalled();
    expect(mockAlert).toHaveBeenCalledWith("Please enter a query.");
    mockAlert.mockRestore();
  });

  test('disables form elements and shows loading text on button when isLoading is true', () => {
    render(<QueryForm onSubmit={mockOnSubmit} isLoading={true} />);

    expect(screen.getByLabelText(/Enter Your Query/i)).toBeDisabled();
    expect(screen.getByLabelText(/Search Type/i)).toBeDisabled();
    expect(screen.getByLabelText(/Number of Results \(Top K\)/i)).toBeDisabled();
    
    const submitButton = screen.getByRole('button', { name: /Searching.../i });
    expect(submitButton).toBeDisabled();
    expect(screen.getByText('Searching...')).toBeInTheDocument();
  });

  test('default values are set correctly', () => {
    render(<QueryForm onSubmit={mockOnSubmit} isLoading={false} />);
    expect((screen.getByLabelText(/Search Type/i) as HTMLSelectElement).value).toBe('Hybrid');
    expect((screen.getByLabelText(/Number of Results \(Top K\)/i) as HTMLInputElement).value).toBe('5');
  });
});
