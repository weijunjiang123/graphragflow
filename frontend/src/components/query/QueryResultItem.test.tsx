import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import QueryResultItem from './QueryResultItem'; // Adjust path as necessary

describe('QueryResultItem', () => {
  const defaultProps = {
    text_snippet: "This is a test snippet.",
  };

  test('renders the text snippet', () => {
    render(<QueryResultItem {...defaultProps} />);
    expect(screen.getByText(defaultProps.text_snippet)).toBeInTheDocument();
  });

  test('renders source document ID if provided', () => {
    const propsWithSource = { ...defaultProps, source_document_id: "doc123" };
    render(<QueryResultItem {...propsWithSource} />);
    expect(screen.getByText(/Source Document: doc123/i)).toBeInTheDocument();
  });

  test('renders node ID and type if provided', () => {
    const propsWithNode = { ...defaultProps, node_id: "node_xyz", node_type: "Person" };
    render(<QueryResultItem {...propsWithNode} />);
    expect(screen.getByText(/Node ID: node_xyz/i)).toBeInTheDocument();
    expect(screen.getByText("Person")).toBeInTheDocument(); // Badge text
  });
  
  test('renders node ID even if node_type is missing', () => {
    const propsWithNodeIdOnly = { ...defaultProps, node_id: "node_abc" };
    render(<QueryResultItem {...propsWithNodeIdOnly} />);
    expect(screen.getByText(/Node ID: node_abc/i)).toBeInTheDocument();
    // Ensure no error if node_type is undefined
  });


  test('renders relevance score if provided', () => {
    const propsWithScore = { ...defaultProps, score: 0.98765 };
    render(<QueryResultItem {...propsWithScore} />);
    expect(screen.getByText(/Relevance Score: 0.9877/i)).toBeInTheDocument(); // toFixed(4)
  });
  
  test('does not render relevance score if score is undefined', () => {
    render(<QueryResultItem {...defaultProps} score={undefined} />);
    expect(screen.queryByText(/Relevance Score:/i)).not.toBeInTheDocument();
  });


  test('renders graph context if provided', () => {
    const graphContext = { name: "Test Node", property: "Value" };
    const propsWithContext = { ...defaultProps, graph_context: graphContext };
    render(<QueryResultItem {...propsWithContext} />);
    
    expect(screen.getByText(/Graph Context:/i)).toBeInTheDocument();
    // Check for stringified JSON content (simplified check)
    expect(screen.getByText(/"name": "Test Node",/i)).toBeInTheDocument();
    expect(screen.getByText(/"property": "Value"/i)).toBeInTheDocument();
  });

  test('does not render source document ID, node ID, score, or graph context if not provided', () => {
    render(<QueryResultItem {...defaultProps} />);
    expect(screen.queryByText(/Source Document:/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Node ID:/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Relevance Score:/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Graph Context:/i)).not.toBeInTheDocument();
  });
});
