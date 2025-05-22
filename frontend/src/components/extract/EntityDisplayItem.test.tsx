import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import EntityDisplayItem from './EntityDisplayItem'; // Adjust path as necessary

describe('EntityDisplayItem', () => {
  test('renders entity text and label', () => {
    const props = { text: "Apple Inc.", label: "ORGANIZATION" };
    render(<EntityDisplayItem {...props} />);
    
    expect(screen.getByText("Apple Inc.")).toBeInTheDocument();
    expect(screen.getByText("ORGANIZATION")).toBeInTheDocument(); // Label is uppercased by component
  });

  test('applies correct CSS classes for label styling based on label type', () => {
    const cases = [
      { label: "PERSON", expectedClassPart: "bg-sky-100" },
      { label: "ORG", expectedClassPart: "bg-indigo-100" },
      { label: "LOCATION", expectedClassPart: "bg-emerald-100" },
      { label: "GPE", expectedClassPart: "bg-emerald-100" }, // GPE also uses emerald
      { label: "DATE", expectedClassPart: "bg-amber-100" },
      { label: "EVENT", expectedClassPart: "bg-rose-100" },
      { label: "PRODUCT", expectedClassPart: "bg-purple-100" },
      { label: "UNKNOWN_TYPE", expectedClassPart: "bg-neutral-100" },
    ];

    cases.forEach(({ label, expectedClassPart }) => {
      render(<EntityDisplayItem text="Test Entity" label={label} />);
      const labelElement = screen.getByText(label.toUpperCase()); // Component uppercases label for display
      expect(labelElement).toHaveClass(expectedClassPart);
      // screen.debug(labelElement); // Optional: for debugging classes
      // Unmount or cleanup might be needed if class list pollution becomes an issue across tests,
      // but React Testing Library usually handles cleanup well.
    });
  });

  test('handles mixed case labels correctly for styling and display', () => {
    const props = { text: "Microsoft", label: "OrGaNiZaTiOn" }; // Mixed case
    render(<EntityDisplayItem {...props} />);
    
    expect(screen.getByText("Microsoft")).toBeInTheDocument();
    const labelElement = screen.getByText("ORGANIZATION"); // Display is uppercased
    expect(labelElement).toBeInTheDocument();
    expect(labelElement).toHaveClass("bg-indigo-100"); // Styling should still match "ORG"
  });
});
