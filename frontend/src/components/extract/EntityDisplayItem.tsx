"use client";

import React from 'react';

interface EntityDisplayItemProps {
  text: string;
  label: string; // e.g., "PERSON", "ORGANIZATION", "LOCATION"
}

// Helper to get a color based on entity type for the badge
const getLabelColor = (label: string): string => {
  const upperLabel = label.toUpperCase();
  if (upperLabel.includes("PERSON")) return "bg-sky-100 text-sky-700";
  if (upperLabel.includes("ORG")) return "bg-indigo-100 text-indigo-700";
  if (upperLabel.includes("LOC") || upperLabel.includes("GPE")) return "bg-emerald-100 text-emerald-700";
  if (upperLabel.includes("DATE")) return "bg-amber-100 text-amber-700";
  if (upperLabel.includes("EVENT")) return "bg-rose-100 text-rose-700";
  if (upperLabel.includes("PRODUCT")) return "bg-purple-100 text-purple-700";
  return "bg-neutral-100 text-neutral-700"; // Default
};

export default function EntityDisplayItem({ text, label }: EntityDisplayItemProps) {
  return (
    <div className="flex items-center justify-between p-3 border border-neutral-200 rounded-md mb-3 bg-white hover:bg-neutral-50 transition-colors duration-150">
      <span className="text-neutral-800 font-medium">{text}</span>
      <span className={`px-3 py-1 text-xs font-semibold rounded-full ${getLabelColor(label)}`}>
        {label.toUpperCase()}
      </span>
    </div>
  );
}
