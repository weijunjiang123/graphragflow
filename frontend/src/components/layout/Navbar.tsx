import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="bg-neutral-100 border-b border-neutral-200 shadow-sm">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex-shrink-0">
            <Link href="/" className="text-2xl font-bold text-blue-600 hover:text-blue-700">
              GraphRAG
            </Link>
          </div>
          <div className="hidden md:flex space-x-6">
            {/* Future: Add SVG icons here */}
            <Link href="/" className="text-neutral-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">
              Home
            </Link>
            <Link href="/documents" className="text-neutral-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">
              Documents
            </Link>
            <Link href="/query" className="text-neutral-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">
              Query
            </Link>
            <Link href="/extract" className="text-neutral-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">
              Extract Entities
            </Link>
            <Link href="/style-guide" className="text-neutral-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">
              Style Guide
            </Link>
          </div>
          {/* Mobile menu button (optional, for future enhancement) */}
          <div className="md:hidden">
            {/* Placeholder for mobile menu icon */}
          </div>
        </div>
      </div>
    </nav>
  );
}
