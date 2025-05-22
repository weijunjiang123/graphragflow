import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="container mx-auto p-4">
      <header className="text-center py-12">
        <h1 className="text-4xl font-bold mb-4">Welcome to GraphRAG</h1>
        <p className="text-lg text-gray-700 mb-8">
          This application provides a comprehensive suite to build, query, and analyze knowledge graphs
          from your documents. Leverage the power of Large Language Models and Graph Databases to
          unlock insights from your textual data.
        </p>
      </header>

      <section className="grid md:grid-cols-3 gap-8 text-center">
        <div className="p-6 border rounded-lg shadow-lg hover:shadow-xl transition-shadow">
          <h2 className="text-2xl font-semibold mb-3">Document Processing</h2>
          <p className="text-gray-600 mb-4">
            Upload your documents (.txt, .pdf, .docx) and transform them into structured knowledge graphs.
            Our pipeline extracts information and relationships, storing them efficiently in Neo4j.
          </p>
          <Link href="/documents" className="text-blue-600 hover:text-blue-800 font-medium">
            Go to Document Processing &rarr;
          </Link>
        </div>

        <div className="p-6 border rounded-lg shadow-lg hover:shadow-xl transition-shadow">
          <h2 className="text-2xl font-semibold mb-3">Query Interface</h2>
          <p className="text-gray-600 mb-4">
            Explore your knowledge graph with powerful querying capabilities. Use vector search for semantic
            similarity, full-text search for keywords, or direct graph traversals for complex patterns.
          </p>
          <Link href="/query" className="text-blue-600 hover:text-blue-800 font-medium">
            Go to Query Interface &rarr;
          </Link>
        </div>

        <div className="p-6 border rounded-lg shadow-lg hover:shadow-xl transition-shadow">
          <h2 className="text-2xl font-semibold mb-3">Entity Extraction</h2>
          <p className="text-gray-600 mb-4">
            Extract key entities (people, organizations, locations, etc.) from any text.
            Understand the main actors and concepts discussed in your content.
          </p>
          <Link href="/extract" className="text-blue-600 hover:text-blue-800 font-medium">
            Go to Entity Extraction &rarr;
          </Link>
        </div>
      </section>

      <section className="text-center py-12">
        <h2 className="text-2xl font-semibold mb-3">Get Started</h2>
        <p className="text-gray-600">
          Choose one of the sections above to begin your journey with GraphRAG.
        </p>
      </section>
    </div>
  );
}
