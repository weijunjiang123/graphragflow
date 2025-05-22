export default function StyleGuidePage() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="mb-10 border-b pb-4">Style Guide</h1>

      {/* Typography */}
      <section className="mb-12">
        <h2 className="mb-6">Typography</h2>
        <div className="space-y-4">
          <h1>Heading 1: The quick brown fox jumps over the lazy dog</h1>
          <h2>Heading 2: The quick brown fox jumps over the lazy dog</h2>
          <h3>Heading 3: The quick brown fox jumps over the lazy dog</h3>
          <h4>Heading 4 (default): The quick brown fox jumps over the lazy dog</h4>
          <p>
            Paragraph: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
            Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
            Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
          </p>
          <p>
            This is a <a href="#">sample link</a> to demonstrate link styling.
          </p>
        </div>
      </section>

      {/* Buttons */}
      <section className="mb-12">
        <h2 className="mb-6">Buttons</h2>
        <div className="flex flex-wrap gap-4 items-center">
          <button className="btn-primary">Primary Button</button>
          <button className="btn-secondary">Secondary Button</button>
          <button className="btn-outline">Outline Button</button>
        </div>
        <div className="mt-4 flex flex-wrap gap-4 items-center">
          <button className="btn-primary" disabled>Primary Disabled</button>
          <button className="btn-secondary" disabled>Secondary Disabled</button>
          <button className="btn-outline" disabled>Outline Disabled</button>
        </div>
      </section>

      {/* Input Fields */}
      <section className="mb-12">
        <h2 className="mb-6">Input Fields</h2>
        <div className="space-y-6 max-w-md">
          <div>
            <label htmlFor="textInput" className="block text-sm font-medium text-neutral-700 mb-1">Text Input</label>
            <input type="text" id="textInput" placeholder="Enter some text" />
          </div>
          <div>
            <label htmlFor="emailInput" className="block text-sm font-medium text-neutral-700 mb-1">Email Input (with error)</label>
            <input type="email" id="emailInput" defaultValue="invalid-email" className="border-red-500 focus:ring-red-500 focus:border-red-500" />
            <p className="mt-1 text-sm text-red-600">This email address is invalid.</p>
          </div>
          <div>
            <label htmlFor="searchInput" className="block text-sm font-medium text-neutral-700 mb-1">Search Input</label>
            <input type="search" id="searchInput" placeholder="Search..." />
          </div>
          <div>
            <label htmlFor="numberInput" className="block text-sm font-medium text-neutral-700 mb-1">Number Input</label>
            <input type="number" id="numberInput" placeholder="0" />
          </div>
          <div>
            <label htmlFor="selectInput" className="block text-sm font-medium text-neutral-700 mb-1">Select Input</label>
            <select id="selectInput">
              <option>Option 1</option>
              <option>Option 2</option>
              <option>Option 3</option>
            </select>
          </div>
          <div>
            <label htmlFor="textareaInput" className="block text-sm font-medium text-neutral-700 mb-1">Textarea</label>
            <textarea id="textareaInput" placeholder="Enter a longer message"></textarea>
          </div>
        </div>
      </section>
      
      {/* Cards */}
      <section className="mb-12">
        <h2 className="mb-6">Cards</h2>
        <div className="grid md:grid-cols-2 gap-6">
            <div className="card">
                <h3 className="!text-neutral-800">Standard Card Title</h3>
                <p>This is some content within a standard card. Cards can be used to group related information or actions.</p>
                <button className="btn-primary mt-4">Action</button>
            </div>
            <div className="card">
                <h3 className="!text-neutral-800">Another Card</h3>
                <p>This card demonstrates the consistent styling applied through the .card class.</p>
                <button className="btn-outline mt-4">Learn More</button>
            </div>
        </div>
      </section>

      {/* Color Palette */}
      <section className="mb-12">
        <h2 className="mb-6">Color Palette</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <h3 className="text-lg font-semibold mb-2 !text-neutral-800">Primary</h3>
            <div className="space-y-2">
              <div className="p-4 bg-primary-light rounded-md text-white text-sm">Light (primary-light)</div>
              <div className="p-4 bg-primary-DEFAULT rounded-md text-white text-sm">Default (primary-DEFAULT)</div>
              <div className="p-4 bg-primary-dark rounded-md text-white text-sm">Dark (primary-dark)</div>
            </div>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-2 !text-neutral-800">Neutrals (Selection)</h3>
            <div className="space-y-2">
              <div className="p-4 bg-neutral-50 rounded-md border border-neutral-200 text-sm">50 (bg-neutral-50)</div>
              <div className="p-4 bg-neutral-100 rounded-md border border-neutral-200 text-sm">100 (bg-neutral-100)</div>
              <div className="p-4 bg-neutral-200 rounded-md text-sm">200 (bg-neutral-200)</div>
              <div className="p-4 bg-neutral-700 rounded-md text-white text-sm">700 (text-neutral-700)</div>
              <div className="p-4 bg-neutral-800 rounded-md text-white text-sm">800 (text-neutral-800)</div>
              <div className="p-4 bg-neutral-900 rounded-md text-white text-sm">900 (text-neutral-900)</div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
