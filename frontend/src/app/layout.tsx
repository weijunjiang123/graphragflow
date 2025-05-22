import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Navbar from "@/components/layout/Navbar"; // Updated import path
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "GraphRAG Frontend",
  description: "Frontend for interacting with the GraphRAG API",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-neutral-50 text-neutral-800 antialiased`}>
        <Navbar />
        <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <footer className="bg-neutral-100 border-t border-neutral-200 text-center p-6 mt-12">
          <p className="text-sm text-neutral-600">&copy; {new Date().getFullYear()} GraphRAG Project. All rights reserved.</p>
        </footer>
      </body>
    </html>
  );
}
