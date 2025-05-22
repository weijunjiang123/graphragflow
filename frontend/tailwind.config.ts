import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#2563EB", // blue-600
          light: "#60A5FA", // blue-400
          dark: "#1D4ED8", // blue-700
        },
        neutral: {
          50: "#F8FAFC",    // slate-50 / custom very light gray
          100: "#F1F5F9",   // slate-100 / custom light gray
          200: "#E2E8F0",   // slate-200 / custom light gray border
          300: "#CBD5E1",   // slate-300
          400: "#94A3B8",   // slate-400
          500: "#64748B",   // slate-500
          600: "#475569",   // slate-600
          700: "#334155",   // slate-700 / custom text dark
          800: "#1E293B",   // slate-800 / custom text darker
          900: "#0F172A",   // slate-900
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', '"Helvetica Neue"', 'Arial', '"Noto Sans"', 'sans-serif', '"Apple Color Emoji"', '"Segoe UI Emoji"', '"Segoe UI Symbol"', '"Noto Color Emoji"'],
      },
      borderRadius: {
        'sm': '0.25rem', // 4px
        'md': '0.5rem',  // 8px
        'lg': '0.75rem', // 12px
        'xl': '1rem',    // 16px
        'full': '9999px',
      },
      boxShadow: {
        sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        DEFAULT: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)',
        md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)',
        lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)',
      }
    },
  },
  plugins: [],
};
export default config;
