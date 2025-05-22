import type { Config } from 'jest';
import nextJest from 'next/jest.js';

const createJestConfig = nextJest({
  // Provide the path to your Next.js app to load next.config.js and .env files in your test environment
  dir: './',
});

// Add any custom config to be passed to Jest
const config: Config = {
  coverageProvider: 'v8',
  testEnvironment: 'jest-environment-jsdom',
  // Add more setup options before each test is run
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  moduleNameMapper: {
    // Handle CSS imports (with CSS modules)
    // https://jestjs.io/docs/webpack#mocking-css-modules
    '^.+\\.module\\.(css|sass|scss)$': 'identity-obj-proxy',

    // Handle CSS imports (without CSS modules)
    '^.+\\.(css|sass|scss)$': '<rootDir>/__mocks__/styleMock.js',

    // Handle image imports
    // https://jestjs.io/docs/webpack#handling-static-assets
    '^.+\\.(png|jpg|jpeg|gif|webp|avif|ico|bmp|svg)$/i': `<rootDir>/__mocks__/fileMock.js`,

    // Handle module aliases (this will be automatically configured by nextJest)
    '^@/components/(.*)$': '<rootDir>/src/components/$1',
    '^@/services/(.*)$': '<rootDir>/src/services/$1',
    '^@/app/(.*)$': '<rootDir>/src/app/$1',
  },
  // Indicates whether the coverage information should be collected while executing the test
  collectCoverage: true,
  // An array of glob patterns indicating a set of files for which coverage information should be collected
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts', // Exclude type definition files
    '!src/**/_app.tsx', // Exclude Next.js special files if not testing them directly
    '!src/**/_document.tsx',
    '!src/services/api.ts', // Exclude api.ts if it's only making fetch calls (integration test instead)
    // Add other exclusions as necessary
  ],
  // The directory where Jest should output its coverage files
  coverageDirectory: 'coverage',
  // A list of paths to modules that run some code to configure or set up the testing framework before each test
  // setupFilesAfterEnv: ['<rootDir>/jest.setup.js'], // Already defined above

  // The glob patterns Jest uses to detect test files
  testMatch: [
    '**/__tests__/**/*.[jt]s?(x)',
    '**/?(*.)+(spec|test).[jt]s?(x)',
  ],

  // A map from regular expressions to paths to transformers
  transform: {
    // Use babel-jest to transpile tests with the next/babel preset
    // https://jestjs.io/docs/configuration#transform-objectstring-pathtotransformer--pathtotransformer-object
    '^.+\\.(js|jsx|ts|tsx)$': ['babel-jest', { presets: ['next/babel'] }],
  },

  // An array of regexp pattern strings that are matched against all source file paths, matched files will skip transformation
  transformIgnorePatterns: [
    '/node_modules/',
    '^.+\\.module\\.(css|sass|scss)$',
  ],
  watchPlugins: [
    "jest-watch-typeahead/filename",
    "jest-watch-typeahead/testname"
  ]
};

// createJestConfig is exported this way to ensure that next/jest can load the Next.js config which is async
export default createJestConfig(config);
