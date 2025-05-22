# STAGE 1: Build the Next.js application
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
COPY frontend/package.json frontend/package-lock.json* ./
# If you use yarn, it would be:
# COPY frontend/package.json frontend/yarn.lock ./

# Install dependencies
RUN npm install
# If you use yarn:
# RUN yarn install

# Copy the rest of the frontend application code
COPY frontend/ ./

# Run the build script
RUN npm run build
# If you use yarn:
# RUN yarn build

# STAGE 2: Setup the production environment
FROM node:20-alpine AS runner

WORKDIR /app

# Set environment variables for Next.js
ENV NODE_ENV=production
# Optionally, if you want to disable telemetry
# ENV NEXT_TELEMETRY_DISABLED 1

# Copy the standalone output from the builder stage
COPY --from=builder /app/.next/standalone ./

# Copy the public folder from the builder stage
COPY --from=builder /app/public ./public

# Copy the static assets from .next/static from the builder stage
COPY --from=builder /app/.next/static ./.next/static

# Next.js recommends a non-root user for security reasons
# Create a non-root user and group
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Set the user for running the application
USER nextjs

# Expose port 3000
EXPOSE 3000

# Command to run the Next.js server
# The server.js file is part of the standalone output
CMD ["node", "server.js"]
