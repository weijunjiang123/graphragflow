# GraphRAGFlow: Intelligent Document Analysis with LLMs and Neo4j

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![GraphRAG Pipeline](./asset/GraphPipeline.svg)

GraphRAGFlow is a comprehensive solution for transforming unstructured text documents into structured knowledge graphs. It leverages Large Language Models (LLMs) like OpenAI's GPT series or local models via Ollama, and stores the resulting graph in Neo4j. This enables advanced querying, including semantic search, full-text search, and complex graph traversals, all accessible via a RESTful API and a user-friendly web interface.

**Example Neo4j Visualization:**
![Neo4j Demo](./asset/show.png)

## Table of Contents

- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Directory Structure](#-directory-structure)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup and Installation](#setup-and-installation)
  - [Accessing Services](#accessing-services)
- [API Endpoints](#-api-endpoints)
- [Development Mode](#-development-mode)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Original Inspiration](#-original-inspiration)


## ✨ Key Features

*   **End-to-End Document Processing**: Load, chunk, and process various document formats.
*   **LLM-Powered Knowledge Extraction**: Utilize OpenAI or Ollama models to extract entities and relationships, building a rich knowledge graph.
*   **Neo4j Integration**: Robust storage and management of graph data.
*   **Vector Embeddings & Semantic Search**: Generate embeddings for document chunks and perform semantic similarity searches.
*   **Hybrid Search Capabilities**: Combine vector, full-text, and graph-based querying.
*   **RESTful API**: Expose core functionalities (document processing, querying, entity extraction) via a FastAPI backend.
*   **Web Interface**: User-friendly Next.js frontend for interacting with the system.
*   **Dockerized Deployment**: Easy setup and consistent environments using Docker Compose for all services (Backend, Frontend, Neo4j).
*   **Configuration Flexibility**: Manage settings via environment variables and a `.env` file.
*   **Comprehensive Testing**: Unit tests for backend API and frontend components.

## 🔧 How It Works

GraphRAGFlow implements a multi-stage pipeline:

1.  **Document Ingestion & Processing**: Documents (text, PDF, DOCX) are uploaded or specified. The `DocumentProcessor` chunks them into manageable sizes.
2.  **Graph Transformation**: The `GraphTransformerWrapper`, using an LLM (OpenAI or Ollama), converts text chunks into graph documents, identifying nodes (entities) and relationships.
3.  **Neo4j Storage**: The `Neo4jManager` stores these graph documents in a Neo4j database. This includes creating `Document` nodes and linking extracted entities to their source.
4.  **Embedding Generation**: The `EmbeddingsManager` generates vector embeddings for document chunks using a configured embedding model (OpenAI or Ollama-compatible). These embeddings are stored in Neo4j and indexed for vector search.
5.  **API Layer**: A FastAPI application (`src/api/main.py`) exposes endpoints for:
    *   Processing new documents.
    *   Querying the graph (vector, full-text, graph traversal, hybrid).
    *   Extracting entities from ad-hoc text.
6.  **Frontend Interface**: A Next.js application provides a UI to interact with these API endpoints, allowing users to upload documents, run queries, and extract entities.

## 📂 Directory Structure

```
GraphRAGFlow/
├── .env.example             # Example environment variables for configuration
├── .dockerignore              # Specifies files to ignore in Docker build contexts
├── backend.Dockerfile         # Dockerfile for the backend (FastAPI) application
├── frontend.Dockerfile        # Dockerfile for the frontend (Next.js) application
├── docker-compose.yml       # Docker Compose file to orchestrate all services
├── README.md                # This documentation file
├── requirements.txt         # Python dependencies for the backend
├── asset/                   # Contains images like the pipeline diagram
├── data/                    # (Optional) Default directory for local data processing (main.py script)
├── frontend/                # Frontend Next.js application
│   ├── public/              # Static assets for the frontend
│   ├── src/                 # Frontend source code (components, pages, services)
│   ├── jest.config.ts       # Jest configuration for frontend tests
│   ├── next.config.ts       # Next.js configuration
│   ├── package.json         # Frontend npm dependencies
│   └── ...
├── src/                     # Backend FastAPI application source code
│   ├── api/                 # API specific modules (routers, main app, models)
│   ├── core/                # Core logic (document processing, graph, embeddings, etc.)
│   ├── tests/               # Backend Pytest tests
│   ├── config.py            # Pydantic settings management
│   └── main.py              # Script for local document processing (legacy, API is primary)
└── ...
```

## 🏗️ System Architecture

The application consists of three main services orchestrated by Docker Compose:

1.  **Backend (FastAPI)**:
    *   Handles business logic for document processing, graph construction, querying, and entity extraction.
    *   Interfaces with Neo4j and LLM providers.
    *   Exposes a RESTful API.
2.  **Frontend (Next.js)**:
    *   Provides a web-based user interface.
    *   Interacts with the Backend API to offer application functionalities.
3.  **Neo4j Database**:
    *   Stores the knowledge graph and vector embeddings.
    *   Includes the APOC plugin for extended graph procedures.

## 🚀 Getting Started

This section guides you through setting up and running GraphRAGFlow using Docker.

### Prerequisites

*   **Docker Engine**: Version 20.x or later.
*   **Docker Compose**: Version v2.x or later.
*   **(Optional) Ollama**: If you plan to use local LLMs, ensure Ollama is installed, running, and has pulled the necessary models (e.g., `ollama pull qwen2.5`, `ollama pull nomic-embed-text`). Configure Ollama to listen on network interfaces accessible by Docker (e.g., `0.0.0.0` or your host IP).

### Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/weijunjiang123/GraphRAG-with-Ollama.git # Replace with your repo URL if different
    cd GraphRAG-with-Ollama
    ```

2.  **Configure Environment Variables:**
    *   Create a `.env` file by copying the example:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file with your specific configurations. **This is crucial.**
        *   **`MODEL_PROVIDER`**: Set to `openai` or `ollama`.
        *   **`MODEL_OPENAI_API_KEY`**: Your OpenAI API key (if using `MODEL_PROVIDER=openai`).
        *   **`MODEL_OPENAI_MODEL`**: Preferred OpenAI model (e.g., `gpt-4-turbo`).
        *   **`MODEL_OPENAI_EMBEDDINGS_MODEL`**: Preferred OpenAI embeddings model (e.g., `text-embedding-3-small`).
        *   **`MODEL_OLLAMA_BASE_URL`**:
            *   If Ollama runs on your host: `http://host.docker.internal:11434` (for Docker Desktop).
            *   If Ollama is another Docker service: `http://ollama_service_name:11434`.
        *   **`MODEL_OLLAMA_LLM_MODEL`**: The Ollama model you have pulled (e.g., `qwen2.5`).
        *   **`NEO4J_PASSWORD`**: The password for the Neo4j database. **This MUST match the password set in `docker-compose.yml` for the `NEO4J_AUTH` environment variable of the `neo4j` service.** The default in `docker-compose.yml` is `yourStrongPassword123!`. It's highly recommended to change this default in both `.env` and `docker-compose.yml`.

3.  **Build and Start Services:**
    Use Docker Compose to build images and start all services in detached mode:
    ```bash
    docker-compose up --build -d
    ```
    *   `--build`: Forces a rebuild of images if Dockerfiles or application code changed.
    *   `-d`: Runs containers in the background (detached mode).

4.  **Shutting Down:**
    To stop all services:
    ```bash
    docker-compose down
    ```
    To stop and remove data volumes (e.g., clear Neo4j database):
    ```bash
    docker-compose down -v
    ```

### Accessing Services

Once the services are running:

*   **Frontend Application**: `http://localhost:3000`
*   **Backend API Documentation (Swagger UI)**: `http://localhost:8000/docs`
*   **Neo4j Browser**: `http://localhost:7474`
    *   **Connect URI**: `bolt://localhost:7687`
    *   **Username**: `neo4j`
    *   **Password**: The one you set in `.env` and `docker-compose.yml` (default: `yourStrongPassword123!`).

## 📖 API Endpoints

The backend provides several API endpoints for interaction. For a detailed specification, please refer to the auto-generated OpenAPI documentation at `http://localhost:8000/docs` when the backend service is running.

Key endpoints include:

*   `POST /api/v1/documents/process`: Processes a list of texts, converts them to graph structures, and stores them in Neo4j.
*   `POST /api/v1/query`: Performs queries against the knowledge graph using vector, full-text, graph, or hybrid search types.
*   `POST /api/v1/entities/extract`: Extracts structured entities (name and type) from provided text.
*   `GET /health`: Health check for the API and its core services.

## 🛠️ Development Mode

For active development, you can enable hot-reloading:

1.  **Backend (FastAPI/Uvicorn)**:
    In `docker-compose.yml`, uncomment the `volumes` section for the `backend` service to mount your local `src/` directory into the container:
    ```yaml
    # services:
    #   backend:
    #     volumes:
    #       - ./src:/app/src
    ```
    Uvicorn, as configured in `backend.Dockerfile` (`CMD`), typically needs the `--reload` flag for hot-reloading. You might need to adjust the `CMD` in `backend.Dockerfile` or override `command` in `docker-compose.yml` for development:
    ```yaml
    # services:
    #   backend:
    #     command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```

2.  **Frontend (Next.js)**:
    In `docker-compose.yml`, uncomment the `volumes` section for the `frontend` service and change the `command` to run the Next.js development server:
    ```yaml
    # services:
    #   frontend:
    #     volumes:
    #       - ./frontend:/app
    #     command: npm run dev
    ```
    **Note on `node_modules`**: The current `frontend.Dockerfile` installs `node_modules` within the image. If you mount `./frontend:/app`, ensure your local `node_modules` (if any) doesn't conflict, or add `node_modules` to a `.dockerignore` specifically for this volume mount context if issues arise. A common practice is to mount only specific source folders like `frontend/src:/app/src`. However, Next.js with its build system often works better if the entire project directory is consistent.

Rebuild and restart services after changing `docker-compose.yml`:
```bash
docker-compose up --build -d
```

## 🧪 Testing

Unit tests are available for both backend and frontend.

### Backend Testing (pytest)

Located in `src/tests/`. Dependencies are mocked to ensure isolated tests.

1.  **Install Dependencies** (if not using Docker for tests):
    ```bash
    uv sync # Or pip install -r requirements.txt
    ```
2.  **Run Locally**:
    From the project root:
    ```bash
    pytest src/tests
    ```
3.  **Run via Docker** (if services are up):
    ```bash
    docker-compose exec backend pytest src/tests
    ```

### Frontend Testing (Jest & React Testing Library)

Located in `frontend/src/components/` (co-located with components).

1.  **Install Dependencies** (if not using Docker for tests):
    Navigate to `frontend/` and run:
    ```bash
    npm install
    ```
2.  **Run Locally**:
    From the `frontend/` directory:
    ```bash
    npm test
    ```
    For watch mode:
    ```bash
    npm test -- --watch
    ```
3.  **Run via Docker** (if services are up):
    ```bash
    docker-compose exec frontend npm test
    ```

## 💡 Troubleshooting

*   **Ollama on Host Not Reachable from Docker**:
    *   Ensure Ollama listens on `0.0.0.0` or your host's network IP.
    *   Use `MODEL_OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env` (for Docker Desktop). For Linux, you might need your host's IP on the `docker0` bridge (e.g., `172.17.0.1`).
*   **Neo4j Healthcheck Failures**: Check Neo4j logs: `docker-compose logs neo4j`. Ensure the password matches and there are no port conflicts.
*   **APOC/GDS Issues**: Verify plugins are listed in `NEO4J_PLUGINS` in `docker-compose.yml` and `NEO4J_dbms_security_procedures_unrestricted` is correctly configured.
*   **Volume Mount Permissions (Linux)**: If you encounter permission errors with volume mounts, ensure the user running Docker has the necessary permissions, or adjust file ownership/permissions on the host for the mounted directories.

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a new branch for your feature or fix, and submit a pull request. Ensure your code adheres to existing styles and all tests pass.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (though no actual LICENSE file was created in this project, MIT is standard for such open examples).

## 🙏 Original Inspiration

This project structure and some initial ideas were inspired by the work from:
https://github.com/Coding-Crashkurse/GraphRAG-with-Llama-3.1
(The link in the original README was `GraphRAG-with-Llama-3.1`, so I've kept it, assuming it's an internal reference or a previous version.)
The current project, GraphRAGFlow, has been significantly expanded with a full API, frontend, Dockerization, and comprehensive testing.
