# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install from requirements.txt (for development)
pip install -r requirements.txt
```

### Running the Server
```bash
# Run the MCP server
python -m rockfish_mcp.server

# Or use the console script
rockfish-mcp
```

### Environment Setup
The application requires these environment variables:
- `ROCKFISH_API_KEY`: Your Rockfish API key (required)
- `ROCKFISH_API_URL`: API URL for Rockfish API (defaults to https://api.rockfish.ai)
- `MANTA_API_URL`: API URL for Manta service (optional - Manta tools only appear if this is set)

Create a `.env` file with these variables for local development:
```bash
# Copy the example and edit with your values
cp .env.example .env
```

## Architecture Overview

This is an MCP (Model Context Protocol) server that provides AI assistants access to the Rockfish machine learning platform API and the Manta dataset testing service. The architecture consists of three main components in a simple, focused structure.

### Project Structure
```
src/rockfish_mcp/
├── __init__.py
├── server.py       # MCP server with tool definitions and routing
├── client.py       # HTTP client for Rockfish API calls
└── manta_client.py # HTTP client for Manta service calls
```

### Core Components

**Server (`server.py`)**: The main MCP server that:
- Defines tools across multiple resource categories
  - Rockfish API: Databases, Worker Sets, Workflows, Models, Projects, Datasets (22 tools, always available)
  - Manta Service: Prompt Management, Data Manipulation, LLM Processing (10 tools, conditional)
- Conditionally loads Manta tools only when `MANTA_API_URL` environment variable is set
- Handles tool registration via `@server.list_tools()` decorator
- Routes tool calls through `@server.call_tool()` decorator:
  - Tools prefixed with `manta_` route to `manta_client`
  - All other tools route to `rockfish_client`
- Manages server initialization and stdio communication with MCP protocol
- Uses global `rockfish_client` (always) and `manta_client` (conditional) instances initialized in `main()`
- Requires `ROCKFISH_API_KEY` environment variable to function

**Client (`client.py`)**: HTTP client wrapper for Rockfish API that:
- Handles Bearer token authentication for all API requests
- Provides async HTTP requests to Rockfish API endpoints via httpx
- Maps MCP tool names to specific HTTP endpoints and methods in `call_endpoint()`
- Uses different HTTP methods (GET, POST, PUT, PATCH, DELETE) based on operation
- Centralizes error handling with `raise_for_status()` and returns formatted responses

**Manta Client (`manta_client.py`)**: HTTP client wrapper for Manta service that:
- Handles Bearer token authentication (uses same `ROCKFISH_API_KEY`)
- Provides async HTTP requests to Manta service endpoints via httpx
- Manages required Manta headers (`X-Organization-ID`, `X-Project-ID`)
- Maps Manta tool names to specific endpoints for:
  - Prompt management (create, get, append, evaluate)
  - Incident injection (spike, magnitude change, outage, ramp)
  - LLM question processing
- Centralizes error handling and returns formatted responses

### Tool Categories and API Mapping

#### Rockfish API Endpoints
The server exposes CRUD operations mapping to these endpoints:
- **Databases**: `/database` endpoints (GET, POST, PUT, DELETE)
- **Worker Sets**: `/worker-set` endpoints (GET, POST, DELETE - no update)
- **Workflows**: `/workflow` endpoints (GET, POST, PUT)
- **Models**: `/models` endpoints (GET, POST, DELETE - note different path)
- **Projects**: `/project` endpoints (GET, POST, PATCH)
- **Datasets**: `/dataset` endpoints (GET, POST, PATCH, DELETE)

#### Manta Service Endpoints (Optional)
The Manta service provides dataset testing and pattern injection capabilities. These tools are only available when `MANTA_API_URL` is configured:
- **Prompt Management**: `/prompts` endpoints (GET, POST, PATCH)
  - Generate and manage test prompts for datasets
  - Evaluate test case results
- **Incident Injection**: Data manipulation endpoints
  - `/instantaneous-spike-data`: Inject sudden spikes
  - `/sustained-magnitude-change-data`: Apply sustained changes
  - `/data-outage-data`: Create data gaps
  - `/value-ramp-data`: Apply gradual changes
  - `/incident-dataset-ids`: List all incident datasets
- **LLM Processing**: `/customer-llm` endpoint
  - Process natural language questions via SQL Agent

### Key Implementation Details

- All API calls are asynchronous using `httpx.AsyncClient` with proper connection handling
- Both clients use a centralized `call_endpoint()` method with if/elif routing for tool dispatch
- Server initialization:
  - Always creates a global `RockfishClient` instance
  - Only creates `MantaClient` instance if `MANTA_API_URL` environment variable is set
  - Manta tools are dynamically added to the tool list only when `manta_client` is initialized
- Tool routing is handled by checking tool name prefix (`manta_` routes to Manta, others to Rockfish)
- Tool schemas are defined inline using JSON Schema format directly in the server
- Error handling returns `types.TextContent` objects for display to users
- Each tool specifies required fields and optional parameters in its input schema
- The clients extract IDs and parameters from arguments and construct appropriate URL paths
- Manta tools require `organization_id` and `project_id` in every request (passed as headers)

Both clients abstract REST API complexity, while the server provides a unified MCP interface that AI assistants can use to interact with Rockfish resources and Manta testing capabilities programmatically.

## API Reference

For complete API documentation, see:
- **Rockfish API**: https://docs.rockfish.ai/openapi.yaml
- **Manta Service**: https://manta.sunset-beach.rockfish.ai/openapi.json