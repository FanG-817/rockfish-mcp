# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install from requirements.txt (for development)
# Note: The requirements.txt uses --find-links to access the Rockfish package repository
pip install -r requirements.txt
```

**Important**: The Rockfish SDK package is hosted on a custom package repository at `https://packages.rockfish.ai`. The [requirements.txt](requirements.txt) file uses `--find-links` directive to enable pip to discover and install packages from this repository.

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
- `ROCKFISH_ORGANIZATION_ID`: Organization ID (optional - uses default if not set)
- `ROCKFISH_PROJECT_ID`: Project ID (optional - uses default if not set)
- `MANTA_API_URL`: API URL for Manta service (optional - Manta tools only appear if this is set)

Create a `.env` file with these variables for local development:
```bash
# Copy the example and edit with your values
cp .env.example .env
```

### Code Formatting
This project uses black and isort for code formatting:
```bash
# Format code before committing
isort src/rockfish_mcp/
black src/rockfish_mcp/

# Check formatting without modifying files
isort --check-only src/rockfish_mcp/
black --check src/rockfish_mcp/
```

### Testing with MCP Inspector
Use the MCP Inspector to test the server before connecting to Claude Desktop:
```bash
# Start the inspector (replace with your actual Python path)
npx @modelcontextprotocol/inspector /path/to/.venv/bin/python -m rockfish_mcp.server
```
The Inspector provides an interactive web interface to test all available tools.

## Architecture Overview

This is an MCP (Model Context Protocol) server that provides AI assistants access to the Rockfish machine learning platform API, the Manta dataset testing service, and the Rockfish SDK for synthetic data generation. The architecture consists of four main components in a simple, focused structure.

### Project Structure
```
src/rockfish_mcp/
├── __init__.py
├── server.py       # MCP server with tool definitions and routing
├── client.py       # HTTP client for Rockfish API calls
├── manta_client.py # HTTP client for Manta service calls
└── sdk_client.py   # SDK client for Rockfish python SDK calls
```

### Core Components

**Server (`server.py`)**: The main MCP server that:
- Defines tools across multiple resource categories
  - Rockfish API: Databases, Worker Sets, Workflows, Models, Projects, Datasets (21 tools, always available)
  - Manta Service: Prompt Management, Data Manipulation, LLM Processing (10 tools, conditional)
  - SDK Tools: Synthetic Data Generation workflow tools (9 tools, always available)
- Conditionally loads Manta tools only when `MANTA_API_URL` environment variable is set
- Handles tool registration via `@server.list_tools()` decorator
- Routes tool calls through `@server.call_tool()` decorator:
  - SDK tools (in `sdk_tools` list) route to `sdk_client`
  - Tools prefixed with `manta_` route to `manta_client`
  - All other tools route to `rockfish_client`
- Manages server initialization and stdio communication with MCP protocol
- Uses global `rockfish_client` (always), `sdk_client` (always), and `manta_client` (conditional) instances initialized in `main()`
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

**SDK Client (`sdk_client.py`)**: Python SDK wrapper for Rockfish workflows that:
- Uses native Rockfish Python SDK (`rockfish` package) instead of HTTP calls
- Provides direct access to Rockfish SDK connection via `rf.Connection.remote()`
- Manages synthetic data generation workflow from end-to-end:
  - Training configuration generation with automatic column type detection
  - Rockfish TabGAN model training workflow execution
  - Model extraction and synthetic data generation
  - Distribution plotting and quality metrics
- Maintains in-memory cache for training configurations (using UUIDs)
- Implements streaming workflow logs with configurable log levels and timeouts
- Returns structured responses with `success` flags and detailed error messages
- Uses PyArrow for efficient dataset manipulation

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

#### SDK Tools (Rockfish Python SDK)
The SDK client provides end-to-end synthetic data generation workflow tools using the native Rockfish Python SDK:
- **Configuration Management**:
  - `obtain_train_config`: Generate training config with automatic column type detection (categorical/continuous/high-cardinality)
  - `update_train_config` [experimental]: Modify hyperparameters (epochs, batch_size, learning rates) or field classifications
- **Workflow Execution**:
  - `start_training_workflow`: Start TabGAN training workflow using cached config
  - `get_workflow_logs`: Stream logs with configurable level (DEBUG/INFO/WARN/ERROR) and timeout
  - `get_trained_model_id`: Extract model ID from completed training workflow
- **Generation**:
  - `start_generation_workflow`: Start generation workflow from trained model
  - `obtain_synthetic_dataset_id`: Extract generated dataset ID from completed workflow
- **Quality Assessment**:
  - `plot_distribution`: Generate distribution plots (bar for categorical, KDE for numerical) comparing datasets
  - `get_marginal_distribution_score`: Calculate similarity score between real and synthetic data distributions

### Key Implementation Details

- All API calls are asynchronous:
  - HTTP clients (`rockfish_client`, `manta_client`) use `httpx.AsyncClient`
  - SDK client uses native async Rockfish SDK via `rf.Connection.remote()`
- All clients use a centralized `call_endpoint()` method with if/elif routing for tool dispatch
- Server initialization:
  - Always creates global `RockfishClient` and `RockfishSDKClient` instances
  - Only creates `MantaClient` instance if `MANTA_API_URL` environment variable is set
  - Manta tools are dynamically added to the tool list only when `manta_client` is initialized
- Tool routing is handled by checking:
  - SDK tools (in `sdk_tools` list) route to `sdk_client`
  - Tools prefixed with `manta_` route to `manta_client`
  - All other tools route to `rockfish_client`
- Tool schemas are defined inline using JSON Schema format directly in the server
- Error handling returns `types.TextContent` objects (or `ImageContent` for plots) for display to users
- Each tool specifies required fields and optional parameters in its input schema
- HTTP clients extract IDs and parameters from arguments and construct appropriate URL paths
- SDK client maintains in-memory cache for training configurations and returns structured error responses
- Manta tools require `organization_id` and `project_id` in every request (passed as headers)

All three clients abstract their respective complexities (REST API, Manta service, Rockfish SDK), while the server provides a unified MCP interface that AI assistants can use to interact with Rockfish resources, Manta testing capabilities, and synthetic data generation workflows programmatically.

## API Reference

For complete API documentation, see:
- **Rockfish API**: https://docs.rockfish.ai/openapi.yaml
- **Manta Service**: https://manta.sunset-beach.rockfish.ai/openapi.json