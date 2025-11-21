# Rockfish MCP Server

A Model Context Protocol (MCP) server that provides access to the Rockfish API, enabling AI assistants to interact with Rockfish's machine learning platform.

## Features

This MCP server provides tools for the following Rockfish resources:

- **Databases**: Create, list, update, and delete databases
- **Worker Sets**: Manage worker sets for distributed processing
- **Workflows**: Create and manage ML workflows
- **Models**: Upload, list, and manage ML models
- **Projects**: Organize and manage projects
- **Datasets**: Create and manage datasets

## Installation

1. Clone the repository and set up your virtual environment (Python 3.12 or below):
```bash
git clone https://github.com/yourusername/rockfish-mcp.git
cd rockfish-mcp
python3.11 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Rockfish API key
```

## Configuration

Create a `.env` file with your Rockfish API credentials:

```env
ROCKFISH_API_KEY=your_api_key_here
ROCKFISH_BASE_URL=https://api.rockfish.ai
```

If you want to use a specific Rockfish Organization and/or Rockfish Project, 
add the following to the `.env` file too: 

```env
ROCKFISH_ORGANIZATION_ID=your_organization_id_here
ROCKFISH_PROJECT_ID=your_project_id_here
```

## Usage

Run the MCP server:

```bash
python -m rockfish_mcp.server
```

Or use the console script:

```bash
rockfish-mcp
```

## Claude Desktop Setup

To use this MCP server with Claude Desktop:

1. **Complete some of the installation steps above** (clone, install dependencies). 
Note that you do not need to start the MCP server manually for using it with Claude Desktop.
Claude Desktop will automatically start it for you when you follow the steps below.
Also, the .env file doesn't need to be created, we will be adding the environment 
variables to the Claude setup.

2. **Find your Claude Desktop configuration directory:**
   - **macOS**: `~/Library/Application Support/Claude/`
   - **Windows**: `%APPDATA%\Claude\`
   - **Linux**: `~/.config/Claude/`

3. **Create or edit the `claude_desktop_config.json` file** in that directory:

Note that setting `ROCKFISH_ORGANIZATION_ID` and `ROCKFISH_PROJECT_ID` is optional.
If you don't set these variables, the default organization and/or default project
will be used.

```json
{
  "mcpServers": {
    "rockfish": {
      "command": "/path/to/your/project/.venv/bin/python",
      "args": ["-m", "rockfish_mcp.server"]
    }
  }
}
```

**Note:** Environment variables (API keys, URLs, organization/project IDs) are configured in the `.env` file in the project root, not in `claude_desktop_config.json`.

4. **Update the paths in the configuration:**
   - Replace `/path/to/your/project/.venv/bin/python` with the actual path to your Python executable
   - All API keys and URLs should be configured in the `.env` file (see step 5)

5. **Get the correct Python path** by running this command in your project directory:
```bash
which python
```

6. **Example configuration** (replace with your actual path):
```json
{
  "mcpServers": {
    "rockfish": {
      "command": "/Users/shane/code/rockfish-mcp/.venv/bin/python",
      "args": ["-m", "rockfish_mcp.server"]
    }
  }
}
```

Configure your API key and URL in the `.env` file:
```bash
ROCKFISH_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
ROCKFISH_API_URL=https://sunset-beach.rockfish.ai
```

7. **Restart Claude Desktop** after making these changes

8. **Test the connection** by asking Claude to list your Rockfish databases or projects

## MCP Inspector Setup

The MCP Inspector is a debugging tool that helps you test your MCP server before connecting it to Claude Desktop.

### Installation

```bash
npx @modelcontextprotocol/inspector
```

### Usage

1. **Start the MCP Inspector:**
```bash
npx @modelcontextprotocol/inspector /Users/shane/code/rockfish-mcp/.venv/bin/python -m rockfish_mcp.server
```

2. **Or create a test script** for easier repeated testing:
```bash
#!/bin/bash
# test-mcp.sh
export ROCKFISH_API_KEY="your_api_key_here"
export ROCKFISH_BASE_URL="https://sunset-beach.rockfish.ai"
npx @modelcontextprotocol/inspector /Users/shane/code/rockfish-mcp/.venv/bin/python -m rockfish_mcp.server
```

Make it executable and run:
```bash
chmod +x test-mcp.sh
./test-mcp.sh
```

3. **The Inspector will open in your browser** and show:
   - Available tools (should show all 32 Rockfish tools)
   - Tool schemas and descriptions
   - Interactive tool testing interface

4. **Test your tools** by:
   - Selecting a tool from the list (e.g., `list_databases`)
   - Filling in required parameters
   - Clicking "Call Tool" to test the API call
   - Viewing the response

### Useful Tools to Test First

- **`list_databases`** - Simple GET request with no parameters
- **`list_projects`** - Another simple list operation
- **`get_database`** - Test with a database ID from the list
- **`create_database`** - Test creating a new resource

### Troubleshooting

- **MCP server not appearing**: Check that the Python path is correct and the virtual environment is activated
- **Authentication errors**: Verify your `ROCKFISH_API_KEY` is correct
- **Connection issues**: Confirm your `ROCKFISH_BASE_URL` is accessible
- **Path issues on Windows**: Use forward slashes or escaped backslashes in JSON paths

## Available Tools

### Database Tools
- `list_databases`: List all databases
- `create_database`: Create a new database
- `get_database`: Get a specific database by ID
- `update_database`: Update a database
- `delete_database`: Delete a database

### Worker Set Tools
- `list_worker_sets`: List all worker sets
- `create_worker_set`: Create a new worker set
- `get_worker_set`: Get a specific worker set by ID
- `delete_worker_set`: Delete a worker set
- `get_worker_set_actions`: List actions that the specific worker set can run
- `list_available_actions`: List all actions available to the user (across all worker sets)

### Workflow Tools
- `list_workflows`: List all workflows
- `create_workflow`: Create and run a new workflow
- `get_workflow`: Get a specific workflow by ID
- `update_workflow`: Update a workflow

### Model Tools
- `list_models`: List all models
- `upload_model`: Upload a new model
- `get_model`: Get a specific model by ID
- `delete_model`: Delete a model

### Organization Tools
- `get_active_organization`: Get the currently active organization
- `list_projects`: List all organizations

### Project Tools
- `get_active_project`: Get the currently active project
- `list_projects`: List all projects
- `create_project`: Create a new project
- `get_project`: Get a specific project by ID
- `update_project`: Update a project

### Dataset Tools
- `list_datasets`: List all datasets
- `create_dataset`: Create a new dataset
- `get_dataset`: Get a specific dataset by ID
- `update_dataset`: Update a dataset
- `delete_dataset`: Delete a dataset
- `get_dataset_schema`: Get dataset metadata present in its schema

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License