import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RockfishClient:
    """Client for interacting with the Rockfish API."""

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.rockfish.ai",
        organization_id=None,
        project_id=None,
    ):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if organization_id:
            self.headers["X-Organization-ID"] = organization_id

        if project_id:
            self.headers["X-Project-ID"] = project_id

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the Rockfish API."""
        url = f"{self.api_url}{endpoint}"

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method, url=url, headers=self.headers, **kwargs
            )
            response.raise_for_status()
            return response.json() if response.content else {}

    async def call_endpoint(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route tool calls to appropriate API endpoints."""

        # Database endpoints
        if tool_name == "list_databases":
            return await self._request("GET", "/database")

        elif tool_name == "create_database":
            return await self._request("POST", "/database", json=arguments)

        elif tool_name == "get_database":
            db_id = arguments["id"]
            return await self._request("GET", f"/database/{db_id}")

        elif tool_name == "update_database":
            db_id = arguments.pop("id")
            return await self._request("PUT", f"/database/{db_id}", json=arguments)

        elif tool_name == "delete_database":
            db_id = arguments["id"]
            return await self._request("DELETE", f"/database/{db_id}")

        # Worker Set endpoints
        elif tool_name == "list_worker_sets":
            return await self._request("GET", "/worker-set")

        elif tool_name == "create_worker_set":
            return await self._request("POST", "/worker-set", json=arguments)

        elif tool_name == "get_worker_set":
            ws_id = arguments["id"]
            return await self._request("GET", f"/worker-set/{ws_id}")

        elif tool_name == "delete_worker_set":
            ws_id = arguments["id"]
            return await self._request("DELETE", f"/worker-set/{ws_id}")

        elif tool_name == "get_worker_set_actions":
            ws_id = arguments["id"]
            return await self._request("GET", f"/worker-set/{ws_id}/actions")

        elif tool_name == "list_available_actions":
            worker_groups = await self._request("GET", "/worker-group")

            workers = []
            for group in worker_groups.get("groups", []):
                if "actions" in group:
                    current_actions = group.get("actions")
                    workers.extend(current_actions)

            return {"actions": workers}

        # Workflow endpoints
        elif tool_name == "list_workflows":
            return await self._request("GET", "/workflow")

        elif tool_name == "create_workflow":
            return await self._request("POST", "/workflow", json=arguments)

        elif tool_name == "get_workflow":
            wf_id = arguments["id"]
            return await self._request("GET", f"/workflow/{wf_id}")

        elif tool_name == "update_workflow":
            wf_id = arguments.pop("id")
            return await self._request("PUT", f"/workflow/{wf_id}", json=arguments)

        # Models endpoints
        elif tool_name == "list_models":
            return await self._request("GET", "/models")

        elif tool_name == "upload_model":
            return await self._request("POST", "/models", json=arguments)

        elif tool_name == "get_model":
            model_id = arguments["id"]
            return await self._request("GET", f"/models/{model_id}")

        elif tool_name == "delete_model":
            model_id = arguments["id"]
            return await self._request("DELETE", f"/models/{model_id}")

        # Organization endpoints
        elif tool_name == "get_active_organization":
            return await self._request("GET", "/organization/active")

        elif tool_name == "list_organizations":
            return await self._request("GET", "/organization")

        # Project endpoints
        elif tool_name == "get_active_project":
            return await self._request("GET", "/project/active")

        elif tool_name == "list_projects":
            return await self._request("GET", "/project")

        elif tool_name == "create_project":
            return await self._request("POST", "/project", json=arguments)

        elif tool_name == "get_project":
            project_id = arguments["id"]
            return await self._request("GET", f"/project/{project_id}")

        elif tool_name == "update_project":
            project_id = arguments.pop("id")
            return await self._request(
                "PATCH", f"/project/{project_id}", json=arguments
            )

        # Dataset endpoints
        elif tool_name == "list_datasets":
            return await self._request("GET", "/dataset")

        elif tool_name == "create_dataset":
            return await self._request("POST", "/dataset", json=arguments)

        elif tool_name == "get_dataset":
            dataset_id = arguments["id"]
            return await self._request("GET", f"/dataset/{dataset_id}")

        elif tool_name == "update_dataset":
            dataset_id = arguments.pop("id")
            return await self._request(
                "PATCH", f"/dataset/{dataset_id}", json=arguments
            )

        elif tool_name == "delete_dataset":
            dataset_id = arguments["id"]
            return await self._request("DELETE", f"/dataset/{dataset_id}")

        # Dataset schema endpoints
        elif tool_name == "get_dataset_schema":
            dataset_id = arguments["id"]
            return await self._request("GET", f"/dataset/{dataset_id}/schema")

        # Query endpoints
        elif tool_name == "execute_query":
            query = arguments["query"]
            project_id = arguments.get("project_id")

            # Prepare headers for query request
            query_headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "text/plain",
            }
            if project_id:
                query_headers["X-Project-ID"] = project_id

            # Make query request with text/plain content
            url = f"{self.api_url}/query"
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method="POST", url=url, headers=query_headers, content=query
                )
                response.raise_for_status()
                return {"result": response.text}

        else:
            raise ValueError(f"Unknown tool: {tool_name}")
