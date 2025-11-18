"""
Recommender service client for Rockfish MCP.

Handles API calls to the Rockfish Recommender service for workflow generation,
dataset property detection, and fidelity checking.
"""

import os
import httpx
from typing import Any


class RecommenderClient:
    """Client for interacting with the Rockfish Recommender API."""

    def __init__(self, api_key: str, project_id: str, organization_id: str, api_url: str):
        """
        Initialize the Recommender client.

        Args:
            api_key: Rockfish API key for authentication
            project_id: Project ID for API requests
            organization_id: Organization ID for API requests
            api_url: Base URL for the Recommender API
        """
        self.api_key = api_key
        self.project_id = project_id
        self.organization_id = organization_id
        self.api_url = api_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Project-ID": project_id,
            "X-Organization-ID": organization_id,
            "Content-Type": "application/json"
        }

    async def call_endpoint(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call a Recommender API endpoint based on the tool name.

        Args:
            tool_name: Name of the MCP tool being called
            arguments: Tool arguments containing request parameters

        Returns:
            API response data

        Raises:
            httpx.HTTPStatusError: If the API request fails
        """
        async with httpx.AsyncClient() as client:
            # Route to appropriate endpoint based on tool name
            if tool_name == "recommender_generate_workflow":
                response = await client.post(
                    f"{self.api_url}/recommendation/generate-workflow",
                    headers=self.headers,
                    json=arguments
                )

            elif tool_name == "recommender_evaluate_workflow":
                response = await client.post(
                    f"{self.api_url}/recommendation/evaluate-workflow",
                    headers=self.headers,
                    json=arguments
                )

            elif tool_name == "recommender_train_workflow":
                response = await client.post(
                    f"{self.api_url}/recommendation/train-workflow",
                    headers=self.headers,
                    json=arguments
                )

            elif tool_name == "recommender_concat_workflow":
                response = await client.post(
                    f"{self.api_url}/recommendation/concat-workflow",
                    headers=self.headers,
                    json=arguments
                )

            elif tool_name == "recommender_tabular_properties":
                response = await client.post(
                    f"{self.api_url}/recommendation/tabular-properties",
                    headers=self.headers,
                    json=arguments
                )

            elif tool_name == "recommender_timeseries_properties":
                response = await client.post(
                    f"{self.api_url}/recommendation/timeseries-properties",
                    headers=self.headers,
                    json=arguments
                )

            elif tool_name == "recommender_dataset_fidelity_score":
                response = await client.post(
                    f"{self.api_url}/recommendation/dataset-fidelity-score",
                    headers=self.headers,
                    json=arguments
                )

            elif tool_name == "recommender_sql_fidelity_checks":
                response = await client.post(
                    f"{self.api_url}/recommendation/sql-fidelity-checks",
                    headers=self.headers,
                    json=arguments
                )

            elif tool_name == "recommender_generate_sources":
                response = await client.post(
                    f"{self.api_url}/recommendation/generate-sources",
                    headers=self.headers,
                    json=arguments
                )

            else:
                raise ValueError(f"Unknown Recommender tool: {tool_name}")

            response.raise_for_status()

            # Debug: Print response details
            # print(f"DEBUG: Status Code: {response.status_code}")
            # print(f"DEBUG: Content-Type: {response.headers.get('content-type')}")
            # print(f"DEBUG: Response Content (first 500 chars): {response.text[:500]}")

            return response.json()
