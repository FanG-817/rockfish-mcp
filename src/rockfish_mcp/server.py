#!/usr/bin/env python3

import asyncio
import logging
from typing import Any, Dict, List, Optional
import os
from dotenv import load_dotenv

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

from .client import RockfishClient
from .manta_client import MantaClient
from .sdk_client import RockfishSDKClient

load_dotenv()

logger = logging.getLogger("rockfish-mcp")

server = Server("rockfish-mcp")

# Global client instances
rockfish_client: Optional[RockfishClient] = None
manta_client: Optional[MantaClient] = None
sdk_client: Optional[RockfishSDKClient] = None


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available Rockfish API and Manta service tools."""
    tools = [
        # Database tools
        types.Tool(
            name="list_databases",
            description="List all databases",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_database",
            description="Create a new database",
            inputSchema={
                "type": "object", 
                "properties": {
                    "name": {"type": "string", "description": "Database name"},
                    "description": {"type": "string", "description": "Database description"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get_database",
            description="Get a specific database by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Database ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="update_database",
            description="Update a database",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Database ID"},
                    "name": {"type": "string", "description": "Database name"},
                    "description": {"type": "string", "description": "Database description"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="delete_database",
            description="Delete a database",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Database ID"}
                },
                "required": ["id"]
            }
        ),
        
        # Worker Set tools
        types.Tool(
            name="list_worker_sets",
            description="List all worker sets",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_worker_set",
            description="Create a new worker set",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Worker set name"},
                    "worker_count": {"type": "integer", "description": "Number of workers"}
                },
                "required": ["name", "worker_count"]
            }
        ),
        types.Tool(
            name="get_worker_set",
            description="Get a specific worker set by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Worker set ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="delete_worker_set",
            description="Delete a worker set",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Worker set ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="get_worker_set_actions",
            description="List the actions for a worker-set",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Worker set ID"}
                },
                "required": ["id"]
            }
        ),

        # Worker tools
        types.Tool(
            name="list_available_actions",
            description="List actions (or workers) from all worker-sets",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        # Workflow tools
        types.Tool(
            name="list_workflows",
            description="List all workflows",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_workflow",
            description="Create and run a new workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "alias": {
                                    "type": "string",
                                    "pattern": "^[\\pL\\pM\\pN -]{2,64}$",
                                    "description": "Alias for this job (used for referencing in inputs)"
                                },
                                "config": {
                                    "type": "object",
                                    "description": "Job configuration (can contain any action config)"
                                },
                                "input": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "List of input aliases this job depends on"
                                },
                                "replicas": {
                                    "type": "integer",
                                    "default": 1,
                                    "description": "Number of replicas for this job"
                                },
                                "worker_name": {
                                    "type": "string",
                                    "pattern": "^[\\pL\\pM\\pN -]{2,64}$",
                                    "description": "Name of the worker to execute this job"
                                },
                                "worker_version": {
                                    "type": "string",
                                    "description": "Version of the worker"
                                }
                            },
                            "required": ["alias", "config", "input", "worker_name", "worker_version"]
                        },
                        "minItems": 1,
                        "description": "Array of jobs that make up the workflow"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "pattern": "^[\\pL\\pM\\pN -]{2,64}$",
                                "description": "Workflow name"
                            },
                            "labels": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "string",
                                    "pattern": "^[\\pL\\pM\\pN _-]{1,64}$"
                                },
                                "description": "Key-value labels for the workflow"
                            }
                        },
                        "required": ["name", "labels"],
                        "description": "Workflow metadata"
                    }
                },
                "required": ["jobs", "metadata"],
                "description": "Workflow definition with jobs and metadata"
            }
        ),
        types.Tool(
            name="get_workflow",
            description="Get a specific workflow by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Workflow ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="update_workflow",
            description="Update a workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Workflow ID"},
                    "name": {"type": "string", "description": "Workflow name"},
                    "definition": {"type": "object", "description": "Workflow definition"}
                },
                "required": ["id"]
            }
        ),
        
        # Models tools
        types.Tool(
            name="list_models",
            description="List all models",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="upload_model",
            description="Upload a new model",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Model name"},
                    "metadata": {"type": "object", "description": "Model metadata"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get_model",
            description="Get a specific model by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Model ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="delete_model",
            description="Delete a model",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Model ID"}
                },
                "required": ["id"]
            }
        ),

        # Organization tools
        types.Tool(
            name="get_active_organization",
            description="Get active organization",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="list_organizations",
            description="List all organizations",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        
        # Project tools
        types.Tool(
            name="get_active_project",
            description="Get active project",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="list_projects",
            description="List all projects",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_project",
            description="Create a new project",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name"},
                    "description": {"type": "string", "description": "Project description"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get_project",
            description="Get a specific project by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Project ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="update_project",
            description="Update a project",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Project ID"},
                    "name": {"type": "string", "description": "Project name"},
                    "description": {"type": "string", "description": "Project description"}
                },
                "required": ["id"]
            }
        ),
        
        # Dataset tools
        types.Tool(
            name="list_datasets",
            description="List all datasets",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_dataset",
            description="Create a new dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset name"},
                    "description": {"type": "string", "description": "Dataset description"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get_dataset",
            description="Get a specific dataset by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="update_dataset",
            description="Update a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID"},
                    "name": {"type": "string", "description": "Dataset name"},
                    "description": {"type": "string", "description": "Dataset description"}
                },
                "required": ["id"]
            }
        ),
        types.Tool(
            name="delete_dataset",
            description="Delete a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID"}
                },
                "required": ["id"]
            }
        ),

        # Dataset schema tools
        types.Tool(
            name="get_dataset_schema",
            description="Get the schema for a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID"}
                },
                "required": ["id"]
            }
        ),
        
        # Query tools
        types.Tool(
            name="execute_query",
            description="Execute a query and return results in CSV format",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to execute"},
                    "project_id": {"type": "string", "description": "Optional project ID to execute the query in"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="query_dataset",
            description="Execute a query against a specific dataset and return results in CSV format",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Dataset ID to query against"},
                    "query": {"type": "string", "description": "The query to execute"},
                    "project_id": {"type": "string", "description": "Optional project ID to execute the query in"}
                },
                "required": ["id", "query"]
            }
        )
    ]

    # Add Manta tools only if Manta client is initialized
    if manta_client:
        manta_tools = [
            # Manta Service - Prompt Management tools
            types.Tool(
            name="manta_get_prompts",
            description="Get natural language prompts and expected answers for a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to get prompts for"},
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_create_prompts",
            description="Create prompts and answers for a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to create prompts for"},
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_append_prompts",
            description="Append prompts to an existing dataset prompt collection",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to append prompts to"},
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_evaluate_test_case",
            description="Compare actual vs expected results for test validation",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The prompt/question that was asked"},
                    "actual_result": {"type": "string", "description": "The actual result obtained"},
                    "expected_result": {"type": "string", "description": "The expected result"},
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["prompt", "actual_result", "expected_result", "organization_id", "project_id"]
            }
        ),

        # Manta Service - Data Manipulation tools (Incident Injection)
        types.Tool(
            name="manta_create_instantaneous_spike",
            description="Create a modified dataset with an instantaneous spike incident injected",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Original dataset ID"},
                    "incident_config": {
                        "type": "object",
                        "description": "Configuration for the spike incident",
                        "properties": {
                            "metadata_predicates": {"type": "object", "description": "Metadata filters to select time series"},
                            "measurements": {"type": "array", "items": {"type": "string"}, "description": "List of measurements to inject spike into"},
                            "incident_start_timestamp": {"type": "string", "description": "ISO timestamp when spike occurs"},
                            "magnitude": {"type": "number", "description": "Magnitude of the spike"}
                        }
                    },
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "incident_config", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_create_sustained_magnitude_change",
            description="Create a modified dataset with a sustained magnitude change incident",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Original dataset ID"},
                    "incident_config": {
                        "type": "object",
                        "description": "Configuration for the sustained magnitude change",
                        "properties": {
                            "metadata_predicates": {"type": "object", "description": "Metadata filters to select time series"},
                            "measurements": {"type": "array", "items": {"type": "string"}, "description": "List of measurements to modify"},
                            "incident_start_timestamp": {"type": "string", "description": "ISO timestamp when change starts"},
                            "incident_end_timestamp": {"type": "string", "description": "ISO timestamp when change ends"},
                            "magnitude": {"type": "number", "description": "Magnitude of the change"}
                        }
                    },
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "incident_config", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_create_data_outage",
            description="Create a modified dataset with a data outage incident (missing data)",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Original dataset ID"},
                    "incident_config": {
                        "type": "object",
                        "description": "Configuration for the data outage",
                        "properties": {
                            "metadata_predicates": {"type": "object", "description": "Metadata filters to select time series"},
                            "measurements": {"type": "array", "items": {"type": "string"}, "description": "List of measurements to create outage for"},
                            "incident_start_timestamp": {"type": "string", "description": "ISO timestamp when outage starts"},
                            "incident_end_timestamp": {"type": "string", "description": "ISO timestamp when outage ends"}
                        }
                    },
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "incident_config", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_create_value_ramp",
            description="Create a modified dataset with a value ramp incident (gradual increase/decrease)",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Original dataset ID"},
                    "incident_config": {
                        "type": "object",
                        "description": "Configuration for the value ramp",
                        "properties": {
                            "metadata_predicates": {"type": "object", "description": "Metadata filters to select time series"},
                            "measurements": {"type": "array", "items": {"type": "string"}, "description": "List of measurements to apply ramp to"},
                            "incident_start_timestamp": {"type": "string", "description": "ISO timestamp when ramp starts"},
                            "incident_end_timestamp": {"type": "string", "description": "ISO timestamp when ramp ends"},
                            "magnitude": {"type": "number", "description": "Total magnitude change over the ramp period"}
                        }
                    },
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "incident_config", "organization_id", "project_id"]
            }
        ),
        types.Tool(
            name="manta_get_incident_dataset_ids",
            description="Get all incident dataset IDs associated with an original dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Original dataset ID"},
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "organization_id", "project_id"]
            }
        ),

        # Manta Service - LLM Processing tools
        types.Tool(
            name="manta_process_llm_questions",
            description="Process natural language questions using SQL Agent functionality against a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to query"},
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of natural language questions to process"
                    },
                    "organization_id": {"type": "string", "description": "Organization ID (required by Manta API)"},
                    "project_id": {"type": "string", "description": "Project ID (required by Manta API)"}
                },
                "required": ["dataset_id", "questions", "organization_id", "project_id"]
            }
        )
        ]
        tools.extend(manta_tools)

    # Add SDK-specific tools
    tools.extend([
        types.Tool(
            name="obtain_tabular_dataset_properties",
            description="Extract properties from a tabular dataset using SDK workflow. Automatically detects PII, association rules, and field properties. Returns dataset ID with properties embedded.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to extract properties from"}
                },
                "required": ["dataset_id"]
            }
        ),
        types.Tool(
            name="obtain_train_config",
            description="Generate training configuration for synthetic data models. Returns a cached train_config_id for use in build_training_workflow. Supports TabGAN (rf_tab_gan) and TimeGAN (rf_time_gan).",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to generate config for"},
                    "model_type": {
                        "type": "string",
                        "enum": ["rf_tab_gan", "rf_time_gan"],
                        "description": "Model type: 'rf_tab_gan' for tabular data or 'rf_time_gan' for time-series data"
                    }
                },
                "required": ["dataset_id", "model_type"]
            }
        ),
        types.Tool(
            name="build_training_workflow",
            description="Build and start a training workflow using a cached config. Automatically detects model type from train_config structure. Use this after obtaining train_config_id from obtain_train_config.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to train on"},
                    "train_config_id": {"type": "string", "description": "Config ID from obtain_train_config (retrieves cached config)"}
                },
                "required": ["dataset_id", "train_config_id"]
            }
        ),
        types.Tool(
            name="update_train_config",
            description="Update training configuration. Modify model hyperparameters (epochs, batch_size, etc.) and/or encoder field classifications (categorical, continuous, ignore). See https://docs.rockfish.ai/model-train.html for config structure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "train_config_id": {
                        "type": "string",
                        "description": "Config ID from obtain_train_config"
                    },
                    "updates": {
                        "type": "object",
                        "properties": {
                            "model_config": {
                                "type": "object",
                                "description": "Model hyperparameters to update (e.g., epochs, batch_size, g_lr, d_lr)",
                                "additionalProperties": True
                            },
                            "encoder_config": {
                                "type": "object",
                                "properties": {
                                    "metadata": {
                                        "type": "object",
                                        "description": "Field name to type mappings. Type: 'categorical', 'continuous', 'ignore', or 'session' (TimeGAN only)",
                                        "additionalProperties": {"type": "string"}
                                    },
                                    "measurements": {
                                        "type": "object",
                                        "description": "(TimeGAN only, TODO: not yet implemented) Time-series measurement field classifications",
                                        "additionalProperties": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "description": "Updates to apply: model_config for hyperparameters, encoder_config for field types"
                    }
                },
                "required": ["train_config_id", "updates"]
            }
        ),
        types.Tool(
            name="get_workflow_status",
            description="Get the current status of a workflow (CREATED, RUNNING, COMPLETED, FAILED, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "Workflow ID to check status"}
                },
                "required": ["workflow_id"]
            }
        ),
        types.Tool(
            name="get_workflow_logs",
            description="Get logs from a workflow execution. Useful for debugging failed workflows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "Workflow ID to get logs from"}
                },
                "required": ["workflow_id"]
            }
        ),
        types.Tool(
            name="get_trained_model_id",
            description="Extract the trained model ID from a completed training workflow. Only works on COMPLETED or FINALIZED workflows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "Training workflow ID"}
                },
                "required": ["workflow_id"]
            }
        ),
        types.Tool(
            name="start_generation_workflow",
            description="Start a workflow to generate synthetic data from a trained TabGAN model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Trained model ID from get_trained_model_id"}
                },
                "required": ["model_id"]
            }
        ),
        types.Tool(
            name="obtain_synthetic_dataset_id",
            description="Extract the synthetic dataset ID from a completed generation workflow. Only works on COMPLETED or FINALIZED workflows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "generation_workflow_id": {"type": "string", "description": "Generation workflow ID from start_generation_workflow"}
                },
                "required": ["generation_workflow_id"]
            }
        ),
        types.Tool(
            name="plot_distribution",
            description="Generate distribution plot comparing datasets for a specific column. Returns base64-encoded PNG image. Automatically chooses bar plot (categorical) or KDE plot (numerical).",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of dataset IDs to compare (e.g., [real_dataset_id, synthetic_dataset_id])"
                    },
                    "column_name": {"type": "string", "description": "Column name to plot distribution for"}
                },
                "required": ["dataset_ids", "column_name"]
            }
        ),
        types.Tool(
            name="get_marginal_distribution_score",
            description="Calculate marginal distribution score between two datasets (real vs synthetic). Lower score means more similar distributions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Exactly 2 dataset IDs: [real_dataset_id, synthetic_dataset_id]"
                    }
                },
                "required": ["dataset_ids"]
            }
        ),
        types.Tool(
            name="html_output",
            description="Generate an HTML report (example/demo tool). Returns HTML content that can be saved to a file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the HTML report"}
                },
                "required": ["name"]
            }
        )
    ])

    return tools


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle tool calls to Rockfish API, SDK, and Manta service."""

    # Route SDK-specific tools to sdk_client
    sdk_tools = [
        "obtain_tabular_dataset_properties",
        "obtain_train_config",
        "update_train_config",
        "build_training_workflow",
        "get_workflow_status",
        "get_workflow_logs",
        "get_trained_model_id",
        "start_generation_workflow",
        "obtain_synthetic_dataset_id",
        "plot_distribution",
        "get_marginal_distribution_score",
        "html_output"
    ]

    if name in sdk_tools:
        if not sdk_client:
            return [types.TextContent(
                type="text",
                text="SDK client not initialized. Please check your API credentials."
            )]

        try:
            result = await sdk_client.call_endpoint(name, arguments)

            # Handle image responses (e.g., from plot_distribution)
            if isinstance(result, dict) and "image" in result and "mimeType" in result:
                # Return both image and descriptive text
                response_parts = [
                    types.ImageContent(
                        type="image",
                        data=result["image"],
                        mimeType=result["mimeType"]
                    )
                ]

                # Add descriptive text if available
                if "column_name" in result:
                    dataset_count = len(result.get("dataset_ids", []))
                    description = f"Distribution plot for column '{result['column_name']}'"
                    if dataset_count > 0:
                        description += f" comparing {dataset_count} dataset(s)"
                    response_parts.append(types.TextContent(
                        type="text",
                        text=description
                    ))

                return response_parts

            # Default: return as text
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            logger.error(f"Error calling {name}: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error calling {name}: {str(e)}"
            )]

    # Route Manta tools to manta_client
    if name.startswith("manta_"):
        if not manta_client:
            return [types.TextContent(
                type="text",
                text="Manta client not initialized. Please check your API credentials."
            )]

        try:
            result = await manta_client.call_endpoint(name, arguments)
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            logger.error(f"Error calling {name}: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error calling {name}: {str(e)}"
            )]

    # Route all other tools to rockfish_client
    if not rockfish_client:
        return [types.TextContent(
            type="text",
            text="Rockfish client not initialized. Please check your API credentials."
        )]

    try:
        result = await rockfish_client.call_endpoint(name, arguments)
        return [types.TextContent(type="text", text=str(result))]
    except Exception as e:
        logger.error(f"Error calling {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error calling {name}: {str(e)}"
        )]


async def main():
    global rockfish_client, manta_client, sdk_client

    # Initialize Rockfish client
    api_key = os.getenv("ROCKFISH_API_KEY")
    api_url = os.getenv("ROCKFISH_API_URL", "https://api.rockfish.ai")
    organization_id = os.getenv("ROCKFISH_ORGANIZATION_ID", None)
    project_id = os.getenv("ROCKFISH_PROJECT_ID", None)

    if not api_key:
        logger.error("ROCKFISH_API_KEY environment variable is required")
        return

    rockfish_client = RockfishClient(
        api_key=api_key,
        api_url=api_url,
        organization_id=organization_id,
        project_id=project_id
    )

    # Initialize SDK client (always initialized when API key is present)
    sdk_client = RockfishSDKClient(
        API_KEY=api_key,
        API_URL=api_url,
        ORGANIZATION_ID=organization_id,
        PROJECT_ID=project_id
    )
    logger.info("SDK client initialized")

    # Initialize Manta client only if MANTA_BASE_URL is configured
    manta_base_url = os.getenv("MANTA_BASE_URL")
    if manta_base_url:
        manta_client = MantaClient(
            api_key=api_key,
            base_url=manta_base_url
        )
        logger.info(f"Manta client initialized with base URL: {manta_base_url}")
    else:
        logger.info("Manta client not initialized (MANTA_BASE_URL not set)")
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rockfish-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

    # Cleanup connections
    if sdk_client:
        await sdk_client.close()


def cli():
    """Console script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()