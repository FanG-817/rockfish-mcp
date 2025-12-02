import asyncio
import base64
import io
import math
import re
from typing import Any, Dict, Optional, Tuple

# Set matplotlib to non-interactive backend BEFORE importing pyplot
# This prevents GUI windows and Python icon from appearing in dock
import matplotlib
import pyarrow as pa
import pyarrow.compute as pc
import rockfish as rf
import rockfish.actions as ra
import rockfish.labs as rl
from rockfish.remote import glue

matplotlib.use("Agg")
import logging
import uuid

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Import validators for comprehensive DataSchema validation
from rockfish_mcp.validators import ValidationError, validate_dataschema_comprehensive


class RockfishSDKClient:
    def __init__(
        self,
        API_KEY: str,
        API_URL: str,
        ORGANIZATION_ID: Optional[str] = None,
        PROJECT_ID: Optional[str] = None,
    ):
        """Initialize SDK client using environment variables via Connection.from_env()."""
        self._conn = rf.Connection.remote(
            API_KEY, api_url=API_URL, organization=ORGANIZATION_ID, project=PROJECT_ID
        )
        self._cache = {}

    async def close(self):
        """Close the SDK connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def call_endpoint(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route tool calls to appropriate SDK methods.

        Raises:
            NotImplementedError: If the operation is not supported by the SDK
            ValueError: If unknown tool name
        """
        if tool_name == "obtain_train_config":
            dataset_id = arguments["dataset_id"]
            model_type = arguments["model_type"]
            # dataset_properties = arguments["dataset_properties"]
            # field_properties_map = arguments["field_properties_map"]
            dataset = await get_local_dataset(self._conn, dataset_id)
            # TODO: Add train_config
            column_metadata = {}
            if model_type == "rf_tab_gan":
                train_config, column_metadata = guess_tab_gan_train_config(dataset)
            elif model_type == "rf_time_gan":
                return {
                    "success": False,
                    "message": f"Model type 'rf_time_gan' is not yet implemented. Currently only 'rf_tab_gan' is supported.",
                    "dataset_id": dataset_id,
                    "model_type": model_type,
                }
            else:
                return {
                    "success": False,
                    "message": f"Model type '{model_type}' is not supported. Currently only 'rf_tab_gan' is supported.",
                    "dataset_id": dataset_id,
                    "model_type": model_type,
                }
            # Serialize Config object to dict using Rockfish converter
            train_config_dict = rf.converter.unstructure(train_config)
            config_id = f"config_{uuid.uuid4()}"
            self._cache[config_id] = train_config_dict

            # Build response with column metadata and warnings
            response = {
                "success": True,
                "dataset_id": dataset_id,
                "train_config_id": config_id,
                "train_config": train_config_dict,
            }

            # Add warning for high cardinality columns
            high_card_cols = column_metadata.get("high_cardinality_columns", [])
            if high_card_cols:
                warning_msg = (
                    f"For now, we are ignoring {len(high_card_cols)} high cardinality columns: {high_card_cols}. "
                    "These columns will be handled in future iterations."
                )
                response["warnings"] = [warning_msg]

            return response
        elif tool_name == "start_training_workflow":
            dataset_id = arguments["dataset_id"]
            train_config_id = arguments["train_config_id"]

            # Check if config exists in cache
            if train_config_id not in self._cache:
                return {
                    "success": False,
                    "message": f"Config ID '{train_config_id}' not found in cache. It may have expired or already been used. Please call obtain_train_config again.",
                    "dataset_id": dataset_id,
                    "train_config_id": train_config_id,
                }

            train_config = self._cache.pop(train_config_id)
            # Detect model type from config
            train_config = rf.converter.unstructure(train_config)
            # NOTICE: unstructured config (dict) returns "tabular-gan" rather than "tabular_gan"
            if "tabular-gan" in train_config:
                train_config = rf.converter.structure(
                    train_config, ra.TrainTabGAN.Config
                )
                train_action = ra.TrainTabGAN(train_config)
            elif "doppelganger" in train_config:
                train_config = rf.converter.structure(
                    train_config, ra.TrainTimeGAN.Config
                )
                train_action = ra.TrainTimeGAN(train_config)
            else:
                return {
                    "success": False,
                    "message": "Unsupported training config format. Currently only RF-Tab-GAN (tabular-gan) and RF-Time-GAN (doppelganger) models are supported.",
                    "dataset_id": dataset_id,
                    "train_config_id": train_config_id,
                }

            load_action = ra.DatasetLoad(dataset_id=dataset_id)
            builder = create_workflow([load_action, train_action])
            train_workflow = await builder.start(self._conn)
            return {"success": True, "train_workflow_id": train_workflow.id()}

        elif tool_name == "get_workflow_logs":
            workflow_id = arguments["workflow_id"]
            log_level_str = arguments.get("log_level", "INFO")
            collection_timeout = arguments.get("timeout", 10)

            # Map string to rf.LogLevel enum
            log_level_map = {
                "DEBUG": rf.LogLevel.DEBUG,
                "INFO": rf.LogLevel.INFO,
                "WARN": rf.LogLevel.WARN,
                "ERROR": rf.LogLevel.ERROR,
            }
            log_level = log_level_map.get(log_level_str, rf.LogLevel.INFO)

            # Get workflow and stream logs
            workflow = await self._conn.get_workflow(workflow_id)
            logs = []

            async def collect_logs():
                nonlocal logs
                async for log in workflow.logs(level=log_level):
                    logs.append(str(log))

            try:
                # Collect logs for specified duration
                await asyncio.wait_for(collect_logs(), timeout=collection_timeout)
            except asyncio.TimeoutError:
                # Expected - collected logs for specified duration
                pass

            # Build response with helpful messages
            result = {
                "workflow_id": workflow_id,
                "logs": logs,
                "count": len(logs),
                "log_level": log_level_str,
            }

            if len(logs) == 0:
                result["message"] = (
                    f"No {log_level_str} logs collected in {collection_timeout}s. Workflow may still be starting. Try waiting longer or increase timeout parameter."
                )
            else:
                result["message"] = (
                    f"Collected {len(logs)} {log_level_str} logs in {collection_timeout}s. Call again to get more logs if workflow is still running."
                )

            return result

        elif tool_name == "get_trained_model_id":
            workflow_id = arguments["workflow_id"]
            workflow = await self._conn.get_workflow(workflow_id)
            status = await workflow.status()
            if status not in {"completed", "finalized"}:
                return {
                    "success": False,
                    "message": f"Workflow is in '{status}' state. This tool only works on COMPLETED or FINALIZED workflows. Please wait for the workflow to complete.",
                    "workflow_id": workflow_id,
                    "status": status,
                }
            model = await workflow.models().last()
            return {"success": True, "workflow_id": workflow_id, "model_id": model.id}
        elif tool_name == "start_generation_workflow":
            model_id = arguments["model_id"]
            generate_rec = rl.steps.GenerateRecommender(self._conn, model=model_id)
            generate_builder = await generate_rec.builder()
            generate_workflow = await generate_builder.start(self._conn)
            return {"generation_workflow_id": generate_workflow.id()}
        elif tool_name == "obtain_synthetic_dataset_id":
            generation_workflow_id = arguments["generation_workflow_id"]
            generation_workflow = await self._conn.get_workflow(generation_workflow_id)
            status = await generation_workflow.status()
            if status not in {"completed", "finalized"}:
                return {
                    "success": False,
                    "message": f"Generation workflow is in '{status}' state. This tool only works on COMPLETED or FINALIZED workflows. Please wait for the workflow to complete.",
                    "generation_workflow_id": generation_workflow_id,
                    "status": status,
                }
            # TODO
            generated_datasets = await generation_workflow.datasets().collect()
            return {
                "success": True,
                "generation_workflow_id": generation_workflow_id,
                "generated_dataset_id(s)": [
                    generated_dataset.id for generated_dataset in generated_datasets
                ],
            }
        elif tool_name == "plot_distribution":
            dataset_ids = arguments["dataset_ids"]
            column_name = arguments["column_name"]

            img_base64 = await plot_distribution(self._conn, dataset_ids, column_name)
            return {
                "image": img_base64,
                "mimeType": "image/png",
                "dataset_ids": dataset_ids,
                "column_name": column_name,
            }
        elif tool_name == "get_marginal_distribution_score":
            dataset_ids = arguments["dataset_ids"]
            dataset = await self._conn.get_dataset(dataset_ids[0])
            dataset = await dataset.to_local(self._conn)
            real_columns = dataset.table.column_names
            synthetic = await self._conn.get_dataset(dataset_ids[1])
            synthetic = await synthetic.to_local(self._conn)
            syn_columns = synthetic.table.column_names

            # Find common and different columns using set operations
            real_columns_set = set(real_columns)
            syn_columns_set = set(syn_columns)
            common_columns = list(real_columns_set & syn_columns_set)
            only_in_real = real_columns_set - syn_columns_set
            only_in_syn = syn_columns_set - real_columns_set

            # Select only common columns from both datasets
            dataset.table = dataset.table.select(common_columns)
            synthetic.table = synthetic.table.select(common_columns)

            # Build informative message about excluded columns
            if only_in_real or only_in_syn:
                msg_parts = []
                if only_in_real:
                    msg_parts.append(
                        f"{', '.join(sorted(only_in_real))} only in real data"
                    )
                if only_in_syn:
                    msg_parts.append(
                        f"{', '.join(sorted(only_in_syn))} only in synthetic data"
                    )
                msg = f"Columns excluded from evaluation: {'; '.join(msg_parts)}"
            else:
                msg = "All columns match between datasets"

            marginal_dist_score = rl.metrics.marginal_dist_score(dataset, synthetic)

            # Check if score is NaN (happens when datasets have missing values)
            if math.isnan(marginal_dist_score):
                return {
                    "success": False,
                    "message": "Dataset contains missing values. Marginal distribution score does not currently support datasets with missing values.",
                    "marginal_distribution_score": None,
                }
            else:
                return {
                    "success": True,
                    "message": msg,
                    "marginal_distribution_score": marginal_dist_score,
                }
        # this tool has not been tested out yet - still experiemental and require udpates
        # For now, it is only for rf-tab-gan
        elif tool_name == "update_train_config":
            train_config_id = arguments["train_config_id"]
            updates = arguments["updates"]

            # Retrieve cached config
            if train_config_id not in self._cache:
                return {
                    "success": False,
                    "message": f"Config ID '{train_config_id}' not found in cache. It may have expired or already been used. Please call obtain_train_config again.",
                    "train_config_id": train_config_id,
                }

            config_dict = self._cache[train_config_id].copy()

            # Detect model type
            if "tabular-gan" in config_dict:
                model_key = "tabular-gan"
            elif "doppelganger" in config_dict:
                model_key = "doppelganger"
            else:
                return {
                    "success": False,
                    "message": "Cannot determine model type from config. Supported model types: 'tabular-gan', 'doppelganger'",
                    "train_config_id": train_config_id,
                }

            changes_applied = {}

            # Update model_config (hyperparameters)
            if "model_config" in updates:
                model_config = updates["model_config"]
                for field, value in model_config.items():
                    if field not in config_dict[model_key]:
                        return {
                            "success": False,
                            "message": f"Field '{field}' not found in {model_key} config. Available fields: {list(config_dict[model_key].keys())}",
                            "train_config_id": train_config_id,
                            "invalid_field": field,
                        }
                    old_value = config_dict[model_key][field]
                    config_dict[model_key][field] = value
                    changes_applied[f"{model_key}.{field}"] = {
                        "old": old_value,
                        "new": value,
                    }

            # Update encoder_config (field classifications)
            if "encoder_config" in updates:
                encoder_config = updates["encoder_config"]

                # Update metadata fields
                if "metadata" in encoder_config:
                    metadata_updates = encoder_config["metadata"]
                    metadata = config_dict["encoder"]["metadata"]

                    for field_name, new_type in metadata_updates.items():
                        # Validate type
                        valid_types = ["categorical", "continuous", "ignore"]
                        if model_key == "doppelganger":
                            valid_types.append("session")

                        if new_type not in valid_types:
                            return {
                                "success": False,
                                "message": f"Invalid type '{new_type}' for field '{field_name}'. Valid types: {valid_types}",
                                "train_config_id": train_config_id,
                                "field_name": field_name,
                                "invalid_type": new_type,
                            }

                        # Find and update field in metadata list
                        field_found = False
                        for field_config in metadata:
                            if field_config["field"] == field_name:
                                old_type = field_config["type"]
                                field_config["type"] = new_type
                                changes_applied[f"encoder.metadata.{field_name}"] = {
                                    "old": old_type,
                                    "new": new_type,
                                }
                                field_found = True
                                break

                        if not field_found:
                            available_fields = [f["field"] for f in metadata]
                            return {
                                "success": False,
                                "message": f"Field '{field_name}' not found in encoder metadata. Available fields: {available_fields}",
                                "train_config_id": train_config_id,
                                "field_name": field_name,
                            }

            # Update cache with modified config
            self._cache[train_config_id] = config_dict

            return {
                "success": True,
                "train_config_id": train_config_id,
                "changes_applied": changes_applied,
                "train_config": config_dict,
            }
        elif tool_name == "validate_data_schema_config":
            data_schema_config = arguments["data_schema_config"]
            entity_labels = arguments.get("entity_labels")

            # Validate schema (all validation handled by _validate_data_schema)
            validation_result = _validate_data_schema(data_schema_config)

            if not validation_result["valid"]:
                # Return validation errors with enhanced details
                error_response = {
                    "success": False,
                    "message": validation_result["summary"],
                    "error": validation_result["error_message"],
                }
                # Add optional fields if present
                if (
                    "suggestion" in validation_result
                    and validation_result["suggestion"]
                ):
                    error_response["suggestion"] = validation_result["suggestion"]
                if "reference" in validation_result and validation_result["reference"]:
                    error_response["reference"] = validation_result["reference"]

                return error_response

            # Validation passed - cache the config
            config_id = f"schema_config_{uuid.uuid4()}"
            cache_entry = {
                "data_schema_config": data_schema_config,
            }
            # TODO: entity_labels?
            if entity_labels:
                cache_entry["entity_labels"] = entity_labels

            self._cache[config_id] = cache_entry

            # Generate summary
            entities = data_schema_config.get("entities", [])
            entity_names = [e.get("name") for e in entities]
            total_columns = sum(len(e.get("columns", [])) for e in entities)
            relationships_count = len(
                data_schema_config.get("entity_relationships", [])
            )

            response = {
                "success": True,
                "schema_config_id": config_id,
                "summary": {
                    "entities_count": len(entities),
                    "entities": entity_names,
                    "total_columns": total_columns,
                    "relationships_count": relationships_count,
                },
                "message": validation_result["summary"],
            }

            return response
        elif tool_name == "start_data_schema_generation_workflow":
            schema_config_id = arguments["schema_config_id"]

            # Check cache
            if schema_config_id not in self._cache:
                return {
                    "success": False,
                    "message": f"Config ID '{schema_config_id}' not found in cache. It may have expired or already been used. Please call validate_data_schema_config again.",
                    "schema_config_id": schema_config_id,
                }

            # Retrieve and remove from cache
            cache_entry = self._cache.pop(schema_config_id)
            data_schema_dict = cache_entry["data_schema_config"]
            entity_labels = cache_entry.get(
                "entity_labels"
            )  # TODO: make use of entity_labels in GenerateFromDataSchema

            # Convert dict to DataSchema (should not fail since we already validated)
            try:
                data_schema = rf.converter.structure(
                    data_schema_dict, ra.ent.DataSchema
                )
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to convert schema config to DataSchema: {str(e)}",
                    "schema_config_id": schema_config_id,
                }

            # Create GenerateFromDataSchema action
            try:
                if entity_labels:
                    generate_action = ra.ent.GenerateFromDataSchema(
                        schema=data_schema, entity_labels=entity_labels
                    )
                else:
                    generate_action = ra.ent.GenerateFromDataSchema(schema=data_schema)

                # Build and start workflow
                builder = create_workflow([generate_action])
                workflow = await builder.start(self._conn)

                return {
                    "success": True,
                    "workflow_id": workflow.id(),
                    "message": f"Started data schema generation workflow: {workflow.id()}. Use get_workflow_logs to monitor progress.",
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to start workflow: {str(e)}",
                    "schema_config_id": schema_config_id,
                }
        else:
            return {
                "success": False,
                "message": f"Unknown SDK tool: '{tool_name}'. This tool is not recognized by the SDK client.",
                "tool_name": tool_name,
            }


# Helper functions


# Workflow helpers
def create_workflow(actions: list[rf.Action]) -> rf.WorkflowBuilder:
    """Create a workflow builder with a linear path of actions."""
    builder = rf.WorkflowBuilder()
    builder.add_path(*actions)
    return builder


async def get_local_dataset(conn, dataset_id: str) -> rf.dataset.LocalDataset:
    """Fetch a dataset and convert it to a local dataset."""
    dataset = await conn.get_dataset(dataset_id)
    dataset = await dataset.to_local(conn)
    return dataset


# Visualization helpers
async def plot_distribution(conn, dataset_ids: list, column_name: str):
    """Plot distribution comparison between real and synthetic data for a given column."""

    def _fig_to_base64(fig):
        """Convert a figure(plot) to a base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        img_str = base64.b64encode(buf.getbuffer()).decode("utf-8")
        buf.close()
        plt.close(fig.fig)  # Close the underlying matplotlib figure to free memory
        return img_str

    if len(dataset_ids) != 2:
        raise ValueError("current only support 2 datasets for comparison plotting")
    # Load dataset and convert to LocalDataset
    dataset = await get_local_dataset(conn, dataset_ids[0])
    synthetic = await get_local_dataset(conn, dataset_ids[1])

    table = dataset.table
    field_type = table[column_name].type

    # Choose plot type based on data characteristics
    if pa.types.is_string(field_type):
        # Categorical/string data → bar plot
        fig = rf.labs.vis.plot_bar([dataset, synthetic], column_name)
    else:
        # Numerical data with enough rows → KDE plot
        fig = rf.labs.vis.plot_kde([dataset, synthetic], column_name)

    img_base64 = _fig_to_base64(fig)
    return img_base64


# Configuration helpers
def guess_tab_gan_train_config(dataset) -> Tuple[ra.TrainTabGAN.Config, dict]:
    """Generate TabGAN training configuration with automatic column type detection."""
    table = dataset.table
    columns = table.column_names
    high_cardinality_columns = []
    categorical_columns = []
    continuous_columns = []
    for column in columns:
        dtype = str(table[column].type)
        # mode='only_valid' exclude null values
        nunique = pc.count_distinct(table[column], mode="only_valid").as_py()
        if dtype in {"string", "bool"}:
            if nunique <= 100:
                categorical_columns.append(column)
            else:
                # Cardinality > 100 is likely to cause OOM so for now, we ignore them in train config.
                # Later, we could do resampling or label encoder to handle them
                high_cardinality_columns.append(column)
        elif nunique <= 10:
            categorical_columns.append(column)
        else:
            continuous_columns.append(column)
    encoder_config = ra.TrainTabGAN.DatasetConfig(
        metadata=[
            ra.TrainTabGAN.FieldConfig(field=col, type="categorical")
            for col in categorical_columns
        ]
        + [
            ra.TrainTabGAN.FieldConfig(field=col, type="ignore")
            for col in high_cardinality_columns
        ]
        + [
            ra.TrainTabGAN.FieldConfig(field=col, type="continuous")
            for col in continuous_columns
        ]
    )
    model_config = ra.TrainTabGAN.TrainConfig(epochs=100)
    train_config = ra.TrainTabGAN.Config(
        encoder=encoder_config, tabular_gan=model_config
    )

    # Return config and column metadata
    column_metadata = {
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "high_cardinality_columns": high_cardinality_columns,
    }
    return train_config, column_metadata


# Entity Data Generator helpers
def _validate_data_schema(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate DataSchema with enhanced error messages.

    Validation is performed in two layers:
    1. SDK validation via rf.converter.structure() - validates types, required fields, enums
    2. validators.py validation - validates business logic (PARAM_TS_07, TS_08, R11, R12)

    All errors are always shown (no limiting).

    Returns:
        Dict with validation results:
        - If valid:
            {
                "valid": True,
                "data_schema": DataSchema,
                "summary": str
            }
        - If invalid:
            {
                "valid": False,
                "error_count": int,
                "error_message": str (numbered list of errors),
                "summary": str
            }
    """
    try:
        # SDK validates all levels automatically via __attrs_post_init__ methods
        data_schema = rf.converter.structure(schema_dict, ra.ent.DataSchema)

        # Run comprehensive validation via validators.py
        comprehensive_errors = validate_dataschema_comprehensive(data_schema)

        if comprehensive_errors:
            # Format errors same way as structure error path
            # Show ALL validator errors (no limiting)
            error_list = []
            for idx, err in enumerate(comprehensive_errors, 1):
                error_list.append(f"{idx}. {err.location}: {err.message}")

            return {
                "valid": False,
                "error_count": len(comprehensive_errors),
                "error_message": "\n\n".join(error_list),
                "summary": f"Validation failed: {len(comprehensive_errors)} error(s)",
            }

        return {
            "valid": True,
            "data_schema": data_schema,
            "summary": f"Validation successful: {len(schema_dict.get('entities', []))} entities",
        }
    except Exception as e:
        error_details = extract_structure_error_details(e)

        # Format errors as numbered list
        error_list = []
        for idx, err in enumerate(error_details["errors"], 1):
            error_list.append(f"{idx}. {err['location']}: {err['error_message']}")

        return {
            "valid": False,
            "error_count": error_details["error_count"],
            "error_message": "\n\n".join(error_list),
            "summary": f"Validation failed: {error_details['summary']}",
        }


def extract_structure_error_details(exc: Exception) -> Dict[str, Any]:
    """
    Extract error message and location from StructureError exception chain.

    Simplified version that extracts only essential information:
    - error_message: The actual error text from the exception
    - location: The JSON path where the error occurred

    Args:
        exc: The exception to parse (typically StructureError)

    Returns:
        {
            "error_count": N,
            "summary": "Found N validation error(s)",
            "errors": [
                {
                    "error_message": "spike_magnitude (5.0) must be in [0, 1]",
                    "location": "$.entities[0].columns[1]"
                },
                ...
            ]
        }

    Example:
        >>> try:
        ...     rf.converter.structure(ts_dict, TimeseriesParams)
        ... except Exception as e:
        ...     details = extract_structure_error_details(e)
        ...     for err in details['errors']:
        ...         print(f"{err['location']}: {err['error_message']}")
    """

    errors = []

    def collect_errors(current_exc, location="$"):
        """Recursively collect error messages and locations."""
        # Check for sub-exceptions (IterableValidationError)
        if hasattr(current_exc, "exceptions") and current_exc.exceptions:
            # Extract location info from wrapper message
            msg = str(current_exc)
            index_match = re.search(r"@ index (\d+)", msg)
            if index_match:
                index = index_match.group(1)
                if "list[Column]" in msg:
                    new_location = f"{location}.columns[{index}]"
                elif "list[Entity]" in msg:
                    new_location = f"{location}.entities[{index}]"
                else:
                    new_location = f"{location}[{index}]"
            else:
                new_location = location

            for sub_exc in current_exc.exceptions:
                collect_errors(sub_exc, new_location)

        # Check for __cause__ chain
        elif hasattr(current_exc, "__cause__") and current_exc.__cause__:
            # Extract location from StructureError if present
            if type(current_exc).__name__ == "StructureError":
                msg = str(current_exc)
                loc_match = re.search(r"@ (\$\.[^\s]+)", msg)
                if loc_match:
                    location = loc_match.group(1)

            collect_errors(current_exc.__cause__, location)

        # Leaf error - add to list
        else:
            # Skip wrapper exceptions
            if type(current_exc).__name__ not in (
                "StructureError",
                "ClassValidationError",
                "IterableValidationError",
                "ExceptionGroup",
            ):
                errors.append({"error_message": str(current_exc), "location": location})

    # Extract top-level location from StructureError
    top_location = "$"
    if type(exc).__name__ == "StructureError":
        top_msg = str(exc)
        loc_match = re.search(r"@ (\$\.[^\s]+)", top_msg)
        if loc_match:
            top_location = loc_match.group(1)

    # Collect all errors
    collect_errors(exc, top_location)

    # If no errors found, use original exception
    if not errors:
        errors = [{"error_message": str(exc), "location": top_location}]

    # Return consistent format
    error_word = "error" if len(errors) == 1 else "errors"
    return {
        "error_count": len(errors),
        "summary": f"Found {len(errors)} validation {error_word}",
        "errors": errors,
    }
