"""
Validation layer for DataSchema configurations.

This module provides business logic validation rules that are NOT covered by the
Rockfish SDK's built-in __attrs_post_init__ methods. These 4 validators catch
errors that would otherwise cause runtime failures during generation.

The SDK already validates:
- Type checking (Layer 1)
- Required fields
- Enum values
- Basic parameter constraints (min < max, ranges [0,1], etc.)

This module adds:
- PARAM_TS_07, TS_08: Peak hour range validation [0,23]
- R11: MEASUREMENT→MEASUREMENT dependency prevention
- R12: MEASUREMENT INDEPENDENT restriction
"""

from dataclasses import dataclass
from typing import Any

from rockfish.actions.ent import (
    Column,
    ColumnCategoryType,
    ColumnType,
    DataSchema,
    Entity,
)
from rockfish.actions.ent.generate import Domain, DomainType, TimeseriesParams

###############################################################################
# Core Validation Types
###############################################################################


@dataclass
class ValidationError:
    """Represents a validation error with context.

    All validation errors are treated as blocking errors (ERROR level).
    """

    rule: str
    message: str
    location: str
    suggestion: str = ""


###############################################################################
# Validator Class
###############################################################################


class DataSchemaValidator:
    """
    Validates DataSchema configurations with business logic rules.

    Only validates rules that are NOT already covered by the Rockfish SDK:
    - PARAM_TS_07: peak_start_hour in [0,23]
    - PARAM_TS_08: peak_end_hour in [0,23]
    - R11: MEASUREMENT→MEASUREMENT dependency prevention
    - R12: MEASUREMENT cannot be INDEPENDENT
    """

    def __init__(self, schema: DataSchema):
        self.schema = schema
        self.errors: list[ValidationError] = []

    def validate_all(self) -> list[ValidationError]:
        """
        Run all validation rules.

        Returns:
            List of ValidationError objects (empty if valid)
        """
        # Validate TimeseriesParams peak hours (PARAM_TS_07, PARAM_TS_08)
        self._validate_domain_params()

        # Validate MEASUREMENT dependencies (R11)
        self._validate_derivation_params()

        # Validate column business rules (R12)
        self._validate_business_rules()

        return self.errors

    def _validate_domain_params(self):
        """Validate Domain parameters for TimeseriesParams peak hours."""
        for entity_idx, entity in enumerate(self.schema.entities):
            for column_idx, column in enumerate(entity.columns):
                if column.domain:
                    location = f"$.entities[{entity_idx}] ({entity.name}) > columns[{column_idx}] ({column.name}) > domain"

                    # Only validate TimeseriesParams
                    if column.domain.type == DomainType.TIMESERIES:
                        self._validate_timeseries_params(column.domain.params, location)

    def _validate_timeseries_params(self, params: TimeseriesParams, location: str):
        """
        Validate TimeseriesParams peak hour ranges (SDK doesn't validate these).

        The SDK validates:
        - min_value < max_value
        - peak_start_hour < peak_end_hour
        - seasonality_strength in [0,1]
        - noise_level in [0,1]
        - spike_probability in [0,1]
        - spike_magnitude in [0,1]

        This validator adds:
        - PARAM_TS_07: peak_start_hour in [0,23]
        - PARAM_TS_08: peak_end_hour in [0,23]
        """
        # PARAM_TS_07: peak_start_hour must be in [0, 23]
        if params.peak_start_hour is not None and not (
            0 <= params.peak_start_hour <= 23
        ):
            self.errors.append(
                ValidationError(
                    rule="PARAM_TS_07",
                    message=f"TimeseriesParams peak_start_hour must be in [0, 23], got: {params.peak_start_hour}",
                    location=location,
                    suggestion="Use a valid hour value between 0 (midnight) and 23 (11 PM)",
                )
            )

        # PARAM_TS_08: peak_end_hour must be in [0, 23]
        if params.peak_end_hour is not None and not (0 <= params.peak_end_hour <= 23):
            self.errors.append(
                ValidationError(
                    rule="PARAM_TS_08",
                    message=f"TimeseriesParams peak_end_hour must be in [0, 23], got: {params.peak_end_hour}",
                    location=location,
                    suggestion="Use a valid hour value between 0 (midnight) and 23 (11 PM)",
                )
            )

    def _validate_derivation_params(self):
        """Validate Derivation parameters and dependencies."""
        for entity_idx, entity in enumerate(self.schema.entities):
            for column_idx, column in enumerate(entity.columns):
                if column.derivation:
                    location = f"$.entities[{entity_idx}] ({entity.name}) > columns[{column_idx}] ({column.name}) > derivation"

                    # R11: Check for MEASUREMENT→MEASUREMENT dependencies - unsolved issue?
                    self._validate_measurement_dependencies(entity, column, location)

    def _validate_measurement_dependencies(
        self, entity: Entity, column: Column, location: str
    ):
        """
        R11: MEASUREMENT derived columns cannot depend on same-entity MEASUREMENT columns.

        This is a critical validator that prevents runtime KeyError during generation.
        MEASUREMENT columns are generated in arbitrary order, so dependencies between
        them in the same entity will fail.

        The SDK does NOT validate this.
        """
        if (
            column.column_category_type == ColumnCategoryType.MEASUREMENT
            and column.derivation
        ):
            # Check each dependency
            for dep_col_name in column.derivation.dependencies:
                # Find dependency column in same entity
                dep_col = next(
                    (col for col in entity.columns if col.name == dep_col_name), None
                )

                if (
                    dep_col
                    and dep_col.column_category_type == ColumnCategoryType.MEASUREMENT
                ):
                    self.errors.append(
                        ValidationError(
                            rule="R11",
                            message=f"MEASUREMENT derived column '{column.name}' cannot depend on another MEASUREMENT column '{dep_col_name}' in the same entity (currently unsupported)",
                            location=location,
                            suggestion=f"Change '{dep_col_name}' to column_category_type='metadata', OR restructure to avoid MEASUREMENT->MEASUREMENT dependencies",
                        )
                    )

    def _validate_business_rules(self):
        """Validate business logic rules."""
        for entity_idx, entity in enumerate(self.schema.entities):
            for column_idx, column in enumerate(entity.columns):
                location = f"$.entities[{entity_idx}] ({entity.name}) > columns[{column_idx}] ({column.name})"

                # R12: MEASUREMENT cannot be INDEPENDENT - will added in the next release
                self._validate_column_business_rules(column, location)

    def _validate_column_business_rules(self, column: Column, location: str):
        """
        Validate column-level business rules.

        R12: MEASUREMENT columns cannot be INDEPENDENT type.

        This combination is currently unsupported by generation but the SDK allows it.
        This validator prevents invalid configurations.
        """
        # R12: MEASUREMENT column cannot be INDEPENDENT
        if (
            column.column_category_type == ColumnCategoryType.MEASUREMENT
            and column.column_type == ColumnType.INDEPENDENT
        ):
            self.errors.append(
                ValidationError(
                    rule="R12",
                    message=f"MEASUREMENT column cannot be INDEPENDENT (currently unsupported)",
                    location=location,
                    suggestion="Change column_type to 'stateful' for time-varying measurements, or change column_category_type to 'metadata' for static values",
                )
            )


###############################################################################
# Public API
###############################################################################


def validate_dataschema_comprehensive(schema: DataSchema) -> list[ValidationError]:
    """
    Comprehensive validation of DataSchema configuration.

    This function validates business logic rules that are NOT covered by the
    Rockfish SDK's built-in __attrs_post_init__ methods.

    Validates:
    - PARAM_TS_07: peak_start_hour in [0,23]
    - PARAM_TS_08: peak_end_hour in [0,23]
    - R11: MEASUREMENT→MEASUREMENT dependency prevention (prevents runtime KeyError)
    - R12: MEASUREMENT INDEPENDENT restriction (unsupported combination)

    The SDK already validates everything else (types, required fields, enums,
    parameter ranges, entity constraints, etc.).

    Args:
        schema: DataSchema object to validate (already structured by SDK)

    Returns:
        List of ValidationError objects (empty if valid)

    Example:
        >>> schema = rf.converter.structure(schema_dict, DataSchema)
        >>> errors = validate_dataschema_comprehensive(schema)
        >>> if errors:
        ...     for err in errors:
        ...         print(f"{err.rule}: {err.message}")
    """
    validator = DataSchemaValidator(schema)
    return validator.validate_all()


def format_validation_errors(errors: list[ValidationError]) -> str:
    """
    Format validation errors for display.

    All errors are treated as ERROR level (blocking).

    Args:
        errors: List of ValidationError objects

    Returns:
        Formatted string with all errors
    """
    if not errors:
        return "No validation errors"

    lines = []
    for idx, err in enumerate(errors, 1):
        lines.append(
            f"{idx}. [ERROR] {err.rule}: {err.message}\n" f"   Location: {err.location}"
        )
        if err.suggestion:
            lines.append(f"   Suggestion: {err.suggestion}")

    return "\n\n".join(lines)
