"""
Comprehensive validation utilities for DataSchema configurations.

This module provides detailed validation beyond what rf.converter.structure() performs,
checking business logic rules, parameter constraints, and semantic relationships.

Validation Layers:
1. Structure validation (rf.converter.structure) - type checking, required fields, enums
2. Parameter validation (this module) - ranges, constraints, logical consistency
3. Business logic validation (this module) - R1-R12 rules
4. Graph validation (planner) - circular dependencies

Usage:
    from ent.validators import validate_dataschema_comprehensive

    errors = validate_dataschema_comprehensive(schema)
    if errors:
        for error in errors:
            print(f"[{error.level}] {error.rule}: {error.message}")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from rockfish.actions.ent import (
    CategoricalParams,
    Column,
    ColumnCategoryType,
    ColumnType,
    DataSchema,
    Derivation,
    DerivationFunctionType,
    Domain,
    DomainType,
    Entity,
    EntityRelationship,
    EntityRelationshipType,
    ExponentialDistParams,
    GlobalTimestamp,
    IDParams,
    MapValuesParams,
    NormalDistParams,
    StateMachineParams,
    TimeseriesParams,
    Transition,
    UniformDistParams,
)


class ValidationLevel(str, Enum):
    """Severity level of validation error."""

    ERROR = "ERROR"  # Must fix - will cause generation to fail
    WARNING = "WARNING"  # Should fix - may cause unexpected behavior
    INFO = "INFO"  # Informational - best practice suggestion


@dataclass
class ValidationError:
    """Structured validation error."""

    level: ValidationLevel
    rule: str  # e.g., "R1", "PARAM_UNIFORM_01", "COL_INDEPENDENT_01"
    message: str
    location: str  # e.g., "entity 'users' > column 'age'"
    suggestion: str = ""  # Optional fix suggestion


class DataSchemaValidator:
    """Comprehensive DataSchema validator."""

    def __init__(self, schema: DataSchema):
        self.schema = schema
        self.errors: list[ValidationError] = []
        self.entity_map = {entity.name: entity for entity in schema.entities}

    def validate_all(self) -> list[ValidationError]:
        """Run all validation checks and return errors."""
        self.errors = []

        # Layer 2: Parameter validation
        self._validate_domain_params()
        self._validate_derivation_params()

        # Layer 3: Business logic (R1-R10)
        self._validate_business_rules()

        # Additional checks
        self._validate_entities()
        self._validate_relationships()
        self._validate_global_timestamp()

        return self.errors

    # =========================================================================
    # Domain Parameter Validation
    # =========================================================================

    def _validate_domain_params(self):
        """Validate all domain parameters."""
        for entity in self.schema.entities:
            for column in entity.columns:
                if column.domain is None:
                    continue

                loc = f"entity '{entity.name}' > column '{column.name}'"

                if column.domain.type == DomainType.ID:
                    self._validate_id_params(column.domain.params, loc)
                elif column.domain.type == DomainType.CATEGORICAL:
                    self._validate_categorical_params(column.domain.params, loc)
                elif column.domain.type == DomainType.UNIFORM_DIST:
                    self._validate_uniform_params(column.domain.params, loc)
                elif column.domain.type == DomainType.NORMAL_DIST:
                    self._validate_normal_params(column.domain.params, loc)
                elif column.domain.type == DomainType.EXPONENTIAL_DIST:
                    self._validate_exponential_params(column.domain.params, loc)
                elif column.domain.type == DomainType.TIMESERIES:
                    self._validate_timeseries_params(column.domain.params, loc)
                elif column.domain.type == DomainType.STATE_MACHINE:
                    self._validate_state_machine_params(column.domain.params, loc)

    def _validate_id_params(self, params: IDParams, location: str):
        """Validate IDParams: template must contain {id}."""
        if "{id}" not in params.template_str:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_ID_01",
                    message=f"IDParams template_str must contain '{{id}}' placeholder, got: '{params.template_str}'",
                    location=location,
                    suggestion="Use a template like 'USER_{id}' or 'item-{id}'",
                )
            )

    def _validate_categorical_params(self, params: CategoricalParams, location: str):
        """Validate CategoricalParams: values not empty, weights match length."""
        if not params.values:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_CAT_01",
                    message="CategoricalParams values list cannot be empty",
                    location=location,
                    suggestion="Provide at least one value in the values list",
                )
            )

        if params.weights is not None:
            if len(params.weights) != len(params.values):
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="PARAM_CAT_02",
                        message=f"CategoricalParams weights length ({len(params.weights)}) must match values length ({len(params.values)})",
                        location=location,
                        suggestion="Either remove weights or ensure it has the same length as values",
                    )
                )

    def _validate_uniform_params(self, params: UniformDistParams, location: str):
        """Validate UniformDistParams: lower < upper."""
        if params.lower >= params.upper:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_UNIFORM_01",
                    message=f"UniformDistParams lower ({params.lower}) must be less than upper ({params.upper})",
                    location=location,
                    suggestion=f"Swap the values or use lower={params.upper}, upper={params.lower + (params.upper - params.lower) * 2}",
                )
            )

    def _validate_normal_params(self, params: NormalDistParams, location: str):
        """Validate NormalDistParams: std > 0."""
        if params.std <= 0:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_NORMAL_01",
                    message=f"NormalDistParams std (standard deviation) must be positive, got: {params.std}",
                    location=location,
                    suggestion="Use a positive value like std=10.0 or std=1.5",
                )
            )

    def _validate_exponential_params(
        self, params: ExponentialDistParams, location: str
    ):
        """Validate ExponentialDistParams: scale > 0."""
        if params.scale <= 0:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_EXP_01",
                    message=f"ExponentialDistParams scale must be positive, got: {params.scale}",
                    location=location,
                    suggestion="Use a positive value like scale=2.0",
                )
            )

    def _validate_timeseries_params(self, params: TimeseriesParams, location: str):
        """Validate TimeseriesParams: 6 range and probability checks."""
        # min_value < max_value
        if params.min_value >= params.max_value:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_TS_01",
                    message=f"TimeseriesParams min_value ({params.min_value}) must be less than max_value ({params.max_value})",
                    location=location,
                    suggestion="Ensure min_value < base_value < max_value",
                )
            )

        # peak_start_hour < peak_end_hour (only relevant for peak_offpeak)
        if params.seasonality_type == "peak_offpeak":
            if params.peak_start_hour >= params.peak_end_hour:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="PARAM_TS_02",
                        message=f"TimeseriesParams peak_start_hour ({params.peak_start_hour}) must be less than peak_end_hour ({params.peak_end_hour})",
                        location=location,
                        suggestion="Use values like peak_start_hour=8, peak_end_hour=22 for business hours",
                    )
                )

            # PARAM_TS_07: peak_start_hour must be in [0, 23]
            if not (0 <= params.peak_start_hour <= 23):
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="PARAM_TS_07",
                        message=f"TimeseriesParams peak_start_hour must be in [0, 23], got: {params.peak_start_hour}",
                        location=location,
                        suggestion="Use a valid hour value between 0 (midnight) and 23 (11 PM)",
                    )
                )

            # PARAM_TS_08: peak_end_hour must be in [0, 23]
            if not (0 <= params.peak_end_hour <= 23):
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="PARAM_TS_08",
                        message=f"TimeseriesParams peak_end_hour must be in [0, 23], got: {params.peak_end_hour}",
                        location=location,
                        suggestion="Use a valid hour value between 0 (midnight) and 23 (11 PM)",
                    )
                )

        # seasonality_strength in [0, 1]
        if not (0.0 <= params.seasonality_strength <= 1.0):
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_TS_03",
                    message=f"TimeseriesParams seasonality_strength must be in [0, 1], got: {params.seasonality_strength}",
                    location=location,
                    suggestion="Use a value between 0.0 (no seasonality) and 1.0 (strong seasonality)",
                )
            )

        # noise_level in [0, 1]
        if not (0.0 <= params.noise_level <= 1.0):
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_TS_04",
                    message=f"TimeseriesParams noise_level must be in [0, 1], got: {params.noise_level}",
                    location=location,
                    suggestion="Use a value between 0.0 (no noise) and 1.0 (high noise)",
                )
            )

        # spike_probability in [0, 1]
        if not (0.0 <= params.spike_probability <= 1.0):
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_TS_05",
                    message=f"TimeseriesParams spike_probability must be in [0, 1], got: {params.spike_probability}",
                    location=location,
                    suggestion="Use a value between 0.0 (no spikes) and 1.0 (frequent spikes)",
                )
            )

        # spike_magnitude in [0, 1]
        if not (0.0 <= params.spike_magnitude <= 1.0):
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_TS_06",
                    message=f"TimeseriesParams spike_magnitude must be in [0, 1], got: {params.spike_magnitude}",
                    location=location,
                    suggestion="Use a value between 0.0 (small spikes) and 1.0 (large spikes)",
                )
            )

    def _validate_state_machine_params(self, params: StateMachineParams, location: str):
        """Validate StateMachineParams: states, transitions, context variables."""
        # initial_state in states
        if params.initial_state not in params.states:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_SM_01",
                    message=f"StateMachineParams initial_state '{params.initial_state}' not in states list: {params.states}",
                    location=location,
                    suggestion=f"Add '{params.initial_state}' to states or use one of: {params.states}",
                )
            )

        # terminal_states all in states
        for terminal in params.terminal_states:
            if terminal not in params.states:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="PARAM_SM_02",
                        message=f"StateMachineParams terminal_state '{terminal}' not in states list: {params.states}",
                        location=location,
                        suggestion=f"Add '{terminal}' to states or remove from terminal_states",
                    )
                )

        # Validate each transition
        for idx, trans in enumerate(params.transitions):
            trans_loc = f"{location} > transition {idx}"
            self._validate_transition(
                trans, params.states, params.context_variables, trans_loc
            )

    def _validate_transition(
        self,
        trans: Transition,
        states: list[str],
        context_vars: dict[str, bool],
        location: str,
    ):
        """Validate a single transition."""
        # source in states
        if trans.source not in states:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_SM_03",
                    message=f"Transition source '{trans.source}' not in states list: {states}",
                    location=location,
                    suggestion=f"Add '{trans.source}' to states or change transition source",
                )
            )

        # dest in states
        if trans.dest not in states:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_SM_04",
                    message=f"Transition dest '{trans.dest}' not in states list: {states}",
                    location=location,
                    suggestion=f"Add '{trans.dest}' to states or change transition dest",
                )
            )

        # probability in (0, 1]
        if not (0.0 < trans.probability <= 1.0):
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_SM_05",
                    message=f"Transition probability must be > 0 and <= 1, got: {trans.probability}",
                    location=location,
                    suggestion="Use a probability value like 0.7 or 0.3",
                )
            )

        # conditions reference valid context vars
        for cond in trans.conditions:
            if cond not in context_vars:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="PARAM_SM_06",
                        message=f"Transition condition '{cond}' not defined in context_variables: {list(context_vars.keys())}",
                        location=location,
                        suggestion=f"Add '{cond}' to context_variables or remove from conditions",
                    )
                )

        # context_updates reference valid context vars
        for key in trans.context_updates:
            if key not in context_vars:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="PARAM_SM_07",
                        message=f"Transition context_update key '{key}' not defined in context_variables: {list(context_vars.keys())}",
                        location=location,
                        suggestion=f"Add '{key}' to context_variables or remove from context_updates",
                    )
                )

    # =========================================================================
    # Derivation Parameter Validation
    # =========================================================================

    def _validate_derivation_params(self):
        """Validate all derivation parameters."""
        for entity in self.schema.entities:
            for column in entity.columns:
                if column.derivation is None:
                    continue

                loc = f"entity '{entity.name}' > column '{column.name}'"

                if column.derivation.function_type == DerivationFunctionType.MAP_VALUES:
                    self._validate_map_values_params(column.derivation.params, loc)

                # Check for unsupported cross-category MEASUREMENT dependencies
                self._validate_measurement_dependencies(entity, column, loc)

    def _validate_map_values_params(self, params: MapValuesParams, location: str):
        """Validate MapValuesParams: mapping not empty, rules have from/to."""
        if not params.mapping:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="PARAM_MAP_01",
                    message="MapValuesParams mapping list cannot be empty",
                    location=location,
                    suggestion='Provide mapping rules like [{"from": "active", "to": "high"}]',
                )
            )

        for idx, rule in enumerate(params.mapping):
            if "from" not in rule:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="PARAM_MAP_02",
                        message=f"MapValuesParams mapping rule {idx} missing 'from' key",
                        location=location,
                        suggestion=f"Add 'from' key to rule: {rule}",
                    )
                )
            if "to" not in rule:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="PARAM_MAP_03",
                        message=f"MapValuesParams mapping rule {idx} missing 'to' key",
                        location=location,
                        suggestion=f"Add 'to' key to rule: {rule}",
                    )
                )

    def _validate_measurement_dependencies(
        self, entity: Entity, column: Column, location: str
    ):
        """
        Validate that MEASUREMENT derived columns don't have same-entity MEASUREMENT dependencies.

        This is currently unsupported and will cause a KeyError at runtime because MEASUREMENT
        columns are generated in an arbitrary order when they don't have explicit dependencies
        tracked in the column graph.
        """
        # Only check MEASUREMENT DERIVED columns
        if column.column_category_type != ColumnCategoryType.MEASUREMENT:
            return
        if column.column_type != ColumnType.DERIVED:
            return
        if column.derivation is None:
            return

        # Build a map of column names to their categories in this entity
        entity_columns = {col.name: col for col in entity.columns}

        # Check each dependency
        for dep_col_name in column.derivation.dependent_columns:
            # Skip cross-entity dependencies (they're fine because dependent entity is generated first)
            if "." in dep_col_name:
                continue

            # Check if this is a same-entity dependency
            dep_col = entity_columns.get(dep_col_name)
            if dep_col is None:
                # Dependency doesn't exist in this entity - will be caught by other validation
                continue

            # Check if the dependency is also a MEASUREMENT column
            if dep_col.column_category_type == ColumnCategoryType.MEASUREMENT:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R11",
                        message=f"MEASUREMENT derived column '{column.name}' cannot depend on another MEASUREMENT column '{dep_col_name}' in the same entity (currently unsupported)",
                        location=location,
                        suggestion=f"Change '{dep_col_name}' to column_category_type='metadata', OR restructure to avoid MEASUREMENT->MEASUREMENT dependencies",
                    )
                )

    # =========================================================================
    # Business Logic Validation (R1-R12)
    # =========================================================================

    def _validate_business_rules(self):
        """Validate R1-R12 business logic rules."""
        # R1: If any entity has Timestamp → GlobalTimestamp required
        entities_with_timestamps = [
            e.name for e in self.schema.entities if e.timestamp is not None
        ]
        if entities_with_timestamps and self.schema.global_timestamp is None:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="R1",
                    message=f"Entities {entities_with_timestamps} have timestamp, but global_timestamp is not defined",
                    location="schema root",
                    suggestion="Add global_timestamp configuration with t_start, t_end, and time_interval",
                )
            )

        # R2-R5, R10-R12: Column-level rules (validated per column)
        for entity in self.schema.entities:
            for column in entity.columns:
                loc = f"entity '{entity.name}' > column '{column.name}'"
                self._validate_column_business_rules(column, loc)

            # R6: Entity with Timestamp → Must have ≥1 measurement column
            if entity.timestamp is not None:
                has_measurement = any(
                    col.column_category_type == ColumnCategoryType.MEASUREMENT
                    for col in entity.columns
                )
                if not has_measurement:
                    self.errors.append(
                        ValidationError(
                            level=ValidationLevel.ERROR,
                            rule="R6",
                            message=f"Entity '{entity.name}' has timestamp but no measurement columns",
                            location=f"entity '{entity.name}'",
                            suggestion="Add at least one column with column_category_type='measurement'",
                        )
                    )

        # R7: ONE_TO_ONE relationship → from.cardinality ≤ to.cardinality
        if self.schema.entity_relationships:
            for rel in self.schema.entity_relationships:
                if rel.relationship_type == EntityRelationshipType.ONE_TO_ONE:
                    from_entity = self.entity_map.get(rel.from_entity)
                    to_entity = self.entity_map.get(rel.to_entity)
                    if from_entity and to_entity:
                        if from_entity.cardinality > to_entity.cardinality:
                            self.errors.append(
                                ValidationError(
                                    level=ValidationLevel.ERROR,
                                    rule="R7",
                                    message=f"ONE_TO_ONE relationship from '{rel.from_entity}' to '{rel.to_entity}': child cardinality ({from_entity.cardinality}) cannot exceed parent cardinality ({to_entity.cardinality})",
                                    location=f"relationship {rel.from_entity} -> {rel.to_entity}",
                                    suggestion=f"Either increase '{rel.to_entity}' cardinality or reduce '{rel.from_entity}' cardinality",
                                )
                            )

        # R8: Entity names must be unique
        entity_names = [e.name for e in self.schema.entities]
        duplicates = [
            name for name in set(entity_names) if entity_names.count(name) > 1
        ]
        if duplicates:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="R8",
                    message=f"Duplicate entity names found: {duplicates}",
                    location="schema root",
                    suggestion="Rename entities to have unique names",
                )
            )

        # R9: Column names unique within entity
        for entity in self.schema.entities:
            column_names = [c.name for c in entity.columns]
            duplicates = [
                name for name in set(column_names) if column_names.count(name) > 1
            ]
            if duplicates:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R9",
                        message=f"Duplicate column names in entity '{entity.name}': {duplicates}",
                        location=f"entity '{entity.name}'",
                        suggestion="Rename columns to have unique names within the entity",
                    )
                )

    def _validate_column_business_rules(self, column: Column, location: str):
        """Validate business rules R2-R5, R10-R12 for a single column."""
        # R2: STATEFUL → must be MEASUREMENT
        if column.column_type == ColumnType.STATEFUL:
            if column.column_category_type != ColumnCategoryType.MEASUREMENT:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R2",
                        message=f"STATEFUL column must be MEASUREMENT category, got: {column.column_category_type}",
                        location=location,
                        suggestion="Change column_category_type to 'measurement'",
                    )
                )

        # R3: STATEFUL → domain must be STATE_MACHINE or TIMESERIES
        if column.column_type == ColumnType.STATEFUL:
            if column.domain is None or column.domain.type not in (
                DomainType.STATE_MACHINE,
                DomainType.TIMESERIES,
            ):
                domain_type = column.domain.type if column.domain else "None"
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R3",
                        message=f"STATEFUL column must have STATE_MACHINE or TIMESERIES domain, got: {domain_type}",
                        location=location,
                        suggestion="Use domain.type='timeseries' or domain.type='state_machine'",
                    )
                )

        # R4: INDEPENDENT → domain CANNOT be STATE_MACHINE or TIMESERIES
        if column.column_type == ColumnType.INDEPENDENT:
            if column.domain and column.domain.type in (
                DomainType.STATE_MACHINE,
                DomainType.TIMESERIES,
            ):
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R4",
                        message=f"INDEPENDENT column cannot have temporal domain ({column.domain.type})",
                        location=location,
                        suggestion="Use a non-temporal domain like 'categorical', 'uniform_dist', or 'id'",
                    )
                )

        # R5: FOREIGN_KEY → must be METADATA
        if column.column_type == ColumnType.FOREIGN_KEY:
            if column.column_category_type != ColumnCategoryType.METADATA:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R5",
                        message=f"FOREIGN_KEY column must be METADATA category, got: {column.column_category_type}",
                        location=location,
                        suggestion="Change column_category_type to 'metadata'",
                    )
                )

        # R12: MEASUREMENT columns cannot be INDEPENDENT (currently unsupported)
        if (
            column.column_category_type == ColumnCategoryType.MEASUREMENT
            and column.column_type == ColumnType.INDEPENDENT
        ):
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="R12",
                    message=f"MEASUREMENT column cannot be INDEPENDENT (currently unsupported)",
                    location=location,
                    suggestion="Change column_type to 'stateful' for time-varying measurements, or change column_category_type to 'metadata' for static values",
                )
            )

        # R10: Column has EXACTLY ONE: domain OR derivation OR neither (FK only)
        has_domain = column.domain is not None
        has_derivation = column.derivation is not None

        if column.column_type in (ColumnType.INDEPENDENT, ColumnType.STATEFUL):
            if not has_domain:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R10",
                        message=f"{column.column_type} column must have domain",
                        location=location,
                        suggestion="Add a domain configuration for this column",
                    )
                )
            if has_derivation:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R10",
                        message=f"{column.column_type} column cannot have derivation",
                        location=location,
                        suggestion="Remove the derivation field",
                    )
                )

        elif column.column_type == ColumnType.DERIVED:
            if has_domain:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R10",
                        message="DERIVED column cannot have domain",
                        location=location,
                        suggestion="Remove the domain field",
                    )
                )
            if not has_derivation:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R10",
                        message="DERIVED column must have derivation",
                        location=location,
                        suggestion="Add a derivation configuration for this column",
                    )
                )

        elif column.column_type == ColumnType.FOREIGN_KEY:
            if has_domain or has_derivation:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="R10",
                        message="FOREIGN_KEY column cannot have domain or derivation",
                        location=location,
                        suggestion="Remove domain and derivation fields (they are auto-populated)",
                    )
                )

    # =========================================================================
    # Entity Validation
    # =========================================================================

    def _validate_entities(self):
        """Validate entity-level constraints."""
        for entity in self.schema.entities:
            loc = f"entity '{entity.name}'"

            # Cardinality must be positive
            if entity.cardinality <= 0:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="ENT_01",
                        message=f"Entity cardinality must be positive, got: {entity.cardinality}",
                        location=loc,
                        suggestion="Use a positive integer like cardinality=100",
                    )
                )

            # Must have at least one column
            if not entity.columns:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="ENT_02",
                        message="Entity must have at least one column",
                        location=loc,
                        suggestion="Add column definitions to this entity",
                    )
                )

    # =========================================================================
    # Relationship Validation
    # =========================================================================

    def _validate_relationships(self):
        """Validate entity relationship constraints."""
        if not self.schema.entity_relationships:
            return

        for rel in self.schema.entity_relationships:
            loc = f"relationship {rel.from_entity} -> {rel.to_entity}"

            # join_columns cannot be empty
            if not rel.join_columns:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="REL_01",
                        message="Relationship join_columns cannot be empty",
                        location=loc,
                        suggestion='Add join_columns like {"user_id": "user_id"}',
                    )
                )

            # from_entity ≠ to_entity
            if rel.from_entity == rel.to_entity:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="REL_02",
                        message=f"Relationship cannot be self-referential: '{rel.from_entity}'",
                        location=loc,
                        suggestion="Create relationships between different entities",
                    )
                )

            # from_entity and to_entity must exist
            if rel.from_entity not in self.entity_map:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="REL_03",
                        message=f"Relationship references unknown from_entity: '{rel.from_entity}'",
                        location=loc,
                        suggestion=f"Use one of: {list(self.entity_map.keys())}",
                    )
                )

            if rel.to_entity not in self.entity_map:
                self.errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        rule="REL_04",
                        message=f"Relationship references unknown to_entity: '{rel.to_entity}'",
                        location=loc,
                        suggestion=f"Use one of: {list(self.entity_map.keys())}",
                    )
                )

    # =========================================================================
    # GlobalTimestamp Validation
    # =========================================================================

    def _validate_global_timestamp(self):
        """Validate GlobalTimestamp constraints."""
        if self.schema.global_timestamp is None:
            return

        gt = self.schema.global_timestamp
        loc = "global_timestamp"

        # time_interval format validation (already done in config.py __attrs_post_init__)
        # Just add a reminder check
        import re

        pattern = r"^\d+(min|hour|day|month)$"
        if not re.match(pattern, gt.time_interval):
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="GT_01",
                    message=f"GlobalTimestamp time_interval format invalid: '{gt.time_interval}'",
                    location=loc,
                    suggestion="Use format like '15min', '1hour', '1day', or '3months'",
                )
            )

        # GT_02: t_start < t_end
        if gt.t_start >= gt.t_end:
            self.errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    rule="GT_02",
                    message=f"GlobalTimestamp t_start ({gt.t_start}) must be less than t_end ({gt.t_end})",
                    location=loc,
                    suggestion="Ensure start time is before end time",
                )
            )


# =============================================================================
# Public API
# =============================================================================


def validate_dataschema_comprehensive(
    schema: DataSchema,
) -> list[ValidationError]:
    """
    Run comprehensive validation on a DataSchema.

    This performs validation beyond rf.converter.structure(), checking:
    - Parameter ranges and constraints (30 rules)
    - Business logic rules (R1-R12)
    - Semantic relationships

    Args:
        schema: DataSchema object to validate

    Returns:
        List of ValidationError objects (empty if valid)

    Example:
        >>> errors = validate_dataschema_comprehensive(schema)
        >>> if errors:
        ...     for err in errors:
        ...         print(f"[{err.rule}] {err.message}")
        ... else:
        ...     print("Schema is valid!")
    """
    validator = DataSchemaValidator(schema)
    return validator.validate_all()


def format_validation_errors(errors: list[ValidationError]) -> str:
    """Format validation errors as a readable report."""
    if not errors:
        return "✅ Schema validation passed!"

    report = [f"❌ Found {len(errors)} validation error(s):\n"]

    for idx, err in enumerate(errors, 1):
        report.append(f"{idx}. [{err.level}] {err.rule}: {err.message}")
        report.append(f"   Location: {err.location}")
        if err.suggestion:
            report.append(f"   Suggestion: {err.suggestion}")
        report.append("")

    return "\n".join(report)
