"""Tests for Section 15 optimization and scaling helpers."""

from __future__ import annotations

import unittest

from optimization import (
    AccelerationBackend,
    BackendRegistry,
    ExecutionProfiler,
    ProfiledOperation,
    ScalingHookManager,
    ScalingWorkload,
    ThresholdScalingHook,
)


class OptimizationLayerTests(unittest.TestCase):
    """Verify Section 15 profiling, backend selection, and scaling hooks."""

    def _build_backend_registry(self) -> BackendRegistry:
        registry = BackendRegistry()
        registry = registry.register(
            AccelerationBackend(
                name="serial_cpu",
                execution_model="serial_cpu",
                supported_components=("physics/forces", "ml"),
                capabilities=("baseline",),
                priority=1,
            )
        )
        registry = registry.register(
            AccelerationBackend(
                name="vectorized_cpu",
                execution_model="vectorized_cpu",
                supported_components=("physics/forces",),
                capabilities=("baseline", "vectorize"),
                priority=3,
            )
        )
        registry = registry.register(
            AccelerationBackend(
                name="gpu_residual",
                execution_model="gpu",
                supported_components=("ml/residual_model.py",),
                capabilities=("baseline", "gpu"),
                priority=2,
            )
        )
        return registry

    def test_execution_profiler_profiles_one_operation_with_warmup(self) -> None:
        profiler = ExecutionProfiler(default_repeats=3, default_warmup_runs=1)
        call_counter = {"count": 0}

        def operation() -> dict[str, int]:
            call_counter["count"] += 1
            return {"count": call_counter["count"]}

        measurement = profiler.profile_operation(
            name="counting_operation",
            operation=operation,
            metadata_factory=lambda result: {"last_count": result["count"] if result is not None else 0},
        )

        self.assertEqual(call_counter["count"], 4)
        self.assertEqual(measurement.repeats(), 3)
        self.assertGreaterEqual(measurement.total_seconds(), 0.0)
        self.assertGreaterEqual(measurement.max_seconds(), measurement.min_seconds())
        self.assertEqual(measurement.metadata["last_count"], 4)

    def test_execution_profiler_builds_multi_operation_report(self) -> None:
        profiler = ExecutionProfiler(default_repeats=2, default_warmup_runs=0)
        report = profiler.profile_report(
            suite_name="unit_profile_suite",
            operations=(
                ProfiledOperation(
                    name="alpha",
                    operation=lambda: "alpha",
                    metadata_factory=lambda result: {"tag": result},
                ),
                ProfiledOperation(
                    name="beta",
                    operation=lambda: "beta",
                    metadata_factory=lambda result: {"tag": result},
                ),
            ),
            metadata={"suite_kind": "unit-test"},
        )

        self.assertEqual(report.measurement_names(), ("alpha", "beta"))
        self.assertEqual(report.measurement_for("beta").metadata["tag"], "beta")
        self.assertEqual(report.metadata["suite_kind"], "unit-test")
        self.assertGreaterEqual(report.total_profiled_seconds(), 0.0)

    def test_backend_registry_selects_highest_priority_matching_backend(self) -> None:
        registry = self._build_backend_registry()

        selection = registry.select_backend(
            "physics/forces/composite.py",
            required_capabilities=("vectorize",),
        )

        self.assertTrue(selection.resolved())
        self.assertEqual(selection.selected_backend, "vectorized_cpu")
        self.assertEqual(selection.unmet_capabilities, ())
        self.assertIn("vectorized_cpu", selection.considered_backends)

    def test_backend_registry_reports_missing_capabilities(self) -> None:
        registry = self._build_backend_registry()

        selection = registry.select_backend(
            "physics/forces/composite.py",
            required_capabilities=("gpu",),
        )

        self.assertFalse(selection.resolved())
        self.assertIn("gpu", selection.unmet_capabilities)
        self.assertIn("none satisfy", selection.rationale.lower())

    def test_backend_registry_honors_preferred_backend_when_valid(self) -> None:
        registry = self._build_backend_registry()

        selection = registry.select_backend(
            "physics/forces/composite.py",
            required_capabilities=("baseline",),
            preferred_backend="serial_cpu",
        )

        self.assertTrue(selection.resolved())
        self.assertEqual(selection.selected_backend, "serial_cpu")
        self.assertIn("preferred backend", selection.rationale.lower())

    def test_scaling_hook_manager_aggregates_scaling_directives(self) -> None:
        registry = self._build_backend_registry()
        backend_selection = registry.select_backend(
            "physics/forces/composite.py",
            required_capabilities=("vectorize",),
        )
        hook = ThresholdScalingHook(
            name="force_scaler",
            component_prefixes=("physics/forces",),
            particle_threshold=100,
            edge_threshold=200,
            parallelism_step=2,
            max_parallelism=8,
            preferred_backend="vectorized_cpu",
        )
        manager = ScalingHookManager(hooks=(hook,))

        plan = manager.evaluate(
            ScalingWorkload(
                target_component="physics/forces/composite.py",
                particle_count=250,
                adaptive_edge_count=220,
                requested_parallelism=1,
            ),
            backend_selection=backend_selection,
        )

        self.assertEqual(plan.recommended_backend, "vectorized_cpu")
        self.assertGreaterEqual(plan.recommended_parallelism, 5)
        self.assertEqual(plan.triggered_hooks, ("force_scaler",))
        self.assertIn("increase_parallelism", {directive.action for directive in plan.directives})
        self.assertIn("chunk_adaptive_edges", {directive.action for directive in plan.directives})
        self.assertIn("prefer_backend", {directive.action for directive in plan.directives})

    def test_scaling_hook_manager_returns_empty_plan_when_no_hook_triggers(self) -> None:
        manager = ScalingHookManager(
            hooks=(
                ThresholdScalingHook(
                    name="replay_scaler",
                    component_prefixes=("ml",),
                    replay_batch_threshold=64,
                    memory_pressure_threshold=0.8,
                ),
            )
        )

        plan = manager.evaluate(
            ScalingWorkload(
                target_component="ml/residual_model.py",
                replay_batch_size=16,
                requested_parallelism=2,
                memory_pressure_fraction=0.2,
            )
        )

        self.assertEqual(plan.recommended_parallelism, 2)
        self.assertEqual(plan.recommended_backend, None)
        self.assertEqual(plan.triggered_hooks, ())
        self.assertEqual(plan.directives, ())


if __name__ == "__main__":
    unittest.main()
