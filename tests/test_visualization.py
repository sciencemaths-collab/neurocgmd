"""Tests for Section 14 visualization, export, and live-dashboard helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from ai_control import ExecutiveController
from benchmarks import BaselineBenchmarkSuite
from compartments import CompartmentDomain, CompartmentRegistry
from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import SimulationId, StateId
from graph import ConnectivityGraph, DynamicEdgeKind, DynamicEdgeState
from scripts.live_dashboard import build_demo_context, write_dashboard_snapshot
from validation import DriftCheckReport, SanityCheckReport, SanityCheckResult
from visualization import DashboardSnapshotView, GraphSnapshotView


class VisualizationTests(unittest.TestCase):
    """Verify Section 14 view adapters and export-safe dashboard generation."""

    def _build_state(self) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.25, 0.0), (2.0, -0.1, 0.0), (3.0, 0.1, 0.0)),
                masses=(1.0, 1.0, 1.0, 1.0),
                velocities=((0.0, 0.0, 0.0),) * 4,
                labels=("A0", "A1", "B0", "B1"),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-vis"),
                state_id=StateId("state-vis"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.2,
            step=2,
            potential_energy=-0.5,
        )

    def _build_graph(self) -> ConnectivityGraph:
        return ConnectivityGraph(
            particle_count=4,
            step=2,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, 2),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.8, 1.05, 1, 2),
                DynamicEdgeState(2, 3, DynamicEdgeKind.ADAPTIVE_LONG_RANGE, 0.6, 1.3, 1, 2),
            ),
        )

    def _build_compartments(self) -> CompartmentRegistry:
        return CompartmentRegistry(
            particle_count=4,
            domains=(
                CompartmentDomain.from_members("A", "domain-a", (0, 1)),
                CompartmentDomain.from_members("B", "domain-b", (2, 3)),
            ),
        )

    def test_graph_snapshot_view_captures_nodes_edges_and_routes(self) -> None:
        snapshot = GraphSnapshotView.from_state_graph(
            self._build_state(),
            self._build_graph(),
            compartments=self._build_compartments(),
        )

        self.assertEqual(snapshot.node_count, 4)
        self.assertEqual(snapshot.active_edge_count, 3)
        self.assertEqual(snapshot.adaptive_edge_count, 2)
        self.assertEqual(snapshot.nodes[0].compartments, ("A",))
        self.assertTrue(snapshot.edges[1].route_label.startswith("inter:"))

    def test_dashboard_snapshot_renders_live_html_shell(self) -> None:
        state = self._build_state()
        graph = GraphSnapshotView.from_state_graph(
            state,
            self._build_graph(),
            compartments=self._build_compartments(),
        )
        validation = SanityCheckReport(
            state_id=state.provenance.state_id,
            checks=(SanityCheckResult(name="ok", passed=True, message="all good"),),
        )
        drift = DriftCheckReport(
            simulation_id=state.provenance.simulation_id,
            reference_state_id=state.provenance.state_id,
            final_state_id=state.provenance.state_id,
            state_count=1,
            max_energy_drift=0.0,
            max_position_displacement=0.0,
            time_monotonic=True,
            step_sequence_ok=True,
        )
        snapshot = DashboardSnapshotView.from_components(
            title="Diagnostic Test",
            problem=__import__("visualization", fromlist=["ProblemStatementView"]).ProblemStatementView(
                title="Problem Test",
                summary="A concrete dashboard objective is attached.",
                objective="Keep the render path for problem statements alive.",
                stage="Encounter Alignment",
                metrics=(
                    __import__("visualization", fromlist=["ObjectiveMetricView"]).ObjectiveMetricView(
                        label="Assembly Score",
                        value="0.500",
                    ),
                ),
                reference_title="Reference",
                reference_summary="Known target values appear here.",
                reference_metrics=(
                    __import__("visualization", fromlist=["ObjectiveMetricView"]).ObjectiveMetricView(
                        label="Target Kd",
                        value="not estimated yet",
                    ),
                ),
                structure_title="Atomistic Alignment",
                structure_summary="A local atomistic-centroid alignment check is attached.",
                structure_metrics=(
                    __import__("visualization", fromlist=["ObjectiveMetricView"]).ObjectiveMetricView(
                        label="Atomistic Centroid RMSD",
                        value="0.321",
                    ),
                ),
                fidelity_title="Shadow Fidelity",
                fidelity_summary="Baseline and shadow-corrected force errors are compared here.",
                fidelity_metrics=(
                    __import__("visualization", fromlist=["ObjectiveMetricView"]).ObjectiveMetricView(
                        label="Force Rms Error",
                        value="1.200 -> 0.800",
                    ),
                ),
            ),
            state=state,
            graph=graph,
            compartments=self._build_compartments(),
            sanity_report=validation,
            drift_report=drift,
            benchmark_report=BaselineBenchmarkSuite(default_repeats=1).run_foundation_suite(
                state=state,
                topology=__import__("topology").SystemTopology(
                    system_id="vis-system",
                    bead_types=(
                        __import__("topology").BeadType(name="bb"),
                    ),
                    beads=tuple(
                        __import__("topology").Bead(
                            bead_id=f"b{index}",
                            particle_index=index,
                            bead_type="bb",
                            label=f"P{index}",
                        )
                        for index in range(state.particle_count)
                    ),
                ),
                forcefield=__import__("forcefields.base_forcefield", fromlist=["BaseForceField"]).BaseForceField(
                    name="vis-forcefield",
                    nonbonded_parameters=(
                        __import__("forcefields.base_forcefield", fromlist=["NonbondedParameter"]).NonbondedParameter(
                            "bb",
                            "bb",
                            sigma=1.0,
                            epsilon=0.0,
                            cutoff=1.0,
                        ),
                    ),
                ),
                force_evaluator=type(
                    "ZeroForce",
                    (),
                    {
                        "name": "zero_force",
                        "classification": "[test]",
                        "evaluate": lambda self, state, topology, forcefield: __import__(
                            "physics.forces.composite", fromlist=["ForceEvaluation"]
                        ).ForceEvaluation(
                            forces=((0.0, 0.0, 0.0),) * state.particle_count,
                            potential_energy=0.0,
                        ),
                    },
                )(),
                graph_manager=__import__("graph").ConnectivityGraphManager(),
                controller=ExecutiveController(),
                previous_graph=self._build_graph(),
                repeats=1,
            ),
        )

        html = snapshot.render_html(refresh_ms=500)
        self.assertIn("Diagnostic Test", html)
        self.assertIn("dashboard.json", html)
        self.assertIn("problem-summary", html)
        self.assertIn("reference_metrics", snapshot.to_dict()["problem"])
        self.assertIn("structure_metrics", snapshot.to_dict()["problem"])
        self.assertIn("fidelity_metrics", snapshot.to_dict()["problem"])
        self.assertIn("setInterval(refresh, refreshMs)", html)

    def test_export_registry_writes_dashboard_bundle(self) -> None:
        state = self._build_state()
        graph = GraphSnapshotView.from_state_graph(
            state,
            self._build_graph(),
            compartments=self._build_compartments(),
        )
        snapshot = DashboardSnapshotView.from_components(
            title="Export Test",
            state=state,
            graph=graph,
            compartments=self._build_compartments(),
        )

        module_path = Path(__file__).resolve().parents[1] / "io" / "export_registry.py"
        spec = importlib.util.spec_from_file_location("test_export_registry_module", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules.setdefault("test_export_registry_module", module)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = module.export_dashboard_snapshot(snapshot, temp_dir, refresh_ms=750)
            payload = module.load_dashboard_payload(Path(temp_dir) / "dashboard.json")

            self.assertEqual(bundle.metadata["refresh_ms"], 750)
            self.assertEqual(payload["title"], "Export Test")
            self.assertEqual(payload["trajectory"]["state_id"], "state-vis")
            self.assertEqual(payload["graph"]["active_edge_count"], 3)

    def test_live_dashboard_script_writes_demo_snapshot(self) -> None:
        context = build_demo_context()
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = write_dashboard_snapshot(context, temp_dir, refresh_ms=600)
            payload = json.loads((Path(temp_dir) / "dashboard.json").read_text(encoding="utf-8"))

            self.assertTrue((Path(temp_dir) / "index.html").exists())
            self.assertEqual(bundle.metadata["refresh_ms"], 600)
            self.assertEqual(payload["title"], "NeuroCGMD Live Dashboard | Spike-ACE2")
            self.assertEqual(payload["problem"]["title"], "Spike-ACE2")
            self.assertEqual(payload["problem"]["reference_title"], "Experimental Reference")
            self.assertEqual(payload["problem"]["structure_title"], "Atomistic Alignment")
            self.assertEqual(payload["problem"]["fidelity_title"], "Shadow Fidelity")
            self.assertTrue(payload["problem"]["fidelity_metrics"])
            self.assertIn("trajectory", payload)
            self.assertIn("graph", payload)


if __name__ == "__main__":
    unittest.main()
