"""Tests for Section 9 memory traces, replay, and episode grouping."""

from __future__ import annotations

import unittest

from compartments import CompartmentDomain, CompartmentRegistry
from core.state import ParticleState
from core.state_registry import LifecycleStage, SimulationStateRegistry
from core.types import SimulationId, StateId
from graph import ConnectivityGraph, DynamicEdgeKind, DynamicEdgeState
from memory import EpisodeKind, EpisodeRegistry, EpisodeStatus, ReplayBuffer, TraceRecord, TraceStore
from plasticity import PairTraceState


class MemoryLayerTests(unittest.TestCase):
    """Verify long-horizon memory remains explicit, bounded, and registry-aligned."""

    def _build_registry(self) -> tuple[SimulationStateRegistry, object, object, object]:
        registry = SimulationStateRegistry(
            created_by="unit-test",
            simulation_id=SimulationId("sim-memory"),
        )
        initial_state = registry.create_initial_state(
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                masses=(1.0, 1.5, 2.0),
                velocities=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, -0.2, 0.0)),
            ),
            time=0.0,
            step=0,
            potential_energy=-1.5,
            stage=LifecycleStage.INITIALIZATION,
        )
        second_state = registry.derive_state(
            initial_state,
            particles=initial_state.particles.with_positions(
                ((0.0, 0.0, 0.0), (1.1, 0.0, 0.0), (2.1, 0.1, 0.0))
            ),
            time=0.1,
            step=1,
            potential_energy=-1.2,
            stage=LifecycleStage.INTEGRATION,
        )
        third_state = registry.derive_state(
            second_state,
            particles=second_state.particles.with_positions(
                ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (2.2, 0.2, 0.0))
            ),
            time=0.2,
            step=2,
            potential_energy=-0.8,
            stage=LifecycleStage.CHECKPOINT,
        )
        return registry, initial_state, second_state, third_state

    def _build_compartments(self) -> CompartmentRegistry:
        return CompartmentRegistry(
            particle_count=3,
            domains=(
                CompartmentDomain.from_members("A", "domain-a", (0, 1)),
                CompartmentDomain.from_members("B", "domain-b", (2,)),
            ),
        )

    def _build_graph(self, step: int) -> ConnectivityGraph:
        return ConnectivityGraph(
            particle_count=3,
            step=step,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, step),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.6, 0.8, 0, step),
                DynamicEdgeState(0, 2, DynamicEdgeKind.ADAPTIVE_LONG_RANGE, 0.2, 2.1, 0, step, active=False),
            ),
        )

    def test_trace_store_summarizes_registered_state_context(self) -> None:
        registry, initial_state, _, _ = self._build_registry()
        graph = self._build_graph(step=0)
        compartments = self._build_compartments()
        traces = (
            PairTraceState(0, 1, 0.8, 0.7, 2, 0, 1),
            PairTraceState(1, 2, 0.6, 0.6, 1, 0, 1),
        )

        store = TraceStore()
        record = store.append_from_registry(
            registry,
            initial_state.provenance.state_id,
            graph=graph,
            plasticity_traces=traces,
            compartments=compartments,
            tags=("baseline", "checkpoint"),
            metadata={"source": "unit-test"},
        )

        self.assertEqual(store.simulation_id, SimulationId("sim-memory"))
        self.assertEqual(record.parent_state_id, None)
        self.assertEqual(record.active_edge_count, 2)
        self.assertEqual(record.structural_edge_count, 1)
        self.assertEqual(record.adaptive_edge_count, 1)
        self.assertEqual(record.plasticity_trace_count, 2)
        self.assertEqual(record.compartment_ids, ("A", "B"))
        self.assertAlmostEqual(record.total_energy(), initial_state.total_energy())
        self.assertEqual(store.get_record(initial_state.provenance.state_id), record)
        self.assertEqual(store.records_by_tag("baseline"), (record,))
        self.assertEqual(TraceStore.from_dict(store.to_dict()).records(), (record,))

    def test_replay_buffer_enforces_capacity_and_score_order(self) -> None:
        simulation_id = SimulationId("sim-memory")
        record_a = TraceRecord(
            record_id="trace-a",
            simulation_id=simulation_id,
            state_id=StateId("state-a"),
            parent_state_id=None,
            stage="initialization",
            step=0,
            time=0.0,
            particle_count=3,
            kinetic_energy=0.1,
            potential_energy=-1.0,
            tags=("seed",),
        )
        record_b = TraceRecord(
            record_id="trace-b",
            simulation_id=simulation_id,
            state_id=StateId("state-b"),
            parent_state_id=StateId("state-a"),
            stage="integration",
            step=1,
            time=0.1,
            particle_count=3,
            kinetic_energy=0.2,
            potential_energy=-0.8,
            tags=("train",),
        )
        record_c = TraceRecord(
            record_id="trace-c",
            simulation_id=simulation_id,
            state_id=StateId("state-c"),
            parent_state_id=StateId("state-b"),
            stage="checkpoint",
            step=2,
            time=0.2,
            particle_count=3,
            kinetic_energy=0.3,
            potential_energy=-0.6,
            tags=("train", "priority"),
        )

        buffer = ReplayBuffer(capacity=2)
        first_item = buffer.add_from_record(record_a, score=1.0, tags=("warmup",))
        buffer.add_from_record(record_b, score=3.0, tags=("train",))
        latest_item = buffer.add_from_record(record_c, score=2.0, tags=("train",))

        self.assertEqual(len(buffer), 2)
        self.assertEqual(buffer.latest(), (latest_item,))
        self.assertEqual(buffer.highest_score(), (buffer.get_item("replay-000002"),))
        self.assertEqual(tuple(item.state_id for item in buffer.items()), (StateId("state-b"), StateId("state-c")))
        self.assertEqual(buffer.items_by_tag("train"), (latest_item, buffer.get_item("replay-000002")))
        self.assertNotIn(first_item.state_id, tuple(item.state_id for item in buffer.items()))
        self.assertEqual(ReplayBuffer.from_dict(buffer.to_dict()).items(), buffer.items())

    def test_episode_registry_tracks_open_append_close_cycle(self) -> None:
        registry, initial_state, second_state, third_state = self._build_registry()

        episode_registry = EpisodeRegistry()
        trajectory = episode_registry.open_from_registry(
            registry,
            initial_state.provenance.state_id,
            kind=EpisodeKind.TRAJECTORY,
            tags=("main-run",),
        )
        updated = episode_registry.append_state(trajectory.episode_id, second_state)
        closed = episode_registry.close_episode(
            trajectory.episode_id,
            final_state=third_state,
            status=EpisodeStatus.CLOSED,
            metadata_updates={"closed_by": "unit-test"},
        )

        self.assertEqual(updated.state_ids, (initial_state.provenance.state_id, second_state.provenance.state_id))
        self.assertEqual(closed.state_ids[-1], third_state.provenance.state_id)
        self.assertEqual(closed.state_steps, (0, 1, 2))
        self.assertEqual(closed.duration_steps(), 2)
        self.assertEqual(episode_registry.open_episodes(), ())
        self.assertEqual(episode_registry.closed_episodes(), (closed,))
        self.assertEqual(
            episode_registry.episodes_for_state(third_state.provenance.state_id),
            (closed,),
        )
        self.assertEqual(EpisodeRegistry.from_dict(episode_registry.to_dict()).all_episodes(), (closed,))


if __name__ == "__main__":
    unittest.main()
