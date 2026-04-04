"""Tests for the integrated hybrid production engine."""

from __future__ import annotations

import unittest

from scripts.live_dashboard import build_live_dashboard_context


class HybridProductionEngineTests(unittest.TestCase):
    """Keep the coordinated production runtime honest and connected."""

    def test_collect_cycle_uses_interconnected_runtime_layers(self) -> None:
        context = build_live_dashboard_context("barnase_barstar")

        cycle = context.engine.collect_cycle(benchmark_repeats=1)

        self.assertEqual(cycle.state.provenance.state_id, context.registry.latest_state().provenance.state_id)
        self.assertGreater(len(cycle.graph.active_edges()), 0)
        self.assertIsNotNone(cycle.chemistry_report)
        self.assertIsNotNone(cycle.qcloud_result)
        self.assertIsNotNone(cycle.benchmark_report)
        self.assertTrue(cycle.sanity_report.passed())
        self.assertGreaterEqual(len(context.trace_store), 1)
        self.assertGreaterEqual(len(context.replay_buffer), 1)
        self.assertGreaterEqual(len(context.engine.episode_registry.open_episodes()), 1)
        self.assertTrue(bool(cycle.final_decision.actions))

    def test_advance_moves_registry_and_collects_new_registered_state(self) -> None:
        context = build_live_dashboard_context("barnase_barstar")
        starting_step = context.registry.latest_state().step

        context.engine.advance(steps=1, record_final_state=False)
        self.assertEqual(context.registry.latest_state().step, starting_step + 1)

        cycle = context.engine.collect_cycle(benchmark_repeats=1)

        self.assertEqual(cycle.state.step, starting_step + 1)
        self.assertGreaterEqual(len(context.trace_store), 2)
        self.assertTrue(cycle.sanity_report.passed())


if __name__ == "__main__":
    unittest.main()
