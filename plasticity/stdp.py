"""Spike-timing dependent plasticity (STDP) rules for synaptic weight updates.

STDP is a biological learning rule where the relative timing of pre- and
post-synaptic spikes determines whether a synapse is strengthened (long-term
potentiation) or weakened (long-term depression).  When the pre-synaptic neuron
fires shortly *before* the post-synaptic neuron, the connection is potentiated;
the reverse order causes depression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp

from core.exceptions import ContractValidationError
from core.types import FrozenMetadata
from graph.edge_models import DynamicEdgeState
from graph.graph_manager import ConnectivityGraph
from plasticity.traces import PairTraceState, build_trace_lookup


# ---------------------------------------------------------------------------
# Timing window
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpikeTimingWindow:
    """Parameterise the asymmetric STDP learning window.

    ``tau_plus`` / ``tau_minus`` control the width of the exponential
    potentiation / depression curves.  ``a_plus`` / ``a_minus`` set the
    peak amplitudes.  When ``weight_dependence`` is enabled the rule becomes
    *multiplicative* STDP, which naturally bounds weights and improves
    stability.
    """

    tau_plus: float = 20.0
    tau_minus: float = 20.0
    a_plus: float = 0.1
    a_minus: float = 0.12
    weight_dependence: bool = True

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.tau_plus <= 0.0:
            issues.append("tau_plus must be positive.")
        if self.tau_minus <= 0.0:
            issues.append("tau_minus must be positive.")
        if self.a_plus < 0.0:
            issues.append("a_plus must be non-negative.")
        if self.a_minus < 0.0:
            issues.append("a_minus must be non-negative.")
        return tuple(issues)

    def compute_delta(self, delta_t: float, current_weight: float) -> float:
        """Return the weight change for a given spike-timing difference.

        Parameters
        ----------
        delta_t:
            ``t_pre - t_post`` measured in simulation steps.  Positive means
            the pre-synaptic spike preceded the post-synaptic spike
            (potentiation); negative means the reverse (depression).
        current_weight:
            The current synaptic weight, used when *weight_dependence* is
            enabled.

        Returns
        -------
        float
            The signed weight delta, already clamped so that the resulting
            weight stays within ``[0, 1]``.
        """

        if delta_t == 0.0:
            return 0.0

        if delta_t > 0.0:
            # Pre before post -> potentiation
            delta_w = self.a_plus * exp(-delta_t / self.tau_plus)
            if self.weight_dependence:
                delta_w *= 1.0 - current_weight
        else:
            # Post before pre -> depression
            delta_w = -self.a_minus * exp(delta_t / self.tau_minus)
            if self.weight_dependence:
                delta_w *= current_weight

        # Clamp so the resulting weight remains in [0, 1].
        new_weight = current_weight + delta_w
        if new_weight > 1.0:
            delta_w = 1.0 - current_weight
        elif new_weight < 0.0:
            delta_w = -current_weight

        return delta_w


# ---------------------------------------------------------------------------
# Core STDP rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class STDPRule:
    """Apply pair-based spike-timing dependent plasticity to a connectivity graph.

    Active edges whose endpoints exhibit above-threshold activity (as recorded
    in pair traces) have their weights adjusted according to the asymmetric
    STDP learning window.
    """

    timing_window: SpikeTimingWindow = field(default_factory=SpikeTimingWindow)
    activity_threshold: float = 0.3
    max_updates_per_step: int = 50
    name: str = "stdp_rule"
    classification: str = "[proposed novel]"

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not (0.0 <= self.activity_threshold <= 1.0):
            issues.append("activity_threshold must lie in the interval [0, 1].")
        if self.max_updates_per_step < 0:
            issues.append("max_updates_per_step must be non-negative.")
        issues += list(self.timing_window.validate())
        return tuple(issues)

    def apply(
        self,
        graph: ConnectivityGraph,
        traces: tuple[PairTraceState, ...],
        *,
        current_step: int,
    ) -> ConnectivityGraph:
        """Return a new graph with STDP-adjusted edge weights.

        For every active edge that has a matching pair trace with activity
        above ``activity_threshold``, the rule estimates a spike-timing
        difference from trace metadata and applies the timing window to
        compute a weight update.

        Parameters
        ----------
        graph:
            The current frozen connectivity graph.
        traces:
            Pair-level activity traces for the current simulation snapshot.
        current_step:
            The global simulation step counter.

        Returns
        -------
        ConnectivityGraph
            A new graph snapshot with updated edge weights and metadata.
        """

        trace_lookup = build_trace_lookup(traces)
        updated_edges: list[DynamicEdgeState] = []
        update_count = 0

        for edge in graph.edges:
            if not edge.active:
                updated_edges.append(edge)
                continue

            pair = edge.normalized_pair()
            trace = trace_lookup.get(pair)

            if (
                trace is None
                or trace.activity_level < self.activity_threshold
                or update_count >= self.max_updates_per_step
            ):
                updated_edges.append(edge)
                continue

            # ----------------------------------------------------------
            # Estimate spike-timing difference from trace information.
            #
            # We treat the *source* spike time as approximated by the
            # trace's ``last_seen_step`` minus a fraction of its
            # ``persistence`` (longer persistence ≈ earlier onset).  The
            # *target* spike time is approximated as ``last_seen_step``
            # scaled by the coactivity level.  The resulting ``delta_t``
            # is positive when the source is estimated to have fired
            # first (potentiation) and negative otherwise.
            # ----------------------------------------------------------
            source_spike_time = trace.last_seen_step - trace.persistence * 0.5
            target_spike_time = trace.last_seen_step - trace.persistence * trace.coactivity_level
            delta_t = source_spike_time - target_spike_time

            delta_w = self.timing_window.compute_delta(delta_t, edge.weight)

            if delta_w == 0.0:
                updated_edges.append(edge)
                continue

            new_weight = max(0.01, min(1.0, edge.weight + delta_w))
            updated_edges.append(
                DynamicEdgeState(
                    source_index=edge.source_index,
                    target_index=edge.target_index,
                    kind=edge.kind,
                    weight=new_weight,
                    distance=edge.distance,
                    active=edge.active,
                    created_step=edge.created_step,
                    last_updated_step=current_step,
                    metadata=edge.metadata.with_updates(
                        {
                            "stdp_delta_w": delta_w,
                            "stdp_delta_t": delta_t,
                            "stdp_step": current_step,
                        }
                    ),
                )
            )
            update_count += 1

        return ConnectivityGraph(
            name=graph.name,
            classification=graph.classification,
            particle_count=graph.particle_count,
            step=current_step,
            edges=tuple(updated_edges),
            metadata=graph.metadata.with_updates(
                {
                    "stdp_updates": update_count,
                    "stdp_rule": self.name,
                }
            ),
        )


# ---------------------------------------------------------------------------
# Homeostatic scaling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HomeostaticScaling:
    """Synaptic scaling that nudges mean active-edge weight toward a target.

    This is a global homeostatic mechanism that prevents runaway potentiation
    or depression by uniformly scaling all active weights so that the
    population mean drifts toward ``target_mean_weight`` at a rate governed
    by ``scaling_rate``.

    Classification: [proposed novel]
    """

    target_mean_weight: float = 0.5
    scaling_rate: float = 0.01
    name: str = "homeostatic_scaling"
    classification: str = "[proposed novel]"

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not (0.0 <= self.target_mean_weight <= 1.0):
            issues.append("target_mean_weight must lie in the interval [0, 1].")
        if self.scaling_rate < 0.0:
            issues.append("scaling_rate must be non-negative.")
        return tuple(issues)

    def apply(self, graph: ConnectivityGraph) -> ConnectivityGraph:
        """Return a new graph with homeostatically scaled active-edge weights.

        Parameters
        ----------
        graph:
            The current frozen connectivity graph.

        Returns
        -------
        ConnectivityGraph
            A new graph snapshot whose active-edge weights have been shifted
            toward ``target_mean_weight``.
        """

        active_edges = graph.active_edges()
        if not active_edges:
            return graph

        mean_weight = sum(e.weight for e in active_edges) / len(active_edges)
        adjustment = self.scaling_rate * (self.target_mean_weight - mean_weight)

        updated_edges: list[DynamicEdgeState] = []
        for edge in graph.edges:
            if not edge.active:
                updated_edges.append(edge)
                continue

            new_weight = max(0.01, min(1.0, edge.weight + adjustment))
            updated_edges.append(
                DynamicEdgeState(
                    source_index=edge.source_index,
                    target_index=edge.target_index,
                    kind=edge.kind,
                    weight=new_weight,
                    distance=edge.distance,
                    active=edge.active,
                    created_step=edge.created_step,
                    last_updated_step=edge.last_updated_step,
                    metadata=edge.metadata.with_updates(
                        {
                            "homeostatic_adjustment": adjustment,
                        }
                    ),
                )
            )

        return ConnectivityGraph(
            name=graph.name,
            classification=graph.classification,
            particle_count=graph.particle_count,
            step=graph.step,
            edges=tuple(updated_edges),
            metadata=graph.metadata.with_updates(
                {
                    "homeostatic_mean_before": mean_weight,
                    "homeostatic_adjustment": adjustment,
                    "homeostatic_rule": self.name,
                }
            ),
        )
