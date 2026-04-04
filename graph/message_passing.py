"""GNN-inspired message passing system for adaptive connectivity graphs.

Implements message passing neural network concepts (Gilmer et al. 2017)
applied to the adaptive graph layer, using only the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sqrt, tanh

from core.exceptions import ContractValidationError
from core.state import SimulationState
from core.types import FrozenMetadata, Vector3
from graph.adjacency_utils import build_adjacency_map
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState
from graph.graph_manager import ConnectivityGraph
from topology.system_topology import SystemTopology


# ---------------------------------------------------------------------------
# Helper math utilities (no external deps)
# ---------------------------------------------------------------------------


def _vec3_magnitude(v: Vector3) -> float:
    """Return the Euclidean magnitude of a 3-vector."""
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vec3_distance(a: Vector3, b: Vector3) -> float:
    """Return the Euclidean distance between two 3-vectors."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return sqrt(dx * dx + dy * dy + dz * dz)


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Inner product of two equal-length tuples."""
    total = 0.0
    for ai, bi in zip(a, b, strict=True):
        total += ai * bi
    return total


def _matvec(matrix: list[list[float]], vec: tuple[float, ...]) -> tuple[float, ...]:
    """Matrix-vector product.  *matrix* is row-major ``[rows][cols]``."""
    result: list[float] = []
    for row in matrix:
        total = 0.0
        for rj, vj in zip(row, vec, strict=True):
            total += rj * vj
        result.append(total)
    return tuple(result)


def _tanh_vec(vec: tuple[float, ...]) -> tuple[float, ...]:
    """Element-wise tanh activation."""
    return tuple(tanh(v) for v in vec)


def _add_vecs(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    """Element-wise addition of two equal-length tuples."""
    return tuple(ai + bi for ai, bi in zip(a, b, strict=True))


def _zeros(n: int) -> tuple[float, ...]:
    """Return a zero-vector of length *n*."""
    return tuple(0.0 for _ in range(n))


def _init_weight_matrix(rows: int, cols: int, seed: int = 42) -> list[list[float]]:
    """Deterministic pseudo-random small-value weight initialization.

    Uses a simple linear congruential generator so the result is
    reproducible and requires no external libraries.
    """
    state = seed
    scale = 1.0 / sqrt(cols)
    matrix: list[list[float]] = []
    for _ in range(rows):
        row: list[float] = []
        for _ in range(cols):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            # Map to [-scale, scale]
            val = ((state / 0x7FFFFFFF) * 2.0 - 1.0) * scale
            row.append(val)
        matrix.append(row)
    return matrix


def _edge_kind_encoding(kind: DynamicEdgeKind) -> float:
    """Encode an edge kind as a scalar in [0, 1]."""
    mapping = {
        DynamicEdgeKind.STRUCTURAL_LOCAL: 0.0,
        DynamicEdgeKind.ADAPTIVE_LOCAL: 0.5,
        DynamicEdgeKind.ADAPTIVE_LONG_RANGE: 1.0,
    }
    return mapping.get(kind, 0.0)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NodeFeature:
    """Immutable feature vector associated with a single particle node."""

    particle_index: int
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EdgeMessage:
    """Immutable message vector sent along one directed edge."""

    source_index: int
    target_index: int
    message: tuple[float, ...]


# ---------------------------------------------------------------------------
# MessagePassingLayer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MessagePassingLayer:
    """One round of message-passing on the adaptive connectivity graph.

    Learnable parameters (``message_weights`` and ``update_weights``) are
    initialised with small deterministic pseudo-random values.  All linear
    algebra is implemented with plain Python loops so that no external
    numerical library is required.
    """

    input_dim: int = 8
    message_dim: int = 8
    edge_feature_dim: int = 4
    name: str = "message_passing_layer"
    classification: str = "[proposed novel]"

    # Learnable parameters -- populated in __post_init__
    message_weights: list[list[float]] = field(default_factory=list, repr=False)
    update_weights: list[list[float]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # W_msg: message_dim x (input_dim + edge_feature_dim)
        if not self.message_weights:
            self.message_weights = _init_weight_matrix(
                self.message_dim, self.input_dim + self.edge_feature_dim, seed=42
            )
        # W_update: input_dim x (input_dim + message_dim)
        if not self.update_weights:
            self.update_weights = _init_weight_matrix(
                self.input_dim, self.input_dim + self.message_dim, seed=137
            )

    # -- public API ---------------------------------------------------------

    def compute_messages(
        self,
        node_features: list[NodeFeature],
        graph: ConnectivityGraph,
    ) -> list[EdgeMessage]:
        """Compute a message for every active directed edge.

        For each active edge ``(s, t)`` the message is::

            m_{s->t} = tanh(W_msg @ concat(source_features, edge_features))

        where ``edge_features = [weight, distance, kind_encoding, age]``.
        """
        feature_map: dict[int, tuple[float, ...]] = {
            nf.particle_index: nf.features for nf in node_features
        }
        messages: list[EdgeMessage] = []
        for edge in graph.active_edges():
            src_feat = feature_map.get(edge.source_index, _zeros(self.input_dim))
            edge_feat = self._edge_features(edge, graph.step)
            concat = src_feat + edge_feat  # tuple concatenation
            msg = _tanh_vec(_matvec(self.message_weights, concat))
            messages.append(EdgeMessage(
                source_index=edge.source_index,
                target_index=edge.target_index,
                message=msg,
            ))
            # Also send a message in the reverse direction (undirected graph)
            tgt_feat = feature_map.get(edge.target_index, _zeros(self.input_dim))
            concat_rev = tgt_feat + edge_feat
            msg_rev = _tanh_vec(_matvec(self.message_weights, concat_rev))
            messages.append(EdgeMessage(
                source_index=edge.target_index,
                target_index=edge.source_index,
                message=msg_rev,
            ))
        return messages

    def aggregate_messages(
        self,
        messages: list[EdgeMessage],
        particle_count: int,
    ) -> list[tuple[float, ...]]:
        """Sum incoming messages per target node.

        Returns a list of length ``particle_count`` where each entry is
        the aggregated message vector for that node.
        """
        aggregated: list[tuple[float, ...]] = [
            _zeros(self.message_dim) for _ in range(particle_count)
        ]
        for msg in messages:
            aggregated[msg.target_index] = _add_vecs(
                aggregated[msg.target_index], msg.message
            )
        return aggregated

    def update_nodes(
        self,
        node_features: list[NodeFeature],
        aggregated_messages: list[tuple[float, ...]],
    ) -> list[NodeFeature]:
        """Update node features from the aggregated neighbourhood messages.

        For each node *i*::

            h'_i = tanh(W_update @ concat(h_i, agg_i))
        """
        updated: list[NodeFeature] = []
        for nf in node_features:
            agg = aggregated_messages[nf.particle_index]
            concat = nf.features + agg
            new_feat = _tanh_vec(_matvec(self.update_weights, concat))
            updated.append(NodeFeature(
                particle_index=nf.particle_index,
                features=new_feat,
            ))
        return updated

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _edge_features(edge: DynamicEdgeState, current_step: int) -> tuple[float, ...]:
        """Build the 4-element edge feature vector."""
        age = float(current_step - edge.created_step)
        # Normalise age with a soft saturation
        norm_age = 1.0 - exp(-0.01 * age) if age > 0.0 else 0.0
        return (
            edge.weight,
            edge.distance,
            _edge_kind_encoding(edge.kind),
            norm_age,
        )


# ---------------------------------------------------------------------------
# MessagePassingGraphUpdater
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MessagePassingGraphUpdater:
    """Use multi-round GNN-style message passing to drive adaptive graph updates.

    After running ``layers`` rounds of message passing the updater:

    1. Scores existing edges via dot-product similarity of endpoint features.
    2. Adjusts edge weights toward the computed scores.
    3. Prunes edges whose weight falls below ``edge_pruning_threshold``.
    4. Proposes new adaptive edges between nearby unconnected particles
       whose node-feature similarity exceeds ``edge_creation_threshold``.
    """

    layers: int = 2
    weight_update_rate: float = 0.05
    edge_creation_threshold: float = 0.7
    edge_pruning_threshold: float = 0.1
    max_new_edges_per_step: int = 4
    message_dim: int = 8
    name: str = "gnn_graph_updater"
    classification: str = "[proposed novel]"

    # Internal message-passing stack -- built on first use
    _mp_layers: list[MessagePassingLayer] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self._mp_layers:
            for idx in range(self.layers):
                self._mp_layers.append(
                    MessagePassingLayer(
                        input_dim=self.message_dim,
                        message_dim=self.message_dim,
                        edge_feature_dim=4,
                        name=f"mp_layer_{idx}",
                        # Vary seed per layer so weights differ
                        message_weights=_init_weight_matrix(
                            self.message_dim, self.message_dim + 4, seed=42 + idx * 97
                        ),
                        update_weights=_init_weight_matrix(
                            self.message_dim, self.message_dim * 2, seed=137 + idx * 97
                        ),
                    )
                )

    # -- public API ---------------------------------------------------------

    def update(
        self,
        state: SimulationState,
        topology: SystemTopology,
        graph: ConnectivityGraph,
    ) -> ConnectivityGraph:
        """Run message passing and return an updated ``ConnectivityGraph``.

        Steps:
        1. Extract node features from the current simulation state.
        2. Run *layers* rounds of message passing.
        3. Score each active edge via dot-product similarity of its endpoints.
        4. Update edge weights toward the score.
        5. Prune weak edges.
        6. Propose new adaptive edges between nearby unconnected particles
           whose feature similarity exceeds the creation threshold.
        7. Return the updated graph.
        """
        if graph.particle_count != state.particle_count:
            raise ContractValidationError(
                "Graph particle_count must match SimulationState particle_count."
            )

        # 1 -- extract initial node features
        node_features = self.extract_node_features(state, topology, graph)

        # 2 -- message passing rounds
        for layer in self._mp_layers:
            messages = layer.compute_messages(node_features, graph)
            aggregated = layer.aggregate_messages(messages, graph.particle_count)
            node_features = layer.update_nodes(node_features, aggregated)

        # Build fast feature lookup
        feat_map: dict[int, tuple[float, ...]] = {
            nf.particle_index: nf.features for nf in node_features
        }

        # 3 + 4 -- score and update existing edges
        updated_edges: list[DynamicEdgeState] = []
        existing_pairs: set[tuple[int, int]] = set()

        for edge in graph.edges:
            pair = edge.normalized_pair()
            existing_pairs.add(pair)

            if not edge.active:
                updated_edges.append(edge)
                continue

            src_feat = feat_map.get(edge.source_index, _zeros(self.message_dim))
            tgt_feat = feat_map.get(edge.target_index, _zeros(self.message_dim))
            score = self._cosine_similarity(src_feat, tgt_feat)

            # Blend current weight toward score
            new_weight = edge.weight + self.weight_update_rate * (score - edge.weight)
            new_weight = max(1e-6, min(1.0, new_weight))

            # 5 -- prune
            if new_weight < self.edge_pruning_threshold:
                updated_edges.append(DynamicEdgeState(
                    source_index=edge.source_index,
                    target_index=edge.target_index,
                    kind=edge.kind,
                    weight=new_weight,
                    distance=edge.distance,
                    active=False,
                    created_step=edge.created_step,
                    last_updated_step=state.step,
                    metadata=edge.metadata.with_updates({"pruned_step": state.step}),
                ))
            else:
                updated_edges.append(DynamicEdgeState(
                    source_index=edge.source_index,
                    target_index=edge.target_index,
                    kind=edge.kind,
                    weight=new_weight,
                    distance=edge.distance,
                    active=True,
                    created_step=edge.created_step,
                    last_updated_step=state.step,
                    metadata=edge.metadata,
                ))

        # 6 -- propose new adaptive edges between nearby unconnected pairs
        new_edges = self._propose_new_edges(
            state, feat_map, existing_pairs, graph.particle_count,
        )
        updated_edges.extend(new_edges)

        # 7 -- assemble the new graph
        metadata = graph.metadata.with_updates({
            "updater": self.name,
            "mp_layers": self.layers,
            "edges_proposed": len(new_edges),
        })
        return ConnectivityGraph(
            name=graph.name,
            classification=graph.classification,
            particle_count=graph.particle_count,
            step=state.step,
            edges=tuple(updated_edges),
            metadata=metadata,
        )

    def extract_node_features(
        self,
        state: SimulationState,
        topology: SystemTopology,
        graph: ConnectivityGraph,
    ) -> list[NodeFeature]:
        """Build a ``NodeFeature`` for every particle from the simulation state.

        The feature vector (length ``message_dim``, padded/truncated to fit)
        encodes:

        * position components (x, y, z)
        * velocity magnitude
        * force magnitude
        * mass
        * degree (number of active neighbours)
        * bead-type hash (simple numeric encoding of the topology bead type)
        """
        adjacency = build_adjacency_map(
            graph.particle_count, graph.edges, active_only=True
        )
        bead_type_map: dict[int, str] = {}
        for bead in topology.beads:
            bead_type_map[bead.particle_index] = bead.bead_type

        features_list: list[NodeFeature] = []
        for idx in range(state.particle_count):
            pos = state.particles.positions[idx]
            vel = state.particles.velocities[idx]
            force = state.particles.forces[idx]
            mass = state.particles.masses[idx]
            degree = float(len(adjacency.get(idx, ())))
            vel_mag = _vec3_magnitude(vel)
            force_mag = _vec3_magnitude(force)

            # Encode bead type as a deterministic hash normalised into [0, 1]
            btype = bead_type_map.get(idx, "")
            # Deterministic: sum of ordinals (stable across Python runs)
            type_hash = float(sum(ord(c) for c in btype) % 1000) / 1000.0

            raw = (
                pos[0], pos[1], pos[2],
                vel_mag,
                force_mag,
                mass,
                degree,
                type_hash,
            )

            # Pad or truncate to message_dim
            if len(raw) < self.message_dim:
                raw = raw + _zeros(self.message_dim - len(raw))
            elif len(raw) > self.message_dim:
                raw = raw[: self.message_dim]

            features_list.append(NodeFeature(
                particle_index=idx,
                features=raw,
            ))
        return features_list

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        """Cosine similarity clamped to [0, 1]."""
        dot_val = _dot(a, b)
        norm_a = sqrt(_dot(a, a))
        norm_b = sqrt(_dot(b, b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        sim = dot_val / (norm_a * norm_b)
        # Clamp to [0, 1] -- negative similarity means repulsion
        return max(0.0, min(1.0, sim))

    def _propose_new_edges(
        self,
        state: SimulationState,
        feat_map: dict[int, tuple[float, ...]],
        existing_pairs: set[tuple[int, int]],
        particle_count: int,
    ) -> list[DynamicEdgeState]:
        """Propose new adaptive edges between nearby unconnected particles.

        Candidates are generated from all pairs within a local distance
        heuristic (based on the mean edge distance in the graph, or a
        default cutoff).  Only the top ``max_new_edges_per_step`` highest-
        scoring candidates that exceed ``edge_creation_threshold`` are kept.
        """
        # Determine a distance cutoff: use 2x the mean existing edge distance
        # or fall back to 2.0 if there are no edges yet.
        total_dist = 0.0
        count = 0
        for pair in existing_pairs:
            i, j = pair
            d = _vec3_distance(
                state.particles.positions[i],
                state.particles.positions[j],
            )
            total_dist += d
            count += 1
        distance_cutoff = (2.0 * total_dist / count) if count > 0 else 2.0

        candidates: list[tuple[float, int, int]] = []
        for i in range(particle_count):
            for j in range(i + 1, particle_count):
                pair = (i, j)
                if pair in existing_pairs:
                    continue
                dist = _vec3_distance(
                    state.particles.positions[i],
                    state.particles.positions[j],
                )
                if dist > distance_cutoff:
                    continue
                feat_i = feat_map.get(i, _zeros(self.message_dim))
                feat_j = feat_map.get(j, _zeros(self.message_dim))
                sim = self._cosine_similarity(feat_i, feat_j)
                if sim > self.edge_creation_threshold:
                    candidates.append((sim, i, j))

        # Sort descending by similarity and take the top candidates
        candidates.sort(key=lambda c: c[0], reverse=True)
        new_edges: list[DynamicEdgeState] = []
        for sim, i, j in candidates[: self.max_new_edges_per_step]:
            dist = _vec3_distance(
                state.particles.positions[i],
                state.particles.positions[j],
            )
            # Map similarity to initial weight in (0, 1]
            init_weight = max(1e-6, min(1.0, sim))
            new_edges.append(DynamicEdgeState(
                source_index=i,
                target_index=j,
                kind=DynamicEdgeKind.ADAPTIVE_LOCAL,
                weight=init_weight,
                distance=dist,
                active=True,
                created_step=state.step,
                last_updated_step=state.step,
                metadata=FrozenMetadata({
                    "origin": "message_passing",
                    "initial_similarity": sim,
                }),
            ))
        return new_edges
