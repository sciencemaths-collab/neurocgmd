"""Tests for Section 3 topology and bead-system models."""

from __future__ import annotations

import unittest

from core.exceptions import ContractValidationError
from core.state import ParticleState
from core.types import BeadId
from topology import Bead, BeadRole, BeadType, Bond, BondKind, SystemTopology


class TopologyTests(unittest.TestCase):
    """Verify static topology alignment and graph-like queries."""

    def _build_topology(self) -> SystemTopology:
        return SystemTopology(
            system_id="toy-system",
            bead_types=(
                BeadType(name="backbone", role=BeadRole.STRUCTURAL),
                BeadType(name="site", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(
                    bead_id=BeadId("b0"),
                    particle_index=0,
                    bead_type="backbone",
                    label="A0",
                    chain_id="A",
                ),
                Bead(
                    bead_id=BeadId("b1"),
                    particle_index=1,
                    bead_type="backbone",
                    label="A1",
                    chain_id="A",
                ),
                Bead(
                    bead_id=BeadId("b2"),
                    particle_index=2,
                    bead_type="site",
                    label="B0",
                    chain_id="B",
                ),
            ),
            bonds=(
                Bond(
                    particle_index_a=0,
                    particle_index_b=1,
                    kind=BondKind.STRUCTURAL,
                    equilibrium_distance=0.47,
                    stiffness=500.0,
                ),
            ),
        )

    def test_topology_builds_neighbors_and_components(self) -> None:
        topology = self._build_topology()

        self.assertEqual(topology.bonded_neighbors(0), (1,))
        self.assertEqual(topology.bonded_neighbors(1), (0,))
        self.assertEqual(topology.bonded_neighbors(2), ())
        self.assertEqual(topology.connected_components(), ((0, 1), (2,)))
        self.assertEqual(topology.bead_by_id(BeadId("b1")).label, "A1")

    def test_topology_rejects_duplicate_bonds(self) -> None:
        with self.assertRaises(ContractValidationError):
            SystemTopology(
                system_id="bad-topology",
                bead_types=(BeadType(name="backbone"),),
                beads=(
                    Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="backbone", label="A0"),
                    Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="backbone", label="A1"),
                ),
                bonds=(
                    Bond(0, 1),
                    Bond(1, 0),
                ),
            )

    def test_topology_rejects_missing_bead_type(self) -> None:
        with self.assertRaises(ContractValidationError):
            SystemTopology(
                system_id="bad-types",
                bead_types=(BeadType(name="backbone"),),
                beads=(
                    Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="site", label="A0"),
                ),
            )

    def test_topology_detects_particle_alignment_mismatch(self) -> None:
        topology = self._build_topology()
        particles = ParticleState(
            positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            masses=(1.0, 1.0),
        )

        self.assertEqual(
            topology.validate_against_particle_state(particles),
            ("ParticleState particle_count does not match SystemTopology bead count.",),
        )

    def test_topology_roundtrip(self) -> None:
        topology = self._build_topology()
        restored = SystemTopology.from_dict(topology.to_dict())
        self.assertEqual(restored, topology)


if __name__ == "__main__":
    unittest.main()
