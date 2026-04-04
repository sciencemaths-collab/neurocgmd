"""Import-safe PDB parsing helpers for local structural-reference workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.types import FrozenMetadata, Vector3, coerce_scalar, coerce_vector3


@dataclass(frozen=True, slots=True)
class PDBAtomRecord(ValidatableComponent):
    """One parsed ATOM/HETATM record from a PDB file."""

    record_type: str
    atom_serial: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_sequence: int
    coordinates: Vector3
    element: str = ""
    occupancy: float = 1.0
    b_factor: float = 0.0
    alt_loc: str = ""
    insertion_code: str = ""
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", coerce_vector3(self.coordinates, "coordinates"))
        object.__setattr__(self, "occupancy", coerce_scalar(self.occupancy, "occupancy"))
        object.__setattr__(self, "b_factor", coerce_scalar(self.b_factor, "b_factor"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.record_type not in {"ATOM", "HETATM"}:
            issues.append("record_type must be ATOM or HETATM.")
        if self.atom_serial <= 0:
            issues.append("atom_serial must be strictly positive.")
        if not self.atom_name.strip():
            issues.append("atom_name must be a non-empty string.")
        if not self.residue_name.strip():
            issues.append("residue_name must be a non-empty string.")
        if not self.chain_id.strip():
            issues.append("chain_id must be a non-empty string.")
        if self.occupancy < 0.0:
            issues.append("occupancy must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "atom_serial": self.atom_serial,
            "atom_name": self.atom_name,
            "residue_name": self.residue_name,
            "chain_id": self.chain_id,
            "residue_sequence": self.residue_sequence,
            "coordinates": list(self.coordinates),
            "element": self.element,
            "occupancy": self.occupancy,
            "b_factor": self.b_factor,
            "alt_loc": self.alt_loc,
            "insertion_code": self.insertion_code,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PDBStructure(ValidatableComponent):
    """Parsed local PDB structure contents."""

    structure_id: str
    atoms: tuple[PDBAtomRecord, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.structure_id.strip():
            issues.append("structure_id must be a non-empty string.")
        if not self.atoms:
            issues.append("atoms must contain at least one parsed atom record.")
        return tuple(issues)

    def atoms_for_chain(self, chain_id: str) -> tuple[PDBAtomRecord, ...]:
        """Return atoms belonging to one chain identifier."""

        return tuple(atom for atom in self.atoms if atom.chain_id == chain_id)

    def first_atom(
        self,
        *,
        chain_id: str,
        residue_sequence: int,
        atom_name: str,
    ) -> PDBAtomRecord:
        """Return the first atom matching the requested location."""

        for atom in self.atoms:
            if (
                atom.chain_id == chain_id
                and atom.residue_sequence == residue_sequence
                and atom.atom_name.strip() == atom_name.strip()
            ):
                return atom
        raise KeyError((chain_id, residue_sequence, atom_name))


def _slice(line: str, start: int, stop: int) -> str:
    return line[start:stop].strip()


def parse_pdb_text(text: str, *, structure_id: str = "pdb_text") -> PDBStructure:
    """Parse ATOM/HETATM records from raw PDB text."""

    atoms: list[PDBAtomRecord] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.startswith(("ATOM", "HETATM")):
            continue
        line = raw_line.rstrip("\n")
        try:
            atom = PDBAtomRecord(
                record_type=_slice(line, 0, 6),
                atom_serial=int(_slice(line, 6, 11)),
                atom_name=_slice(line, 12, 16),
                alt_loc=_slice(line, 16, 17),
                residue_name=_slice(line, 17, 20),
                chain_id=_slice(line, 21, 22) or "?",
                residue_sequence=int(_slice(line, 22, 26)),
                insertion_code=_slice(line, 26, 27),
                coordinates=(
                    float(_slice(line, 30, 38)),
                    float(_slice(line, 38, 46)),
                    float(_slice(line, 46, 54)),
                ),
                occupancy=float(_slice(line, 54, 60) or 1.0),
                b_factor=float(_slice(line, 60, 66) or 0.0),
                element=_slice(line, 76, 78),
                metadata={"line_number": line_number},
            )
        except ValueError as exc:
            raise ContractValidationError(f"Unable to parse PDB atom line {line_number}: {raw_line!r}") from exc
        atoms.append(atom)
    return PDBStructure(
        structure_id=structure_id,
        atoms=tuple(atoms),
        metadata={"record_count": len(atoms)},
    )


def load_pdb_file(path: str | Path, *, structure_id: str | None = None) -> PDBStructure:
    """Load and parse a local PDB file from disk."""

    resolved = Path(path).expanduser().resolve()
    return parse_pdb_text(
        resolved.read_text(encoding="utf-8"),
        structure_id=structure_id or resolved.stem,
    )
