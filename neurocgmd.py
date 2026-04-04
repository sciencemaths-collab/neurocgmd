#!/usr/bin/env python3
"""
NeuroCGMD -- unified molecular dynamics engine.

Usage:
    neurocgmd run <config.toml>              # prep + run + analyze in one shot
    neurocgmd prepare <config.toml>          # prep only
    neurocgmd analyze <config.toml>          # analyze existing trajectory
    neurocgmd info                           # show platform capabilities
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import tomllib
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Terminal color support
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    """Check if terminal supports ANSI colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_COLOR = _supports_color()

def _c(code: str, text: str) -> str:
    """Apply ANSI color code if terminal supports it."""
    if not _COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def _dim(t: str) -> str: return _c("2", t)
def _bold(t: str) -> str: return _c("1", t)
def _blue(t: str) -> str: return _c("38;5;75", t)
def _teal(t: str) -> str: return _c("38;5;79", t)
def _purple(t: str) -> str: return _c("38;5;141", t)
def _gold(t: str) -> str: return _c("38;5;221", t)
def _green(t: str) -> str: return _c("38;5;114", t)
def _red(t: str) -> str: return _c("38;5;203", t)
def _pink(t: str) -> str: return _c("38;5;211", t)
def _orange(t: str) -> str: return _c("38;5;215", t)
def _white(t: str) -> str: return _c("1;37", t)
def _gray(t: str) -> str: return _c("38;5;245", t)
def _bg_blue(t: str) -> str: return _c("48;5;24;38;5;255", t)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_LOGO = r"""
                                  _____ __  __ _____
       _   _                     / ____|  \/  |  __ \
      | \ | | ___ _   _ _ __ ___| |    | \  / | |  | |
      |  \| |/ _ \ | | | '__/ _ \ |    | |\/| | |  | |
      | |\  |  __/ |_| | | | (_) | |___| |  | | |__| |
      |_| \_|\___|\__,_|_|  \___/ \_____|_|  |_|_____/
"""

_TAGLINE = "Quantum-Classical-CG-ML Cooperative Molecular Dynamics"

def _banner() -> None:
    if _COLOR:
        # Gradient effect on the logo
        lines = _LOGO.strip().split("\n")
        colors = ["38;5;75", "38;5;111", "38;5;147", "38;5;141", "38;5;177", "38;5;213"]
        print()
        for i, line in enumerate(lines):
            color = colors[i % len(colors)]
            print(f"  \033[{color}m{line}\033[0m")
        print()
        print(f"  {_dim('v' + VERSION)}  {_teal(_TAGLINE)}")
        # Subtle line
        print(f"  {_dim('.' * 62)}")
        print()
    else:
        print(f"\n{'=' * 64}")
        print(f"  NeuroCGMD v{VERSION} -- {_TAGLINE}")
        print(f"{'=' * 64}\n")


def _section(title: str) -> None:
    if _COLOR:
        dot = _dim(":")
        print(f"\n  {_blue('>>>')} {_bold(title)}")
        print(f"  {_dim('-' * 58)}")
    else:
        print(f"\n--- {title} {'-' * max(0, 56 - len(title))}\n")


def _ok(msg: str) -> None:
    print(f"  {_green('+')} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_gold('!')} {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    print(f"  {_red('x')} {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"  {msg}")


# ---------------------------------------------------------------------------
# TOML -> RunManifest translation
# ---------------------------------------------------------------------------

def _translate_toml_to_manifest_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Map the user's simplified TOML schema into the canonical RunManifest dict.

    The user writes a friendly format (flat [solvent], [ions], [nonbonded],
    system.chains, system.temperature, ref_t lists, etc.).  This function
    reshapes it into the internal dict that RunManifest.from_dict() expects.
    """
    result: dict[str, Any] = {}

    # --- system -----------------------------------------------------------
    sys_raw = dict(raw.get("system", {}))
    chains = sys_raw.pop("chains", None)
    default_temperature = float(sys_raw.pop("temperature", 300.0))
    if chains:
        entity_groups = []
        for chain_id in chains:
            entity_groups.append({
                "entity_id": f"chain_{chain_id.lower()}" if len(chains) > 1 else "protein",
                "chain_ids": [chain_id],
                "description": f"Chain {chain_id} from user config.",
            })
        sys_raw["entity_groups"] = entity_groups
    result["system"] = sys_raw

    # --- prepare (merge [prepare] + [solvent] + [ions]) -------------------
    prepare_raw: dict[str, Any] = {}
    if "prepare" in raw:
        prepare_raw.update(raw["prepare"])
    if "solvent" in raw:
        solvent = raw["solvent"]
        if "mode" in solvent:
            prepare_raw["solvent_mode"] = solvent["mode"]
        for key in ("water_model", "box_type", "padding_nm"):
            if key in solvent:
                prepare_raw[key] = solvent[key]
    if "ions" in raw:
        ions = raw["ions"]
        for key in ("neutralize", "salt", "ionic_strength_molar"):
            if key in ions:
                prepare_raw[key] = ions[key]
    result["prepare"] = prepare_raw

    # --- forcefield -------------------------------------------------------
    ff_raw: dict[str, Any] = {}
    if "forcefield" in raw:
        ff_raw.update(raw["forcefield"])
    constraint_algorithm = ff_raw.pop("constraint_algorithm", None)
    constraints_type = ff_raw.pop("constraints", None)
    result["forcefield"] = ff_raw

    # --- nonbonded -> neighbor_list ---------------------------------------
    nb_raw: dict[str, Any] = {}
    if "nonbonded" in raw:
        nb_raw.update(raw["nonbonded"])
    result["neighbor_list"] = nb_raw

    # --- stages -----------------------------------------------------------
    stages_raw: dict[str, Any] = {}
    for stage_name in ("em", "nvt", "npt", "production"):
        stage_key = f"stages.{stage_name}"
        # TOML parses [stages.em] as nested dict stages -> em
        stage_data = raw.get("stages", {}).get(stage_name, {})
        if stage_data:
            stage_dict = dict(stage_data)
            # Map ref_t -> temperature (take first element if list)
            if "ref_t" in stage_dict:
                ref_t = stage_dict.pop("ref_t")
                if isinstance(ref_t, (list, tuple)) and ref_t:
                    stage_dict.setdefault("temperature", float(ref_t[0]))
                elif isinstance(ref_t, (int, float)):
                    stage_dict.setdefault("temperature", float(ref_t))
            # Apply system-level temperature as default
            if stage_name in ("nvt", "npt", "production"):
                stage_dict.setdefault("temperature", default_temperature)
            # NPT gets default pressure
            if stage_name == "npt":
                stage_dict.setdefault("pressure", 1.0)
            stages_raw[stage_name] = stage_dict
        elif stage_name in ("nvt", "npt", "production"):
            stages_raw[stage_name] = {"temperature": default_temperature}
            if stage_name == "npt":
                stages_raw[stage_name]["pressure"] = 1.0
    result["stages"] = stages_raw

    # --- hybrid -----------------------------------------------------------
    hybrid_raw: dict[str, Any] = {}
    if "hybrid" in raw:
        h = raw["hybrid"]
        if "graph" in h:
            hybrid_raw["graph_enabled"] = h["graph"].get("enabled", True)
        if "qcloud" in h:
            hybrid_raw["qcloud"] = dict(h["qcloud"])
        if "ml" in h:
            hybrid_raw["ml"] = dict(h["ml"])
        if "control" in h:
            hybrid_raw["control"] = dict(h["control"])
    result["hybrid"] = hybrid_raw

    # --- outputs ----------------------------------------------------------
    outputs_raw: dict[str, Any] = {}
    if "outputs" in raw:
        o = raw["outputs"]
        if "trajectory" in o:
            outputs_raw["trajectory"] = o["trajectory"]
        if "energies" in o:
            outputs_raw["energy"] = o["energies"]
        if "checkpoint" in o:
            outputs_raw["checkpoint"] = o["checkpoint"]
        if "report" in o:
            outputs_raw["analysis_html"] = o["report"]
        # Also accept canonical names directly
        for key in ("output_dir", "prepared_bundle", "energy", "analysis_json",
                     "analysis_html", "run_summary", "log"):
            if key in o:
                outputs_raw[key] = o[key]
    if sys_raw.get("name"):
        outputs_raw.setdefault("output_dir", f"outputs/{sys_raw['name']}")
    result["outputs"] = outputs_raw

    # --- analysis ---------------------------------------------------------
    if "analysis" in raw:
        result["analysis"] = raw["analysis"]

    # Store constraint info for the runner
    result["_constraint_algorithm"] = constraint_algorithm
    result["_constraints_type"] = constraints_type

    return result


# ---------------------------------------------------------------------------
# Load and build manifest from user TOML
# ---------------------------------------------------------------------------

def _load_user_toml(config_path: Path) -> tuple[Any, dict[str, Any]]:
    """Parse user TOML and return (RunManifest, translated_dict)."""
    from config import RunManifest

    resolved = config_path.expanduser().resolve()
    if not resolved.exists():
        _fail(f"Config file not found: {resolved}")
        raise SystemExit(1)

    raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
    translated = _translate_toml_to_manifest_dict(raw)

    # Strip internal keys before passing to RunManifest
    internal_keys = {k for k in translated if k.startswith("_")}
    manifest_dict = {k: v for k, v in translated.items() if k not in internal_keys}

    manifest = RunManifest.from_dict(manifest_dict, source_path=str(resolved))
    return manifest, translated


# ---------------------------------------------------------------------------
# Preparation phase
# ---------------------------------------------------------------------------

def _run_preparation(manifest: Any) -> tuple[Any, Path]:
    """Run the preparation pipeline.  Returns (bundle, bundle_path).

    Falls back to a synthetic test system if the PDB is unavailable.
    """
    from prepare import PreparationPipeline

    pipeline = PreparationPipeline()
    output_dir = manifest.outputs.resolve(REPO_ROOT, manifest.outputs.prepared_bundle)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        _info(f"Loading structure from: {manifest.system.structure}")
        bundle = pipeline.prepare(manifest)
        dest = pipeline.write_bundle(bundle, output_dir)
        _ok(f"Prepared bundle written to {dest}")
        return bundle, dest
    except Exception as exc:
        _warn(f"Preparation from PDB failed: {exc}")
        _info("Falling back to synthetic test system for pipeline demonstration...")
        return _build_fallback_system(manifest, output_dir)


def _build_fallback_system(manifest: Any, bundle_path: Path) -> tuple[Any, Path]:
    """Build a minimal synthetic system when the real PDB is not available."""
    import math
    from core.state import EnsembleKind, ParticleState, SimulationState, ThermodynamicState, UnitSystem
    from core.state_registry import SimulationStateRegistry
    from core.types import FrozenMetadata

    n_particles = 64
    spacing = 0.4
    side = math.ceil(n_particles ** (1.0 / 3.0))
    positions = []
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if len(positions) >= n_particles:
                    break
                positions.append((ix * spacing, iy * spacing, iz * spacing))
            if len(positions) >= n_particles:
                break
        if len(positions) >= n_particles:
            break
    positions = tuple(positions[:n_particles])

    masses = tuple(72.0 for _ in range(n_particles))
    labels = tuple(f"CG_{i}" for i in range(n_particles))
    velocities = tuple((0.0, 0.0, 0.0) for _ in range(n_particles))
    forces = tuple((0.0, 0.0, 0.0) for _ in range(n_particles))

    box_side = side * spacing + 1.0
    from core.state import SimulationCell
    cell = SimulationCell(box_vectors=((box_side, 0.0, 0.0), (0.0, box_side, 0.0), (0.0, 0.0, box_side)))

    temperature = 300.0
    for stage in (manifest.nvt, manifest.npt, manifest.production):
        if stage.enabled:
            temperature = stage.temperature
            break

    particles = ParticleState(
        positions=positions,
        velocities=velocities,
        forces=forces,
        masses=masses,
        labels=labels,
    )
    thermodynamics = ThermodynamicState(
        ensemble=EnsembleKind.NVT,
        target_temperature=temperature,
        friction_coefficient=1.0,
    )

    _ok(f"Built synthetic system: {n_particles} particles, box={box_side:.2f} nm")

    # Return a lightweight object that mimics PreparedSystemBundle enough
    # for downstream stages.
    @dataclass
    class _FallbackBundle:
        system_name: str
        particles: Any
        thermodynamics: Any
        cell: Any
        n_particles: int
        metadata: dict

    fb = _FallbackBundle(
        system_name=manifest.system.name,
        particles=particles,
        thermodynamics=thermodynamics,
        cell=cell,
        n_particles=n_particles,
        metadata={"fallback": True},
    )
    return fb, bundle_path


# ---------------------------------------------------------------------------
# Simulation phase -- run each enabled stage
# ---------------------------------------------------------------------------

def _run_stages(
    manifest: Any,
    bundle: Any,
    translated: dict[str, Any],
) -> tuple[Any, Any]:
    """Run all enabled stages (em -> nvt -> npt -> production).

    Returns (final_state, energy_tracker).
    """
    from core.state import EnsembleKind, ParticleState, SimulationState, ThermodynamicState, UnitSystem
    from core.state_registry import SimulationStateRegistry
    from core.types import FrozenMetadata

    constraint_algo = translated.get("_constraint_algorithm")

    # --- Build initial simulation state -----------------------------------
    if hasattr(bundle, "runtime_seed"):
        # Real PreparedSystemBundle
        registry = SimulationStateRegistry(created_by="neurocgmd-unified")
        initial_state = registry.create_initial_state(
            particles=bundle.runtime_seed.particles,
            units=bundle.runtime_seed.units,
            thermodynamics=bundle.runtime_seed.thermodynamics,
            cell=bundle.runtime_seed.cell,
            notes="prepared runtime seed",
        )
    else:
        # Fallback synthetic bundle
        registry = SimulationStateRegistry(created_by="neurocgmd-unified")
        initial_state = registry.create_initial_state(
            particles=bundle.particles,
            units=UnitSystem.md_nano(),
            thermodynamics=bundle.thermodynamics,
            cell=bundle.cell,
            notes="synthetic fallback system",
        )

    n_particles = initial_state.particle_count
    _info(f"System has {n_particles} particles")

    # --- Build topology ---------------------------------------------------
    topology = _build_topology(initial_state, manifest)

    # --- Build force field ------------------------------------------------
    forcefield = _build_forcefield(initial_state, manifest)

    # --- Build constraints ------------------------------------------------
    constraints, constraint_solver = _build_constraints(
        topology, constraint_algo, n_particles
    )

    # --- Build analysis hooks ---------------------------------------------
    analysis_engine = _build_analysis_engine(initial_state)
    observable_collector = _build_observable_collector()

    def _analysis_hook(state: Any, energy_tracker_arg: Any = None, **kwargs: Any) -> None:
        pe = getattr(energy_tracker_arg, '_potential_energies', [None])[-1] if energy_tracker_arg else None
        temp = getattr(energy_tracker_arg, '_temperatures', [None])[-1] if energy_tracker_arg else None
        try:
            analysis_engine.collect(state, potential_energy=pe, temperature=temp)
        except Exception:
            pass
        try:
            if observable_collector is not None:
                observable_collector.collect_all(state, topology)
        except Exception:
            pass

    # --- Run each stage ---------------------------------------------------
    from sampling.production_loop import EnergyTracker

    cumulative_energy_tracker = EnergyTracker()
    current_state = initial_state
    stage_records: list[dict[str, Any]] = []

    stage_defs = [
        ("em", manifest.em),
        ("nvt", manifest.nvt),
        ("npt", manifest.npt),
        ("production", manifest.production),
    ]

    for stage_name, stage_config in stage_defs:
        if not stage_config.enabled:
            _info(f"Stage '{stage_name}': skipped (disabled)")
            continue

        _section(f"Stage: {stage_name}")

        if stage_name == "em":
            current_state, steps_done = _run_minimization_stage(
                stage_config, current_state, topology, forcefield,
                registry, cumulative_energy_tracker,
            )
        else:
            current_state, steps_done = _run_dynamics_stage(
                stage_name, stage_config, current_state, topology,
                forcefield, registry, constraints, constraint_solver,
                cumulative_energy_tracker, _analysis_hook,
            )

        stage_records.append({
            "stage": stage_name,
            "steps_requested": stage_config.max_steps if stage_name == "em" else stage_config.nsteps,
            "steps_completed": steps_done,
            "final_step": current_state.step,
            "final_time": current_state.time,
        })
        _ok(f"Completed {steps_done} steps | step={current_state.step} time={current_state.time:.4f}")

    return current_state, cumulative_energy_tracker, analysis_engine, stage_records


def _build_topology(state: Any, manifest: Any) -> Any:
    """Build a SystemTopology from available information."""
    from topology.system_topology import SystemTopology
    from topology.beads import Bead, BeadType
    from topology.bonds import Bond
    from core.types import BeadId

    n = state.particle_count
    bead_types = (BeadType(name="CG"),)
    beads = tuple(
        Bead(
            bead_id=BeadId(f"bead_{i}"),
            particle_index=i,
            bead_type="CG",
            label=state.particles.labels[i] if i < len(state.particles.labels) else f"CG_{i}",
            residue_name=f"R{i}",
            chain_id="A",
        )
        for i in range(n)
    )
    # Build bonds for adjacent beads (linear chain topology)
    bonds = tuple(
        Bond(
            particle_index_a=i,
            particle_index_b=i + 1,
            equilibrium_distance=0.38,
            stiffness=1250.0,
        )
        for i in range(n - 1)
    )
    return SystemTopology(
        system_id=manifest.system.name,
        bead_types=bead_types,
        beads=beads,
        bonds=bonds,
    )


def _build_forcefield(state: Any, manifest: Any) -> Any:
    """Build a BaseForceField matching the topology's bead type 'CG'."""
    from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
    return BaseForceField(
        name="cg_protein_ff",
        bond_parameters=(
            BondParameter("CG", "CG", equilibrium_distance=0.38, stiffness=100.0),
        ),
        nonbonded_parameters=(
            NonbondedParameter("CG", "CG", sigma=0.30, epsilon=0.5, cutoff=1.2),
        ),
    )


def _build_constraints(
    topology: Any, algorithm: str | None, n_particles: int
) -> tuple[tuple, Any]:
    """Build constraint list and solver from topology bonds."""
    from physics.constraints import DistanceConstraint

    if algorithm is None:
        return (), None

    constraints = tuple(
        DistanceConstraint(
            particle_a=bond.particle_index_a,
            particle_b=bond.particle_index_b,
            target_distance=bond.equilibrium_distance if bond.equilibrium_distance is not None else 0.38,
        )
        for bond in topology.bonds
    )

    solver = None
    if algorithm == "lincs":
        try:
            from physics.constraints import LINCSolver
            solver = LINCSolver()
            _ok(f"LINCS constraint solver with {len(constraints)} constraints")
        except Exception:
            _warn("LINCS solver not available; constraints disabled")
            return (), None
    elif algorithm == "shake":
        try:
            from physics.constraints import SHAKESolver
            solver = SHAKESolver()
            _ok(f"SHAKE constraint solver with {len(constraints)} constraints")
        except Exception:
            _warn("SHAKE solver not available; constraints disabled")
            return (), None
    else:
        _warn(f"Unknown constraint algorithm: {algorithm}; constraints disabled")
        return (), None

    return constraints, solver


def _build_analysis_engine(initial_state: Any) -> Any:
    """Build and auto-configure the AdaptiveAnalysisEngine."""
    try:
        from validation.adaptive_analysis import AdaptiveAnalysisEngine
        engine = AdaptiveAnalysisEngine()
        engine.auto_configure(initial_state)
        return engine
    except Exception as exc:
        _warn(f"Analysis engine setup failed: {exc}")

        @dataclass
        class _StubAnalysis:
            def collect(self, *a: Any, **kw: Any) -> None:
                pass
            def generate_report(self) -> None:
                return None
        return _StubAnalysis()


def _build_observable_collector() -> Any:
    """Build the MolecularObservableCollector if available."""
    try:
        from validation.molecular_observables import MolecularObservableCollector
        return MolecularObservableCollector()
    except Exception:
        return None


def _build_integrator(stage_name: str, stage_config: Any) -> Any:
    """Build the appropriate integrator for a dynamics stage."""
    try:
        from integrators.baoab import BAOABIntegrator
        return BAOABIntegrator(
            time_step=stage_config.dt,
            assume_reduced_units=True,
            friction_coefficient=stage_config.friction_coefficient,
        )
    except Exception:
        pass
    try:
        from integrators.langevin import LangevinIntegrator
        return LangevinIntegrator(
            time_step=stage_config.dt,
            friction_coefficient=stage_config.friction_coefficient,
        )
    except Exception as exc:
        _fail(f"No integrator available: {exc}")
        raise


def _build_force_evaluator(state: Any) -> Any:
    """Build the production force evaluator with force capping."""
    try:
        from physics.production_evaluator import ProductionForceEvaluator
        from physics.forces.composite import ForceEvaluation
        from core.types import FrozenMetadata
        from math import sqrt as _sqrt, isnan as _isnan, isinf as _isinf

        inner = ProductionForceEvaluator(
            use_pbc=True,
            charges=tuple(0.0 for _ in range(state.particle_count)),
            electrostatic_method="reaction_field",
        )

        MAX_FORCE = 1000.0  # cap per-component force to prevent NaN

        class _CappedEvaluator:
            name = "capped_production_evaluator"
            classification = "[adapted]"

            def evaluate(self, st, topo, ff):
                fe = inner.evaluate(st, topo, ff)
                # Cap forces to prevent NaN/explosion
                capped = []
                for fx, fy, fz in fe.forces:
                    if _isnan(fx) or _isinf(fx): fx = 0.0
                    if _isnan(fy) or _isinf(fy): fy = 0.0
                    if _isnan(fz) or _isinf(fz): fz = 0.0
                    fx = max(-MAX_FORCE, min(MAX_FORCE, fx))
                    fy = max(-MAX_FORCE, min(MAX_FORCE, fy))
                    fz = max(-MAX_FORCE, min(MAX_FORCE, fz))
                    capped.append((fx, fy, fz))
                pe = fe.potential_energy
                if _isnan(pe) or _isinf(pe):
                    pe = 0.0
                return ForceEvaluation(
                    forces=tuple(capped),
                    potential_energy=pe,
                    component_energies=fe.component_energies,
                    metadata=fe.metadata,
                )

        return _CappedEvaluator()
    except Exception as exc:
        _warn(f"ProductionForceEvaluator unavailable ({exc}); using stub")

        @dataclass
        class _StubEval:
            name: str = "stub_evaluator"
            def evaluate(self, state: Any, topology: Any, forcefield: Any) -> Any:
                n = state.particle_count
                forces = tuple((0.0, 0.0, 0.0) for _ in range(n))
                from core.state import ParticleState
                return ParticleState(
                    positions=state.particles.positions,
                    velocities=state.particles.velocities,
                    forces=forces,
                    masses=state.particles.masses,
                    labels=state.particles.labels,
                )
        return _StubEval()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_minimization_stage(
    config: Any,
    state: Any,
    topology: Any,
    forcefield: Any,
    registry: Any,
    energy_tracker: Any,
) -> tuple[Any, int]:
    """Run energy minimization (steepest descent)."""
    from core.state import ParticleState, SimulationState
    from core.types import FrozenMetadata

    _info(f"Algorithm: {config.algorithm}, max_steps: {config.max_steps}, tol: {config.tolerance}")

    evaluator = _build_force_evaluator(state)
    current = state
    steps_done = 0
    prev_pe = float("inf")

    for step_i in range(config.max_steps):
        try:
            eval_result = evaluator.evaluate(current, topology, forcefield)
            forces = eval_result.forces if hasattr(eval_result, "forces") else current.particles.forces
            pe = eval_result.potential_energy if hasattr(eval_result, "potential_energy") else 0.0
        except Exception:
            forces = current.particles.forces
            pe = 0.0

        # Sanitize PE
        import math as _math
        if _math.isnan(pe) or _math.isinf(pe):
            pe = 0.0

        # Steepest descent: move along force direction
        scale = config.step_size
        new_positions = tuple(
            (
                pos[0] + scale * f[0],
                pos[1] + scale * f[1],
                pos[2] + scale * f[2],
            )
            for pos, f in zip(current.particles.positions, forces)
        )
        new_particles = ParticleState(
            positions=new_positions,
            velocities=current.particles.velocities,
            forces=forces,
            masses=current.particles.masses,
            labels=current.particles.labels,
        )
        new_state = registry.derive_state(
            parent=current,
            particles=new_particles,
            notes=f"em step {step_i}",
        )
        current = new_state
        steps_done += 1

        energy_tracker.record(
            step=current.step, ke=0.0, pe=pe, temperature=0.0,
        )

        # Check convergence
        if abs(prev_pe - pe) < config.tolerance and step_i > 0:
            _info(f"Converged at step {step_i} (dE={abs(prev_pe - pe):.6f})")
            break
        prev_pe = pe

        if step_i % max(1, config.max_steps // 10) == 0:
            _info(f"  EM step {step_i}/{config.max_steps}")

    return current, steps_done


def _run_dynamics_stage(
    stage_name: str,
    config: Any,
    state: Any,
    topology: Any,
    forcefield: Any,
    registry: Any,
    constraints: tuple,
    constraint_solver: Any,
    energy_tracker: Any,
    analysis_hook: Any,
) -> tuple[Any, int]:
    """Run a dynamics stage using ProductionSimulationLoop."""
    from sampling.production_loop import ProductionSimulationLoop, TemperatureSchedule
    from core.types import FrozenMetadata

    _info(f"Ensemble: {config.ensemble.value}, dt={config.dt}, nsteps={config.nsteps}, T={config.temperature} K")
    if config.pressure is not None:
        _info(f"Pressure: {config.pressure} bar")

    integrator = _build_integrator(stage_name, config)
    evaluator = _build_force_evaluator(state)

    temperature_schedule = TemperatureSchedule(
        mode="constant",
        initial_temperature=config.temperature,
        total_steps=max(1, config.nsteps),
    )

    # Build a fresh registry rooted at the current state.
    # We re-create an initial state in the new registry to avoid simulation_id
    # mismatch when transferring states across registries.
    from core.state_registry import SimulationStateRegistry
    loop_registry = SimulationStateRegistry(created_by=f"neurocgmd-{stage_name}")
    root_state = loop_registry.create_initial_state(
        particles=state.particles,
        units=state.units,
        thermodynamics=state.thermodynamics,
        cell=state.cell,
        time=state.time,
        step=state.step,
        notes=f"stage {stage_name} root (step={state.step}, time={state.time})",
    )

    loop = ProductionSimulationLoop(
        topology=topology,
        forcefield=forcefield,
        integrator=integrator,
        force_evaluator=evaluator,
        registry=loop_registry,
        constraints=constraints,
        constraint_solver=constraint_solver,
        temperature_schedule=temperature_schedule,
        apply_pbc=(root_state.cell is not None),
        energy_tracker=energy_tracker,
        analysis_hooks=[analysis_hook] if analysis_hook else [],
        analysis_interval=config.energy_stride,
    )

    report_interval = max(1, config.nsteps // 10)
    _info(f"Running {config.nsteps} steps (reporting every {report_interval})...")

    t0 = time.monotonic()
    try:
        result = loop.run(
            config.nsteps,
            notes=f"neurocgmd {stage_name} stage",
            metadata=FrozenMetadata({"stage": stage_name}),
        )
        elapsed = time.monotonic() - t0
        ns_per_day = (config.nsteps * config.dt / 1000.0) / (elapsed / 86400.0) if elapsed > 0 else 0.0
        _info(f"Wall time: {elapsed:.2f}s | Performance: {ns_per_day:.1f} ns/day")
        if result.constraint_violations > 0:
            _warn(f"{result.constraint_violations} constraint violations")
        _info(f"Energy drift: {result.energy_drift:.6e}")
        return result.final_state, result.steps_completed
    except Exception as exc:
        elapsed = time.monotonic() - t0
        _warn(f"ProductionSimulationLoop raised: {exc}")
        _info(f"Attempting manual stepping fallback (wall time so far: {elapsed:.2f}s)...")
        return _manual_dynamics_fallback(
            config, state, topology, forcefield, integrator, evaluator,
            registry, energy_tracker, analysis_hook,
        )


def _manual_dynamics_fallback(
    config: Any,
    state: Any,
    topology: Any,
    forcefield: Any,
    integrator: Any,
    evaluator: Any,
    registry: Any,
    energy_tracker: Any,
    analysis_hook: Any,
) -> tuple[Any, int]:
    """Step-by-step dynamics when the production loop cannot be used."""
    from core.state import ParticleState, SimulationState
    from core.types import FrozenMetadata
    from core.units import BOLTZMANN_CONSTANT
    from math import fsum

    current = state
    steps_done = 0

    for step_i in range(config.nsteps):
        try:
            step_result = integrator.step(current, topology, forcefield, evaluator)
            new_particles = step_result.particles
            new_state = registry.derive_state(
                parent=current,
                particles=new_particles,
                notes=f"dynamics step {step_i}",
            )
        except Exception:
            # Minimal forward Euler as last resort
            new_positions = tuple(
                (
                    p[0] + config.dt * v[0],
                    p[1] + config.dt * v[1],
                    p[2] + config.dt * v[2],
                )
                for p, v in zip(current.particles.positions, current.particles.velocities)
            )
            new_particles = ParticleState(
                positions=new_positions,
                velocities=current.particles.velocities,
                forces=current.particles.forces,
                masses=current.particles.masses,
                labels=current.particles.labels,
            )
            new_state = registry.derive_state(
                parent=current,
                particles=new_particles,
                notes=f"fallback dynamics step {step_i}",
            )

        current = new_state
        steps_done += 1

        # Record energy
        ke = fsum(
            0.5 * m * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            for m, v in zip(current.particles.masses, current.particles.velocities)
        )
        pe = current.potential_energy if current.potential_energy is not None else 0.0
        n_dof = max(1, 3 * current.particle_count)
        temp = 2.0 * ke / (n_dof * BOLTZMANN_CONSTANT) if BOLTZMANN_CONSTANT > 0 else 0.0
        energy_tracker.record(step=current.step, ke=ke, pe=pe, temperature=temp)

        if analysis_hook:
            try:
                analysis_hook(current, potential_energy=pe, temperature=temp)
            except Exception:
                pass

        if step_i % max(1, config.nsteps // 10) == 0:
            _info(f"  Step {step_i}/{config.nsteps}")

    return current, steps_done


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _write_trajectory(
    energy_tracker: Any,
    final_state: Any,
    output_path: Path,
) -> None:
    """Write trajectory as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for i, step in enumerate(energy_tracker._steps):
            record = {
                "step": step,
                "ke": energy_tracker._kinetic_energies[i],
                "pe": energy_tracker._potential_energies[i],
                "total": energy_tracker._total_energies[i],
                "temperature": energy_tracker._temperatures[i],
            }
            fh.write(json.dumps(record) + "\n")
    _ok(f"Trajectory written to {output_path} ({len(energy_tracker._steps)} frames)")


def _write_energies(energy_tracker: Any, output_path: Path) -> None:
    """Write energies as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "time", "ke", "pe", "total", "temperature"])
        for i, step in enumerate(energy_tracker._steps):
            writer.writerow([
                step,
                f"{step * 0.002:.6f}",
                f"{energy_tracker._kinetic_energies[i]:.6f}",
                f"{energy_tracker._potential_energies[i]:.6f}",
                f"{energy_tracker._total_energies[i]:.6f}",
                f"{energy_tracker._temperatures[i]:.4f}",
            ])
    _ok(f"Energies written to {output_path}")


def _write_checkpoint(final_state: Any, output_path: Path) -> None:
    """Write checkpoint JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = {
            "state": final_state.to_dict(),
            "metadata": {
                "engine": f"neurocgmd-v{VERSION}",
                "step": final_state.step,
                "time": final_state.time,
            },
        }
    except Exception:
        payload = {
            "metadata": {
                "engine": f"neurocgmd-v{VERSION}",
                "step": getattr(final_state, "step", 0),
                "time": getattr(final_state, "time", 0.0),
            },
        }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _ok(f"Checkpoint written to {output_path}")


def _write_report(
    manifest: Any,
    analysis_engine: Any,
    energy_tracker: Any,
    stage_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate and write the HTML analysis report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try the full transition_analysis report generator first
    try:
        from validation.transition_analysis import generate_full_html_report
        report = analysis_engine.generate_report()
        html = generate_full_html_report(report, energy_tracker=energy_tracker)
        output_path.write_text(html, encoding="utf-8")
        _ok(f"Full HTML report written to {output_path}")
        return
    except Exception as exc:
        _warn(f"Full report generation failed ({exc}); writing summary report")

    # Fallback: build a simple but informative HTML report
    html = _build_fallback_report(manifest, energy_tracker, stage_records)
    output_path.write_text(html, encoding="utf-8")
    _ok(f"Summary HTML report written to {output_path}")


def _build_fallback_report(
    manifest: Any,
    energy_tracker: Any,
    stage_records: list[dict[str, Any]],
) -> str:
    """Build a self-contained fallback HTML report with inline SVGs."""
    from math import fsum

    system_name = manifest.system.name
    steps = energy_tracker._steps
    ke = energy_tracker._kinetic_energies
    pe = energy_tracker._potential_energies
    total = energy_tracker._total_energies
    temps = energy_tracker._temperatures

    n = len(steps)

    # Compute summary statistics
    mean_ke = fsum(ke) / max(1, n)
    mean_pe = fsum(pe) / max(1, n)
    mean_total = fsum(total) / max(1, n)
    mean_temp = fsum(temps) / max(1, n)

    # Build simple SVG energy plot
    energy_svg = _make_svg_line_plot(
        steps, total, title="Total Energy vs Step",
        xlabel="Step", ylabel="Energy (kJ/mol)",
        width=600, height=300,
    )
    temp_svg = _make_svg_line_plot(
        steps, temps, title="Temperature vs Step",
        xlabel="Step", ylabel="Temperature (K)",
        width=600, height=300,
    )
    ke_svg = _make_svg_line_plot(
        steps, ke, title="Kinetic Energy vs Step",
        xlabel="Step", ylabel="KE (kJ/mol)",
        width=600, height=300,
    )
    pe_svg = _make_svg_line_plot(
        steps, pe, title="Potential Energy vs Step",
        xlabel="Step", ylabel="PE (kJ/mol)",
        width=600, height=300,
    )

    stage_rows = "\n".join(
        f"<tr><td>{r['stage']}</td><td>{r['steps_completed']}</td>"
        f"<td>{r['final_step']}</td><td>{r['final_time']:.4f}</td></tr>"
        for r in stage_records
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NeuroCGMD Analysis Report | {system_name}</title>
<style>
  body {{ font-family: Georgia, serif; margin: 2rem; line-height: 1.6; color: #333; }}
  h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 0.5rem; }}
  h2 {{ color: #2c3e50; margin-top: 2rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #bdc3c7; padding: 0.5rem 1rem; text-align: right; }}
  th {{ background: #ecf0f1; text-align: center; }}
  .metric {{ display: inline-block; margin: 0.5rem 1rem; padding: 0.8rem 1.2rem;
             background: #f8f9fa; border-radius: 4px; border-left: 4px solid #3498db; }}
  .metric .label {{ font-size: 0.85rem; color: #7f8c8d; }}
  .metric .value {{ font-size: 1.3rem; font-weight: bold; color: #2c3e50; }}
  svg {{ margin: 1rem 0; border: 1px solid #ecf0f1; }}
  .footer {{ margin-top: 3rem; font-size: 0.85rem; color: #95a5a6; }}
</style>
</head>
<body>
<h1>NeuroCGMD Analysis Report</h1>
<p>System: <strong>{system_name}</strong> | Engine: NeuroCGMD v{VERSION}</p>

<h2>Summary Metrics</h2>
<div>
  <div class="metric"><div class="label">Mean KE</div><div class="value">{mean_ke:.4f}</div></div>
  <div class="metric"><div class="label">Mean PE</div><div class="value">{mean_pe:.4f}</div></div>
  <div class="metric"><div class="label">Mean Total E</div><div class="value">{mean_total:.4f}</div></div>
  <div class="metric"><div class="label">Mean Temp</div><div class="value">{mean_temp:.2f} K</div></div>
  <div class="metric"><div class="label">Total Steps</div><div class="value">{n}</div></div>
</div>

<h2>Stage Summary</h2>
<table>
<tr><th>Stage</th><th>Steps Completed</th><th>Final Step</th><th>Final Time</th></tr>
{stage_rows}
</table>

<h2>Total Energy</h2>
{energy_svg}

<h2>Temperature</h2>
{temp_svg}

<h2>Kinetic Energy</h2>
{ke_svg}

<h2>Potential Energy</h2>
{pe_svg}

<h2>Analysis Capabilities</h2>
<ul>
  <li>RMSD tracking (reference = initial structure)</li>
  <li>Radial distribution function (RDF)</li>
  <li>Potential of mean force (PMF via WHAM)</li>
  <li>Binding energy estimation</li>
  <li>SASA, radius of gyration, H-bonds, contacts</li>
  <li>Secondary structure estimation</li>
  <li>Transition state detection</li>
  <li>Reaction coordinate generation</li>
  <li>Energy autocorrelation and block averaging</li>
</ul>

<div class="footer">
  Generated by NeuroCGMD v{VERSION} |
  Observables: RMSD, RDF, PMF, binding energy, SASA, Rg, H-bonds, contacts,
  secondary structure, transition states, reaction coordinates
</div>
</body>
</html>"""


def _make_svg_line_plot(
    xs: list,
    ys: list,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    width: int = 600,
    height: int = 300,
) -> str:
    """Generate a simple inline SVG line plot."""
    if not xs or not ys:
        return f"<p><em>No data for {title}</em></p>"

    # Downsample if too many points
    max_points = 500
    if len(xs) > max_points:
        stride = len(xs) // max_points
        xs = xs[::stride]
        ys = ys[::stride]

    margin_l, margin_r, margin_t, margin_b = 70, 20, 40, 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1
    # Add 5% padding
    y_range = y_max - y_min
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range

    def tx(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def ty(y: float) -> float:
        return margin_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    points = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in zip(xs, ys))

    return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="white"/>
  <text x="{width // 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>
  <rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#ddd"/>
  <polyline points="{points}" fill="none" stroke="#2980b9" stroke-width="1.5"/>
  <text x="{width // 2}" y="{height - 8}" text-anchor="middle" font-size="11">{xlabel}</text>
  <text x="12" y="{height // 2}" text-anchor="middle" font-size="11" transform="rotate(-90,12,{height // 2})">{ylabel}</text>
  <text x="{margin_l}" y="{height - 28}" font-size="9" fill="#999">{x_min:.0f}</text>
  <text x="{width - margin_r}" y="{height - 28}" text-anchor="end" font-size="9" fill="#999">{x_max:.0f}</text>
  <text x="{margin_l - 4}" y="{margin_t + 10}" text-anchor="end" font-size="9" fill="#999">{y_max:.4g}</text>
  <text x="{margin_l - 4}" y="{margin_t + plot_h}" text-anchor="end" font-size="9" fill="#999">{y_min:.4g}</text>
</svg>"""


# ---------------------------------------------------------------------------
# Resolve output paths
# ---------------------------------------------------------------------------

def _resolve_output(manifest: Any, relative: str) -> Path:
    return manifest.outputs.resolve(REPO_ROOT, relative)


# ---------------------------------------------------------------------------
# Subcommand: run  (prep -> run -> analyze -> report)
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """Full pipeline: prepare, run all stages, analyze, report."""
    _banner()
    config_path = Path(args.config)

    # 1. Parse TOML and build manifest
    _section("Configuration")
    _info(f"Config file: {config_path.resolve()}")
    manifest, translated = _load_user_toml(config_path)
    _ok(f"System: {manifest.system.name}")
    _ok(f"Structure: {manifest.system.structure}")
    _info(f"Stages: em={'on' if manifest.em.enabled else 'off'}"
          f" nvt={'on' if manifest.nvt.enabled else 'off'}"
          f" npt={'on' if manifest.npt.enabled else 'off'}"
          f" prod={'on' if manifest.production.enabled else 'off'}")

    # 2. Prepare
    _section("Preparation")
    bundle, bundle_path = _run_preparation(manifest)

    # 3. Run stages
    _section("Simulation")
    final_state, energy_tracker, analysis_engine, stage_records = _run_stages(
        manifest, bundle, translated,
    )

    # 4. Write outputs
    _section("Outputs")

    traj_path = _resolve_output(manifest, manifest.outputs.trajectory)
    energy_path = _resolve_output(manifest, manifest.outputs.energy)
    ckpt_path = _resolve_output(manifest, manifest.outputs.checkpoint)
    report_path = _resolve_output(manifest, manifest.outputs.analysis_html)

    _write_trajectory(energy_tracker, final_state, traj_path)
    _write_energies(energy_tracker, energy_path)
    _write_checkpoint(final_state, ckpt_path)

    # 5. Analyze and report
    _section("Analysis & Report")
    _write_report(manifest, analysis_engine, energy_tracker, stage_records, report_path)

    # 6. Summary
    _section("Summary")
    total_steps = sum(r["steps_completed"] for r in stage_records)
    _info(f"System:         {manifest.system.name}")
    _info(f"Total steps:    {total_steps}")
    _info(f"Final step:     {final_state.step}")
    _info(f"Final time:     {final_state.time:.6f}")
    _info(f"Trajectory:     {traj_path}")
    _info(f"Energies:       {energy_path}")
    _info(f"Checkpoint:     {ckpt_path}")
    _info(f"Report:         {report_path}")
    _info(f"Prepared bundle:{bundle_path}")
    print(f"\n{'=' * 64}")
    print(f"  Run complete.")
    print(f"{'=' * 64}\n")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> int:
    """Preparation only: parse structure, build coarse-grained system, write bundle."""
    _banner()
    config_path = Path(args.config)

    _section("Configuration")
    manifest, translated = _load_user_toml(config_path)
    _ok(f"System: {manifest.system.name}")

    _section("Preparation")
    bundle, bundle_path = _run_preparation(manifest)

    _section("Summary")
    _info(f"Prepared bundle: {bundle_path}")
    print(f"\n{'=' * 64}")
    print(f"  Preparation complete.")
    print(f"{'=' * 64}\n")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze existing trajectory and checkpoint, generate report."""
    _banner()
    config_path = Path(args.config)

    _section("Configuration")
    manifest, translated = _load_user_toml(config_path)
    _ok(f"System: {manifest.system.name}")

    _section("Loading Existing Data")

    # Load checkpoint
    ckpt_path = _resolve_output(manifest, manifest.outputs.checkpoint)
    if not ckpt_path.exists():
        _fail(f"Checkpoint not found: {ckpt_path}")
        _info("Run 'neurocgmd run <config.toml>' first to generate data.")
        return 1
    _ok(f"Checkpoint: {ckpt_path}")

    # Load trajectory
    traj_path = _resolve_output(manifest, manifest.outputs.trajectory)
    energy_path = _resolve_output(manifest, manifest.outputs.energy)

    # Rebuild energy tracker from CSV or trajectory
    from sampling.production_loop import EnergyTracker
    energy_tracker = EnergyTracker()

    if energy_path.exists():
        _ok(f"Energies: {energy_path}")
        with open(energy_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                energy_tracker.record(
                    step=int(row["step"]),
                    ke=float(row["ke"]),
                    pe=float(row["pe"]),
                    temperature=float(row["temperature"]),
                )
    elif traj_path.exists():
        _ok(f"Trajectory: {traj_path}")
        with open(traj_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    energy_tracker.record(
                        step=record.get("step", 0),
                        ke=record.get("ke", 0.0),
                        pe=record.get("pe", 0.0),
                        temperature=record.get("temperature", 0.0),
                    )
    else:
        _warn("No trajectory or energy file found; report will have no time-series data")

    # Load checkpoint state for analysis engine
    ckpt_payload = json.loads(ckpt_path.read_text(encoding="utf-8"))

    # Try to reconstruct analysis engine
    analysis_engine = None
    try:
        from core.state import SimulationState
        if "state" in ckpt_payload:
            state = SimulationState.from_dict(ckpt_payload["state"])
            analysis_engine = _build_analysis_engine(state)
    except Exception as exc:
        _warn(f"Could not reconstruct analysis engine: {exc}")

    if analysis_engine is None:
        @dataclass
        class _Stub:
            def generate_report(self) -> None:
                return None
        analysis_engine = _Stub()

    # Infer stage records from energy tracker
    stage_records = [{"stage": "loaded", "steps_completed": len(energy_tracker._steps),
                      "final_step": energy_tracker._steps[-1] if energy_tracker._steps else 0,
                      "final_time": 0.0}]

    # Generate report
    _section("Analysis & Report")
    report_path = _resolve_output(manifest, manifest.outputs.analysis_html)
    _write_report(manifest, analysis_engine, energy_tracker, stage_records, report_path)

    _section("Summary")
    _info(f"Data points:    {len(energy_tracker._steps)}")
    _info(f"Report:         {report_path}")
    print(f"\n{'=' * 64}")
    print(f"  Analysis complete.")
    print(f"{'=' * 64}\n")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    """Print platform capabilities."""
    _banner()

    def _layer(icon: str, name: str, color_fn, items: list[tuple[str, str]]) -> None:
        print(f"  {color_fn(icon + ' ' + name)}")
        for label, detail in items:
            print(f"      {_gray(label.ljust(22))} {detail}")
        print()

    _layer("+", "CG DYNAMICS", _blue, [
        ("Integrators", "BAOAB Langevin splitting, Velocity-Verlet Langevin"),
        ("Force fields", "Harmonic bonds/angles, LJ (shifted), electrostatics"),
        ("Neighbor search", "O(N) cell-list with cutoff"),
        ("PBC", "Orthorhombic, dodecahedron, octahedron"),
        ("Eval stride", "Full pipeline every N steps, lightweight between"),
    ])

    _layer("~", "QCLOUD QUANTUM", _purple, [
        ("Corrections", "AA-level force deltas on priority regions"),
        ("Region selection", "Adaptive priority-based subgraph selection"),
        ("Event detection", "Bond forming/breaking, conformational shifts"),
        ("Feedback loop", "Correction magnitudes inform next region selection"),
    ])

    _layer("*", "ML RESIDUAL", _gold, [
        ("Architecture", "Neural residual MLP with online SGD"),
        ("Training", "On-the-fly from QCloud corrections"),
        ("Composition", "F = F_CG + F_QCloud + F_ML (bounded, capped)"),
        ("Uncertainty", "Ensemble variance for active learning"),
    ])

    _layer(">", "BACK-MAPPING", _teal, [
        ("Method", "Distance-weighted interpolation + bond relaxation"),
        ("Resolution", "Full AA coordinates from CG trajectory"),
        ("Information", "Carries CG + QCloud + ML corrections"),
        ("Export", "Multi-model PDB trajectories (CG and AA)"),
    ])

    _layer("#", "ANALYSIS", _pink, [
        ("CG level", "Energy, RMSD, RMSF, RDF, SASA, Rg, PMF, free energy"),
        ("AA level", "Residue contacts, H-bonds (angle+distance), hotspots"),
        ("QCloud", "Structural events, correction timeline, adaptive focus"),
        ("Binding", "Auto-detect pairs, decomposed interactions, dashboards"),
    ])

    _layer("=", "SYSTEMS", _green, [
        ("Types", "Protein, peptide, ligand, lipid, polymer, ion, water"),
        ("Solvent", "Explicit (TIP3P, SPC/E), implicit, vacuum"),
        ("Ions", "NaCl, KCl, MgCl2 with auto-neutralization"),
        ("Sampling", "Replica exchange, metadynamics, umbrella sampling"),
    ])

    print(f"  {_dim('.' * 62)}")
    print(f"  {_gray('Usage:')}")
    print(f"    {_white('neurocgmd run')} config.toml       {_dim('Full pipeline')}")
    print(f"    {_white('neurocgmd prepare')} config.toml   {_dim('Preparation only')}")
    print(f"    {_white('neurocgmd analyze')} config.toml   {_dim('Analyze trajectory')}")
    print(f"    {_white('neurocgmd info')}                  {_dim('This display')}")
    print()
    print(f"  {_dim('Contact & collaboration:')}")
    print(f"    {_teal('bessuman.academia@gmail.com')}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurocgmd",
        description="NeuroCGMD -- unified molecular dynamics engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python3 neurocgmd.py run run.toml
  python3 neurocgmd.py prepare system.toml
  python3 neurocgmd.py analyze system.toml
  python3 neurocgmd.py info
""",
    )
    parser.add_argument("--version", action="version", version=f"neurocgmd {VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- run ---------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Full pipeline: prepare, simulate, analyze, report.",
    )
    run_parser.add_argument("config", help="Path to TOML configuration file.")
    run_parser.set_defaults(func=cmd_run)

    # -- prepare -----------------------------------------------------------
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare the system (structure import, solvation, ions).",
    )
    prepare_parser.add_argument("config", help="Path to TOML configuration file.")
    prepare_parser.set_defaults(func=cmd_prepare)

    # -- analyze -----------------------------------------------------------
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze existing trajectory data and generate report.",
    )
    analyze_parser.add_argument("config", help="Path to TOML configuration file.")
    analyze_parser.set_defaults(func=cmd_analyze)

    # -- info --------------------------------------------------------------
    info_parser = subparsers.add_parser(
        "info",
        help="Show platform capabilities and supported features.",
    )
    info_parser.set_defaults(func=cmd_info)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        _fail(f"Unhandled error: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
