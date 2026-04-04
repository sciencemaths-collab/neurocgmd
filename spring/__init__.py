# SPRING: Structured Piece Relaxation via Iterative eNergy-Guided descent
# A universal optimization architecture, integrated with NeuroCGMD.

from spring.engine import Piece, Spring, SpringEngine, Problem, Snapshot
from spring.bridges.neurocgmd_bridge import (
    MDPoweredSolver,
    MDSystemAsSPRING,
    spring_to_md,
    md_to_spring,
)
