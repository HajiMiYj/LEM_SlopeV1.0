# -*- coding: utf-8 -*-
from core.solvers.base import BaseLEMSolver
from core.solvers.fellenius import FelleniusSolver
from core.solvers.bishop import BishopSolver
from core.solvers.janbu import JanbuSolver
from core.solvers.spencer import SpencerSolver
from core.solvers.morgenstern_price import MorgensternPriceSolver

__all__ = [
    "BaseLEMSolver",
    "FelleniusSolver",
    "BishopSolver",
    "JanbuSolver",
    "SpencerSolver",
    "MorgensternPriceSolver"
]

FelleniusSolver.supports_non_circular = False
BishopSolver.supports_non_circular = False
JanbuSolver.supports_non_circular = True
SpencerSolver.supports_non_circular = True
MorgensternPriceSolver.supports_non_circular = True

