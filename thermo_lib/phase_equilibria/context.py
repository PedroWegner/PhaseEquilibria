# Pode ficar em um novo arquivo: thermo_lib/phase_equilibria/context.py
from dataclasses import dataclass
from ..eos.eos_abc import EquationOfState
from .workers.stability import StabilityCriteriaWorker

@dataclass
class CalculationContext:
    eos_model: EquationOfState
    criteria_worker: StabilityCriteriaWorker
