from .eos.eos_abc import EquationOfState
from .eos.cubic_eos import CubicEquationOfState
from .eos.concrete_models import *
from .workers.cubic_workers.factory import CubicWorkerFactory, CubicWorkerSet

class EoSFactory:
    @staticmethod
    def get_eos_model(model_name: str) -> EquationOfState:
        normalized_name = model_name.strip().upper()

        model_constantes = {}

        if normalized_name in ("PENG-ROBINSON", "PENGROBINSON", "PR"):
            model_constantes = {
                'delta1': 1 + 2**0.5,
                'delta2': 1 - 2**0.5,
                'omega1': 0.45724,
                'omega2': 0.07780,
                'm_func': lambda omega: 0.37464 + 1.57454226 * omega - 0.26992 * omega**2
            }

            workers = CubicWorkerFactory.create_workers(**model_constantes)
            return CubicEquationOfState(workers=workers)
        
        if normalized_name in ("SOAVE-REDLICH-KWONG", "SOAVEREDLICHKWONG", "SRK"):
            return SoaveRedlichKwong()

        if normalized_name in ('VAN DER WAALS', 'VDW'):
            return VanDerWaals()
        
        if normalized_name in ("REDLICH-KWONG", "REDLICHKWONG", "RK"):
            raise NotImplementedError('Redlich-Kwong is not implemented yet')
        
        raise ValueError(f"The model {model_name} is not implemented")