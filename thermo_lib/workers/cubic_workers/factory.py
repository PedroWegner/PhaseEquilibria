from .parameter_worker import CubicParametersWorker
from .Z_solver_worker import SolveZWorker
from .core_model_worker import CubicCoreModelWorker
from .helmholtz_derivatives_worker import CubicHelmholtzDerivativesWorker
from .pressure_derivatives_worker import PressureDerivativesWorker
from .fugacity_worker import FugacityWorker
from .residual_properties_worker import ResidualPropertiesWorker

from dataclasses import dataclass

@dataclass
class CubicWorkerSet:
    params: CubicParametersWorker
    z_solver: SolveZWorker
    core_model:  CubicCoreModelWorker
    helmholtz_derivatives: CubicHelmholtzDerivativesWorker
    pressure_derivatives: PressureDerivativesWorker
    fugacity: FugacityWorker
    residual_props: ResidualPropertiesWorker


class CubicWorkerFactory:
    @staticmethod
    def create_workers(delta1: float, delta2: float, omega1: float, omega2: float, m_func: callable) -> CubicWorkerSet:
        params_worker = CubicParametersWorker(omega1=omega1, omega2=omega2, m=m_func)
        solver_Z_worker = SolveZWorker(delta1=delta1, delta2=delta2)
        core_model_worker =  CubicCoreModelWorker(delta1=delta1, delta2=delta2)
        
        return CubicWorkerSet(
            params = params_worker,
            z_solver = solver_Z_worker,
            core_model = core_model_worker,
            helmholtz_derivatives = CubicHelmholtzDerivativesWorker(),
            pressure_derivatives = PressureDerivativesWorker(),
            fugacity = FugacityWorker(),
            residual_props = ResidualPropertiesWorker()
        )