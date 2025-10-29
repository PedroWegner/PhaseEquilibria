from ..constants import RGAS_SI
from ..state import State
from .eos_abc import EquationOfState
from ..workers.cubic_workers.factory import CubicWorkerSet
# from ..workers.cubic_workers.parameter_worker import CubicParametersWorker
# from ..workers.cubic_workers.Z_solver_worker import SolveZWorker
# from ..workers.cubic_workers.core_model_worker import CubicCoreModelWorker
# from ..workers.cubic_workers.helmholtz_derivatives_worker import CubicHelmholtzDerivativesWorker
# from ..workers.cubic_workers.pressure_derivatives_worker import PressureDerivativesWorker
# from ..workers.cubic_workers.fugacity_worker import FugacityWorker
# from ..workers.cubic_workers.residual_properties_worker import ResidualPropertiesWorker

class CubicEquationOfState(EquationOfState):
    def __init__(self, workers: CubicWorkerSet):
        # self.delta1 = delta1
        # self.delta2 = delta2
        # self.omega1 = omega1
        # self.omega2 = omega2
        # self.m_func = m_func

        # self.params_worker = CubicParametersWorker(omega1=self.omega1, omega2=self.omega2, m=self.m_func)
        # self.solver_Z_worker = SolveZWorker(delta1=self.delta1, delta2=self.delta2)
        # self.core_model_worker =  CubicCoreModelWorker(delta1=self.delta1, delta2=self.delta2)
        # self.helmholtz_derivatives_worker = CubicHelmholtzDerivativesWorker()
        # self.pression_derivatives_worker = PressureDerivativesWorker()
        # self.fugacity_worker = FugacityWorker()
        # self.residual_props_worker = ResidualPropertiesWorker()
        self.workers = workers

    # All the abstract methods are constructed below
    def calculate_from_TP(self, state: State, is_vapor: bool) -> None:
        if state.T is None or state.P is None:
            raise ValueError('Temperature and Pressure must be inputed to use this method')
        state.params = self.workers.params.params_to_dict(state=state)
        Z = self.workers.z_solver.get_Z(state=state) # PONTO DE APOIO, eu preciso mudar o solver_Z, porque cada eos tem um solver diferente.....
        if is_vapor:
            state.Z = max(Z)
        else: 
            state.Z = min(Z)

        state.Vm = state.Z * RGAS_SI * state.T / state.P
        state.V = state.Vm * state.n
    
    def calculate_fugacity(self, state: State) -> None:
        if state.Z is None:
            raise ValueError('The thermodynamic state was not calculated, there is no value for compressibility')
        
        state.core_model = self.workers.core_model.core_model_to_dict(state=state)
        state.helmholtz_derivatives = self.workers.helmholtz_derivatives.helmholtz_derivatives_to_dict(state=state)
        state.P_derivatives = self.workers.pressure_derivatives.P_derivatives_to_dict(state=state)
        state.fugacity_dict = self.workers.fugacity.fugacity_to_dict(state=state)
        pass
    
    def calculate_pressure(self, state: State) -> None:
        if state.T is None or state.Vm is None:
            raise ValueError('Temperature (T) and Molar Volume (Vm) must be different from None') 
        
        state.params = self.workers.params.params_to_dict(state=state)
        b = state.params['b_mix']
        a = state.params['a_mix']
        delta1 = self.workers.z_solver.delta1
        delta2 = self.workers.z_solver.delta2
        state.P = RGAS_SI * state.T / (state.Vm - b) - a / ((state.Vm + delta1 * b) * (state.Vm + delta2 * b))
    
    def calculate_mixture_parameters(self, state: State) -> None:
        if state.T is None:
            raise ValueError('Temperature (T) must be inputed')
        
        state.params = self.workers.params.params_to_dict(state=state)
    # All the abstract methods are constructed above

    