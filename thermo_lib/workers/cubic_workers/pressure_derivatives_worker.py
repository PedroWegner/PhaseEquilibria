from ...constants import RGAS_SI
from ...state import State, HelmholtzResult, PressureResult

class PressureDerivativesWorker:
    def __init__(self):
        pass

    def _calculate_P_derivatives(self, state: State) -> PressureResult:
        dF_dVV = state.helmholtz_results.dF_dVV
        dF_dTV = state.helmholtz_results.dF_dTV
        dF_dniV = state.helmholtz_results.dF_dniV

        dP_dV = - RGAS_SI * state.T * dF_dVV - state.n * RGAS_SI * state.T / (state.V**2)
        dP_dT = - RGAS_SI * state.T * dF_dTV + state.P / state.T
        dP_dni = - RGAS_SI * state.T * dF_dniV + RGAS_SI * state.T / state.V

        return PressureResult(
            dP_dV=dP_dV,
            dP_dT=dP_dT,
            dP_dni=dP_dni
        )
        
    def P_derivatives_to_dict(self, state: State) -> PressureResult:
        pressure_results = self._calculate_P_derivatives(state=state)
        return pressure_results