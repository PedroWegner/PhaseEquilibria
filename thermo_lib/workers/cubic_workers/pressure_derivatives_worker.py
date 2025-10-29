from ...constants import RGAS_SI
from ...state import State

class PressureDerivativesWorker:
    def __init__(self):
        self.derivatives = {}

    def _calculate_P_derivatives(self, state: State) -> None:
        dF_dVV = state.helmholtz_derivatives['dF_dVV']
        dF_dTV = state.helmholtz_derivatives['dF_dTV']
        dF_dniV = state.helmholtz_derivatives['dF_dniV']

        self.derivatives['dP_dV'] = - RGAS_SI * state.T * dF_dVV - state.n * RGAS_SI * state.T / (state.V**2)
        self.derivatives['dP_dT'] = - RGAS_SI * state.T * dF_dTV + state.P / state.T
        self.derivatives['dP_dni'] = - RGAS_SI * state.T * dF_dniV + RGAS_SI * state.T / state.V
        
    def P_derivatives_to_dict(self, state: State):
        self.derivatives = {}
        self._calculate_P_derivatives(state=state)
        return self.derivatives