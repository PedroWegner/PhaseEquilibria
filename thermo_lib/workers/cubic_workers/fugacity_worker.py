from ...constants import RGAS_SI
from ...state import State
from typing import Dict
import numpy as np

class FugacityWorker:
    def __init__(self):
        self.fugacity_dict = {}

    def _calculate_fugacity(self, state: State) -> None:
        dF_dni = np.array(state.helmholtz_derivatives['dF_dni'])
        self.fugacity_dict['lnphi'] = dF_dni.reshape(-1) - np.log(state.Z)
        self.fugacity_dict['phi'] = np.exp(self.fugacity_dict['lnphi'])

    def _caculate_fugacity_derivatives(self, state: State) -> None:
        dF_dniT = np.array(state.helmholtz_derivatives['dF_dniT'])
        dF_dninj = state.helmholtz_derivatives['dF_dninj']
        dP_dV = state.P_derivatives['dP_dV']
        dP_dT = state.P_derivatives['dP_dT']
        dP_dni = np.array(state.P_derivatives['dP_dni'])
        # Volume parcial molar
        Vi = np.array(- dP_dni / dP_dV).reshape(-1)
        n_dlnphi_dni = state.n * dF_dninj + 1 + (state.n * np.outer(dP_dni, dP_dni)) / (RGAS_SI * state.T * dP_dV)

        self.fugacity_dict['dlnphi_dT'] = dF_dniT + 1 / state.T - Vi * dP_dT / (RGAS_SI * state.T)    
        self.fugacity_dict['dlnphi_dP'] = Vi / (RGAS_SI * state.T) - 1 / state.P
        self.fugacity_dict['n_dlnphi_dni'] = n_dlnphi_dni
        self.fugacity_dict['dlnphi_dni'] = n_dlnphi_dni / state.n

    def fugacity_to_dict(self, state: State) -> Dict[str, any]:
        self.fugacity_dict = {}
        self._calculate_fugacity(state=state)
        self._caculate_fugacity_derivatives(state=state)
        return self.fugacity_dict