from ...constants import RGAS_SI
from ...state import State, FugacityResults
from typing import Dict
import numpy as np

class FugacityWorker:
    def __init__(self):
        # self.fugacity_dict = {}
        pass

    def _calculate_fugacity(self, state: State) -> tuple:
        dF_dni = state.helmholtz_results.dF_dni
        lnphi = dF_dni.reshape(-1) - np.log(state.Z)
        phi = np.exp(lnphi)

        return lnphi, phi

    def _caculate_fugacity_derivatives(self, state: State) -> None:

        dF_dniT = state.helmholtz_results.dF_dniT
        dF_dninj = state.helmholtz_results.dF_dninj
        dP_dV = state.pressure_results.dP_dV
        dP_dT = state.pressure_results.dP_dT
        dP_dni = state.pressure_results.dP_dni

        # Volume parcial molar
        Vi = np.array(- dP_dni / dP_dV).reshape(-1)
        n_dlnphi_dni = state.n * dF_dninj + 1 + (state.n * np.outer(dP_dni, dP_dni)) / (RGAS_SI * state.T * dP_dV)

        dlnphi_dT = dF_dniT + 1 / state.T - Vi * dP_dT / (RGAS_SI * state.T)    
        dlnphi_dP = Vi / (RGAS_SI * state.T) - 1 / state.P
        dlnphi_dni = n_dlnphi_dni / state.n

        return n_dlnphi_dni, dlnphi_dT, dlnphi_dP, dlnphi_dni

    def fugacity_to_dict(self, state: State) -> tuple:
        lnphi, phi = self._calculate_fugacity(state=state)
        n_dlnphi_dni, dlnphi_dT, dlnphi_dP, dlnphi_dni = self._caculate_fugacity_derivatives(state=state)
        fugacity_results = FugacityResults(
            ln_phi=lnphi,
            phi=phi,
            dlnphi_dT=dlnphi_dT,
            dlnphi_dP=dlnphi_dP,
            dlnphi_dni=dlnphi_dni,
            n_dlnphi_dni=n_dlnphi_dni,
        )
        return fugacity_results