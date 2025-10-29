from ...constants import RGAS_SI
from ...state import State
from typing import Dict
import numpy as np

class ResidualPropertiesWorker:
    def __init__(self):
        self.residual_dict = {}

    def _calculate_residual_properties(self, state: State, core_model: Dict[str, any]) -> None:
        Sr_TVn = (- state.T * state.helmholtz_derivatives['dF_dT'] - core_model['F']) * RGAS_SI
        Ar = core_model['F'] * state.T * RGAS_SI

        self.residual_dict['Sr'] = Sr_TVn + state.n * RGAS_SI * np.log(state.Z)  
        self.residual_dict['Hr'] = Ar + state.T * Sr_TVn + state.P * state.V - state.n * RGAS_SI * state.T
        self.residual_dict['Gr'] = Ar + state.P * state.V - state.n * RGAS_SI * state.T * (1 + np.log(state.Z))

        self.residual_dict['F'] = core_model['F']

    def residual_props_to_dict(self, state: State, core_model: Dict[str, any]) -> Dict[str, float]:
        self.residual_dict = {}
        self._calculate_residual_properties(state=state, core_model=core_model)
        return self.residual_dict