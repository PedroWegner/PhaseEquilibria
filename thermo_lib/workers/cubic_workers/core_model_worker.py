from ...constants import RGAS_SI
from ...state import State
from typing import Dict
import numpy as np

class CubicCoreModelWorker:
    def __init__(self, delta1: float, delta2: float):
        self.delta1 = delta1
        self.delta2 = delta2

        self._core_model = {}

    def _calculate_F(self, state: State, D: float) -> None:
        f =self._core_model['f'] 
        fV =self._core_model['fV']
        fB = self._core_model['fB']
        fVV =self._core_model['fVV']
        fBV = self._core_model['fBV'] 
        fBB=self._core_model['fBB']

        g =self._core_model['g'] 
        gV =self._core_model['gV']
        gB = self._core_model['gB']
        gVV =self._core_model['gVV']
        gBV= self._core_model['gBV'] 
        gBB =self._core_model['gBB']
        
        #Fn FB e FD
        self._core_model['F'] = - state.n * g - D * f / state.T
        self._core_model['Fn'] = -g
        self._core_model['FT'] = D * f / state.T**2
        self._core_model['FV'] = - state.n * gV - D * fV / state.T
        self._core_model['FB'] =  - state.n * gB - D * fB / state.T
        self._core_model['FD'] = - f / state.T
        self._core_model['FnV'] = - gV
        self._core_model['FnB'] = -gB
        self._core_model['FTT'] = - 2 * self._core_model['FT'] / state.T
        self._core_model["FBT"] = D * fB / state.T**2
        self._core_model['FDT'] = f / state.T**2
        self._core_model['FBV'] = - state.n * gBV - D * fBV / state.T
        self._core_model['FBB'] = - state.n * gBB - D *fBB / state.T
        self._core_model['FDV'] = - fV / state.T
        self._core_model['FBD'] = - fB / state.T
        self._core_model['FTV'] = D * fV / state.T**2
        self._core_model['FVV'] = - state.n * gVV - D * fVV / state.T
    
    def _calculate_f_functions(self, state: State, B: float) -> None:
        f = 1 / (RGAS_SI * B * (self.delta1 - self.delta2)) * np.log((state.V + self.delta1 * B) / (state.V + self.delta2 * B))
        fV = (1 / (RGAS_SI * B * (self.delta1 - self.delta2))) * (1 /(state.V + self.delta1 * B) - 1 /(state.V + self.delta2 * B))
        fB = - (f + fV * state.V) / B
        fVV = (1 / (RGAS_SI * B * (self.delta1 - self.delta2))) * (-1 /(state.V + self.delta1 * B)**2 + 1 /(state.V + self.delta2 * B)**2)
        fBV = - (2 * fV + state.V * fVV) / B
        fBB = - (2 * fB + state.V * fBV) / B

        self._core_model['f'] = f
        self._core_model['fV'] = fV
        self._core_model['fB'] = fB
        self._core_model['fVV'] = fVV
        self._core_model['fBV'] = fBV
        self._core_model['fBB'] = fBB

    def _calculate_g_functions(self, state: State, B: float) -> None:
        self._core_model['g'] = np.log(1 - B / state.V)
        self._core_model['gV'] = B / (state.V * (state.V - B))
        self._core_model['gB'] = - 1 / (state.V - B)
        self._core_model['gVV'] = - 1 / (state.V - B)**2 + 1 / state.V**2
        self._core_model['gBV'] = 1 / (state.V - B)**2
        self._core_model['gBB'] = - 1 / (state.V - B)**2

    def core_model_to_dict(self, state: State) -> Dict[str, any]:
        self._core_model = {}
        B = state.params['b_mix'] * state.n
        D = state.params['a_mix'] * state.n**2
        self._calculate_f_functions(state=state, B=B)
        self._calculate_g_functions(state=state, B=B)
        self._calculate_F(state=state, D=D)
        return self._core_model
