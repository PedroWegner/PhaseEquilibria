from ...constants import RGAS_SI
from ...state import State
import numpy as np

class SolveZWorker:
    def __init__(self, delta1: float, delta2: float):
        self.delta1 = delta1
        self.delta2 = delta2
        
    def _solver_Z(self, A: float, B: float):
        # 1. Parametros do modelos
        delta = self.delta1 + self.delta2
        delta_inv = self.delta1 * self.delta2
        # 2. Solucao analitica da cubica
        a1 = B * (delta - 1) - 1
        a2 = B**2 * (delta_inv - delta) - B * delta + A
        a3 = - (B**2 * delta_inv * (B + 1) + A * B)
        _Q = (3 * a2 - a1**2) / 9
        _R = (9 * a1 * a2 - 27 * a3 -2 *a1**3)/54
        _D = _Q**3 + _R**2
        if _D < 0:
            theta = np.arccos(_R / np.sqrt(-_Q**3))
            x1 = 2 * np.sqrt(-_Q) * np.cos(theta / 3)  - a1 /3
            x2 = 2 * np.sqrt(-_Q) * np.cos((theta + 2 * np.pi) / 3) - a1 /3
            x3 = 2 * np.sqrt(-_Q) * np.cos((theta + 4 * np.pi) / 3) - a1 /3
        else:
            _S = np.cbrt(_R + np.sqrt(_D))
            _T = np.cbrt(_R - np.sqrt(_D))
            x1 = _S + _T - (1/3) * a1
            x2 = (-1/2)*(_S + _T) - (1/3) * a1 + (1/2) * 1j * np.sqrt(3) * (_S - _T)
            x3 = (-1/2)*(_S + _T) - (1/3) * a1 - (1/2) * 1j * np.sqrt(3) * (_S - _T)
        # 3. Limpeza das raizes obtidas
        Z = [x1, x2, x3]
        Z = [r.real for r in Z if np.isclose(r.imag, 0) and r.real > 0]
        return sorted(Z)

    def get_Z(self, state: State) -> tuple:
        A = state.params['a_mix'] * state.P / (RGAS_SI * state.T)**2
        B = state.params['b_mix'] * state.P / (RGAS_SI * state.T)
        Z = self._solver_Z(A=A, B=B)
        return Z

    def get_Z_(self, state: State, params) -> tuple:
        state = state
        # print(state.params)
        A = params['a_mix'] * state.P / (RGAS_SI * state.T)**2
        B = params['b_mix'] * state.P / (RGAS_SI * state.T)
        Z = self._solver_Z(A=A, B=B)
        is_vapor = None
        if len(Z) == 1:
            # print("O sistema só tem uma fase possível")
            if Z[0] < 0.5:
                # print("O estado só pode ser liquido")
                is_vapor = False
            else: 
                # print("O sistema só pode ser vapor")
                is_vapor = True
        else: 
            is_vapor = state.is_vapor
        return (Z, is_vapor)
