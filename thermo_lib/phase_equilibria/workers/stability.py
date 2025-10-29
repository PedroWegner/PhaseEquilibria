from ...eos.eos_abc import EquationOfState
from ...state import State

import numpy as np
import copy

class StabilityCriteriaWorker:
    def __init__(self, eos_model: EquationOfState):
        self.eos_model = eos_model

    @staticmethod
    def _calculate_B_matrix(state: State) -> np.ndarray:
        n_array = state.n * state.z
        I = np.identity(len(state.mixture.components))
        I = I / n_array
        B = np.sqrt(np.outer(state.z, state.z)) * (I + state.helmholtz_derivatives['dF_dninj'])
        return B

    @staticmethod
    def _calculate_eingen(B: float) -> tuple[float, float]:
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        min_eigenvalue_index = np.argmin(eigenvalues)
        lambda1 = eigenvalues[min_eigenvalue_index]
        u = eigenvectors[:, min_eigenvalue_index]
        return lambda1, u

    def _calculate_c(self, u: np.ndarray, state: State, eta: float=0.0001) -> float:
        delta = eta * u * np.sqrt(state.z)
        n_pos = state.z + delta
        state_pos = self._obtain_state(n=n_pos, state=state)
        B_pos = self._calculate_B_matrix(state=state_pos)
        lambda1_pos, _ = self._calculate_eingen(B=B_pos)

        n_neg = state.z - delta
        state_neg = self._obtain_state(n=n_neg, state=state)
        B_neg = self._calculate_B_matrix(state=state_neg)
        lambda1_neg, _ = self._calculate_eingen(B=B_neg)
        
        c = (lambda1_pos - lambda1_neg) / (2 * eta)
        return c

    def _obtain_state(self, n: np.ndarray, state: State) -> State:
        state_local = copy.deepcopy(state)
        state_local.n = np.sum(n)
        state_local.z = n / np.sum(n)
        state_local.Vm = state_local.V / state_local.n
        self.eos_model.calculate_from_TVm(state=state_local)
        self.eos_model.calculate_fugacity(state=state_local)
        return state_local
    
    def get_criteria(self, state: State) -> tuple[float, float]:
        B = self._calculate_B_matrix(state=state)
        lambda1, u = self._calculate_eingen(B=B)
        c = self._calculate_c(u=u, state=state)
        return lambda1, c