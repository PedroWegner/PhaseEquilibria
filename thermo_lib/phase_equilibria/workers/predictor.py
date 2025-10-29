from ...state import State
from ..context import CalculationContext
import numpy as np
"""
A ideia é desestruturar uma classe state para uma stateless, porque a ela nao precisa de fato calcular uma jacobiana, 
só precisa da informação dela...
"""
class ContinuationPredictorWorker:
    def __init__(self, context: CalculationContext):
        self.context = context
            
    def _calculate_sensitivity_vector(self, jacobian: np.ndarray, spec_var_index: int):
            # J = self._calculate_jacobian(state=state, spec_var_index=spec_var_index)
            J = jacobian
            F = np.zeros(3)  #   --------------------------> ponto de apoio
            F[spec_var_index] = -1

            dX_dS = np.linalg.solve(J, -F)
            # print(dX_dS)

            return dX_dS

    def calculate_next_step(self, jacobian: np.ndarray, spec_var_index: int, X: np.ndarray, iter_newton: int):
        dX_dS = self._calculate_sensitivity_vector(jacobian=jacobian, spec_var_index=spec_var_index)
        delta_S = 0.001
        # delta_S_max = 0.05

        # if iter_newton <= 3:
        #     delta_S = min(delta_S*1.25, delta_S_max)
        # elif iter_newton >= 5:
        #     delta_S = delta_S / 2

        if iter_newton <= 3:
            delta_S = 0.0075
        elif iter_newton >= 5:
            delta_S = 0.00075

        X = X + dX_dS * delta_S
        spec_var_index_new = np.argmax(np.abs(dX_dS))

        spec_var_value_new = X[spec_var_index_new]

        if spec_var_index_new == 1 or spec_var_index_new == 2:
            spec_var_value_new = np.exp(spec_var_value_new)
        return X, spec_var_index_new, spec_var_value_new