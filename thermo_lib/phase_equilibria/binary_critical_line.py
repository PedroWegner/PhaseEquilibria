from .workers.solver import NewtonSolverWorker
from .context import CalculationContext
from .workers.stability import StabilityCriteriaWorker
from .workers.predictor import ContinuationPredictorWorker
from ..eos.eos_abc import EquationOfState
from ..state import State
import copy
import numpy as np

class CriticalLineCalculator:
    """
    Essa classe está especificamente para cubicas!!!!! 
    """
    def __init__(self, eos_model: EquationOfState):
        critical_worker = StabilityCriteriaWorker(eos_model=eos_model)

        self.context = CalculationContext(
            eos_model=eos_model,
            criteria_worker=critical_worker
        )

        self.solver = NewtonSolverWorker(context=self.context)
        self.predictor = ContinuationPredictorWorker(context=self.context)
        print('construido com sucesso')

    def initial_guess(self, state: State) -> tuple:
        print(state.T)
        state.n = 1.0
        Tc = np.array([c.Tc for c in state.mixture.components])
        state.T = np.sum(state.z * Tc)
        self.context.eos_model.calculate_mixture_parameters(state=state)
        # self.eos_engine.calculate_params(state=state)
        state.Vm = 2.5 * state.params['b_mix']
        # self.eos_engine.calculate_state_2(state=state)
        self.context.eos_model.calculate_from_TVm(state=state)
        print(state.T, state.Vm)
        return state.T, state.Vm

    def trace_line(self, initial_state: State, spec_var_index: int=0, spec_var_value: float=0.001) -> tuple:
        """
        index = 0 -> variavel especificada é a composicao
        index = 1 -> variavel especificada é temperatura
        index = 2 -> variavel especificada é o volume molar
        """
        PTVm = []
        current_state = copy.deepcopy(initial_state)
        

        for ponto_idx in range(1200):
            converged_state, X_old, iter_newton, jacobian = self.solver.newton_solver(
                state_guess=current_state,
                spec_var_index=spec_var_index,
                spec_var_value=spec_var_value
            )
            if converged_state is None:
                print('possivelmente um ponto criondenbar!')
                break

            PTVm.append(np.array([converged_state.T - 273.15, converged_state.P / 10**5, converged_state.Vm]))
            # PTVm.append(self._get_PTVm(state=converged_state))
            if converged_state.z[0] >= 0.99 or converged_state.P > 2e9: # Pressão em bar
                print("Condição de parada da linha atingida.")
                break
            
            X_new, spec_var_index_new, spec_var_value_new = self.predictor.calculate_next_step(
                jacobian=jacobian,
                spec_var_index=spec_var_index,
                X=X_old,
                iter_newton=iter_newton
            )
            # X_new, spec_var_index_new, spec_var_value_new = self._calculate_next_step(state=converged_state, spec_var_index=spec_var_index, X=X_old, iter_newton=iter_newton)
            
            if X_new[0] > 1.0:
                X_new[0] = 0.999
            elif X_new[0] < 0.0:
                X_new[0] = 0.0

            spec_var_index = spec_var_index_new
            spec_var_value = spec_var_value_new

            current_state.z = np.array([X_new[0], 1 - X_new[0]])
            current_state.T = np.exp(X_new[1])
            current_state.Vm = np.exp(X_new[2])
            # print(ponto_idx)
        return PTVm
    


