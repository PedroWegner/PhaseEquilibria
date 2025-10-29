from ...state import State
from ..context import CalculationContext
import numpy as np
import copy


class NewtonSolverWorker:
    def __init__(self, context: CalculationContext):
        self.context = context
        
    def _calculate_jacobian(self, state: State, spec_var_index: int) -> np.ndarray:
        # Define os passos
        h_log = 1e-4
        h_z1 = 1e-4

        # Defini a jacobiana
        num_com = len(state.z)
        J = np.zeros((3,3), dtype=np.float64)  # PONTO DE APOIO

        # Pertubacao na composição
        state_z1_pos = copy.deepcopy(state) 
        z1_pos = np.array([state.z[0] + h_z1, state.z[1] - h_z1])
        state_z1_pos.z = z1_pos
        self.context.eos_model.calculate_from_TVm(state=state_z1_pos)
        self.context.eos_model.calculate_fugacity(state=state_z1_pos)

        state_z1_neg = copy.deepcopy(state)
        z1_neg = np.array([state.z[0] - h_z1, state.z[1] + h_z1])
        state_z1_neg.z = z1_neg
        self.context.eos_model.calculate_from_TVm(state=state_z1_neg)
        self.context.eos_model.calculate_fugacity(state=state_z1_neg)

        # Pertubacao na temperatura
        state_T_pos = copy.deepcopy(state) 
        state_T_pos.z = state.z.copy()
        state_T_pos.T *= (1 + h_log)
        self.context.eos_model.calculate_from_TVm(state=state_T_pos)
        self.context.eos_model.calculate_fugacity(state=state_T_pos)

        state_T_neg = copy.deepcopy(state) 
        state_T_neg.z = state.z.copy()
        state_T_neg.T *= (1 - h_log)
        self.context.eos_model.calculate_from_TVm(state=state_T_neg)
        self.context.eos_model.calculate_fugacity(state=state_T_neg)

        # Pertubacao no volume
        state_Vm_pos = copy.deepcopy(state) 
        state_Vm_pos.z = state.z.copy()
        state_Vm_pos.Vm *= (1 + h_log)
        self.context.eos_model.calculate_from_TVm(state=state_Vm_pos)
        self.context.eos_model.calculate_fugacity(state=state_Vm_pos)

        state_Vm_neg = copy.deepcopy(state) 
        state_Vm_neg.z = state.z.copy()
        state_Vm_neg.Vm *= (1 - h_log)
        self.context.eos_model.calculate_from_TVm(state=state_Vm_neg)
        self.context.eos_model.calculate_fugacity(state=state_Vm_neg)
        
        # As derivadsa dos criterios
        # Calcula os critérios para cada estado perturbado
        b_z1_pos, c_z1_pos = self.context.criteria_worker.get_criteria(state=state_z1_pos)
        b_z1_neg, c_z1_neg = self.context.criteria_worker.get_criteria(state=state_z1_neg)
        b_T_pos, c_T_pos = self.context.criteria_worker.get_criteria(state=state_T_pos)
        b_T_neg, c_T_neg = self.context.criteria_worker.get_criteria(state=state_T_neg)
        b_Vm_pos, c_Vm_pos = self.context.criteria_worker.get_criteria(state=state_Vm_pos)
        b_Vm_neg, c_Vm_neg = self.context.criteria_worker.get_criteria(state=state_Vm_neg)

        # preenche a linha da especificada
        J[0, spec_var_index] = 1.0
        # Preenche a primeira linha (derivadas de b = λ₁)
        J[1, 0] = (b_z1_pos - b_z1_neg) / (2 * h_z1)      # ∂b/∂z₁
        J[1, 1] = (b_T_pos - b_T_neg) / (2 * h_log)       # ∂b/∂(ln T)
        J[1, 2] = (b_Vm_pos - b_Vm_neg) / (2 * h_log)     # ∂b/∂(ln Vm)

        # J[1, 3] = 0   --------------------------> ponto de apoio

        # Preenche a segunda linha (derivadas de c)
        J[2, 0] = (c_z1_pos - c_z1_neg) / (2 * h_z1)      # ∂c/∂z₁
        J[2, 1] = (c_T_pos - c_T_neg) / (2 * h_log)       # ∂c/∂(ln T)
        J[2, 2] = (c_Vm_pos - c_Vm_neg) / (2 * h_log)     # ∂c/∂(ln Vm)

        # J[2, 3] = 0  # --------------------------> ponto de apoio

        # # PONTO DE APOIO   #--------------------------> ponto de apoio
        # J[3, 0] = - (np.log(state_z1_pos.P) - np.log(state_z1_neg.P)) / (2 * h_z1)   #--------------------------> ponto de apoio
        # J[3, 1] = - (np.log(state_T_pos.P) - np.log(state_T_neg.P)) / (2 * h_log)   #--------------------------> ponto de apoio
        # J[3, 2] = - (np.log(state_Vm_pos.P) - np.log(state_Vm_neg.P)) / (2 * h_log)   #--------------------------> ponto de apoio
        # J[3, 3] = 1   #--------------------------> ponto de apoio
        # # FIM DO PONTO DE APOIO
        return J

    def _calculate_system_of_equations(self, variables: list, state_template: State, spec_var_index: int, S_target: float) -> list:
        """
        Esta é a função que define o sistema a ser zerado.
        Ela recebe as variáveis [z₁, lnT, lnV] e retorna o vetor de erros [b, c, g].
        """
        z1, lnT, lnV = variables  # lnP    --------------------------> ponto de apoio
        
        # Cria o estado atual para a iteração
        current_state = copy.deepcopy(state_template)
        current_state.z = state_template.z.copy()
        current_state.z[0] = z1; current_state.z[1] = 1.0 - z1
        current_state.T = np.exp(lnT)
        current_state.Vm = np.exp(lnV)
        self.context.eos_model.calculate_from_TVm(state=current_state)
        self.context.eos_model.calculate_fugacity(state=current_state)

        # Calcula os critérios b e c
        b, c = self.context.criteria_worker.get_criteria(state=current_state)
        
        # f4 = lnP - np.log(current_state.P) #  --------------------------> ponto de apoio

        # --- CÁLCULO CORRETO DE 'g' ---
        # Compara a variável correspondente do estado atual com o valor alvo 'S'.
        if spec_var_index == 0:
            g = current_state.z[0] - S_target
        elif spec_var_index == 1:
            g = np.log(current_state.T) - S_target
        elif spec_var_index == 2:
            g = np.log(current_state.Vm) - S_target
        elif spec_var_index == 3:
            g = 1 - S_target   # --------------------------> ponto de apoio
        else:
            # Segurança
            g = 0

        return np.array([g, b, c], dtype=np.float64)

    def newton_solver(self, state_guess: State, spec_var_index: int, spec_var_value: float, max_iter: int=50, tol: float=1e-9):
        local_state = copy.deepcopy(state_guess)

        # Se vai ser temperatura ou volume molar, precisa empacotar num ln
        if spec_var_index == 1 or spec_var_index == 2 or spec_var_index == 3:  # PONTO DE APOIO
            S_target = np.log(spec_var_value)
        else:
            S_target = spec_var_value

        if local_state.P is None:
            self.context.eos_model.calculate_from_TVm(state=local_state)
            self.context.eos_model.calculate_fugacity(state=local_state)

        X = np.array([
            local_state.z[0],
            np.log(local_state.T),
            np.log(local_state.Vm),
            ])

        for i in range(max_iter):
            F = self._calculate_system_of_equations(
                variables=X,
                state_template=local_state,
                spec_var_index=spec_var_index,
                S_target=S_target
                )
            local_state.z[0] = X[0]; local_state.z[1] = 1.0 - X[0]
            local_state.T = np.exp(X[1])
            local_state.Vm = np.exp(X[2])
            self.context.eos_model.calculate_from_TVm(state=local_state)
            self.context.eos_model.calculate_fugacity(state=local_state)
            
            # Verifica a convergência
            norm_F = np.linalg.norm(F)
            J = self._calculate_jacobian(state=local_state, spec_var_index=spec_var_index)

            if norm_F < tol:
                # 
                # print(f"Convergência atingida em {i+1} iterações, sendo X = {X}")
                print(f"{i:<8} | {norm_F:.3e}, {local_state.z[0]}, {local_state.T -273.15}, {local_state.P /10**5}, Z={local_state.Z} , Vm={local_state.Vm}")

                # Retorna o último estado consistente calculado dentro de _system_of_equations
                return local_state, X, i+1, J

            
            delta_X = np.linalg.solve(J, -F)

            X = X + delta_X
        print('Newton falhou!')
        return None, None, None, None