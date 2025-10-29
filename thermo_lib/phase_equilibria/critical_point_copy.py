from ..eos.eos_abc import EquationOfState
from ..state import State

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
        eigenvalues, eigenvectors = np.linalg.eigh(B) # ponto de analise!!!!
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
        return state_local
    
    def get_criteria(self, state: State) -> tuple[float, float]:
        B = self._calculate_B_matrix(state=state)
        lambda1, u = self._calculate_eingen(B=B)
        c = self._calculate_c(u=u, state=state)
        return lambda1, c
    

    

class CriticalPointSolver:
    def __init__(self, EoS_Engine: ModeloPengRobinson):
        self.eos_engine = EoS_Engine()

    def initial_guess(self, state: State) -> tuple:
        state.n = 1.0
        Tc = np.array([c.Tc for c in state.mixture.components])
        state.T = np.sum(state.z * Tc)
        self.eos_engine.calculate_params(state=state)
        state.Vm = 2.5 * state.params['b_mix']
        self.eos_engine.calculate_state_2(state=state)
        return state.T, state.Vm

    def _calculate_B_matrix(self, state: State) -> np.ndarray:
        n_array = state.n * state.z
        I = np.identity(len(state.mixture.components))
        I = I / n_array
        B = np.sqrt(np.outer(state.z, state.z)) * (I + state.helmholtz_derivatives['dF_dninj'])
        return B
    
    def _calculate_eingen(self, B: np.ndarray):
        eigenvalues, eigenvectors = np.linalg.eigh(B) # ponto de analise!!!!
        min_eigenvalue_index = np.argmin(eigenvalues)
        lambda1 = eigenvalues[min_eigenvalue_index]
        u = eigenvectors[:, min_eigenvalue_index]
        return lambda1, u
    
    def _obtain_state(self, n: np.ndarray, state: State) -> State:
        state_local = copy.deepcopy(state) #
        state_local.n = np.sum(n)
        state_local.z = n / np.sum(n)
        state_local.Vm = state_local.V / state_local.n
        self.eos_engine.calculate_params(state=state_local)
        self.eos_engine.calculate_state_2(state=state_local)
        return state_local

    def _calculate_c(self, u:np.ndarray, state: State, eta: float=0.0001):
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

    def _objective_function(self, vars, mixture: Mixture, z: np.ndarray, is_vapor: bool):
        T, Vm = vars
        local_state = State(mixture=mixture, T=T, Vm=Vm, z=z.copy(), is_vapor=is_vapor)
        local_state.n = 1
        self.eos_engine.calculate_params(state=local_state)
        self.eos_engine.calculate_state_2(state=local_state)

        local_B = self._calculate_B_matrix(state=local_state)
        local_lambda, local_u = self._calculate_eingen(B=local_B)
        local_c = self._calculate_c(u=local_u, state=local_state)

        FO = local_lambda**2 + local_c**2
        return FO

    def solver_optimize(self, state: State, T: float, Vm: float) -> None:
        vars0 = [T, Vm]
        result = minimize(fun=self._objective_function,
                          x0=vars0,
                          args=(state.mixture, state.z, state.is_vapor),
                          method='Nelder-Mead')
        T, Vm = result.x
        print('Minimize, total de iterações: ', result.nit)
        state.T = T
        state.Vm = Vm
        state.n = 1
        self.eos_engine.calculate_params(state=state)
        self.eos_engine.calculate_state_2(state=state)
        return state.T, state.P, Vm
    
    def _get_criteria(self, state: State) -> tuple:
        B = self._calculate_B_matrix(state=state)
        lambda1, u = self._calculate_eingen(B=B)
        c = self._calculate_c(u=u, state=state)
        return lambda1, c
    
    def _calculate_jacobian(self, state: State, spec_var_index: int) -> np.ndarray:
        
        # Define os passos
        h_log = 1e-4
        h_z1 = 1e-4

        # Defini a jacobiana
        num_com = len(state.z)
        J = np.zeros((3,3), dtype=np.float64)  # PONTO DE APOIO

        # Pertubacao na composição
        state_z1_pos = copy.deepcopy(state) #self._copy_state(state=state)
        z1_pos = np.array([state.z[0] + h_z1, state.z[1] - h_z1])
        state_z1_pos.z = z1_pos
        self.eos_engine.calculate_params(state=state_z1_pos)
        self.eos_engine.calculate_state_2(state=state_z1_pos)

        state_z1_neg = copy.deepcopy(state) #self._copy_state(state=state)
        z1_neg = np.array([state.z[0] - h_z1, state.z[1] + h_z1])
        state_z1_neg.z = z1_neg      
        self.eos_engine.calculate_params(state=state_z1_neg)
        self.eos_engine.calculate_state_2(state=state_z1_neg)

        # Pertubacao na temperatura
        state_T_pos = copy.deepcopy(state) #self._copy_state(state=state)
        state_T_pos.z = state.z.copy()
        state_T_pos.T *= (1 + h_log)
        self.eos_engine.calculate_params(state=state_T_pos)
        self.eos_engine.calculate_state_2(state=state_T_pos)

        state_T_neg = copy.deepcopy(state) #self._copy_state(state=state)
        state_T_neg.z = state.z.copy()
        state_T_neg.T *= (1 - h_log)
        self.eos_engine.calculate_params(state=state_T_neg)
        self.eos_engine.calculate_state_2(state=state_T_neg)


        # Pertubacao no volume
        state_Vm_pos = copy.deepcopy(state) #self._copy_state(state=state)
        state_Vm_pos.z = state.z.copy()
        state_Vm_pos.Vm *= (1 + h_log)
        self.eos_engine.calculate_params(state=state_Vm_pos)
        self.eos_engine.calculate_state_2(state=state_Vm_pos)

        state_Vm_neg = copy.deepcopy(state) #self._copy_state(state=state)
        state_Vm_neg.z = state.z.copy()
        state_Vm_neg.Vm *= (1 - h_log)
        self.eos_engine.calculate_params(state=state_Vm_neg)
        self.eos_engine.calculate_state_2(state=state_Vm_neg)
        
        # As derivadsa dos criterios
        # Calcula os critérios para cada estado perturbado
        b_z1_pos, c_z1_pos = self._get_criteria(state_z1_pos)
        b_z1_neg, c_z1_neg = self._get_criteria(state_z1_neg)
        b_T_pos, c_T_pos = self._get_criteria(state_T_pos)
        b_T_neg, c_T_neg = self._get_criteria(state_T_neg)
        b_Vm_pos, c_Vm_pos = self._get_criteria(state_Vm_pos)
        b_Vm_neg, c_Vm_neg = self._get_criteria(state_Vm_neg)

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
        self.eos_engine.calculate_params(state=current_state)
        self.eos_engine.calculate_state_2(state=current_state)

        # Calcula os critérios b e c
        # b, u = self._calculate_eingen(self._calculate_B_matrix(state=current_state))
        # c = self._calculate_c(u, current_state)
        b, c = self._get_criteria(state=current_state)
        
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

    def _newton_solver(self, state_guess: State, spec_var_index: int, spec_var_value: float, max_iter: int=50, tol: float=1e-6):
        local_state = copy.deepcopy(state_guess)

        # Se vai ser temperatura ou volume molar, precisa empacotar num ln
        if spec_var_index == 1 or spec_var_index == 2 or spec_var_index == 3:  # PONTO DE APOIO
            S_target = np.log(spec_var_value)
        else:
            S_target = spec_var_value

        if local_state.P is None:
            self.eos_engine.calculate_params(state=local_state)
            self.eos_engine.calculate_state_2(state=local_state)

        X = np.array([
            local_state.z[0],
            np.log(local_state.T),
            np.log(local_state.Vm),
            ])

        for i in range(max_iter):
            F = self._calculate_system_of_equations(variables=X, state_template=local_state, spec_var_index=spec_var_index, S_target=S_target)
            local_state.z[0] = X[0]; local_state.z[1] = 1.0 - X[0]
            local_state.T = np.exp(X[1])
            local_state.Vm = np.exp(X[2])
            self.eos_engine.calculate_params(state=local_state)
            self.eos_engine.calculate_state_2(state=local_state)
            

            # Verifica a convergência
            norm_F = np.linalg.norm(F)
            if norm_F < tol:
                # 
                # print(f"Convergência atingida em {i+1} iterações, sendo X = {X}")
                print(f"{i:<8} | {norm_F:.3e}, {local_state.z[0]}, {local_state.T -273.15}, {local_state.P /10**5}, {local_state.Z} ,{local_state.Vm}")

                # Retorna o último estado consistente calculado dentro de _system_of_equations
                return local_state, X, i+1

            J = self._calculate_jacobian(state=local_state, spec_var_index=spec_var_index)
            delta_X = np.linalg.solve(J, -F)

            X = X + delta_X
        return None, None, None
    
    def _calculate_sensitivity_vector(self, state: State, spec_var_index: int):
        J = self._calculate_jacobian(state=state, spec_var_index=spec_var_index)
        F = np.zeros(3)  #   --------------------------> ponto de apoio
        F[spec_var_index] = -1

        dX_dS = np.linalg.solve(J, -F)
        # print(dX_dS)

        return dX_dS

    def _calculate_next_step(self, state: State, spec_var_index: int, X: np.ndarray, iter_newton: int):
        dX_dS = self._calculate_sensitivity_vector(state=state, spec_var_index=spec_var_index)
        delta_S = 0.001
        delta_S_max = 0.05
        if iter_newton <= 3:
            delta_S = min(delta_S*1.25, delta_S_max)
        elif iter_newton >= 5:
            delta_S = delta_S / 2

        # if iter_newton <= 3:
        #     delta_S = 0.00075
        # elif iter_newton >= 5:
        #     delta_S = 0.000075

        X = X + dX_dS * delta_S
        spec_var_index_new = np.argmax(np.abs(dX_dS))

        spec_var_value_new = X[spec_var_index_new]

        if spec_var_index_new == 1 or spec_var_index_new == 2:
            spec_var_value_new = np.exp(spec_var_value_new)
        return X, spec_var_index_new, spec_var_value_new

    def _get_PTVm(self, state: State) -> tuple:
        PTVm = np.array([state.T - 273.15, state.P / 10**5, state.Vm]) # [ºC, bar, mol m-3]
        return PTVm

    def calcula_2(self, state: State, spec_var_index: int=0, spec_var_value: float=0.001) -> tuple:
        """
        index = 0 -> variavel especificada é a composicao
        index = 1 -> variavel especificada é temperatura
        index = 2 -> variavel especificada é o volume molar
        """
        PTVm = []
        current_state = copy.deepcopy(state)
        

        for ponto_idx in range(5000):
            converged_state, X_old, iter_newton = self._newton_solver(state_guess=current_state, spec_var_index=spec_var_index, spec_var_value=spec_var_value)
            if converged_state is None:
                print('possivelmente um ponto criondenbar!')
                break

            PTVm.append(self._get_PTVm(state=converged_state))
            if converged_state.z[0] >= 0.99 or converged_state.P > 2e9: # Pressão em bar
                print("Condição de parada da linha atingida.")
                break

            X_new, spec_var_index_new, spec_var_value_new = self._calculate_next_step(state=converged_state, spec_var_index=spec_var_index, X=X_old, iter_newton=iter_newton)
            
            if X_new[0] > 1.0:
                X_new[0] = 0.999
            elif X_new[0] < 0.0:
                X_new[0] = 0.0

            spec_var_index = spec_var_index_new
            spec_var_value = spec_var_value_new

            current_state.z = np.array([X_new[0], 1 - X_new[0]])
            current_state.T = np.exp(X_new[1])
            current_state.Vm = np.exp(X_new[2])
            print(ponto_idx)
        return PTVm
    