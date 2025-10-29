from CubicEoS_module import * 
import numpy as np
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
import numdifftools as nd
from time import time
import copy
class tpdCalculator:
    def __init__(self, EoS_Engine):
        self.eos_engine = EoS_Engine()
        pass
    
    def _calculate_tm(self, W: np.ndarray, state_local: State, di: np.ndarray) -> float:
        z = W / np.sum(W)
        state_local.z = z
        self.eos_engine.calculate_state(state=state_local)
        tm = 1 + np.sum(W * (np.log(W) + np.log(state_local.fugacity_dict['phi']) - (di + 1)))

        return tm

    def _get_initial_guess(self, state: State):
        Pc = np.array([c.Pc for c in state.mixture.components])
        Tc = np.array([c.Tc for c in state.mixture.components])
        omega = np.array([c.omega for c in state.mixture.components])
        K = (Pc / state.P) * np.exp(5.373 * (1 + omega) * (1 - Tc / state.T))
        # estimativas inicias para fase vapor e liquida, teremos que testar as duas mesmo
        y = K * state.z
        x = state.z / K

        y = y / np.sum(y)
        x = x / np.sum(x)

        return x, y 

    def _calculate_g_hessian(self, W: np.ndarray, state_local: State, di: np.ndarray):
        z = W / np.sum(W)
        state_local.z = z

        self.eos_engine.calculate_state(state=state_local)
        I = np.identity(len(di))
        gi = np.sqrt(W)*(np.log(W) + np.log(state_local.fugacity_dict['phi']) - di)

        Hij = I + np.sqrt(np.outer(W, W)) * state_local.fugacity_dict['dlnphi_dni'] 
        
        return gi, Hij  

    def _find_minimum(self, state_local: State, di: np.ndarray):
        W0 = state_local.z
        alpha_K = 2 * np.sqrt(W0)
        S = np.identity(len(state_local.z))

        for _ in range(30):
            Wk = (alpha_K / 2)**2
            gK, Hk = self._calculate_g_hessian(W=Wk, state_local=state_local, di=di)

            if np.linalg.norm(gK) < 1e-6:
                print('convergiu')
                break

            eta = 0.0
            delta_alpha = 0.0
            for _ in range(15):
                H_mod = Hk + eta * S
                try:
                    delta_alpha = np.linalg.solve(H_mod, -gK)
                    break
                except np.linalg.LinAlgError:
                    eta = eta*10 if eta > 0 else 1e-6
            if delta_alpha is None:
                print("deu ruim")

            alpha_K += delta_alpha
        
        W = (alpha_K / 2)**2
        tm = self._calculate_tm(W=W, state_local=state_local, di=di)
        return W, tm

    def _get_di(self, state: State):
        di = np.log(state.z) + np.log(state.fugacity_dict['phi'])
        return di

    def check_stability(self, state: State):
        x, y = self._get_initial_guess(state=state)
        liq_state = State(mixture=state.mixture,
                          T=state.T,
                          P=state.P,
                          z=x,
                          is_vapor=False)
        vap_state = State(mixture=state.mixture,
                          T=state.T,
                          P=state.P,
                          z=y,
                          is_vapor=True)
        
        di = self._get_di(state=state)

        W_liq, tm_liq = self._find_minimum(state_local=liq_state, di=di)
        W_vap, tm_vap = self._find_minimum(state_local=vap_state, di=di)

        print(W_liq, W_liq/np.sum(W_liq), tm_liq)
        print(W_vap, W_vap / np.sum(W_vap),tm_vap)

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
        J = np.zeros((3,3), dtype=np.float64)

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

        # Preenche a segunda linha (derivadas de c)
        J[2, 0] = (c_z1_pos - c_z1_neg) / (2 * h_z1)      # ∂c/∂z₁
        J[2, 1] = (c_T_pos - c_T_neg) / (2 * h_log)       # ∂c/∂(ln T)
        J[2, 2] = (c_Vm_pos - c_Vm_neg) / (2 * h_log)     # ∂c/∂(ln Vm)

        # --- 4. LINHA DA ESPECIFICAÇÃO (g) ---
        # A derivada da função de especificação g = X_s - S
        return J

    def _calculate_system_of_equations(self, variables: list, state_template: State, spec_var_index: int, S_target: float) -> list:
        """
        Esta é a função que define o sistema a ser zerado.
        Ela recebe as variáveis [z₁, lnT, lnV] e retorna o vetor de erros [b, c, g].
        """
        z1, lnT, lnV = variables
        
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
        
        # --- CÁLCULO CORRETO DE 'g' ---
        # Compara a variável correspondente do estado atual com o valor alvo 'S'.
        if spec_var_index == 0:
            g = current_state.z[0] - S_target
        elif spec_var_index == 1:
            g = np.log(current_state.T) - S_target
        elif spec_var_index == 2:
            g = np.log(current_state.Vm) - S_target
        else:
            # Segurança
            g = 0

        return np.array([g, b, c], dtype=np.float64)


    def _newton_solver(self, state_guess: State, spec_var_index: int, spec_var_value: float, max_iter: int=50, tol: float=1e-6):
        local_state = copy.deepcopy(state_guess)

        # Se vai ser temperatura ou volume molar, precisa empacotar num ln
        if spec_var_index == 1 or spec_var_index == 2:
            S_target = np.log(spec_var_value)
        else:
            S_target = spec_var_value

        X = np.array([local_state.z[0], np.log(local_state.T), np.log(local_state.Vm)])

        # print("--- Iniciando Solver de Newton para Ponto Único ---")
        # print(f"Iteração | Norma de F")
        # print("-" * 30)

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
                # print("-" * 30)
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
        F = np.zeros(3)
        F[spec_var_index] = -1

        dX_dS = np.linalg.solve(J, -F)
        # print(dX_dS)

        return dX_dS

    def _calculate_next_step(self, state: State, spec_var_index: int, X: np.ndarray, iter_newton: int):
        dX_dS = self._calculate_sensitivity_vector(state=state, spec_var_index=spec_var_index)

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
        # print("novo X=", X)
        # print("novo spec_i=", spec_var_index_new)
        # print("novo spec_v=", spec_var_value_new)
        return X, spec_var_index_new, spec_var_value_new

    def _get_PTVm(self, state: State) -> tuple:
        PTVm = np.array([state.T - 273.15, state.P / 10**5, state.Vm]) # [ºC, bar, mol m-3]
        return PTVm

    def calcula_2(self, state: State) -> tuple:
        PTVm = []
        current_state = copy.deepcopy(state)
        spec_var_index = 0 # Primeiro ponto sempre será a composição do componente com Tc superior
        spec_var_value = 0.001

        for ponto_idx in range(5000):
            converged_state, X_old, iter_newton = self._newton_solver(state_guess=current_state, spec_var_index=spec_var_index, spec_var_value=spec_var_value)
            
            if converged_state is None:
                print('possivelmente um ponto criondenbar!')
                break

            PTVm.append(self._get_PTVm(state=converged_state))
            if converged_state.z[0] >= 0.99 or converged_state.P > 500e5: # Pressão em bar
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
    

class HighPressurCriticalLineSolver:
    def __init__(self, EoS_Engine: ModeloPengRobinson):
        self.eos_engine = EoS_Engine()

    def _calculate_B_matrix_PT(self, state: State):
        # self.eos_engine.calculate_params(state=state)
        # self.eos_engine.calculate_state_2(state=state)
        I = np.identity(len(state.mixture.components))
        B = I + np.sqrt(np.outer(state.z, state.z)) * state.fugacity_dict['dlnphi_dni']
        return B

    def _calculate_eingen(self, B: np.ndarray):
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        min_eigenvalue_index = np.argmin(eigenvalues)
        lambda1 = eigenvalues[min_eigenvalue_index]
        u = eigenvectors[:, min_eigenvalue_index]
        return lambda1, u
    
    def _get_min_eigenvalue_TP(self, state: State):
        B = self._calculate_B_matrix_PT(state=state)
        lambda_min, _ = self._calculate_eingen(B=B)

        return lambda_min
    
    def _find_unstable_state(self, state_template: State, P: float=1e8, T: float=300):
        z_range = np.linspace(0.001, 0.999, 50)
        min_lambda_overall = float('inf')
        z1_s = -1

        for z1_test in z_range:
            current_state = copy.deepcopy(state_template)
            current_state.z = np.array([z1_test, 1 - z1_test])
            current_state.P = P
            current_state.T = T
            Z = min(self.eos_engine._get_Z(state=current_state))
            self.eos_engine.calculate_state_3(state=current_state, Z=Z)

            lambda_val = self._get_min_eigenvalue_TP(state=current_state)
            if lambda_val < min_lambda_overall:
                min_lambda_overall = lambda_val
                z1_s = z1_test

        return z1_s, min_lambda_overall
    
    def _find_critical_temperature(self, state_template: State, z_s: np.ndarray, P: float, T_search: tuple):
        def lambda_as_function_of_T(temp_K: float) -> float:
            # Cria um estado com a T variável e z, P fixos
            current_state = copy.deepcopy(state_template)
            current_state.z = z_s
            current_state.T = temp_K
            current_state.P = P
            Z = min(self.eos_engine._get_Z(state=current_state))
            self.eos_engine.calculate_state_3(state=current_state, Z=Z)
            # Retorna o autovalor mínimo para esta temperatura
            return self._get_min_eigenvalue_TP(state=current_state)

        try: 
            T_crit = brentq(f=lambda_as_function_of_T, a=T_search[0], b=T_search[1], xtol=1e-6)
            return T_crit
        except ValueError:
            print('não achou nada')
            return None


    def calcula(self, state: State):
        lambda_min = self._get_min_eigenvalue_TP(state=state)
        print(lambda_min)


class tester:
    def __init__(self, EoS_Engine: ModeloPengRobinson):
        self.eos_engine = EoS_Engine()

    def _objective_function(self, vars, component: Component, T: float):
        P = vars[0]
        local_mixture = Mixture(components=[component], k_ij=0.0, l_ij=0.0)
        vapor_state = State(mixture=local_mixture, z=np.array([1.0]), T=T, P=P, is_vapor=True)
        liquid_state = State(mixture=local_mixture, z=np.array([1.0]), T=T, P=P, is_vapor=False)
        Z = self.eos_engine._get_Z(state=vapor_state)

        if len(Z) < 3:
            return 1e15
    
        liquid_state.params = vapor_state.params.copy()
        
        self.eos_engine.calculate_state_3(state=vapor_state, Z=max(Z))
        self.eos_engine.calculate_state_3(state=liquid_state, Z=min(Z))

        vapor_phi = vapor_state.fugacity_dict['phi']
        liquid_phi = liquid_state.fugacity_dict['phi']

        FO = (np.log(liquid_phi) - np.log(vapor_phi))**2
        return FO

    def calcula(self, component: Component, T: float, P: float) -> tuple:
        vars0 = [P]
        # fo = self._objective_function(vars0, component=component, T=T)
        result = minimize(fun=self._objective_function,
                          x0=vars0,
                          args=(component, T),
                          method='Nelder-Mead',
                          options={'maxiter': 1000,
                                   'xatol': 1e-15,
                                   'fatol': 1e-15,
                                   })
        return result.x[0]

if __name__ == '__main__':  
    metano = Component(name='Methane', Tc=190.6, Pc=45.99e5, omega=0.012)    
    dioxide = Component(name='Carbon Dioxide', Tc=304.2, Pc=73.83e5, omega=0.224)
    sulfeto = Component(name='sulfeto de hidrogenio', Tc=373.5, Pc=89.63e5, omega=0.094)
    k_ij = 0.08
    k_ij = np.array([[0, k_ij],[k_ij,0]])
    mixture = Mixture([metano, sulfeto], k_ij=k_ij, l_ij=0.0)
    z = np.array([0.001, 0.999])
    trial_state = State(mixture=mixture, T=280, P=25e5, z=z, is_vapor=True, n=1.0)

    # critical_calc = CriticalPointSolver(EoS_Engine=ModeloPengRobinson)
    # critical_calc.initial_guess(state=trial_state)
    high_pressure_calculator = HighPressurCriticalLineSolver(EoS_Engine=ModeloPengRobinson)
    z_s, _ = high_pressure_calculator._find_unstable_state(state_template=trial_state)
    z_s_vector = np.array([z_s, 1 - z_s])
    T_critica_estimada = high_pressure_calculator._find_critical_temperature(state_template=trial_state, z_s=z_s_vector, P=1e8, T_search=[150, 300])
    print(T_critica_estimada)