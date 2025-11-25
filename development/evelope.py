from thermo_lib.components import Component, Mixture
from thermo_lib.factory import EoSFactory
from thermo_lib.state import State
import numpy as np
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy

def _compute_lnKi(mixture:Mixture, T:float, P:float):
    Tc = np.array([c.Tc for c in mixture.components])
    Pc = np.array([c.Pc for c in mixture.components])
    omega = np.array([c.omega for c in mixture.components])

    ln_Ki = np.log(Pc / P) + 5.373 * (1 + omega) * (1 - Tc / T)
    return ln_Ki

def _compute_x_y(z:np.ndarray, Ki:np.ndarray, beta:float):
    x = z / (1 - beta + beta * Ki)
    y = Ki * z / (1 - beta + beta * Ki)

    return x, y

def _compute_f(state:State, beta:float, X:np.ndarray, spec_var_index:int, S:float):
    """
    Aqui é para representar as C+2 funções
    spec_var_index: 
    0 -> lnP
    1 -> lnT
    C + 2 -> Ki
    """
    local_state = deepcopy(state)
    z = local_state.z
    lnP, lnT, lnKi = X[0], X[1], X[2:]
    P, T, Ki = np.exp(lnP), np.exp(lnT), np.exp(lnKi)

    local_state.T, local_state.P = T, P

    x, y = _compute_x_y(z=z, Ki=Ki, beta=beta)

    # A F2 vai ser o balanço de massa
    f2 = np.sum(y - x)

    x = x / np.sum(x)
    y = y / np.sum(y)

    # calcula o liquido
    local_state.z = x
    eos_model.calculate_from_TP(state=local_state, is_vapor=False)
    eos_model.calculate_fugacity(state=local_state)
    lnphi_l = local_state.fugacity_result.ln_phi

    # calcula o vapor
    local_state.z = y
    eos_model.calculate_from_TP(state=local_state, is_vapor=True)
    eos_model.calculate_fugacity(state=local_state)
    lnphi_v = local_state.fugacity_result.ln_phi

    # a F1 vai ser a especificação
    if spec_var_index == 0:
        f1 = lnP - S
    elif spec_var_index == 1:
        f1 = lnT - S
    else:
        # Para caso seja um dos Ki, precisa melhorar isso
        f1 = X[spec_var_index] - S
    
    

    fC_2 = lnKi + lnphi_v - lnphi_l
    return np.hstack([f1, f2, fC_2]), x, y


def _compute_jacobian_f(state:State, beta:float, X:np.ndarray, spec_var_index:int, C:int):
    J = np.zeros((C + 2, C + 2))

    local_state = deepcopy(state)
    z = local_state.z
    n = local_state.n
    lnP, lnT, lnKi = X[0], X[1], X[2:]
    P, T, Ki = np.exp(lnP), np.exp(lnT),np.exp(lnKi)
    local_state.T = T
    local_state.P = P

    x, y = _compute_x_y(z=z, Ki=Ki, beta=beta)


    x = x / np.sum(x)
    y = y / np.sum(y)
    
    # Derivada de x e y com relacao a Ki
    D = 1 - beta + beta * Ki
    dx_dlnK = - (beta * Ki * z) / D**2 
    dy_dlnK = Ki * z * (1 - beta) / D**2
    # Funcao da especificacao
    J[0, spec_var_index] = 1.0

    # balanço de massa
    J[1,2:] = dy_dlnK - dx_dlnK


    # Agora do equilibrio... talvez tenha que fazer numericamente, o que é feio... enfim

    # calcula o liquido
    local_state.z = x
    eos_model.calculate_from_TP(state=local_state, is_vapor=False)
    eos_model.calculate_fugacity(state=local_state)
    dlnphi_dT_l = local_state.fugacity_result.dlnphi_dT
    dlnphi_dP_l = local_state.fugacity_result.dlnphi_dP
    n_dlnphi_dn_l = local_state.fugacity_result.n_dlnphi_dni

    # calcula o vapor
    local_state.z = y
    eos_model.calculate_from_TP(state=local_state, is_vapor=True)
    eos_model.calculate_fugacity(state=local_state)
    dlnphi_dT_v = local_state.fugacity_result.dlnphi_dT
    dlnphi_dP_v = local_state.fugacity_result.dlnphi_dP
    n_dlnphi_dn_v = local_state.fugacity_result.n_dlnphi_dni

    for i in range(C):
        row = i + 2
        J[row, 0] = P * (dlnphi_dP_v[i] - dlnphi_dP_l[i])
        J[row, 1] = T * (dlnphi_dT_v[i] - dlnphi_dT_l[i])

        J[row, 2:] = np.eye(C)[i] + (n_dlnphi_dn_v[i, :] * dy_dlnK) - (n_dlnphi_dn_l[i, :] * dx_dlnK)

    
    # print(30*'*')
    # print("Derivada 'analitica' que voce me passou.......")
    # print(J)
    return J

def _compute_num_jacobian_f(state:State, beta:float, X:np.ndarray, spec_var_index:int, S:float, C:int, epislon:float=1e-6):
    J = np.zeros((C + 2, C + 2))
    for i in range(C+2):
        X_orig = X[i]

        temp_plus = X_orig + epislon
        h_plus = temp_plus - X_orig

        X_pert = X.copy()

        X_pert[i] = temp_plus
        f_plus, _, _ = _compute_f(state=state, beta=beta, X=X_pert, spec_var_index=spec_var_index, S=S)

        temp_minus = X_orig - epislon
        h_minus = X_orig - temp_minus

        X_pert[i] = temp_minus
        f_minus, _, _ = _compute_f(state=state, beta=beta, X=X_pert, spec_var_index=spec_var_index, S=S)

        J[:, i] = (f_plus - f_minus) / (h_plus + h_minus)

    # print(30*'*')
    # print("Derivada numerica")
    # print(J)
    # print(30*'*')
    return J




def newton_solver(state_guess:State, X:np.ndarray, beta:float, spec_var_index:int, S:float, max_iter:int=50, tol:float=1e-9):
        local_state = deepcopy(state_guess)
        C = len(local_state.z)

        
        for i in range(max_iter):
            lnP, lnT, lnKi = X[0], X[1], X[2:]
            P, T, Ki = np.exp(lnP), np.exp(lnT),np.exp(lnKi)

            f, x, y = _compute_f(state=local_state, beta=beta, X=X, spec_var_index=spec_var_index, S=S)
            norm_f = np.linalg.norm(f) # calcula a norma, ver convergencia
            J = _compute_jacobian_f(state=local_state, beta=beta, X=X, spec_var_index=spec_var_index, C=C)
            # J = _compute_num_jacobian_f(state=local_state, beta=beta, X=X, spec_var_index=spec_var_index, S=S, C=C)
            if norm_f < tol:
                # 
                # print(f"Convergência atingida em {i+1} iterações, sendo X = {X}")
                # print(f"{i:<8} | {norm_f:.3e}, {P/10**5}, {T}, {x}, {y}")

                # Retorna o último estado consistente calculado dentro de _system_of_equations
                return local_state, X, i+1, J

            
            delta_X = np.linalg.solve(J, -f)

            X = X + delta_X
        print('Newton falhou!')
        return None, None, None, None


def _compute_sensibility_vector(J:np.ndarray, C:int):
    df_dS = np.zeros(C + 2)
    df_dS[0] = -1.0

    dX_dS = np.linalg.solve(J, -df_dS)

    new_spec_index = np.argmax(np.abs(dX_dS))

    return dX_dS, new_spec_index


def _compute_cubic_prediction(X_curr:np.ndarray, X_old:np.ndarray, dX_dS_curr:np.ndarray, dX_dS_old:np.ndarray, step_curr:float,
                               step_old:float):
    h = step_old
    d = X_curr
    c = dX_dS_curr

    delta_X = X_curr - X_old 

    a_sens = h * (dX_dS_curr + dX_dS_old)
    b_sens = h * (2 * dX_dS_curr + dX_dS_old)

    a = (-2 * delta_X + a_sens) / h**3
    b = (-3 * delta_X + b_sens) / h**2
    s = step_curr
    X_next = (a * s**3) + (b * s**2) + (c * s) + d

    return X_next

def trace_envelope(state:State, beta:float, X_initial:np.ndarray, spec_var_index:int, S:float):
    local_state = deepcopy(state)
    C = len(local_state.mixture.components)

    X_space = []
    J_space = []
    dX_dS_space = []
    S_space = []

    _, X0, n_iter, J0 = newton_solver(state_guess=local_state, X=X_initial, beta=beta, spec_var_index=spec_var_index, S=S)
    dX_dS0, new_spec_index = _compute_sensibility_vector(J=J0, C=C)

    X_space.append(X0)
    J_space.append(J0)
    dX_dS_space.append(dX_dS0)
    S_new = S + np.log(1.01)
    delta_S = S_new - S
    X_new = X0 + dX_dS0 * delta_S

    _, X1, n_iter, J1 = newton_solver(state_guess=trial_state, X=X_new, beta=beta, spec_var_index=new_spec_index, S=S_new)
    dX_dS1, new_spec_index = _compute_sensibility_vector(J=J1, C=C)
    
    X_space.append(X1)
    J_space.append(J1)
    dX_dS_space.append(dX_dS1)


    step_old = delta_S
    
    

    X_curr = X1
    X_old = X0
    dX_dS_curr = dX_dS1
    dX_dS_old = dX_dS0

    for _ in range(19):
        print(_, new_spec_index)
        if n_iter < 3:
            step_curr = 1.25 * step_old
        elif n_iter >= 5:
            step_curr = 0.5 * step_old
        else:
            step_curr = step_old

        X_guess = _compute_cubic_prediction(X_curr=X_curr, X_old=X_old, dX_dS_curr=dX_dS_curr, dX_dS_old=dX_dS_old,
                                        step_curr=step_curr, step_old=step_old)
        S_guess = X_guess[new_spec_index]
        _, X_new, n_iter, J = newton_solver(state_guess=trial_state, X=X_guess, beta=beta, spec_var_index=new_spec_index, S=S_guess)
        dX_dS_new, next_spec_index = _compute_sensibility_vector(J=J, C=C)
        
        # guarda info
        X_space.append(X_new)

        # aqui arrumo as coisas
        X_old = X_curr
        dX_dS_old = dX_dS_curr
        X_curr = X_new
        dX_dS_curr = dX_dS_new
        step_old = step_curr
        new_spec_index = next_spec_index
    

    return X_space

if __name__ == '__main__':
    # 1. Define the critical properties for two molecules
    methane = Component(name='CH4', Tc=190.6, Pc=45.99e5, omega=0.012)    
    nitrogen = Component(name='N2', Tc=126.2, Pc=34.00e5, omega=0.038)
    ethane = Component(name='C2H6', Tc=305.3, Pc=48.72e5, omega=0.100)

    # 2. Construct a mixture with two molecules
    mixture = Mixture([methane, ethane], k_ij=0.0, l_ij=0.0)

    # 3. This is defined, the algorithm start considering x_1 equals or near to zero
    z = np.array([0.97, 0.03])

    # 4. Set a state, the temperature and molar volume don't change the final result
    T = 167.0 # K
    P = 6e5 # Pa
    C = len(mixture.components)
    trial_state = State(mixture=mixture, T=T, P=P, z=z, n=1)

    # 5. Set the equation of state
    eos_factory = EoSFactory()
    eos_model = eos_factory.get_eos_model(model_name='PR')

    eos_model.calculate_from_TP(state=trial_state, is_vapor=True)
    eos_model.calculate_fugacity(state=trial_state)


    lnKi = _compute_lnKi(mixture=mixture, T=T, P=P)
    beta = 1.0

    X = np.hstack([np.log(P), np.log(T), lnKi])
    f = _compute_f(state=trial_state, beta=1.0, X=X, spec_var_index=0, S=np.log(5e5))
    print(f)
    J = _compute_jacobian_f(state=trial_state, beta=beta, X=X, spec_var_index=0, C=C)
    J_num = _compute_num_jacobian_f(state=trial_state, beta=1.0, X=X, spec_var_index=0, S=np.log(5e5), C=C)

    S = np.log(5e5)
    X_space = trace_envelope(state=trial_state, beta=beta, X_initial=X, spec_var_index=0, S=S)

    # to = time()
    # S = np.log(5e5)
    # local_state, X, _, J = newton_solver(state_guess=trial_state, X=X, beta=beta, spec_var_index=0, S=S)
    # tf = time()
    # dX_dS, new_spec_index = _compute_sensibility_vector(J=J, C=C)

    # Snew = np.log(7e5)
    # delta_S = Snew - S
    # X_new = X + dX_dS * delta_S
    # local_new, X_new_new, _, J_new = newton_solver(state_guess=trial_state, X=X_new, beta=beta, spec_var_index=new_spec_index, S=Snew)
    # dX_dS_new, new_spec_index_new = _compute_sensibility_vector(J=J_new, C=C)
    
    # print(X)
    # print(X_new)
    # print(X_new_new)
    # X = _compute_cubic_prediction(X_new=X_new_new, X_old=X, dX_dS_new=dX_dS_new, dX_dS_old=dX_dS, step_new=np.log(2), step_old=delta_S)
    # print('tempo total: ', tf-to)
    # print(X)

    X_space = np.array(X_space)

    T_space = []
    P_space = []

    for i in range(len(X_space)):
        T_space.append(np.exp(X_space[i,1]) - 273.15)
        P_space.append(np.exp(X_space[i,0]) / 10**5)

    print(P_space)
    plt.figure(figsize=(10, 8))
    # plt.scatter(x_exp, P_exp, marker='x', color='navy', label=f'T={T} [K]')
    # plt.scatter(y_exp, P_exp, marker='x', color='navy')
    plt.plot(T_space, P_space, color='k', linewidth=0.85)
    # plt.plot(x_space, P_space, color='k', linewidth=0.85)
    # plt.ylabel(ylabel=r'$P\;/\;bar$')
    # plt.xlabel(xlabel=r'$x_{CO_{2}}\;/\;y_{CO_{2}}$')
    # plt.xlim(left=0.0, right=1.0)
    # plt.ylim(bottom=P_space[0]*0.8)
    # plt.legend(loc='upper left')
    plt.show()