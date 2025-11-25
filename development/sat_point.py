from thermo_lib.state import State
from thermo_lib.components import Component, Mixture
import numpy as np
from thermo_lib.eos.eos_abc import EquationOfState
from thermo_lib.factory import EoSFactory
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import minimize
import openpyxl as pyxl
import os 
from Pc_saft import *

class PressureSatEngine:
    def __init__(self, eos_model: EquationOfState):
        # 1. Inicializa a misutra testada, a Engine Termodinamica (Equacao de Estado)
        # A EoS precisa ter o metodo calculate_from_TP e a derivada dlnphi_dP implementados
        self.eos_model = eos_model

        
    def _compute_Q_dQ_dP(self, T:float, P:float, z:np.ndarray, y:np.ndarray, state:State):
        # 1. Cria uma copia do estado independente e atribui os valores de P e T da iteracao
        local_state = deepcopy(state)
        local_state.T = T
        local_state.P = P
        
        try:
            # 2. testa a fase liquida
            local_state.z = z
            self.eos_model.calculate_from_TP(state=local_state, is_vapor=False)
            self.eos_model.calculate_fugacity(state=local_state)
            lnphi_l = local_state.fugacity_result.ln_phi
            dlnphi_dP_l = local_state.fugacity_result.dlnphi_dP
            
            # 3. testa a fase vapor
            local_state.z = y
            self.eos_model.calculate_from_TP(state=local_state, is_vapor=True)
            self.eos_model.calculate_fugacity(state=local_state)
            lnphi_v = local_state.fugacity_result.ln_phi
            dlnphi_dP_v = local_state.fugacity_result.dlnphi_dP
        except ValueError:
            # Esse except so esta aqui para poder testar o ponto de orvalho durante a otimizacao do parametro...
            # 2. testa a fase liquida
            local_state.z = z
            self.eos_model.calculate_from_TP(state=local_state, is_vapor=True)
            self.eos_model.calculate_fugacity(state=local_state)
            lnphi_l = local_state.fugacity_result.ln_phi
            dlnphi_dP_l = local_state.fugacity_result.dlnphi_dP
            
            # 3. testa a fase vapor
            local_state.z = y
            self.eos_model.calculate_from_TP(state=local_state, is_vapor=False)
            self.eos_model.calculate_fugacity(state=local_state)
            lnphi_v = local_state.fugacity_result.ln_phi
            dlnphi_dP_v = local_state.fugacity_result.dlnphi_dP

        # # 2. testa a fase liquida
        # local_state.z = z
        # self.eos_model.calculate_from_TP(state=local_state, is_vapor=False)
        # self.eos_model.calculate_fugacity(state=local_state)
        # lnphi_l = local_state.fugacity_result.ln_phi
        # dlnphi_dP_l = local_state.fugacity_result.dlnphi_dP
        
        # # 3. testa a fase vapor
        # local_state.z = y
        # self.eos_model.calculate_from_TP(state=local_state, is_vapor=True)
        # self.eos_model.calculate_fugacity(state=local_state)
        # lnphi_v = local_state.fugacity_result.ln_phi
        # dlnphi_dP_v = local_state.fugacity_result.dlnphi_dP

        # 4. Aplica os limites para evitar overflow
        diff_lnphi = np.clip(lnphi_l - lnphi_v, -50.0, 50.0)
        K = np.exp(diff_lnphi)
        dK_dP = K * (dlnphi_dP_l - dlnphi_dP_v)
        aux_y = z * K
        Q = 1.0 - np.sum(aux_y)
        dQ_dP =  - np.sum(z * dK_dP)
        
        return Q, dQ_dP, aux_y

    def calculate_pressure_saturation(self, state:State, T:float, z:np.ndarray, P_guess:float, y_guess:np.ndarray,
                                      max_iter:int=1500, P_tol:float=1e-6, y_tol:float=1e-5) -> any:
        """
        Aqui eh para implementar a logica melhor
        """
        P = P_guess
        y = np.copy(y_guess)


        MIN_DERIVATIVE = 1e-20
        DAMPING_FACTOR = 0.5
        # state = State(mixture=mixture, T=T, P=P, z=None)
        for _ in range(max_iter):
            # 1. Obtem os valores do problema
            Q2, dQ2_dP, aux_y = self._compute_Q_dQ_dP(T=T, P=P, z=z, y=y, state=state)

            
            if abs(dQ2_dP) < MIN_DERIVATIVE:
            # dQ_dP é zero. Newton é impossível.
            # Usamos o "Plano B": um passo de fallback (ex: 10% de P)
            # A direção do passo é -np.sign(Q)
                delta_P = -np.sign(Q2) * P * 0.1 
                # print(f"Iter {_}: Aviso dQ/dP é nulo. Usando fallback.")
            else:
                # A derivada é segura. Usamos o "Plano A" (Newton)
                delta_P = - Q2 / dQ2_dP


            max_step = P * DAMPING_FACTOR
            if abs(delta_P) > max_step:
                delta_P = max_step * np.sign(delta_P)
                # print(f"Iter {_}: Damping ativado. Passo limitado.")


            # 2. Calcula a nova pressao e composicao da fase incipiente
            P_new = P + delta_P

            if abs(P_new - P) > 0.5 * P:
                P_new = P + 0.5 * P * np.sign(P_new - P)
            if P_new <= 0:
                P_new = P / 2.0

            y_new = aux_y / np.sum(aux_y)

            # 3. Determina o erro para analise de convergencia
            P_error = abs((P_new - P) / P)
            y_error = np.linalg.norm(y_new - y)

            if P_error < P_tol and y_error < y_tol:
                return P_new, y_new
        
            P, y = P_new, y_new
        return P, y



def FO(vars, state:State, x_exp:np.ndarray, T_exp:np.ndarray):
    local_state = deepcopy(state)
    kij = vars[0]
    lij = vars[1]
    # kij = vars[0]
    # print('kij=',kij)
    # k_ij = np.array([[0, kij],[kij, 0]])    
    # local_state.mixture.k_ij = k_ij
    FO = 0.0
    dict_T = {}
    for T, P_exp in T_exp.items():
        k_ij = np.array([[0, kij],[kij, 0]])
        l_ij = np.array([[0, lij],[lij, 0]])
        local_state.mixture.k_ij = k_ij
        local_state.mixture.l_ij = l_ij
        y_space = []
        P_space = []
        y = np.array([0.90, 0.10])
        if not T in dict_T:
            dict_T[T] = []
        for i, P in enumerate(P_exp):
            x1 = x_exp[i]
            x = np.array([x1, 1 - x1])
            P_guess = P * 10**5
            P, y = pressure_sat_calculator.calculate_pressure_saturation(T=T, z=x, P_guess=P_guess, y_guess=y, state=local_state)
            y_space.append(y[0])
            P_space.append(P / 10**5)
            
        
        P_space = np.array(P_space)
        P_exp = np.array(P_exp)
        # FO += np.sum(abs((P_exp - P_space) / P_exp))
        FO += np.sum((P_exp - P_space)**2)
    return FO




if __name__ == '__main__':
    # 1. Define the critical properties for two molecules
    isooctano = Component(name='C8H18', Tc=543.9, Pc=25.68e5, omega=0.304)    
    dioxide = Component(name='CO2', Tc=304.2, Pc=73.83e5, omega=0.224)
   
    # 2. Construct a mixture with two molecules
    mixture = Mixture([dioxide, isooctano], k_ij=None, l_ij=0.0)
    T = 343
    P = 97300
    state = State(mixture=mixture, T=T, P=P, z=None)

    # 5. Set the equation of state and the saturation pressure calculator
    eos_factory = EoSFactory()
    eos_model = eos_factory.get_eos_model(model_name='PR')  

    pressure_sat_calculator = PressureSatEngine(eos_model=eos_model)
