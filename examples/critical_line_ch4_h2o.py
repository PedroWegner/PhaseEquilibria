
from thermo_lib.components import Component, Mixture
from thermo_lib.factory import EoSFactory
from thermo_lib.phase_equilibria.binary_critical_line import CriticalLineCalculator
from thermo_lib.state import State
from thermo_lib.eos.eos_abc import EquationOfState
import numpy as np
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
from scipy.optimize import minimize


class ELV_pure:
    def __init__(self, eos_model: EquationOfState):
        self.eos_model = eos_model

    def _compute_FO(self, state: State, P: float):
        local_state = deepcopy(state)
        local_state.P = P

        # Calcula a compressibilidade e fugacidade da fase vapor
        self.eos_model.calculate_from_TP(state=local_state, is_vapor=True)
        self.eos_model.calculate_fugacity(state=local_state)
        phi_v = local_state.fugacity_dict['phi']

        # Calcula a compressibilidade e fugacidade da fase liquida
        self.eos_model.calculate_from_TP(state=local_state, is_vapor=False)
        self.eos_model.calculate_fugacity(state=local_state)
        phi_l= local_state.fugacity_dict['phi']

        FO = (np.log(phi_l) - np.log(phi_v))**2
        return FO
        

    def _compute_dFO_dP(self, state: State, P: float, eta: float=0.001):
        P_pos = P + eta
        FO_pos = self._compute_FO(state=state, P=P_pos)
        P_neg = P - eta
        FO_neg = self._compute_FO(state=state, P=P_neg)

        dFO_dP = (FO_pos - FO_neg) / (2 * eta)
        return dFO_dP
        

    def calcule_by_NR(self, state: State, T: float, P_k: float, max_iter: int=250, tol: float=1e-6) -> float:
        state_local = deepcopy(state)
        state_local.T = T
        for _ in range(max_iter):
            FO = self._compute_FO(state=state_local, P=P_k)
            dFO = self._compute_dFO_dP(state=state_local, P=P_k)
            P_k1 = P_k - FO / dFO

            if np.abs(P_k- P_k1) < tol:
                break
            P_k = P_k1
        
        return P_k[0]

from time import time
if __name__ == '__main__':
    # 1. Define the critical properties for two molecules
    ch4 = Component(name='ch4', Tc=190.6, Pc=45.99e5, omega=0.012)
    water = Component(name='water', Tc=647.1, Pc=220.55e5, omega=0.345)

    # 2. Define binary parameters (k_ij and l_ij)
    kij = 0.0
    k_ij = np.array([[0, kij],[kij,0]])

    # 3. Construct a mixture with two molecules
    mixture = Mixture([ch4, water], k_ij=k_ij, l_ij=0.0)

    # 4. This is defined, the algorithm start considering x_1 equals or near to zero
    z = np.array([0.001, 0.999])

    # 5. Set a state, the temperature and molar volume don't change the final result
    trial_state = State(mixture=mixture, T=280, Vm=25e5, z=z)

    # 6. Set the equation of state
    eos_factory = EoSFactory()
    eos_model = eos_factory.get_eos_model(model_name='PR')

    # 7. Construct a critical line calculator (this method is implemented considering cubic equations)
    critical_line_calculator = CriticalLineCalculator(eos_model=eos_model)
    critical_line_calculator.initial_guess(state=trial_state)
    # 8. Set initial guess, this can change the final result or the time to reach
    z_alvo = 0.001
    T_alvo = trial_state.T
    Vm_alvo = trial_state.Vm
    z_chute = z_alvo * 1.05
    T_chute = T_alvo * 1.05
    Vm_chute = Vm_alvo * 0.98

    # 9. Start tracing the binary critical line
    trial_state.T = T_chute
    trial_state.Vm = Vm_chute
    trial_state.z = np.array([z_chute, 1 - z_chute])
    PTVm = critical_line_calculator.trace_line(initial_state=trial_state)

    


    # 10. Linha ELV pura
    T_water = np.linspace(550, 644.5, 300)
    T_ch4 =  np.linspace(150, 187, 300)
    P_water_0 = 60e5
    P_ch4_0 = 3e5
    P_water = []
    T_water_c = []

    P_ch4 = []
    T_ch4_c = []

    T_test = 350 # K
    P_test = 3e5 # Pa
    z_test = np.array([1.0])
    mixture_test = Mixture(components=[water], k_ij=0.0, l_ij=0.0)
    water_state = State(mixture=mixture_test, z=z_test, T=T_test, P=P_test)
    elv_pure = ELV_pure(eos_model=eos_model)
    mixture_test = Mixture(components=[ch4], k_ij=0.0, l_ij=0.0)
    co2_state = State(mixture=mixture_test, z=z_test, T=T_test, P=P_water_0)


    print("--- Newton-Raphson---")
    for T in T_water:
        P_water_0 = elv_pure.calcule_by_NR(state=water_state, T=T, P_k=P_water_0)
        P_water.append(P_water_0 / 10**5)
        T_water_c.append(T - 273.15)
    plt.plot(T_water_c, P_water, linewidth=1.25, color='black')

    for T in T_ch4:
        print(T)
        P_ch4_0 = elv_pure.calcule_by_NR(state=co2_state, T=T, P_k=P_ch4_0)
        P_ch4.append(P_ch4_0/ 10**5)
        T_ch4_c.append(T - 273.15)

    # 11. Plotting the binary line
    x = []
    y = []
    for i in PTVm:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x, y, linestyle='--', linewidth=1.25, color='blue', label="V-L")

    plt.plot(T_water_c, P_water, linewidth=1.25, color='black')
    plt.plot(T_ch4_c, P_ch4, linewidth=1.25, color='black')
    plt.xlabel('T / ºC')
    plt.ylabel('P / bar')
    plt.ylim(top=560, bottom=30)
    plt.show()

    