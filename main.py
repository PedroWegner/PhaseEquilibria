
from thermo_lib.components import Component, Mixture
from thermo_lib.factory import EoSFactory
from thermo_lib.phase_equilibria.binary_critical_line import CriticalLineCalculator
from thermo_lib.state import State
import numpy as np
import matplotlib.pyplot as plt
from time import time

if __name__ == '__main__':



    sulfeto = Component(name='H2S', Tc=373.5, Pc=89.63e5, omega=0.094)    
    dioxide = Component(name='CO2', Tc=304.2, Pc=73.83e5, omega=0.224)
    kij = 0.099
    k_ij = np.array([[0, kij],[kij,0]])
    mixture = Mixture([dioxide, sulfeto], k_ij=k_ij, l_ij=0.0)
    z = np.array([0.001, 0.999])

    trial_state = State(mixture=mixture, T=280, Vm=25e5, z=z)

    eos_factory = EoSFactory()
    eos_model = eos_factory.get_eos_model(model_name='PR')

    critical_line_calculator = CriticalLineCalculator(eos_model=eos_model)
    critical_line_calculator.initial_guess(state=trial_state)

    t02 = time()

    z_alvo = 0.001
    T_alvo = 304.2
    Vm_alvo = 9.8e-5

    z_chute = z_alvo * 1.05
    T_chute = T_alvo * 1.1
    Vm_chute = Vm_alvo * 0.98
    trial_state.T = T_chute
    trial_state.Vm = Vm_chute
    trial_state.z = np.array([z_chute, 1 - z_chute])
    PTVm = critical_line_calculator.trace_line(initial_state=trial_state)
    tf2 = time()

    x = []
    y = []
    for i in PTVm:
        x.append(i[0])
        y.append(i[1])
    # plt.plot(T_space, P_space, linestyle='--', linewidth=1.25, color='red', label="V-L")
    plt.plot(x, y, linestyle='--', linewidth=1.25, color='blue', label="V-L")
    # plt.plot(T_sulfur_space, P_sulfur_space, color='k', linewidth=1.0)
    # plt.plot(T_dioxide_space, P_dioxide_space, color='k', linewidth=1.0)
    # plt.scatter(Tc_space, Pc_space, marker='x', color='k')
    plt.xlabel('T / ºC')
    plt.ylabel('P / bar')
    print(tf2 - t02)
    plt.show()