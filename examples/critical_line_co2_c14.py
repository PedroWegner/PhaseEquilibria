
from thermo_lib.components import Component, Mixture
from thermo_lib.factory import EoSFactory
from thermo_lib.phase_equilibria.binary_critical_line import CriticalLineCalculator
from thermo_lib.state import State
import numpy as np
import matplotlib.pyplot as plt
from time import time

if __name__ == '__main__':
    # 1. Define the critical properties for two molecules
    tetra = Component(name='C14H30', Tc=693.3, Pc=15.73e5, omega=0.498)    
    dioxide = Component(name='CO2', Tc=304.2, Pc=73.83e5, omega=0.224)
    isoc8 = Component(name='iso-octane', Tc=544.0, Pc=25.68e5, omega=0.302)

    # 2. Define binary parameters (k_ij and l_ij)
    kij = 0.0
    k_ij = np.array([[0, kij],[kij,0]])

    # 3. Construct a mixture with two molecules
    mixture = Mixture([dioxide, isoc8], k_ij=k_ij, l_ij=0.0)

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

    # 10. Plotting the binary line
    x = []
    y = []
    for i in PTVm:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x, y, linestyle='--', linewidth=1.25, color='blue', label="V-L")
    plt.xlabel('T / ºC')
    plt.ylabel('P / bar')
    plt.show()