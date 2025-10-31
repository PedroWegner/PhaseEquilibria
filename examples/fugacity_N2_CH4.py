
from thermo_lib.components import Component, Mixture
from thermo_lib.factory import EoSFactory
from thermo_lib.phase_equilibria.binary_critical_line import CriticalLineCalculator
from thermo_lib.state import State
import numpy as np
import matplotlib.pyplot as plt
from time import time

if __name__ == '__main__':
    # 1. Define the critical properties for two molecules
    methane = Component(name='CH4', Tc=190.6, Pc=45.99e5, omega=0.012)    
    nitrogen = Component(name='N2', Tc=126.2, Pc=34.00e5, omega=0.038)

    # 2. Construct a mixture with two molecules
    mixture = Mixture([nitrogen, methane], k_ij=0.0, l_ij=0.0)

    # 3. This is defined, the algorithm start considering x_1 equals or near to zero
    z = np.array([0.40, 0.60])

    # 4. Set a state, the temperature and molar volume don't change the final result
    T = 200.0 # K
    P = 30e5 # Pa
    trial_state = State(mixture=mixture, T=T, P=P, z=z)

    # 5. Set the equation of state
    eos_factory = EoSFactory()
    eos_model = eos_factory.get_eos_model(model_name='PR')

    eos_model.calculate_from_TP(state=trial_state, is_vapor=True)
    eos_model.calculate_fugacity(state=trial_state)
    # print(trial_state.fugacity_dict['n_dlnphi_dni'])
    print('dF_dV com Peng-Robison = ',trial_state.helmholtz_derivatives['dF_dV'])
    # # print('dF_dP com Peng-Robison = ',trial_state.helmholtz_derivatives['dF_dP'])
    print('dF_dVV com Peng-Robison = ',trial_state.helmholtz_derivatives['dF_dVV'])
    print('dF_dniV com Peng-Robison = ',trial_state.helmholtz_derivatives['dF_dniV'])
    print('dF_dninj com Peng-Robison = ',trial_state.helmholtz_derivatives['dF_dninj'])
    print('dlnphi com Peng-Robison = ',trial_state.fugacity_dict['dlnphi_dni'])

    