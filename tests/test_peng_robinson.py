import numpy as np
import pytest
from thermo_lib.factory import EoSFactory
from thermo_lib.components import Component, Mixture
from thermo_lib.state import State

def test_pr_liquid_vapor_butane():
    """
    Esse metodo testa a equacao de Peng-Robinson para o calculo do volume molar (cm3 mol-1) do butano
    Exemplo 3.9 Van Ness
    """
    butano = Component(name='Methane', Tc=425.1, Pc=37.96e5, omega=0.200)
    trial_mixture = Mixture(components=[butano], k_ij=np.array([0.0]), l_ij=np.array([0.0]))
    T = 350 # K
    P = 9.4573e5 # Pa
    z = np.array([1.0])
    trial_state = State(mixture=trial_mixture, z=z, T=T, P=P)

    eos_factory = EoSFactory()
    eos_model = eos_factory.get_eos_model(model_name='PR')

    expected_Vm_v = 2486 / 100**3 # cm3 mol-1
    expected_Vm_l = 112.6 / 100**3 # cm3 mol-1
    
    eos_model.calculate_from_TP(state=trial_state, is_vapor=True)
    calculate_Vm_v = trial_state.Vm
    eos_model.calculate_from_TP(state=trial_state, is_vapor=False)
    calculate_Vm_l = trial_state.Vm
    
    assert calculate_Vm_v == pytest.approx(expected_Vm_v, abs=1e-4)
    assert calculate_Vm_l == pytest.approx(expected_Vm_l, abs=1e-4)

