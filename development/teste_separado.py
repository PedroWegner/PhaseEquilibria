from Pc_saft import *
from thermo_lib.components import Component, Mixture

def test_gibbs_duhem(state: State) -> None:
    ni = state.n * state.z
    dlnphi_dni = state.fugacity_result.dlnphi_dni
    res = ni @ dlnphi_dni
    _test = np.allclose(res, 0, atol=1e-6)
    if _test:
        print("Teste de Gibbs-Duhem passou (Eq. 34, cap .2)")
    else:
        print("Gibbs-Duhem é uma fraude aqui")

T = 200 # K
P = 30e5 # Pa

nitrogenio = Component(
    name='N2',
    Tc=None,
    Pc=None,
    omega=None,
    sigma=3.3130,
    epsilon=90.96,
    segment=1.2053
)

metano = Component(
    name='CH4',
    Tc=None,
    Pc=None,
    omega=None,
    sigma=3.7039,
    epsilon=150.03,
    segment=1.000
)

mixture = Mixture(
    components=[nitrogenio, metano],
    k_ij=0.0,
    l_ij=0.0
    )

state_trial = State(
    mixture=mixture,
    z=np.array([0.4, 0.6]),
    T=T,
    P=P
)


# seta a calc
pc_saft_engine = PCSaft(workers=None)
pc_saft_engine.calculate_from_TP(state=state_trial, is_vapor=True)
pc_saft_engine.calculate_fugacity(state=state_trial)

print(state_trial.V)
# print('dF_dV com PC-SAFT = ',state_trial.helmholtz_result.dF_dV)
# print('dF_dP com PC-SAFT = ', state_trial.helmholtz_result.dF_dP)
# print('dF_dVV com PC-SAFT = ',state_trial.helmholtz_result.dF_dVV)
# print('dF_dni com PC-SAFT = ',state_trial.helmholtz_result.dF_dni)
# print('dF_dniV com PC-SAFT = ',state_trial.helmholtz_result.dF_dniV)
print('dF_dninj com PC-SAFT = ',state_trial.helmholtz_result.dF_dninj)
# print('dlnphi com PC-SAFT = ',state_trial.fugacity_result.dlnphi_dni)

test_gibbs_duhem(state=state_trial)

