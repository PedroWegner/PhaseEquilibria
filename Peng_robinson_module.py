import numpy as np
from scipy.optimize import  fsolve, minimize
import matplotlib.pyplot as plt
from sympy.physics.units import stefan

# EQUAÇÕES DO VAN NESS
sigma_PR = 1 + np.sqrt(2)
epsilon_PR = 1 - np.sqrt(2)
omega_PR = 0.07780
psi_PR = 0.45724
R = 8.3144621

class Peng_robinson_state():
    def __init__(self,T: float, P: float, ncomp: int, x: list[float], Tc: list[float], Pc: list[float], omega: list[float]):
        # VARIAVEIS DA MISTURA
        self.ncomp = ncomp
        self.x = np.array(x)
        self.Tc = np.array(Tc)
        self.Pc = np.array(Pc)
        self.omega = np.array(omega)
        # PROPRIEDADES DO SISTEMA
        self.T = T
        self.P = P
        self.Z = type[float]
        #
        self.a = type[float]
        self.b = type[float]
        self.beta = type[float]
        self.q = type[float]
        # ARRAYS PARA CALCULAR A COMPRESSIBILIDADE
        self.a_i = np.zeros(self.ncomp)
        self.b_i = np.zeros(self.ncomp)
        self.a_ij = np.zeros((self.ncomp, self.ncomp))
        self.b_ij = np.zeros((self.ncomp, self.ncomp))
        self.k_ij = np.zeros((self.ncomp, self.ncomp))
        self.l_ij = np.zeros((self.ncomp, self.ncomp))

        self.I = type[float]
        self.prime_a_i = np.zeros(self.ncomp)
        self.prime_b_i = np.zeros(self.ncomp)
        self.prime_q_i = np.zeros(self.ncomp)
        self.ln_phi = np.zeros(self.ncomp)
        self.phi = np.zeros(self.ncomp)

def update_state(state: Peng_robinson_state, liq: bool) -> None:
    calc_ind_parameters(state=state)
    calc_combining_rule(state=state)
    calc_parameter_mixture(state=state)
    if liq:
        compressibility(state=state, Z_init=state.beta)
    else:
        compressibility(state=state, Z_init=1.0e0)
    prime_p_i(state=state)

def calc_ind_parameters(state: Peng_robinson_state) -> None:
    global omega_PR, psi_PR, R
    # EQ 3.44
    state.b_i = omega_PR * R * (state.Tc / state.Pc)
    #
    alpha = (1 + (0.37464 + 1.54223 * state.omega - 0.26992 * state.omega**2) * (1 - np.sqrt(state.T / state.Tc)))**2
    state.a_i = psi_PR * (R * state.Tc)**2 * alpha / state.Pc

def calc_combining_rule(state: Peng_robinson_state) -> None:
    # aqui teria que adicionar regra para b_ij
    state.a_ij = np.sqrt(state.a_i[:, np.newaxis] * state.a_i[np.newaxis, :]) * (1 - state.k_ij)
    # state.b_ij = (state.b_i[:, np.newaxis] + state.b_i[np.newaxis, :]) / 2 * (1 - state.l_ij)
    state.a = np.sum(state.x[:, None] * state.x[None, :] * state.a_ij)
    state.b = np.sum(state.x * state.b_i)

def calc_parameter_mixture(state: Peng_robinson_state) -> None:
    global R
    state.beta = state.b * state.P / (R * state.T)
    state.q = state.a / (state.b * state.T * R)

def prime_p_i(state: Peng_robinson_state) -> None:
    global sigma_PR, epsilon_PR
    state.I = np.log((state.Z + sigma_PR * state.beta) / (state.Z + epsilon_PR * state.beta)) / (sigma_PR - epsilon_PR)
    state.prime_a_i = 2 * np.sum(state.x[np.newaxis, :] * state.a_ij, axis=1) - state.a
    state.prime_b_i = state.b_i
    state.prime_q_i = state.q * (1 + state.prime_a_i / state.a - state.prime_b_i / state.b)


def compressibility(state: Peng_robinson_state, Z_init: float) -> None:
    global sigma_PR, epsilon_PR
    if Z_init == state.beta:
        f_z = lambda Z, b, q: Z - (b + (Z + epsilon_PR * b) * (Z + sigma_PR * b) * (1 + b - Z) / (q * b))
    else:
        f_z = lambda Z, b, q: Z - (1 + b - q * b * (Z - b) /((Z + epsilon_PR * b) * (Z + sigma_PR * b)))
    Z = fsolve(func=f_z, x0=Z_init, args=(state.beta, state.q))
    state.Z = Z[0]

def fugacity(state: Peng_robinson_state) -> None:
    state.ln_phi = (state.b_i / state.b) * (state.Z - 1) - np.log(state.Z - state.beta) - state.prime_q_i * state.I
    state.phi = np.exp(state.ln_phi)

def res_K(P: float, state_liq, state_gas) -> float:
    state_liq.P = P
    state_gas.P = P
    update_state(state=state_liq, liq=True)
    update_state(state=state_gas, liq=False)
    fugacity(state=state_liq)
    fugacity(state=state_gas)
    K = np.sum(state_liq.phi * state_liq.x / state_gas.phi)
    state_gas.x = (state_liq.phi * state_liq.x / state_gas.phi) / K
    res = (1 - K)**2
    return res


