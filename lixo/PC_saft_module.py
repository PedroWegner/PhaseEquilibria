import numpy as np
from scipy.optimize import  minimize
import matplotlib.pyplot as plt
from time import time
import openpyxl as pyxl
import os

time_0 = time()
# PARAMETROS GLOBAIS DO MODELO PC-SAFT
saft_a0 = np.array([0.910563145, 0.636128145, 2.686134789, -26.54736249, 97.75920878, -159.5915409, 91.29777408])
saft_a1 = np.array([-0.308401692, 0.186053116, -2.503004726, 21.41979363, -65.25588533, 83.31868048, -33.74692293])
saft_a2 = np.array([-0.090614835, 0.452784281, 0.596270073, -1.724182913, -4.130211253, 13.77663187, -8.672847037])
saft_b0 = np.array([0.724094694, 2.238279186, -4.002584949, -21.00357682, 26.85564136, 206.5513384, -355.6023561])
saft_b1 = np.array([-0.575549808, 0.699509552, 3.892567339, -17.21547165, 192.6722645, -161.8264617, -165.2076935])
saft_b2 = np.array([0.097688312, -0.255757498, -9.155856153, 20.64207597, -38.80443005, 93.62677408, -29.66690559])
K_boltz = 1.380649e-23 # m2 kg s-2 K-1
N_Avogrado = 6.02214076e23 # mol-1
R = K_boltz * N_Avogrado # J mol-1 K-1
#Teste
class PC_saft_state():
    def __init__(self,
                T: float,
                P: float,
                ncomp: int,
                x: list[float],
                m: list[float],
                sigma: list[float],
                epsilon: list[float],
                liquid: bool):
        # VARIAVEIS DA MISTURA
        self.ncomp = ncomp
        self.x = np.array(x)
        self.m = np.array(m)
        self.sigma = np.array(sigma)
        self.epsilon = np.array(epsilon)
        self.liquid = liquid

        # PROPRIEDADES DO ESTADO
        self.T = T
        self.P = P
        self.eta = type[float]
        self.rho = type[float]
        self.helmholtz_hs = type[float]
        self.helmholtz_hc = type[float]
        self.helmholtz_disp = type[float]
        self.helmholtz_res = type[float]
        self.Z_hs = type[float]
        self.Z_hc = type[float]
        self.Z_disp = type[float]
        self.Z = type[float]
        # VARIAVEIS PARA CALCULAR HELMHOLTZ RESIDUAL E COMPRESSIBILIDADE
        self.I_1 = type[float]
        self.I_2 = type[float]
        self.m2es3 = type[float]
        self.m2e2s3 = type[float]
        self.C_1 = type[float]
        self.C_2 = type[float]
        self.prime_I_1 = type[float]
        self.prime_I_2 = type[float]
        self.mean_m = type[float]

        # ARRAYS PARA CALCULAR HELMHOLTZ RESIDUAL E COMPRESSIBILIDADE
        self.d = np.zeros(self.ncomp)
        self.zeta = np.zeros(4)
        self.a_m = np.zeros(7)
        self.b_m = np.zeros(7)
        self.g_ij = np.zeros((self.ncomp, self.ncomp))
        self.s_ij = np.zeros((self.ncomp, self.ncomp))
        self.e_ij = np.zeros((self.ncomp, self.ncomp))
        self.k_ij = np.zeros((self.ncomp, self.ncomp))
        print(self.k_ij)
        self.grad_g_ij_rho = np.zeros((self.ncomp, self.ncomp))

        # ARRAYS PARA CALCULAR A FUGACIDADE
        self.phi = np.zeros(self.ncomp)
        self.ln_phi = np.zeros(self.ncomp)
        self.chemical_pow = np.zeros(self.ncomp)
        self.zeta_x = np.zeros((4 , self.ncomp))
        self.prime_helmholtz_hs_x = np.zeros(self.ncomp)
        self.prime_helmholtz_hc_x = np.zeros(self.ncomp)
        self.prime_helmholtz_disp_x = np.zeros(self.ncomp)
        self.prime_helmholtz_res_x = np.zeros(self.ncomp)
        self.grad_g_ij_x = np.zeros((self.ncomp, self.ncomp, self.ncomp))
        self.m2es3_x = np.zeros(self.ncomp)
        self.m2e2s3_x = np.zeros(self.ncomp)
        self.C_1_x = np.zeros(self.ncomp)
        self.C_2_x = np.zeros(self.ncomp)
        self.I_1_x = np.zeros(self.ncomp)
        self.I_2_x = np.zeros(self.ncomp)
        self.a_x = np.zeros((7, self.ncomp))
        self.b_x = np.zeros((7, self.ncomp))


# ************************************************************
# INICIO DAS FUNCOES PARA CALCULO DA HELMHOLTZ E COMPRESSIBILIDADE
# ************************************************************
def calc_diameter_T(state: PC_saft_state) -> None:
    # EQ (A.9)
    state.d = state.sigma * (1.0e0 - 0.12e0 * np.exp(- 3.0e0 * state.epsilon / state.T))

def calc_mean_m(state: PC_saft_state):
    # EQ (A.5)
    state.mean_m = np.sum(state.x * state.m)

def calc_combining_rules(state: PC_saft_state) -> None:
    # EQ (A.14)
    state.s_ij = (state.sigma[:, np.newaxis] + state.sigma[np.newaxis, :]) / 2
    # EQ (A.15)
    state.e_ij = np.sqrt(state.epsilon[:, np.newaxis] * state.epsilon[np.newaxis, :])*(1 - state.k_ij)

def calc_zeta(state: PC_saft_state) -> None:
    zeta_aux = np.pi * state.rho / 6
    exp = np.arange(4).reshape(4, 1)
    # EQ (A.8)
    state.zeta = zeta_aux * np.sum(state.x * state.m * state.d ** exp, axis=1)

def calc_hard_sphere(state: PC_saft_state) -> None:
    aux_1 = 1 - state.zeta[3]
    aux_2 = 3 * state.zeta[2] / aux_1**2
    aux_3 = 2 * state.zeta[2]**2 / aux_1**3
    d_ij = state.d[:, np.newaxis] * state.d[np.newaxis, :] / (state.d[:, np.newaxis] + state.d[np.newaxis, :])
    # EQ (A.7)
    state.g_ij = 1.0e0 / aux_1 + d_ij * aux_2 + d_ij**2 * aux_3

def calc_ab_m(state: PC_saft_state) -> None:
    aux_1 = (state.mean_m - 1.0e0) / state.mean_m
    aux_2 = aux_1 * (state.mean_m - 2.0e0) / state.mean_m
    # EQ (A.18)
    state.a_m = saft_a0  + aux_1 * saft_a1 + aux_2 * saft_a2
    # EQ (A.19)
    state.b_m = saft_b0  + aux_1 * saft_b1 + aux_2 * saft_b2

def calc_pertubation_integral(state: PC_saft_state) -> None:
    exp = np.arange(7)
    # EQ (A.16)
    state.I_1 = np.sum(state.a_m * state.eta**exp)
    # EQ (A.17)
    state.I_2 = np.sum(state.b_m * state.eta**exp)

def calc_prime_pertubation_integral(state: PC_saft_state) -> None:
    exp = np.arange(7)
    # EQ (A.29)
    state.prime_I_1 = np.sum(state.a_m * (exp + 1) * state.eta**exp)
    # EQ (A.30)
    state.prime_I_2 = np.sum(state.b_m * (exp + 1) * state.eta**exp)

def calc_C_12(state:PC_saft_state) -> None:
    # EQ (A.11)
    aux_1 = 1e0 + state.mean_m * ((8e0 * state.eta - 2e0 * state.eta**2e0) / (1e0 - state.eta)**4e0)
    aux_2 = (1e0 - state.mean_m) * (20e0 * state.eta - 27e0 * state.eta**2e0 + 12e0 * state.eta**3e0 - 2e0 * state.eta**4e0)
    aux_2 = aux_2 / ((1e0 - state.eta) * (2e0 - state.eta))**2e0
    state.C_1 = 1e0 / (aux_1 + aux_2)
    # EQ (A.31)
    aux_1 = state.mean_m * (-4 * state.eta**2 + 20 * state.eta + 8) / (1 - state.eta)**5
    aux_2 = (1 - state.mean_m) * (2 * state.eta**3 + 12 * state.eta**2 - 48 * state.eta + 40)
    aux_2 = aux_2 / ((1 - state.eta) * (2 - state.eta))**3
    state.C_2 = - state.C_1**2 * (aux_1 + aux_2)


def calc_abbr_mes(state: PC_saft_state) -> None:
    x_ij = state.x[:, np.newaxis] * state.x[np.newaxis, :]
    m_ij = state.m[:, np.newaxis] * state.m[np.newaxis, :]
    e_ij = state.e_ij / state.T
    # EQ (A.12)
    state.m2es3 = np.sum(x_ij * m_ij * e_ij * state.s_ij**3)
    # EQ (A.13)
    state.m2e2s3 = np.sum(x_ij * m_ij * e_ij**2 * state.s_ij**3)

def calc_grad_g_rho(state: PC_saft_state) -> None:
    d_ij = state.d[:, np.newaxis] * state.d[np.newaxis, :] / (state.d[:, np.newaxis] + state.d[np.newaxis, :])
    # EQ (A.27)
    zeta_aux = 1e0 - state.zeta[3]
    aux_1 = state.zeta[3] / zeta_aux**2e0
    aux_2 = 3e0 * state.zeta[2] / zeta_aux**2e0 + 6e0 * state.zeta[2] * state.zeta[3] / zeta_aux**3e0
    aux_3 = 4e0 * state.zeta[2]**2e0 / zeta_aux**3e0 + 6e0 * state.zeta[2]**2e0 * state.zeta[3] / zeta_aux**4e0
    state.grad_g_ij_rho = aux_1 + d_ij * aux_2 + d_ij**2 * aux_3

def helmholtz_residual(state: PC_saft_state) -> None:
    # EQ (A.6)
    zeta_aux = 1e0 - state.zeta[3]
    aux_1 = 3e0 * state.zeta[1] * state.zeta[2] / zeta_aux
    aux_2 = state.zeta[2]**3e0 / (state.zeta[3] * zeta_aux**2e0)
    aux_3 = (state.zeta[2]**3e0 / state.zeta[3]**2e0 - state.zeta[0])*np.log(zeta_aux)
    state.helmholtz_hs = (1e0 / state.zeta[0]) * (aux_1 + aux_2 + aux_3)
    # EQ (A.4)
    sum_aux = np.sum(state.x * (state.m - 1) * np.log(np.diagonal(state.g_ij))) # ALTERA AQUI SE NECESSARIO
    state.helmholtz_hc = state.mean_m * state.helmholtz_hs - sum_aux

    # EQ (A.10)
    aux_1 = - 2 * np.pi * state.rho * state.I_1 * state.m2es3
    aux_2 = - np.pi * state.rho * state.mean_m * state.C_1 * state.I_2 * state.m2e2s3
    state.helmholtz_disp = aux_1 + aux_2

    # EQ (A.3)
    state.helmholtz_res = state.helmholtz_hc + state.helmholtz_disp

def compressibility(state: PC_saft_state) -> None:
    # EQ (A.26)
    zeta_aux = 1 - state.zeta[3]
    aux_1 = state.zeta[3] / zeta_aux
    aux_2 = 3 * state.zeta[1] * state.zeta[2] / (state.zeta[0] * zeta_aux**2)
    aux_3 = (3 * state.zeta[2]**3 - state.zeta[3] * state.zeta[2]**3) / (state.zeta[0] * zeta_aux**3)
    state.Z_hs = aux_1 + aux_2 + aux_3

    # EQ (A.25)
    sum_aux = np.sum(state.x * (state.m - 1) * (np.diagonal(state.g_ij))**-1 * np.diagonal(state.grad_g_ij_rho))
    state.Z_hc = state.mean_m * state.Z_hs - sum_aux

    # EQ (A.28)
    aux_1 = - 2 * np.pi * state.rho * state.prime_I_1 * state.m2es3
    aux_2 = - np.pi * state.rho * state.mean_m * state.m2e2s3
    aux_2 *= (state.C_1 * state.prime_I_2 + state.C_2 * state.eta * state.I_2)
    state.Z_disp = aux_1 + aux_2

    # EQ (A.24)
    state.Z = 1 + state.Z_hc + state.Z_disp

# ************************************************************
# FIM DAS FUNCOES PARA CALCULO DA HELMHOLTZ E COMPRESSIBILIDADE
# ************************************************************
# ************************************************************
# INICIO DAS FUNCOES PARA CALCULO DA FUGACIDADE
# ************************************************************

def calc_zeta_x(state: PC_saft_state) -> None:
    exp_aux = np.arange(4).reshape(4, 1)
    # EQ (A.34)
    state.zeta_x = (np.pi * state.rho / 6) * (state.m * state.d**exp_aux)

def calc_helmholtz_hardchain_x(state: PC_saft_state) -> None:
    calc_zeta_x(state=state)

    # Auxiliares da EQ (A.37) ****
    d_ii = state.d / 2
    zeta_aux = 1 - state.zeta[3]
    aux_1 = (state.zeta_x[3, :] / zeta_aux**2)[:, None]
    aux_2 = ((3 * state.zeta_x[2, :] / zeta_aux**2 + 6 * state.zeta[2] * state.zeta_x[3, :] / zeta_aux**3))[:, None]
    aux_3 = ((4 * state.zeta[2] * state.zeta_x[2, :] / zeta_aux**3 + 6 * state.zeta[2]**2 * state.zeta_x[3, :] / zeta_aux**4))[:, None]
    # EQ (A.37) ****
    state.grad_g_ij_x = aux_1 + d_ii * aux_2 + d_ii**2 * aux_3  

    # Auxiliares da EQ (A.36)
    aux_1 = - state.zeta_x[0, :] * state.helmholtz_hs / state.zeta[0]
    aux_2 = (3 * state.zeta[2]**2 * state.zeta_x[2, :] * state.zeta[3] - 2 * state.zeta[2]**3 * state.zeta_x[3, :]) / state.zeta[3]**3 - state.zeta_x[0, :]
    aux_2 = aux_2 * np.log(zeta_aux)
    aux_2 += 3 * (state.zeta_x[1, :] * state.zeta[2] + state.zeta[1] * state.zeta_x[2, :]) / zeta_aux
    aux_2 += 3 * state.zeta[1] * state.zeta[2] * state.zeta_x[3, :] / zeta_aux**2
    aux_2 += 3 * state.zeta[2]**2 * state.zeta_x[2, :] / (state.zeta[3] * zeta_aux**2)
    aux_2 += state.zeta[2]**3 * state.zeta_x[3, :] * (3 * state.zeta[3] - 1) / (state.zeta[3]**2 * zeta_aux**3)
    aux_2 += (state.zeta[0] - state.zeta[2]**3 / state.zeta[3]**2) * (state.zeta_x[3, :] / zeta_aux)
    # EQ (A.36)
    state.prime_helmholtz_hs_x = aux_1 + (1 / state.zeta[0]) * aux_2

    # Auxiliar EQ (A.35)
    aux_1= np.sum(state.x * (state.m - 1) * (1/np.diagonal(state.g_ij)) * state.grad_g_ij_x, axis=1) 
    # EQ (A.35)
    aux_2 = - (state.m - 1)*np.log(np.diagonal(state.g_ij))
    state.prime_helmholtz_hc_x = state.m * state.helmholtz_hs + state.mean_m * state.prime_helmholtz_hs_x - aux_1 + aux_2

def calc_helmholtz_disp_x(state: PC_saft_state) -> None:
    # EQ (A.44)
    m_x = (state.m / state.mean_m**2)[:, None]
    state.a_x =  m_x * saft_a1 + m_x*(3 - 4 / state.mean_m) * saft_a2
    # EQ (A.45)
    state.b_x =  m_x * saft_b1 + m_x*(3 - 4 / state.mean_m) * saft_b2
    
    e_aux = (np.arange(7))[:, None]
    # EQ (A.42)
    state.I_1_x = np.sum(state.a_m[:, None] * e_aux * state.zeta_x[3, :] * state.eta**(e_aux - 1) + state.a_x.T * state.eta**e_aux, axis=0)
    # EQ (A.43)
    state.I_2_x = np.sum(state.b_m[:, None] * e_aux * state.zeta_x[3, :] * state.eta**(e_aux - 1) + state.b_x.T * state.eta**e_aux, axis=0)

    # Auxiliares da EQ (A.41)
    aux_1 = state.m * (8 * state.eta - 2 * state.eta**2) / (1 - state.eta)**4
    aux_2 = - state.m * (20 * state.eta - 27 * state.eta**2 + 12 * state.eta**3 - 2 * state.eta**4) / ((1 - state.eta) * (2 - state.eta))**2
    # EQ (A.41)
    state.C_1_x = state.C_2*state.zeta_x[3, :] - state.C_1**2 * (aux_1 + aux_2)
    
    # Auxiliares da EQ (A.39) & (A.40)
    sum_aux_1 = np.sum(state.x * state.m * (state.e_ij/ state.T) * state.s_ij**3, axis=1)
    sum_aux_2 = np.sum(state.x * state.m * (state.e_ij/ state.T)**2 * state.s_ij**3, axis=1)

    # EQ (A.39)
    state.m2es3_x = 2 * state.m * sum_aux_1    
    # EQ (A.40)
    state.m2e2s3_x = 2 * state.m * sum_aux_2


    # Auxiliares da EQ (A.38)
    aux_1 = - 2 * np.pi * state.rho * (state.I_1_x * state.m2es3 + state.I_1 * state.m2es3_x)
    aux_2 = (state.m * state.C_1 * state.I_2 + state.mean_m * state.C_1_x * state.I_2 + state.mean_m * state.C_1 * state.I_2_x)
    aux_3 = - np.pi * state.rho * ( aux_2 * state.m2e2s3 + state.mean_m * state.C_1 * state.I_2 * state.m2e2s3_x)
    state.prime_helmholtz_disp_x = aux_1 + aux_3

def calc_helmholtz_res_x(state: PC_saft_state) -> None:
    calc_helmholtz_hardchain_x(state=state)
    calc_helmholtz_disp_x(state=state)
    state.prime_helmholtz_res_x = state.prime_helmholtz_hc_x + state.prime_helmholtz_disp_x

def calc_fugacity_coef(state: PC_saft_state) -> None:
    calc_helmholtz_res_x(state=state)
    # Auxiliar d EQ (A.33)
    sum_aux = - np.sum(state.x * state.prime_helmholtz_res_x)
    # EQ (A.33)
    state.chemical_pow = state.helmholtz_res + (state.Z - 1) + state.prime_helmholtz_res_x + sum_aux
    # EQ (A.32)
    state.ln_phi = state.chemical_pow - np.log(state.Z)
    state.phi = np.exp(state.ln_phi)
# ************************************************************
# FIM DAS FUNCOES PARA CALCULO DA FUGACIDADE
# ************************************************************

# ************************************************************
# INICIO FUNCOES DE CALCULAR O ETA
# ************************************************************
def calc_rho(state: PC_saft_state) -> None:
    sum_aux = np.sum(state.x * state.m * state.d**3)
    state.rho = (6 * state.eta / np.pi) * sum_aux**-1

def calc_P(state: PC_saft_state) -> None:
    global K_boltz
    P = state.Z * K_boltz * state.T * state.rho * 1.0e10**3
    return P

def res_pressao(eta: float, state: PC_saft_state) -> float:
    state.eta = eta
    calc_state(state)
    P_calc = calc_P(state)
    return np.abs(1 - state.P / P_calc)


def calc_eta(state: PC_saft_state) -> float:
    if state.liquid:
        eta_0 = 0.5e0
    else:
        eta_0 = 1e-10
    eta = minimize(res_pressao, x0=eta_0, args=(state), method='Nelder-Mead', tol=1e-8)
    state.eta = eta.x[0]

# ************************************************************
# FIM FUNCOES DE CALCULAR O ETA
# ************************************************************

# ************************************************************
# INICIO DE UM TESTE 
# ************************************************************
def Nada_a_ver(state: PC_saft_state):
    i = np.arange(4).reshape(4, 1)
    prime_zeta = np.sum(state.x * state.m * state.d**i, axis=1) / np.sum(state.x * state.m * state.d**3)
    d_ij = state.d[:, np.newaxis] * state.d[np.newaxis, :] / (state.d[:, np.newaxis] + state.d[np.newaxis, :])


    #PRIME Z_hs
    aux_1 = prime_zeta[3] / (1 - state.zeta[3])**2
    alpha = state.zeta[1] * state.zeta[2]
    beta = state.zeta[0] * (1 - state.zeta[3])**2
    prime_alpha = state.zeta[2] * prime_zeta[1] + state.zeta[1] * prime_zeta[2]
    prime_beta = prime_zeta[0] * (1 - state.zeta[3])**2 - 2 * state.zeta[0] * (1 - state.zeta[3]) * prime_zeta[3]
    aux_2 = 3 * (prime_alpha * beta - alpha * prime_beta) / beta**2
    alpha = 3 * state.zeta[2]**3 - state.zeta[3] * state.zeta[2]**3
    beta = state.zeta[0] * (1 - state.zeta[3])**3
    prime_alpha = 9 * state.zeta[2]**2 * prime_zeta[2] - (prime_zeta[3] * state.zeta[2]**3 + 3 * state.zeta[3] * state.zeta[2] * prime_zeta[3])
    prime_beta = prime_zeta[0] * (1 - state.zeta[3]) **3 - 3*state.zeta[0]*(1-state.zeta[3])**2 * prime_zeta[3]
    aux_3 = (prime_alpha * beta - alpha * prime_beta) / beta**2
    prime_Z_hs = aux_1 + aux_2 + aux_3

    # prime_gij_ehta vai ter que pegar a diagonal
    aux_1 = prime_zeta[3] * (1 + state.zeta[3]) / (1 - state.zeta[3])**3
    alpha = 6 * state.zeta[2] * state.zeta[3]
    beta = (1 - state.zeta[3])**3
    prime_alpha = 6 * (state.zeta[3] * prime_zeta[2] + state.zeta[2]*prime_zeta[3])
    prime_beta = - 3 * (1 - state.zeta[3])**2 * prime_zeta[3]
    aux_2 = (prime_alpha * beta - alpha * prime_beta) / beta**2
    alpha =  4 * state.zeta[2]**2
    beta = (1 - state.zeta[3])**4
    prime_alpha = 6 * (2 * state.zeta[2] * state.zeta[3] * prime_zeta[2] + state.zeta[2]**2 * prime_zeta[3])
    prime_beta = - 4 * (1 - state.zeta[3])**3 * prime_zeta[3]
    aux_3 = (prime_alpha * beta - alpha * prime_beta) / beta**2
    prime_g_ij_ehta = aux_1 + d_ij * aux_2 + d_ij**2 * aux_3

    # 1 / g_ij -> prime_f
    aux_1 = -(1 - state.zeta[3])**2
    
    alpha = 3 * state.zeta[2]
    beta = (1- state.zeta[3])**2
    prime_alpha = 3 * prime_zeta[2]
    prime_beta = - 2 * (1 - state.zeta[3]) * prime_zeta[3]
    aux_2 = (prime_alpha * beta - alpha * prime_beta) / beta**2
    alpha = 2 * state.zeta[2]**2
    beta = (1 - state.zeta[3])**3
    prime_alpha = 4 * state.zeta[2] * prime_zeta[2]
    prime_beta = -3 * (1 - state.zeta[3])**2
    aux_3 = (prime_alpha * beta - alpha * prime_beta) / beta**2
    f = 1 / state.g_ij
    prime_f = -1 * state.g_ij**2 * (aux_1 + d_ij * aux_2 + d_ij**2 * aux_3)**-2

    sum_aux = - np.sum(state.x * state.m * (f.diagonal()*prime_g_ij_ehta.diagonal() + prime_f.diagonal() * state.grad_g_ij_rho.diagonal()))

    prime_Z_hc = state.mean_m * prime_Z_hs + sum_aux


    j = np.arange(7)

    prime_2_I_1 = np.sum(state.a_m * (j + 1) * j * state.eta**(i-1))
    prime_2_I_2 = np.sum(state.b_m * (j + 1) * j * state.eta**(i-1))


    alpha = 8 * state.eta - 2 * state.eta**2
    beta = (1 - state.eta)**4
    prime_alpha = 8 - 4 * state.eta
    prime_beta = - 4 * (1 - state.eta)**3
    aux_1 = (prime_alpha * beta - alpha * prime_beta) / beta**2
    alpha = 20 * state.eta - 27 * state.eta**2 + 12 * state.eta**3 - 2 * state.eta**4
    beta = (2 - 3 * state.eta + state.eta**2)**2
    prime_alpha = 20 - 54 * state.eta + 36 * state.eta**2 - 8 * state.eta**3
    prime_beta = 2 * (2 - 3 * state.eta + state.eta**2) * (-3 + 2 * state.eta)
    aux_2 = 20 * state.eta - 27 * state.eta**2 + 12 * state.eta**3 - 2 * state.eta**4
    T_1 = state.mean_m * aux_1 + (1 - state.mean_m) * aux_2


    alpha = - 4 * state.eta**2 + 20 * state.eta + 8
    beta = (1 - state.eta)**5
    prime_alpha = - 8 * state.eta + 20
    prime_beta = - 5 * (1 - state.eta)**4
    aux_1 = (prime_alpha * beta - alpha * prime_beta) / beta**2
    alpha = 2 * state.eta**3 + 12 * state.eta**2 - 48*state.eta + 10
    beta = (2 - 3 * state.eta + state.eta**2)**3
    prime_alpha = 6 * state.eta**2 + 24 * state.eta - 48
    prime_beta = 3 * (2 - 3 * state.eta + state.eta**2)**2 * (-3 + 2 * state.eta)
    aux_2 = (prime_alpha * beta - alpha * prime_beta) / beta**2
    prime_T_1 = state.mean_m * aux_1 + (1 - state.mean_m)*aux_2
    prime_C_2 = - 2 * state.C_1 * T_1 * state.C_2 - state.C_1**2 * prime_T_1




    aux_1 = -12 * state.eta * np.sum(state.x * state.m * state.d**3)**-1 * state.m2es3 * (state.prime_I_1 + state.eta * prime_2_I_1)
    primeiro_termo = state.C_1 * state.prime_I_2 + state.eta*(state.C_2* state.prime_I_2 + state.C_1 * prime_2_I_2)
    segundo_termo = 2 * state.eta * state.C_2 * state.prime_I_2 + state.eta**2 * (prime_C_2 * state.I_2 + state.C_2 * prime_2_I_2)
    aux_2 = - 6 * np.sum(state.x * state.m * state.d**3)**-1 * state.m2e2s3 * (primeiro_termo + segundo_termo)
    prime_Z_disp = aux_1 + aux_2

    prime_Z = prime_Z_hc + prime_Z_disp
    prime_rho = 6 * np.sum(state.x * state.m * state.d**3)**-1 / np.pi
    prime_P = (state.Z * prime_rho + state.rho * prime_Z) * (K_boltz * state.T * (10**10)**3)

    return prime_P



def calc_prime_P(state: PC_saft_state) -> float:
    """
    Implementa a derivada da pressao em relacao aa variavel ehta;
    Essa implementacao eh usada para aplicar metodo de Newton-Raphson sem derivada numerica e
    para evitar a utilizacao de metodo de otimizacao sem derivada 
    """
    sum_3 = (np.sum(state.x * state.m * state.d**3))**-1
    prime_rho = (6 / np.pi) * sum_3
    n = np.arange(4).reshape(4, 1)
    prime_zeta = np.sum(state.x * state.m * state.d**n, axis=1) * sum_3
    zeta_aux = (1 - state.zeta[3])

    # Derivada de Z_hs em relacao ao ehta
    aux_1 = 1 / zeta_aux**2
    aux_2 = 3 * state.zeta[1] * state.zeta[2] * (1 + state.zeta[3]) / (state.zeta[0] * state.zeta[3] * zeta_aux**3)
    aux_3 = 6 * state.zeta[2]**3 / (state.zeta[0] * state.zeta[3] * zeta_aux**4)
    prime_Z_hs = aux_1 + aux_2 + aux_3

    d_ij = state.d[:, np.newaxis] * state.d[np.newaxis, :] / (state.d[:, np.newaxis] + state.d[np.newaxis, :])
    # Grad_grad_g_ij_hs * rho
    aux_1 = (1 + state.zeta[3]) / zeta_aux**3
    aux_2 = (3 * state.zeta[2] * (1 + state.zeta[3])) / (state.zeta[3] * zeta_aux**3)
    aux_2 += (6 * state.zeta[2] * (2 + state.zeta[3])) / (zeta_aux**4)
    aux_3 = (4 * state.zeta[2]**2 * (2 + state.zeta[3])) / (state.zeta[3] * zeta_aux**4)
    aux_3 += (6 * state.zeta[2]**2 * (3 + state.zeta[3])) / zeta_aux**5
    grad_grad_g_ij = aux_1 + d_ij * aux_2 + d_ij**2 * aux_3

    # Calcula funcao auxiliar para calcular a derivada de Z_hc em relacao ao ehta
    f = (state.g_ij)**-1
    aux_1 = 1 / zeta_aux**2
    aux_2 = 3 * state.zeta[2] * (1 + state.zeta[3]) / (state.zeta[3] * zeta_aux**3)
    aux_3 = 2 * state.zeta[2]**2 * (2 + state.zeta[3]) / (state.zeta[3] * zeta_aux**4)
    prime_aux = aux_1 + d_ij * aux_2 + d_ij**2 * aux_3
    prime_f = - f**2 * prime_aux
    # Derivada do Z_hc
    sum_aux = np.sum(state.x * (state.m - 1) * (state.grad_g_ij_rho.diagonal() * prime_f.diagonal() + (np.diagonal(f)) * grad_grad_g_ij.diagonal()))
    prime_Z_hc = state.mean_m * prime_Z_hs - sum_aux

    # print(prime_f)
    # alpha = (1 - state.zeta[3])**-1
    # prime_alpha = (1 - state.zeta[3])**-2
    # prime_f_1 = prime_alpha
    # alpha = 3 * state.zeta[2]
    # prime_alpha = 3 * prime_zeta[2]
    # gamma = (1 - state.zeta[3])**2
    # prime_gamma = - 2 * (1 - state.zeta[3]) * prime_zeta[3]
    # prime_f_2 = (prime_alpha * gamma - alpha * prime_gamma) / gamma**2
    # alpha = 2 * state.zeta[2]**2
    # prime_alpha = 4 * state.zeta[2] * prime_zeta[2]
    # gamma = (1 - state.zeta[3])**3
    # prime_gamma = - 3 * (1 - state.zeta[3])**2 * prime_zeta[3]
    # prime_f_3 = (prime_alpha * gamma - alpha * prime_gamma) / gamma**2
    # prime_f = - state.g_ij**-2 * (prime_f_1 + d_ij*prime_f_2 + d_ij**2*prime_f_3)
    # print(prime_f)

    # sum_aux = np.sum(state.x * (state.m - 1) * (state.grad_g_ij_rho.diagonal() * prime_f.diagonal() + (np.diagonal(state.g_ij))**-1 * grad_grad_g_ij.diagonal()))
    # prime_Z_hc = state.mean_m * prime_Z_hs - sum_aux
    # print(prime_Z_hc)
    

    # Calcula a derivada de Z_disp em relacao ao ehta
    j = np.arange(7)
    prime_2_I_1 = np.sum(state.a_m * (j + 1) * j * state.eta**(j-1))
    prime_2_I_2 = np.sum(state.b_m * (j + 1) * j * state.eta**(j-1))
    prime_I_2 = np.sum(state.b_m * j * state.eta**(j-1))
    prime_rho_eta = 12 * state.eta * sum_3 / np.pi


    alpha = (- 4 * state.eta**2 + 20 * state.eta + 8) 
    prime_alpha = state.mean_m * (- 8 * state.eta + 20)
    gamma = (1 - state.eta)**5
    prime_gamma = - 5 * (1 - state.eta)**4
    h_1 = alpha / gamma
    prime_h_1 = (prime_alpha * gamma - alpha * prime_gamma) / gamma**2
    alpha = (1 - state.mean_m) * (2 * state.eta**3 + 12 * state.eta**2 - 48 * state.eta + 40)
    prime_alpha = (1 - state.mean_m) * (6 * state.eta**2 + 24 * state.eta - 48)
    gamma = (2 - 3 * state.eta + state.eta**2)**2
    prime_gamma = 2 * (2 - 3 * state.eta + state.eta**2) * (2 * state.eta - 3)
    h_2 = alpha / gamma
    prime_h_2 = (prime_alpha * gamma - alpha * prime_gamma) / gamma**2
    h = h_1 + h_2
    prime_h = prime_h_1 + prime_h_2
    prime_C_2 = - 2 * state.C_1 * state.C_2 * h - state.C_1**2 * prime_h

    prime_Z_disp_1 = - 2 * np.pi * state.m2es3 * (prime_rho * state.prime_I_1 + state.rho * prime_2_I_1)
    prime_Z_disp_2_1 = prime_rho * state.C_1 * state.prime_I_2 + state.rho * (state.C_2 * state.prime_I_2 + state.C_1 * prime_2_I_2)
    prime_Z_disp_2_2 = prime_rho_eta * (state.C_2 * state.I_2) + state.rho * state.eta * (prime_C_2 * state.I_2 + state.C_2 * prime_I_2)
    prime_Z_disp_2 = - np.pi * state.mean_m * state.m2e2s3 * (prime_Z_disp_2_1 + prime_Z_disp_2_2)
    prime_Z_disp = prime_Z_disp_1 + prime_Z_disp_2

    prime_Z = prime_Z_hc + prime_Z_disp
    
    prime_P = K_boltz * state.T * 1.0e10**3 * (prime_rho * state.Z + state.rho * prime_Z)

    return prime_P
    

def update_test(state: PC_saft_state) -> None:
    """
    Implementacao para convergir o ehta aa pressao solicitada
    Aqui tem a implementacao do metodo de Newton-Raphson com maximo de iteracao de 250 e tolerancia de 1e-10
    """

    # Estimativas iniciais sugeridas por Gross & Sadowski (2001)
    if state.liquid:
        state.eta = 0.5
    else:
        state.eta = 1e-8
    for i in range(250):
        calc_state(state=state)
        P = calc_P(state=state)
        dP = calc_prime_P(state=state)
        FO = 1 - P / state.P
        dFO = - dP / state.P
        eta_old = state.eta
        state.eta = state.eta - FO / dFO

        if np.abs(eta_old - state.eta) < 1e-10:
            break

    # Atualiza o real estado do sistema com ehta convergido
    calc_state(state=state)
    calc_fugacity_coef(state=state)

# ********************* ***************************************
# FIM DO TESTE
# ************************************************************


# ************************************************************
# INICIO UPDATE STATE
# ************************************************************
def calc_state(state: PC_saft_state) -> None:
    calc_diameter_T(state=state)
    calc_rho(state)
    calc_mean_m(state=state)
    calc_zeta(state=state)
    calc_combining_rules(state=state)
    calc_hard_sphere(state=state)
    calc_ab_m(state=state)
    calc_pertubation_integral(state=state)
    calc_prime_pertubation_integral(state=state)
    calc_C_12(state=state)
    calc_abbr_mes(state=state)
    calc_grad_g_rho(state=state)
    helmholtz_residual(state=state)
    compressibility(state=state)

def update_state(state):
    calc_eta(state=state)
    calc_state(state=state)
    calc_fugacity_coef(state=state)
# ************************************************************
# FIM UPDATE STATE
# ************************************************************

def res_K(P: float, state_liq, state_gas) -> float:
    state_liq.P = P
    state_gas.P = P
    update_test(state=state_liq)
    update_test(state=state_gas)
    K = np.sum(state_liq.phi * state_liq.x / state_gas.phi)
    state_gas.x = (state_liq.phi * state_liq.x / state_gas.phi) / K
    res = (1 - K)**2
    return res


def generate_mole_fractions(n, num_points=100):
    k = num_points - 1 
    indices = np.array(list(multichoose(n, k))) 
    fractions = indices / k 
    return fractions

def multichoose(n, k):
    if n == 1:
        yield (k,)
    else:
        for i in range(k + 1):
            for tail in multichoose(n - 1, k - i):
                yield (i,) + tail


if __name__ == "__main__":
    m =  [0.680, 1.0000]
    s = [3.54, 3.7039]
    e = [31.57, 150.03]
    T_o = 173.65 # K
    P_o = 35e5
    x = 0.0168
    x_l = [x, 1 - x]
    y = 0.148
    y_l = [y, 1-y]
    liquid = PC_saft_state(T=T_o,
                           P=P_o,
                           ncomp=2,
                           x=x_l,
                           m=m,
                           sigma=s,
                           epsilon=e,
                           liquid=True)


    gas = PC_saft_state(T=T_o,
                        P=P_o,
                        ncomp=2,
                        x=y_l,
                        m=m,
                        sigma=s,
                        epsilon=e,
                        liquid=False)
    # k_ij = -0.06
    # liquid.k_ij[0][1] = k_ij
    # gas.k_ij[0][1] = k_ij
    # liquid.k_ij[1][0] = k_ij
    # gas.k_ij[1][0] = k_ij
    fractions_matrix = np.array(generate_mole_fractions(n=2, num_points=225))
    # fractions_matrix = fractions_matrix[fractions_matrix[:,0] >= 0.0001]
    fractions_matrix = fractions_matrix[fractions_matrix[:,0] <= 0.30]
    print(fractions_matrix)
    P_o = 30e5
    x_space = []
    y_space = []
    P_space = []
    for x in fractions_matrix:
        liquid.x = np.array(x)
        P_o = minimize(fun=res_K, x0=P_o, args=(liquid, gas), method='Nelder-Mead').x[0]
        x_space.append(liquid.x[0])
        y_space.append(gas.x[0])
        P_space.append(P_o * 1e-5)
        print(liquid.x[0], gas.x[0], P_o*1e-5)

    # wb = pyxl.Workbook()
    # st = wb.active
    # st['A1'] = 'P_exp'
    # st['B1'] = 'x_exp'
    # st['C1'] = 'y_exp'
    # for i in range(len(x_space)):
    #     st[f'A{i + 2}'] = P_space[i]
    #     st[f'B{i + 2}'] = x_space[i]
    #     st[f'C{i + 2}'] = y_space[i]
    
    # wb.save((os.path.dirname(os.path.abspath(__file__)) + f'\\data\\methane_co2_{T_o}K_kij={k_ij}.xlsx'))
        
        
    # P_exp = [130.779896, 127.6256499,123.9861352, 120.5892548, 117.1923744,110.1559792, 103.3622184,
    #           96.32582322, 86.13518198, 82.73830156, 68.90814558, 55.0779896, 41.24783362, 34.21143847,
    #           27.66031196, 20.62391681, 14.07279029, 10.43327556, 7.279029463, 5.82322357, 4.610051993]
    # x_exp = [0.703549061,0.645093946, 0.613778706, 0.586638831, 0.557411273, 0.519832985, 0.482254697, 0.446764092,
    #         0.400835073, 0.382045929, 0.315240084, 0.252609603, 0.183716075, 0.150313152, 0.118997912, 0.08559499,
    #         0.052192067, 0.033402923, 0.014613779, 0.008350731, 0.004175365,]
    # y_exp = [0.770354906, 0.805845511, 0.82045929, 0.835073069, 0.845511482, 0.855949896, 0.866388309, 0.87473904,
    #           0.881002088, 0.881002088, 0.881002088, 0.876826722, 0.860125261, 0.8434238, 0.822546973, 0.780793319,
    #           0.699373695, 0.622129436, 0.455114823, 0.331941545, 0.131524008]
    # critical_p = [0.722]
    # critical_P = [131.99]
    P_exp = [ 35.2,40.1,59.9,70.1,80.6,82.7,99.2,102,106.9]
    x_exp = [0.0168,0.0257,0.0709,0.0938,0.124,0.131,0.19,0.205,0.225]
    y_exp = [0.148,0.206,0.362,0.406,0.424,0.424,0.42,0.411,0.401]

    plt.figure(figsize=(5, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['text.color'] = 'black'
    plt.plot(x_space, P_space, color='#333333', linewidth=0.8, label=f'PC-SAFT')
    plt.plot(y_space, P_space, color='#333333', linewidth=0.8)
    plt.scatter(x_exp, P_exp, marker='o', edgecolors='#333333', facecolor='none', linewidths=0.5, label='Experimental')
    plt.scatter(y_exp, P_exp, marker='o', edgecolors='#333333', facecolor='none', linewidths=0.5)
    # # # plt.scatter(critical_p, critical_P, marker='+', color='#333333', linewidths=0.5)
    plt.xlabel(r'$x_{H_{2}}\;\;y_{H_{2}}$')
    plt.ylabel(r'$P\;(bar)$')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend()
    time_f = time()
    print(time_f - time_0)

    plt.show()