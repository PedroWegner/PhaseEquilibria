import numpy as np
from scipy.optimize import  minimize
import matplotlib.pyplot as plt


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
    aux_3 = 2 * state.zeta[2]**2 / aux_1**2
    d_ij = state.d[:, np.newaxis] * state.d[np.newaxis, :] / (state.d[:, np.newaxis] + state.d[np.newaxis, :])
    # EQ (A.7)
    state.g_ij = 1.0e0 / aux_1 + d_ij * aux_2 + d_ij**2 * aux_3

def calc_ab_m(state: PC_saft_state) -> None:
    aux_1 = (state.mean_m - 1.0e0) / state.mean_m
    aux_2 = aux_1 * (state.mean_m - 2.0e0) / state.mean_m
    # EQ (A.18)
    state.a_m = saft_a0  + aux_1 * saft_a1 +aux_2 * saft_a2
    # EQ (A.19)
    state.b_m = saft_b0  + aux_1 * saft_b1 +aux_2 * saft_b2

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
    aux_1 = 1 + state.mean_m * ((8 * state.eta - 2 * state.eta**2) / (1 - state.eta)**4)
    aux_2 = (1 - state.mean_m) * (20 * state.eta - 27 * state.eta**2 + 12 * state.eta**3 - 2 * state.eta**4)
    aux_2 = aux_2 / ((1 - state.eta) * (2 - state.eta))**2
    state.C_1 = 1 / (aux_1 + aux_2)
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
    # EQ (A.12)
    state.m2e2s3 = np.sum(x_ij * m_ij * e_ij**2 * state.s_ij**3)

def calc_grad_g_rho(state: PC_saft_state) -> None:
    d_ij = state.d[:, np.newaxis] * state.d[np.newaxis, :] / (state.d[:, np.newaxis] + state.d[np.newaxis, :])
    # EQ (A.27)
    zeta_aux = 1 - state.zeta[3]
    aux_1 = state.zeta[3] / zeta_aux**2
    aux_2 = 3 * state.zeta[2] / zeta_aux**2 + 6 * state.zeta[2] * state.zeta[3] / zeta_aux**3
    aux_3 = 4 * state.zeta[2]**2 / zeta_aux**3 + 6 * state.zeta[2]**2 * state.zeta[3] / zeta_aux**4
    state.grad_g_ij_rho = aux_1 + d_ij * aux_2 + d_ij**2 * aux_3

def helmholtz_residual(state: PC_saft_state) -> None:
    # EQ (A.6)
    zeta_aux = 1 - state.zeta[3]
    aux_1 = 3 * state.zeta[1] * state.zeta[2] / zeta_aux
    aux_2 = state.zeta[2]**3 / (state.zeta[3] * zeta_aux**2)
    aux_3 = (state.zeta[2]**3 / state.zeta[3]**2 - state.zeta[0])*np.log(zeta_aux)
    state.helmholtz_hs = (1 / state.zeta[0]) * (aux_1 + aux_2 + aux_3)

    # EQ (A.4)
    sum_aux = np.sum(state.x * (state.m - 1) * np.log(np.diagonal(state.g_ij)))
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
    n_values = np.arange(4)
    # EQ (A.34)
    state.zeta_x = (np.pi * state.rho / 6) * (state.m * state.d ** n_values[:, None])

def calc_paramets_x(state: PC_saft_state) -> None:
    m_x = state.m / state.mean_m**2
    # EQ (A.44)
    state.a_x = m_x[None, :] * saft_a1[:, None] + m_x[None, :] * (3 - 4 / state.mean_m) * saft_a2[:, None]
    # EQ (A.45)
    state.b_x = m_x[None, :] * saft_b1[:, None] + m_x[None, :] * (3 - 4 / state.mean_m) * saft_b2[:, None]

    i_v = (np.arange(7))[:, None]
    # EQ (A.42)
    state.I_1_x = np.sum(state.a_m[:, None] * i_v * state.zeta_x[3, :] * state.eta**(i_v - 1) + state.a_x * state.eta**i_v, axis=0)
    # EQ (A.43)
    state.I_2_x = np.sum(state.b_m[:, None] * i_v * state.zeta_x[3, :] * state.eta**(i_v - 1) + state.b_x * state.eta**i_v, axis=0)

    # AUXILIARES DA EQ (A.41)
    aux_1 = state.m * (8 * state.eta - 2 *state.eta**2) / (1 - state.eta)**4
    aux_2 = - state.m * (20 * state.eta - 27 * state.eta**2 + 12 * state.eta**3 - 2 * state.eta**4)
    aux_2 = aux_2 /  ((1 - state.eta) * (2 - state.eta))**2
    # EQ (A.41)
    state.C_1_x = state.C_2 * state.zeta_x[3, :] - state.C_1**2 * (aux_1 + aux_2)

    # AUXILIARES DA EQ (A.39) E (A.40)
    sum_aux_1 = (np.sum(state.x[None, :] * state.m[None, :] * (state.e_ij / state.T) * state.s_ij**3, axis=1))
    sum_aux_2 = (np.sum(state.x[None, :] * state.m[None, :] * (state.e_ij / state.T)**2 * state.s_ij**3, axis=1))
    state.m2es3_x = 2 * state.m * sum_aux_1
    state.m2e2s3_x = 2 * state.m * sum_aux_2


def calc_chemical_pow(state: PC_saft_state) -> None:
    zeta_aux = 1 - state.zeta[3]
    # AUXILIARES DA EQ (A.36)
    aux_1 = - state.zeta_x[0, :] * state.helmholtz_hs / state.zeta[0]
    aux_2 = 3 * (state.zeta_x[1, :] * state.zeta[2] + state.zeta[1] * state.zeta_x[2, :]) / zeta_aux
    aux_2 += 3 * state.zeta[1] * state.zeta[2] * state.zeta_x[3, :] / zeta_aux**2
    aux_2 +=  3 * state.zeta[2]**2 * state.zeta_x[2, :] / (state.zeta[3] * zeta_aux**2)
    aux_2 += state.zeta[2]**3 * state.zeta_x[3, :] * (3 * state.zeta[3] - 1) / (state.zeta[3]**2 * zeta_aux**3)
    aux_2 += (state.zeta[0] - state.zeta[2]**3 / state.zeta[3]**2) * state.zeta_x[3, :] / zeta_aux
    aux_3 = (3 * state.zeta[2]**2 * state.zeta_x[2, :] * state.zeta[3] - 2 * state.zeta[2]**3 * state.zeta_x[3, :]) / state.zeta[3]**3
    aux_3 -= state.zeta_x[0, :]
    aux_3 *= np.log(zeta_aux)
    # EQ (A.36)
    state.prime_helmholtz_hs_x = aux_1 + (1 / state.zeta[0]) * (aux_2 + aux_3)

    # AUXILIARES DA EQ (A.37)
    aux_1 = state.zeta_x[3, :] / zeta_aux**2
    aux_2 = 3 * state.zeta_x[2, :] / zeta_aux**2 + 6 * state.zeta[2] * state.zeta_x[3, :] / zeta_aux**3
    aux_3 = 4 * state.zeta[2] * state.zeta_x[2, :] / zeta_aux**3 + 6 * state.zeta[2]**2 * state.zeta_x[3, :] / zeta_aux**4
    d_ij = state.d[:, np.newaxis] * state.d[np.newaxis, :] / (state.d[:, np.newaxis] + state.d[np.newaxis, :])
    # EQ (A.37)
    state.grad_g_ij_x = aux_1[:, np.newaxis, np.newaxis] + d_ij * aux_2[:, np.newaxis, np.newaxis]  + d_ij**2 * aux_3[:, np.newaxis, np.newaxis]

    # AUXILIARES DA EQ (A.35)
    sum_aux = np.sum(state.x * (state.m - 1) * (state.g_ij.diagonal())**-1 * state.grad_g_ij_x[:, :, 0].diagonal(), axis=0)
    aux_1 = - (state.m - 1) * np.log(state.g_ij.diagonal())
    # EQ (A.35)
    state.prime_helmholtz_hc_x = state.m * state.helmholtz_hs + state.mean_m * state.prime_helmholtz_hs_x - sum_aux + aux_1

    # AUXILIARES DA EQ (A.38)
    aux_1 = - 2 * np.pi * state.rho * (state.I_1_x * state.m2es3 + state.I_1 * state.m2es3_x)
    aux_2 = state.m * state.C_1 * state.I_2 + state.mean_m * state.C_1_x * state.I_2 + state.mean_m * state.C_1 * state.I_2_x
    aux_2 = aux_2 * state.m2e2s3 + state.mean_m * state.C_1 * state.I_2 * state.m2e2s3_x
    aux_2 = - np.pi * state.rho * aux_2
    state.prime_helmholtz_disp_x = aux_1 + aux_2

    # EQUACAO IMPLICITA
    state.prime_helmholtz_res_x = state.prime_helmholtz_hc_x  + state.prime_helmholtz_disp_x

    # AUXILIARES DA EQ(A.33)
    sum_aux = np.sum(state.x * state.prime_helmholtz_res_x)
    # EQ (A.33)
    state.chemical_pow = state.helmholtz_res + (state.Z - 1) + state.prime_helmholtz_res_x - sum_aux

def calc_fugacity(state: PC_saft_state) -> None:
    calc_zeta_x(state=state)
    calc_paramets_x(state=state)
    calc_chemical_pow(state=state)
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
    calc_fugacity(state=state)
# ************************************************************
# FIM UPDATE STATE
# ************************************************************



if __name__ == "__main__":
    # Abaixo eh um teste EXEMPLO 13.7 DO VAN
    # m = [1.2053, 1.0000]
    # sigma = [3.3130, 3.7039]
    # epsilon_ = [90.96, 150.03]
    m = [1.000, 2.3316]
    sigma = [3.7039, 3.7086]
    epsilon_ = [150.03, 222.88]
    T_o = 310.93  # K
    P_o = 73.5e5
    x_o = 0.25
    x = [x_o, 1 - x_o]
    
    state = PC_saft_state(T=T_o,
                                 P=P_o,
                                 ncomp=2,
                                 x=x,
                                 m=m,
                                 sigma=sigma,
                                 epsilon=epsilon_,
                                 liquid=True)
    update_state(state=state)
    print(state.Z)
    print(state.phi)

    # m = [2.3316]
    # sigma = [3.7086]
    # epsilon_ = [222.88]
    # T_o = 350  # K
    # P_o = 9.7543e5

    # state = PC_saft_state(T=T_o,
    #                              P=P_o,
    #                              ncomp=1,
    #                              x=[1.0],
    #                              m=m,
    #                              sigma=sigma,
    #                              epsilon=epsilon_,
    #                              liquid=True)
    
    # update_state(state=state)
    # print(state.Z)
    # print((state.Z * R * T_o / P_o)*100**3)
