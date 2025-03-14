import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
# !!! variaveis globais !!! #
g_omega =  0.07780
g_psi =  0.45724
g_epsilon =1 - 2**0.5
g_sigma = 1 + 2**0.5
g_R = 8.314
def calc_a_b(T, Tc, Pc, omega):
    m = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = 1 #(1 + m * (1 - (T / Tc)**0.5))**2
    a = g_psi * alpha * (g_R * Tc)**2 / Pc
    b = g_omega * g_R * Tc / Pc

    return a, b

def P_V(V, a, b, T):
    return (g_R * T) / (V - b)  - a / (V**2 + 2*b*V - b**2) #((V + g_epsilon*b) * (V + g_sigma*b))

def dP_dV(V, a, b, T):
    return a*(2*V + 2*b) / (V**2 +2*b*V - b**2)**2 - (g_R * T) / (V - b)**2

def pontos_criticos(a, b, T):
    # 70/100**3 e 180/100**3
    raizes = fsolve(dP_dV, x0=[(70/100**3), (150/100**3)], args=(a, b, T))
    return raizes


def graph_P_V(T, Tc, Pc, omega):
    V = np.linspace((50/100**3), (400/100**3), 1500)
    a, b = calc_a_b(T=T, Tc=Tc, Pc=Pc, omega=omega)
    P = P_V(V, a, b, T)

    V_c = pontos_criticos(a=a, b=b, T=T)
    P_c = P_V(V_c, a, b, T)

    plt.figure(figsize=(5, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['text.color'] = 'black'
    plt.plot((V*100**3), (P/10**6), color='#333333', linewidth=0.3)
    plt.scatter((V_c*100**3), (P_c/10**6), color='#333333', linewidths=0.5)
    plt.xlabel(r'$V\;(cm^{3}\;mol^{-1}$')
    plt.ylabel(r'$P\;(MPa)$')
    plt.xlim(left=0, right = 400.0)
    plt.ylim(bottom=1.0, top=5.0)
    plt.grid(False)

    # !!! agora preciso o fazer o negocio !!! #
    if V_c[0] > V_c[1]:
        v_l = V_c[1]
        v_g = V_c[0]

    cp_g = 34.942 - 3.9957*10**(-2)*T +1.9184*10**(-4)*T**2 - 1.5303*10**(-7)*T**3 + 3.9321*10**(-11)*T**4
    cp_l = 3.48*16.043


if __name__ == '__main__':
    Tc = 196.06 # K
    Pc = 45.99*10**5 # Pa
    omega = 0.012
    T = 175.00 # K

    graph_P_V(T=T, Tc=Tc, Pc=Pc, omega=omega)