import matplotlib.pyplot as plt
import numpy as np



# def compressibily(eta):
#     return (1 + eta + eta**2) / (1 - eta)**3

# def pressure(eta):
#     return (1 + 2 * eta + 3 * eta**2) / (1 - eta)**2

# def carnahan_starling(eta):
#     return (1 + eta + eta**2 - eta**3) / (1 - eta)**3


# eta = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.47, 0.49]
# P_rhokT = [1.229, 1.532, 1.933, 2.463, 3.166, 4.103, 5.372, 7.114, 9.544, 10.82, 12.44]


# eta_g = np.linspace(0, 0.5, 500)

# P_rhokT_compressibilty = compressibily(eta=eta_g)
# P_rhokT_pressure = pressure(eta=eta_g)
# P_rhokT_carnahan_starling = carnahan_starling(eta=eta_g)

# plt.figure(figsize=(10, 8))
# plt.scatter(eta, P_rhokT, marker='x', color='navy', zorder=5)
# plt.plot(eta_g, P_rhokT_compressibilty, color='darkorange', linewidth=0.95, label=r'Via compressibilidade')
# plt.plot(eta_g, P_rhokT_pressure, color='darkgreen', linewidth=0.95, label=r'Via pressão')
# plt.plot(eta_g, P_rhokT_carnahan_starling, color='purple', linewidth=0.95, label=r'Via Carnahan-Starling')
# plt.ylabel(ylabel=r'$\frac{P}{ρ\;k_{B}\;T}$')
# plt.xlabel(xlabel=r'$η$')
# plt.xlim(left=0.0, right=0.5)
# plt.ylim(bottom=0, top=max(P_rhokT_compressibilty))
# plt.legend(loc='upper left')
# plt.show()
def B2sw_anal(T, bo, Rsw, e_kB):
    return bo * (1 + (1 - np.exp(e_kB / T)) * (Rsw**3 - 1))

def B2sw_top(T, bo, Rsw, e_kB):
    return bo * (1 + (e_kB / T) * (1 - Rsw**3))

def B2sw_top_2(T, bo, Rsw, e_kB):
    return bo * (1 - (Rsw**3 - 1) * (e_kB / T + (1/2) * (e_kB / T)**2))

bo = 0.054
Rsw = 1.5
e_kB = 150

T = np.linspace(-10, 600, 1500)
B2sw_anal_T = B2sw_anal(T=T, bo=bo, Rsw=Rsw, e_kB=e_kB)
B2sw_top_T = B2sw_top(T=T, bo=bo, Rsw=Rsw, e_kB=e_kB)
B2sw_top_2_T = B2sw_top_2(T=T, bo=bo, Rsw=Rsw, e_kB=e_kB)
plt.figure(figsize=(10, 8))
plt.plot(T, B2sw_anal_T, color='darkgreen', linewidth=0.95, label=r'Solução exata')
plt.plot(T, B2sw_top_T, color='purple', linewidth=0.95, label=r'Via Teoria da Perturbação (truncado no primeiro termo)')
plt.plot(T, B2sw_top_2_T, color='darkorange', linewidth=0.95, label=r'Via Teoria da Perturbação (truncado no segundo termo)')
plt.ylabel(ylabel=r'$B_{2}\;[L\;mol^{-1}]$')
plt.xlabel(xlabel=r'$T [K]$')
plt.xlim(left=0.0, right=600)
plt.ylim(bottom=-1, top=0.1)
plt.legend(loc='lower right')
plt.show()