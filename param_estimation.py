import numpy as np
from scipy.optimize import minimize
from Peng_robinson_module import *
import os

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

# ******
# Teste | Methane + CO2 em 271.0e0
# *****
def func_res(k_ij, liquid, gas, x_exp, y_exp, P_exp, T_exp):
    res_P = 0.0e0
    for T in T_exp:
        liquid.T = T
        gas.T = T
        # AQUI PODE SER INTRODUZIDO A IDEIA DO PARAMETRO DEPEDENTE DA TEMPERATURA!!!
        liquid.k_ij[0][1] = k_ij
        gas.k_ij[0][1] = k_ij
        liquid.k_ij[1][0] = k_ij
        gas.k_ij[1][0] = k_ij
        for i in range(len(x_exp)):
            liquid.x = np.array([x_exp[i], 1 - x_exp[i]])
            gas.x = np.array([y_exp[i], 1 - y_exp[i]]) # O que esta abaixo deve ser tratado como estimativa
            # Colocar P_exp[i] eh meio aproximar para a solucao que quero
            P = minimize(fun=res_K, x0=P_exp[i], args=(liquid, gas), method="Nelder-Mead") 
            res_P += ((P_exp[i] - P.x[0])**2)*1e-5
    return res_P


# DONNELY & KATZ, 1954
T_exp = [271.0e0] #K
x_exp = [0.0675, 0.084, 0.103, 0.16, 0.157, 0.165, 0.191]
y_exp = [0.235, 0.3, 0.329, 0.367, 0.369, 0.387, 0.39]
P_exp = [50.537e5, 55.9156e5, 59.983e5, 68.119e5, 68.394e5, 72.531e5, 76.392e5]

Tc =[190.6, 304.2]
Pc = [45.99e5, 73.83e5]
omega = [0.012, 0.224]
liquid = Peng_robinson_state(T=None,
                        P=None,
                        ncomp=2,
                        x=None,
                        Tc=Tc,
                        Pc=Pc,
                        omega=omega,
                        liquid=True)

gas = Peng_robinson_state(T=None,
                            P=None,
                            ncomp=2,
                            x=None,
                            Tc=Tc,
                            Pc=Pc,
                            omega=omega,
                            liquid=False)


k_ij_0 = 0.125 # Estimativa inicial, se fosse n != 3, deveria ser um vetor unidimensional
k_ij_calc = minimize(fun=func_res, x0=k_ij_0, args=(liquid, gas, x_exp, y_exp, P_exp, T_exp), method='Nelder-Mead')
print(k_ij_calc)
print(k_ij_calc.x)

# ***
# TESTE DO GRAFICO
# ****
liquid.k_ij[0][1] = k_ij_calc.x
gas.k_ij[0][1] = k_ij_calc.x
liquid.k_ij[1][0] = k_ij_calc.x
gas.k_ij[1][0] = k_ij_calc.x
print(liquid.k_ij)
liquid.T = T_exp[0]
gas.T = T_exp[0]
P_o = P_exp[0]

fractions_matrix = np.array(generate_mole_fractions(n=2, num_points=150))
fractions_matrix = fractions_matrix[fractions_matrix[:,0] <= 0.36]
x_space = []
y_space = []
P_space = []

for x in fractions_matrix:
    liquid.x = np.array(x)
    P_o = minimize(fun=res_K, x0=P_o, args=(liquid, gas), method='Nelder-Mead').x[0]
    x_space.append(liquid.x[0])
    y_space.append(gas.x[0])
    P_space.append(P_o * 1e-5)


P_exp = [50.53778268, 55.91560949, 59.98345284, 68.11913955, 68.39492554, 72.53171539, 76.39271925]

plt.figure(figsize=(5, 6))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['text.color'] = 'black'
plt.plot(x_space, P_space, color='#333333', linewidth=0.9, label=f'Peng-Robinson')
plt.plot(y_space, P_space, color='#333333', linewidth=0.9)
plt.scatter(x_exp, P_exp, marker='o', edgecolors='#333333', facecolor='none', linewidths=0.5, label='Donnelly e Katz, 1954')
plt.scatter(y_exp, P_exp, marker='o', edgecolors='#333333', facecolor='none', linewidths=0.5)

plt.xlabel(r'$x_{CH_{4}}\;\;y_{CH_{4}}$')
plt.ylabel(r'$P\;(bar)$')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.legend()
plt.show()