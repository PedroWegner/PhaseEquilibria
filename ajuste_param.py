from Peng_robinson_module import *
import numpy as np
import itertools
import os
from scipy.special import comb
import openpyxl

# x1 = np.linspace(0.0001, 0.72, 350)
# x_linspace = np.array([x1, 1 - x1]).T





# plt.figure(figsize=(5, 6))
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 12
# plt.rcParams['text.color'] = 'black'
# plt.plot(x_space, P_space, color='#333333', linewidth=0.3)
# plt.plot(y_space, P_space, color='#333333', linewidth=0.3)

# plt.xlabel(r'$x\;\;y$')
# plt.ylabel(r'$P\;(bar)$')
# plt.ylim(bottom=0)
# plt.xlim(left=0)
#
#
# plt.grid(False)
# # plt.show()


def generate_mole_fractions(n, num_points=100):
    """
    Gera frações molares uniformemente distribuídas para uma mistura com n componentes.

    Parâmetros:
    - n: número de componentes.
    - num_points: número de subdivisões (maior = mais precisão, mas mais lento).

    Retorna:
    - matriz (m, n) com todas as possíveis frações molares.
    """
    k = num_points - 1  # Divisões do espaço
    indices = np.array(list(multichoose(n, k)))  # Gera as partições inteiras de k em n partes
    fractions = indices / k  # Normaliza para obter frações molares entre 0 e 1
    return fractions

def multichoose(n, k):
    """
    Gera todas as maneiras de somar k usando n números inteiros não negativos.
    Isso resolve o problema de gerar todas as possíveis frações molares.
    """
    if n == 1:
        yield (k,)
    else:
        for i in range(k + 1):
            for tail in multichoose(n - 1, k - i):
                yield (i,) + tail

# Exemplo: Gerando frações molares para uma mistura de 5 componentes
fractions_matrix = np.array(generate_mole_fractions(n=2, num_points=800))
fractions_matrix = fractions_matrix[fractions_matrix[:,0] <= 0.36]
print(fractions_matrix)



# CH2 CO2
T = 271 #K
P_0 = 32e5
Tc = [190.6, 304.2]
Pc = [45.99e5, 73.83e5]
omega = [0.012, 0.224]
state_liq = Peng_robinson_state(
    T=T,
    P=P_0,
    ncomp=2,
    x=fractions_matrix[0],
    Tc=Tc,
    Pc=Pc,
    omega=omega
)

state_gas = Peng_robinson_state(
    T=T,
    P=P_0,
    ncomp=2,
    x=fractions_matrix[0],
    Tc=Tc,
    Pc=Pc,
    omega=omega
)
state_liq.k_ij[0][1] = 0.12
state_liq.k_ij[0][1] = 0.12
state_gas.k_ij[0][1] = 0.12
state_gas.k_ij[0][1] = 0.12
state_liq.k_ij[1][0] = 0.12
state_liq.k_ij[1][0] = 0.12
state_gas.k_ij[1][0] = 0.12
state_gas.k_ij[1][0] = 0.12

x_space_1 = []
y_space_1 = []
P_space_1 = []
for x in fractions_matrix:
    state_liq.x = np.array(x)
    P_0 = minimize(fun=res_K, x0=P_0, args=(state_liq, state_gas), method='Nelder-Mead').x[0]
    x_space_1.append(state_liq.x[0])
    y_space_1.append(state_gas.x[0])
    P_space_1.append(P_0* 1e-5)


wb = openpyxl.load_workbook((os.path.dirname(os.path.abspath(__file__)) + f'\\data\\methane_co2_{T}.xlsx'))
sheet = wb.active

x_exp_1 = []
y_exp_1 = []
P_exp_1 = []
for l in filter(None, sheet.iter_rows(min_row=1, values_only=True)):
    P_exp_1.append(l[2])
    x_exp_1.append(l[0])
    y_exp_1.append(l[1])


plt.figure(figsize=(5, 6))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['text.color'] = 'black'
plt.plot(x_space_1, P_space_1, color='#333333', linewidth=0.9, label=f'PR - 271 K')
plt.plot(y_space_1, P_space_1, color='#333333', linewidth=0.9)
plt.scatter(x_exp_1, P_exp_1, marker='o', edgecolors='#333333', facecolor='none', linewidths=0.5, label='Donnelly e Katz, 1954')
plt.scatter(y_exp_1, P_exp_1, marker='o', edgecolors='#333333', facecolor='none', linewidths=0.5)

plt.xlabel(r'$x_{CH_{4}}\;\;y_{CH_{4}}$')
plt.ylabel(r'$P\;(bar)$')
plt.ylim(bottom=30, top=95)
plt.xlim(left=0, right=0.4)
plt.legend()


# P_lit = [130.779896, 127.6256499,123.9861352, 120.5892548, 117.1923744,110.1559792, 103.3622184,
#          96.32582322, 86.13518198, 82.73830156, 68.90814558, 55.0779896, 41.24783362, 34.21143847,
#          27.66031196, 20.62391681, 14.07279029, 10.43327556, 7.279029463, 5.82322357, 4.610051993]
# x_lit = [0.703549061,0.645093946, 0.613778706, 0.586638831, 0.557411273, 0.519832985, 0.482254697, 0.446764092,
#          0.400835073, 0.382045929, 0.315240084, 0.252609603, 0.183716075, 0.150313152, 0.118997912, 0.08559499,
#          0.052192067, 0.033402923, 0.014613779, 0.008350731, 0.004175365,]
# y_lit = [0.770354906, 0.805845511, 0.82045929, 0.835073069, 0.845511482, 0.855949896, 0.866388309, 0.87473904,
#          0.881002088, 0.881002088, 0.881002088, 0.876826722, 0.860125261, 0.8434238, 0.822546973, 0.780793319,
#          0.699373695, 0.622129436, 0.455114823, 0.331941545, 0.131524008]
# critical_p = [0.722]
# critical_P = [131.99]

# plt.scatter(critical_p, critical_P, marker='+', color='#333333', linewidths=0.5)

plt.grid(False)
plt.show()