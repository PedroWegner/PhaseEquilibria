import numpy as np

from PC_saft_module import *
import os


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

fractions_matrix = np.array(generate_mole_fractions(n=2, num_points=50))
fractions_matrix = fractions_matrix[fractions_matrix[:,0] <= 0.85]



m = [1.0000, 2.3316]
sigma = [3.7039, 3.7086]
epsilon_ = [150.03, 222.88]
T_o = 310.93 # K
P_o = 3e5

state_liq = PC_saft_state(
    T=T_o,
    P=P_o,
    ncomp=2,
    x=fractions_matrix[0],
    m=m,
    sigma=sigma,
    epsilon=epsilon_
)

state_gas = PC_saft_state(
    T=T_o,
    P=P_o,
    ncomp=2,
    x=fractions_matrix[0],
    m=m,
    sigma=sigma,
    epsilon=epsilon_
)

x_space = []
y_space = []
P_space = []
for x in fractions_matrix:
    state_liq.x = np.array(x)
    P_o = minimize(fun=res_K, x0=P_o, args=(state_liq, state_gas), method='Nelder-Mead').x[0]
    x_space.append(state_liq.x[0])
    y_space.append(state_gas.x[0])
    P_space.append(P_o * 1e-5)


P_lit = [130.779896, 127.6256499,123.9861352, 120.5892548, 117.1923744,110.1559792, 103.3622184,
         96.32582322, 86.13518198, 82.73830156, 68.90814558, 55.0779896, 41.24783362, 34.21143847,
         27.66031196, 20.62391681, 14.07279029, 10.43327556, 7.279029463, 5.82322357, 4.610051993]
x_lit = [0.703549061,0.645093946, 0.613778706, 0.586638831, 0.557411273, 0.519832985, 0.482254697, 0.446764092,
         0.400835073, 0.382045929, 0.315240084, 0.252609603, 0.183716075, 0.150313152, 0.118997912, 0.08559499,
         0.052192067, 0.033402923, 0.014613779, 0.008350731, 0.004175365,]
y_lit = [0.770354906, 0.805845511, 0.82045929, 0.835073069, 0.845511482, 0.855949896, 0.866388309, 0.87473904,
         0.881002088, 0.881002088, 0.881002088, 0.876826722, 0.860125261, 0.8434238, 0.822546973, 0.780793319,
         0.699373695, 0.622129436, 0.455114823, 0.331941545, 0.131524008]
critical_p = [0.722]


plt.figure(figsize=(5, 6))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['text.color'] = 'black'
plt.plot(x_space, P_space, color='#333333', linewidth=0.9, label=f'')
plt.plot(y_space, P_space, color='#333333', linewidth=0.9)
plt.scatter(x_lit, P_lit, marker='o', edgecolors='#333333', facecolor='none', linewidths=0.5, label='Donnelly e Katz, 1954')
plt.scatter(y_lit, P_lit, marker='o', edgecolors='#333333', facecolor='none', linewidths=0.5)


plt.xlabel(r'$x_{CH_{4}}\;\;y_{CH_{4}}$')
plt.ylabel(r'$P\;(bar)$')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.legend()
plt.show()