import EoS
from EoS import  *
import  numpy as np
import matplotlib.pyplot as plt

x_space = []
y_space = []
P_space = []
T_space = []
def bubble_p(T: float,
             P_est: float,
             ncomp: int,
             Tc: list,
             Pc: list,
             omega: list,
             h: int = 15,
             max_iter: int = 3000,
             tol=1e-4):
    global x_space, y_space, P_space
    # !!! aqui eu sei que esta ruim, eh bom pensar numa funcao que possa facilitar esse negocio !!!
    x_l = [[0.0] * ncomp for _ in range(h)]
    y_l = False
    x_aux = np.linspace(0.0, 1.0, h)
    for i in range(h):
        x_l[i][0] = float(x_aux[i])
        x_l[i][1] = 1 - x_aux[i]

    _P = P_est
    for x_i in x_l:
        if not y_l:
            y_l = x_i
        # calcula fugacidade das fases
        print('---')
        phi_l = solve_fugacity(T=T, P=_P, ncomp=ncomp, x=x_i, Tc=Tc, Pc=Pc, omega=omega, vapor=False)
        print(EoS.Z, phi_l)
        phi_v = solve_fugacity(T=T, P=_P, ncomp=ncomp, x=y_l, Tc=Tc, Pc=Pc, omega=omega, vapor=True)
        print(EoS.Z, phi_v)
        k = sum(((phi_l[i] / phi_v[i]) * x_i[i]) for i in range(ncomp))
        y_l = [((phi_l[i] / phi_v[i]) * x_i[i] / k) for i in range(ncomp)]
        # ((phi_l[i] / phi_v[i]) * x_i[i] / k)
        # print('---')
        # print(x_i)
        # print(f'a_ind: {EoS.a_ind}, b_ind: {EoS.b_ind}, a_ij: {EoS.a_ij} e b_ij: {EoS.b_ij}')
        # print(f'a_par: {EoS.a_par}, b_par: {EoS.b_par}, q_par: {EoS.q_par}')
        # print(f'beta: {EoS.beta}, q: {EoS.q}, Z: {EoS.Z}')
        # print(phi_l)
        # print(phi_v)
        # print('---')

        print('x: ', x_i, _P)
        print('y: ', y_l)
        for i in range(max_iter):
            if k > 1:
                _P *= 1.0002
            else:
                _P *= 0.9998
            # calcula fugacidade das fases
            phi_l = solve_fugacity(T=T, P=_P, ncomp=ncomp, x=x_i, Tc=Tc, Pc=Pc, omega=omega, vapor=False)
            phi_v = solve_fugacity(T=T, P=_P, ncomp=ncomp, x=y_l, Tc=Tc, Pc=Pc, omega=omega, vapor=True)
            if i == 0:
                print(EoS.Z, EoS.phi, EoS.ln_phi)
            k = sum(((phi_l[j] / phi_v[j]) * x_i[j]) for j in range(ncomp))

            y_l = [((phi_l[j] / phi_v[j]) * x_i[j] / k) for j in range(ncomp)]
            if abs(1 - k) < tol:
                break
        print(y_l, _P)
        x_space.append(x_i[0])
        y_space.append(y_l[0])
        P_space.append(_P / 10**5)
    print(x_space)
    print(y_space)
    print(P_space)


def graph_generator():
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
    critical_P = [131.99]
    plt.figure(figsize=(5, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['text.color'] = 'black'
    plt.plot(x_space, P_space,color='#333333', linewidth=0.3)
    plt.plot(y_space,P_space,color='#333333', linewidth=0.3)
    plt.scatter(x_lit, P_lit, marker='^', edgecolors='#333333', facecolor='none', linewidths=0.5)
    plt.scatter(y_lit, P_lit, marker='o', edgecolors='#333333', facecolor='none', linewidths=0.5)
    plt.scatter(critical_p, critical_P, marker='+', color='#333333', linewidths=0.5)
    plt.xlabel(r'$x\;\;y$')
    plt.ylabel(r'$P\;(bar)$')
    plt.xlim(left=0, right=1.0)

    plt.grid(False)
    plt.show()