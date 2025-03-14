from webbrowser import Error
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from time import time

class Component:
    def __init__(self, name: str, Tc: float, Pc: float, omega: float, mole_fraction_l: float = 0.0, mole_fraction_g: float =0.0):
        self.name = name
        self.mole_fraction_g = mole_fraction_g
        self.mole_fraction_l = mole_fraction_l
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.fugacity_g = 0.0
        self.fugacity_l = 0.0


        # if mole_fraction_g:
        #     if mole_fraction_g < 0 or mole_fraction_g > 1:
        #         raise ValueError('Molar fraction must be between 0 and 1')
        # else:
        #     if mole_fraction_l < 0 or mole_fraction_l > 1:
        #         raise ValueError('Molar fraction must be between 0 and 1')
        #     if not mole_fraction_l:
        #         raise Error("precisa coisar")



class PR:
    def __init__(self, components: list[Component], T: float = 0.0, P: float = 0.0, k_ij_M: list[list[float]] = None,
                 l_ij_M: list[list[float]] = None):
        self.components = components
        self.k_ij_M = k_ij_M or [[0.0]*len(components) for _ in range(len(components))]
        self.l_ij_M = l_ij_M or [[0.0]*len(components) for _ in range(len(components))]
        self.T = T
        self.P = P
        self._sigma = 1 + np.sqrt(2)
        self._epsilon = 1 - np.sqrt(2)
        self._omega = 0.07780
        self._psi = 0.45724
        self._Zc = 0.30740
        self.R = 8.314  # J mol-1 k-1
        self._params = None
        self._V_g = 0.0
        self._V_l = 0.0
        #
        self._calculate_parameters()

    def _validade_molar_fraction(self):
        if True:
            total = sum(c.mole_fraction_g for c in self.components)
        else:
            total = sum(c.mole_fraction_l for c in self.components)
        if not abs(total - 1) < 1e-6:
            raise ValueError("Sum of mole must be equal 1")

    def _calculate_parameters(self):
        params = {}

        for comp in self.components:
            alpha = (1 + (0.37464 + 1.54226*comp.omega -0.26992*comp.omega**2) * (1 - (self.T / comp.Tc)**0.5))**2
            a = (self._psi * alpha * (self.R * comp.Tc)**2) / comp.Pc
            b = (self._omega * self.R * comp.Tc) / comp.Pc
            print('alpha: ', alpha)
            params[comp.name] = {'a': a,
                                 'b': b}

        self._params = params

    def _mixture_coefficients(self, vapor: bool= True):
        a_mix = 0
        b_mix = 0
        params = self._params
        n = len(self.components)

        for i in range(n):
            comp_i = self.components[i]
            if vapor:
                x_i = comp_i.mole_fraction_g
            else:
                x_i = comp_i.mole_fraction_l
            # b_i = params[comp_i.name]['b']
            # b_mix += x_i*b_i
            for j in range(n):
                comp_j = self.components[j]
                if vapor:
                    x_j = comp_j.mole_fraction_g
                else:
                    x_j = comp_j.mole_fraction_l
                a_i = params[comp_i.name]['a']
                a_j = params[comp_j.name]['a']
                b_i = params[comp_i.name]['b']
                b_j = params[comp_j.name]['b']
                # parametros binarios
                a_ij = ((a_i*a_j)**0.5)*(1-self.k_ij_M[i][j]) # parametro a_ij
                b_ij = ((b_i + b_j)/2)*(1-self.l_ij_M[i][j]) # parametro b_ij
                # parametros da mistura
                a_mix += x_i * x_j * a_ij
                b_mix += x_i * x_j * b_ij

        return a_mix, b_mix

    def Z_equation_v(self, Z, B, Q):
        return Z - (1 + B - Q * B * (Z - B) / ((Z+self._epsilon*B)*(Z+self._sigma*B)))

    def Z_equation_l(self,Z, B, Q):
        return Z - (B +(Z+self._epsilon*B)*(Z+self._sigma*B)*((1+B-Z)/(Q*B)))

    def _solve_for_Z(self, vapor: bool = True):

        a_mix, b_mix = self._mixture_coefficients(vapor)
        beta = (b_mix * self.P) / (self.R * self.T)
        q = a_mix / (b_mix * self.R * self.T)
        if vapor:
            Z_0 = [1.0]
            Z = fsolve(self.Z_equation_v, Z_0, args=(beta, q))
        else:
            Z_0 = [beta]
            Z = fsolve(self.Z_equation_l, Z_0, args=(beta, q))
        self.Z = Z[0]
        return Z[0], a_mix, b_mix, beta, q

    def _fugacity(self, vapor: bool = True):
        if vapor:
            Z, a_mix, b_mix, beta, q = self._solve_for_Z(vapor=vapor)
            self._V_g = Z * self.R * self.T / self.P # m3 mol-1
        else:
            Z, a_mix, b_mix, beta, q = self._solve_for_Z(vapor=vapor)
            self._V_l = Z * self.R * self.T / self.P  # m3 mol-1
        params = self._params
        I = (1/(self._sigma - self._epsilon))*np.log((Z+self._sigma*beta)/(Z+self._epsilon*beta))
        n = len(self.components)
        for i in range(n):
            comp_i = self.components[i]
            a_i = params[comp_i.name]['a']
            b_i = params[comp_i.name]['b']
            a_i_par = -a_mix
            b_i_par = -b_mix
            for j in range(n):
                comp_j = self.components[j]
                if vapor:
                    y_j = comp_j.mole_fraction_g
                else:
                    y_j = comp_j.mole_fraction_l
                a_j = params[comp_j.name]['a']
                b_j = params[comp_j.name]['b']
                # parametros a e b parciais
                a_i_par += 2*y_j*(a_i*a_j)**0.5*(1-self.k_ij_M[i][j])
                b_i_par += y_j*(b_i + b_j)

            q_i_par = q*(1 + a_i_par/a_mix - b_i_par/b_mix)
            params[comp_i.name]['a_par'] = a_i_par
            params[comp_i.name]['q_par'] = q_i_par
            params[comp_i.name]['b_par'] = b_i_par
            fug_i = np.exp((b_i_par/b_mix)*(Z - 1) -np.log(Z-beta)-q_i_par*I)
            a_i_par = 0
            b_i_par = 0
            if vapor:
                comp_i.fugacity_g = fug_i
            else:
                comp_i.fugacity_l = fug_i

if __name__ == '__main__':
    "Teste de fugacidades, grandeza"
    nitrogenio = Component(
        name='nitrogenio',
        mole_fraction_g=0.4,
        mole_fraction_l=0.4,
        Tc=126.2,
        Pc=34.00 * 10 ** 5,
        omega=0.038
    )

    metano = Component(
        name='metano',
        mole_fraction_g=0.6,
        mole_fraction_l=0.6,
        Tc=190.6,
        Pc=45.99 * 10 ** 5,
        omega=0.012
    )


    list_com = [nitrogenio, metano]
    PR_oes = PR(
        components=list_com,
        T=200.00,
        P=30*10**5
    )

    PR_oes._fugacity(vapor=True)
    print("Z: ", PR_oes.Z)
    print(nitrogenio.fugacity_g)
    print(metano.fugacity_g)




    ##
    "Abaixo tem teste para ver se as equacoes cubicas estao resolvendo certo (Exemplo 3.9 Van Ness)"
    t_0 = time()
    butano = Component(
        name='butano',
        mole_fraction_g=1.0,
        mole_fraction_l=1.0,
        Tc=425.1,
        Pc=37.96 * 10 ** 5,
        omega=0.2
    )
    PR_oes = PR(
        components=[butano],
        T=350,
        P=9.4573*10**5
    )

    PR_oes._fugacity(vapor=True)




    "Teste de BOL P (Exemplo 13.7 VAN NESS), considerando x1 = 0.2"

    # #TESTE DO BOLHA P
    # T = 310.93 # K
    # # estimativa
    #
        # y = 0.6
        # x = 0.2
    # metano = Component(
    #     name='metano',
    #     mole_fraction_l=x,
    #     mole_fraction_g= y,
    #     Tc=190.6,
    #     Pc=45.99 * 10 ** 5,
    #     omega=0.012
    # )
    # n_butano = Component(
    #     name='n_butano',
    #     mole_fraction_l=1-x,
    #     mole_fraction_g= 1 - y,
    #     Tc=425.1,
    #     Pc=37.96 * 10 ** 5,
    #     omega=0.200
    # )
    #
    # P = 25*10**5 # Pa
    #
    # PR_e = PR(components=[metano, n_butano],
    #           T=T,
    #           P=P)
    # PR_e._fugacity(vapor=False)
    # PR_e._fugacity(vapor=True)

    #
    # y = (metano.fugacity_l/metano.fugacity_g)*x / ((metano.fugacity_l/metano.fugacity_g)*x + (n_butano.fugacity_l/n_butano.fugacity_g)*(1-x))
    # cond_K = (metano.fugacity_l/metano.fugacity_g)*x + (n_butano.fugacity_l/n_butano.fugacity_g)*(1-x)
    #
    # n = 0
    # cond = True
    # while cond and n <= 2000:
    #     if cond_K > 1:
    #         P = P * 1.002
    #     else:
    #         P = P * 0.998
    #     PR_e.P = P
    #
    #     metano.mole_fraction_g = y
    #     n_butano.mole_fraction_g = 1 - y
    #     PR_e._fugacity(vapor=False)
    #     PR_e._fugacity(vapor=True)
    #     y = (metano.fugacity_l / metano.fugacity_g) * x / (
    #                 (metano.fugacity_l / metano.fugacity_g) * x + (n_butano.fugacity_l / n_butano.fugacity_g) * (1 - x))
    #     cond_K = (metano.fugacity_l / metano.fugacity_g) * x + (n_butano.fugacity_l / n_butano.fugacity_g) * (1 - x)
    #
    #     if abs(1 - cond_K) < 1e-3:
    #         cond = False
    #
    #     n += 1
    #



    "Teste para geração da figura 13.13 (Van Ness)"
    P_graph = []
    x_graph = []
    y_graph = []
    x_space = np.linspace(0.0, 1.0, 15)
    T = 310.93 #K
    P = 3*10**5 #Pa
    metano = Component(
        name='metano',
        mole_fraction_l=0.99607108,
        mole_fraction_g= 0.99607108,
        Tc=190.6,
        Pc=45.99 * 10 ** 5,
        omega=0.012
    )
    n_butano = Component(
        name='n_butano',
        mole_fraction_l=0.003928910,
        mole_fraction_g=0.003928910,
        Tc=425.1,
        Pc=37.96 * 10 ** 5,
        omega=0.20
    )
    k_ij = [
        [0.0, 0.0],
        [0.0, 0.0]
    ]
    l_ij = [
        [0.0, 0.0],
        [0.0, 0.0]
    ]
    t_0 = time()
    PR_oes = PR(components=[metano, n_butano],
                T=T,
                P=16517526.827650828,
                k_ij_M=k_ij,
                l_ij_M=l_ij)
    PR_oes._fugacity(vapor=True)
    print(f'Z_g: {PR_oes.Z}, phi: {metano.fugacity_g} e {n_butano.fugacity_g}')
    PR_oes._fugacity(vapor=False)
    print(f'Z_l: {PR_oes.Z}, phi: {metano.fugacity_l} e {n_butano.fugacity_l}')
    # # # primeira estimativa do y1
    y = x_space[0]
    # for i, x in enumerate(x_space):
    #
    #     metano.mole_fraction_l = x
    #     metano.mole_fraction_g = y
    #     n_butano.mole_fraction_l = 1 - x
    #     n_butano.mole_fraction_g = 1 - y
    #     # calcunado as fugacidades nas fases
    #     PR_oes._fugacity(vapor=False) #fase liquida
    #     print(PR_oes.Z)
    #     PR_oes._fugacity(vapor=True) # fase vapor
    #     print(PR_oes.Z)
    #     K1 = metano.fugacity_l/metano.fugacity_g
    #     K2 = n_butano.fugacity_l / n_butano.fugacity_g
    #     y = K1 * x / (K1*x + K2*(1-x)) # normalizando o novo y
    #     cond_K = K1*x + K2*(1-x) # analisando a condicao K = 1
    #     # agora inicia os loopings de convergencia
    #     print('as fugacidades')
    #     print(metano.fugacity_l, n_butano.fugacity_l)
    #     print(metano.fugacity_g, n_butano.fugacity_g)
    #     n = 0
    #     cond = True
    #     print(cond_K)
    #     while cond and n < 2000:
    #         if cond_K > 1:
    #             P = P * 1.0002
    #         else:
    #             P = P * 0.9998
    #         PR_oes.P = P
    #         metano.mole_fraction_g = y
    #         n_butano.mole_fraction_g = 1 - y
    #         PR_oes._fugacity(vapor=False)
    #         PR_oes._fugacity(vapor=True)
    #         K1 = metano.fugacity_l / metano.fugacity_g
    #         K2 = n_butano.fugacity_l / n_butano.fugacity_g
    #         y = K1 * x / (K1*x + K2*(1-x))
    #         cond_K = K1*x + K2*(1-x)
    #
    #         if abs(1 - cond_K) < 1e-4:
    #             cond = False
    #
    #         n += 1
    #
    #     P_graph.append(P*10**(-5))
    #     x_graph.append(x)
    #     y_graph.append(y)
    #
    # #
    # #
    # #
    # # #
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
    # # # P_lit = [58.4, 65.5, 70.1, 72.4, 75.9]
    # # # x_lit = [0.3994, 0.499, 0.5990, 0.6992, 0.7991]
    # # P_lit = [63.2, 66.6, 68.5, 72.5, 78.5, 92.0, 108.7, 132.0, 161.8, 177.2, 184.6, 104.9]
    # # x_lit = [0.3989, 0.4015, 0.4563, 0.4982, 0.5046, 0.5990, 0.6196, 0.6476, 0.6992, 0.7497, 0.7992, 0.9695]
    # #
    # # # gerando o grafico
    # plt.figure(figsize=(5, 6))
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.size'] = 12
    # plt.rcParams['text.color'] = 'black'
    # plt.plot(x_graph,P_graph,color='#333333', linewidth=0.3)
    # plt.plot(y_graph,P_graph,color='#333333', linewidth=0.3)
    # # plt.scatter(x_lit, P_lit, marker='^', edgecolors='#333333', facecolor='none', linewidths=0.5)
    # # plt.scatter(y_lit, P_lit, marker='o', edgecolors='#333333', facecolor='none', linewidths=0.5)
    # # plt.scatter(critical_p, critical_P, marker='+', color='#333333', linewidths=0.5)
    # plt.xlabel(r'$x\;\;y$')
    # plt.ylabel(r'$P\;(bar)$')
    # plt.xlim(left=0, right=1.0)
    #
    # plt.grid(False)
    # plt.show()
    #
