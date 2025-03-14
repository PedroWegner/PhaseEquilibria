import copy
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from time import time


"""
Esse código foi gerado apenas a partir da EoS de Peng-Robinson, caso seja necessario testar outra EoS, basta alterar
as funcoes sinalizadas.
"""
R = 8.314

class Compound:
    def __init__(self, name: str, z: float, Tc: float, Pc: float, omega: float, T: float = None):
        self.name = name
        self.z = z
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.b_ind = 0.0

        # Parametros parciais
        self.a_par = 0.0
        self.b_par = 0.0
        self.q_par = 0.0

        # fugacidade
        self.fugacity = 0.0

        self._calc_b()
        if T is not None:
            self.T = T
            self._calc_a()
        else:
            self.T = None
            self.a_ind = 0.0

    def _calc_b(self):
        """
        Esse funcao nao tem retorno, ela eh utilizada para calcular os parametros b individual, que nao depende da
        temperatura na PR-EoS
        :return:
        """
        self.b_ind = 0.07780 * R * self.Tc / self.Pc

    def _calc_alpha(self):
        """
        Calcuna o alpha, cuja dependencia eh a temperatura, do modelo.
        :return:
        """
        kappa = 0.37464 + 1.54226*self.omega -0.26992*self.omega**2
        return (1 + kappa * (1 - (self.T / self.Tc)**0.5))**2

    def _calc_a(self):
        """
        Calcula o parametro a
        :return:
        """
        alpha = self._calc_alpha()
        self.a_ind = 0.45724 * alpha * (R * self.Tc)**2 / self.Pc

    def update_a(self, T):
        self.T = T
        self._calc_a()

    def update_a_par(self, a_par):
        self.a_par = a_par

    def update_b_par(self, b_par):
        self.b_par = b_par

    def update_q_par(self, q_par):
        self.q_par = q_par

    def update_fugacity(self, fugacity):
        self.fugacity = fugacity

    def update_molar_fraction(self, z):
        """
        Essa funcao eh usada para ficar alterando a composicao do componente num flash
        :param z:
        :return:
        """
        self.z = z

class Mixture:
    def __init__(self, compounds: list[Compound],  k_ij_M: list[list[float]] = None):
        self.compounds = compounds
        self._n_comp = len(self.compounds)
        self.k_ij_M = k_ij_M or [[0.0] * self._n_comp for _ in range(self._n_comp)]
        # self._validate_mixture() # talvez tenha que tirar isso no flash negativo.
        self.a_mix = 0.0
        self.b_mix = 0.0
        self.calc_a_mix()
        self.calc_b_mix()

    def _validate_mixture(self):
        sum_z = sum(comp.z for comp in self.compounds)
        if not abs(sum_z - 1) < 1e-6:
            raise ValueError("Sum of mole must be equal 1")

    def get_compositon(self):
        return [comp.z for comp in self.compounds]

    def calc_cross_a(self):
        """
        Aqui pode-se alterar a forma com a qula o a_ij eh calculado
        :return: list[list[float]]
        """
        cross_a = []
        for i in range(self._n_comp):
            cross_a.append([])
            for j in range(self._n_comp):

                cross_a[i].append((self.compounds[i].a_ind * self.compounds[j].a_ind)**0.5*(1-self.k_ij_M[i][j]))
        return cross_a

    def calc_cross_b(self):
        """
        Nao foi implementado nenhuma calculo para o b cruzado, porque no Van Ness nao mostra
        :return:
        """
        cross_b = []
        for i in range(self._n_comp):
            cross_b.append([])
            for j in range(self._n_comp):
                cross_b[i].append((self.compounds[i].b_ind + self.compounds[j].b_ind)/2)
        return cross_b


    def calc_a_mix(self):
        """
        Nesta funcao, pode-se alterar a regra de mistura utilizada, foi implementada a de Van der Waals
        :return:
        """
        cross_a = self.calc_cross_a()
        a_mix = 0
        for i in range(self._n_comp):
            z_i = self.compounds[i].z
            for j in range(self._n_comp):
                z_j = self.compounds[j].z
                a_mix += (z_i * z_j) * cross_a[i][j]
        self.a_mix = a_mix

    def calc_b_mix(self):
        """
        Nesta funcao, pode-se alterar a regra de mistura de b
        :return:
        """
        cross_b = self.calc_cross_b()
        if cross_b is not None:
            b_mix = 0
            for i in range(self._n_comp):
                z_i = self.compounds[i].z
                for j in range(self._n_comp):
                    z_j = self.compounds[j].z
                    b_mix += (z_i * z_j) * cross_b[i][j]
        else:
            b_mix = sum(comp.b_ind * comp.z for comp in self.compounds)

        self.b_mix = b_mix

    def calc_a_par(self):

        for comp_i in self.compounds:
            a_aux = - self.a_mix
            for comp_j in self.compounds:
                z_j = comp_j.z
                a_aux += 2*z_j*(comp_i.a_ind * comp_j.a_ind)**0.5
            comp_i.update_a_par(a_aux)

    def calc_b_par(self):
        """
        Essa funcao depende da regra de mistura utilizada.
        Caso ela seja: b_mix = y_i * y_j * b_ij, o b parcial sera muito semelhante ao a parcial
        :return:
        """
        for comp_i in self.compounds:
            b_aux = - self.b_mix
            for comp_j in self.compounds:
                z_j = comp_j.z
                b_aux += z_j*(comp_i.b_ind + comp_j.b_ind)
            comp_i.update_b_par(b_aux)

    def calc_q_par(self, q):
        for comp in self.compounds:
            q_aux = q * (1 + comp.a_par / self.a_mix - comp.b_par / self.b_mix) # AQUI FOI ALTERADO
            comp.update_q_par(q_aux)

    def update_mixture(self):
        self.calc_a_mix()
        self.calc_b_mix()


class PR_EoS:
    def __init__(self, mixture: Mixture, T: float, p: float, vapor: bool=True):
        self.mixture = mixture
        self.T = T
        self.p = p
        self._e = 1 - 2 ** 0.5
        self._s = 1 + 2 ** 0.5
        self.vapor = vapor

    def update_pression(self, p):
        self.p = p

    def Z_function(self, Z, beta, q):
        """
        Implements the compressibility equation for solving using fsolve or something similar.
        :param Z: Compresibility
        :param beta: model's parameter
        :param q: model's parameter
        :return:
        """
        return Z - (1 + beta - q * beta * (Z - beta) / ((Z + self._e * beta) * (Z + self._s * beta)))

    def Z_function_L(self, Z, beta, q):
        """
        Implements the compressibility equation for solving using fsolve or something similar.
        :param Z: Compresibility
        :param beta: model's parameter
        :param q: model's parameter
        :return:
        """
        return Z - (beta + (Z + self._e * beta) * (Z + self._s*beta) * ((1 + beta - Z) / (beta * q)))


    def solve_Z(self):
        a_mix = self.mixture.a_mix
        b_mix = self.mixture.b_mix

        self.beta = b_mix * self.p / (R * self.T)
        self.q = a_mix / (b_mix * R * self.T)
        if self.vapor:
            Z_0 = [1.0]
            Z = fsolve(self.Z_function, Z_0, args=(self.beta, self.q))
        else:
            Z_0 = [self.beta]
            Z = fsolve(self.Z_function_L, Z_0, args=(self.beta, self.q))
        # print('Z: ', Z, ' when b: ', b_mix, ' when a: ', a_mix)
        self.Z = Z[0]

    def solve_fugacity(self):
        self.solve_Z()
        I = (1 / (self._s - self._e))*np.log((self.Z + self._s * self.beta) / (self.Z + self._e * self.beta))
        self.mixture.calc_a_par()
        self.mixture.calc_b_par()
        self.mixture.calc_q_par(self.q)

        phi_matrix = []
        for comp in self.mixture.compounds:
            ln_phi_aux = (comp.b_ind / self.mixture.b_mix) * (self.Z - 1) - np.log(self.Z - self.beta) - comp.q_par * I # AQUI TAMBEM FOI ALTERADO
            # abaixo converto para a fugacidade
            phi_matrix.append(np.exp(ln_phi_aux))
            comp.update_fugacity(np.exp(ln_phi_aux))
        return phi_matrix


class Flash:
    def __init__(self, mixture: Mixture, pr_eos: PR_EoS, T: float, p: float):
        """
        :param mixture: A mistura que se desejar colocar no flash
        :param pr_eos: A equacao de estado de Peng-Robinson
        :param T: Temperatura
        :param p: Pressao
        """
        self.mixture = mixture
        self.pr_eos = PR_EoS(self.mixture, T=T, p=p)
        self.x_matrix = 0.0
        self.y_matrix = 0.0
        self.z_matrix = self.mixture.get_compositon()

        # Estimativas iniciais do flash
        self.V = 0.60 # Trata-se de uma estimativa inicial
        self.K_matrix = np.array([7, 2.4, 0.8, 0.3]) # trata-se de uma estimativa inicial

    def rachford_rice_equation(self, v):
        """
        Implementacao da equacao de Rachford e Rice, 1952
        :param v:
        :return:
        """
        return np.sum(self.z_matrix * (self.K_matrix - 1) / (1 + v * (self.K_matrix - 1)))

    def rachford_rice_derivation(self, v):
        """
        Para avaliar a derivada de Rachfrod e Rice, 1952
        :param v:
        :return:
        """
        return - (np.sum(self.z_matrix * (self.K_matrix - 1)**2 / (1 + v * (self.K_matrix - 1))**2))

    def vapor_solve(self, tol=1e-8, max_iter=250):
        """
        Resolve a funcao de Rachford e Rice, 1952
        :param tol:
        :param max_iter:
        :return:
        """
        v = self.V
        # aplicacao do metodo d newton
        for i in range(max_iter):
            f = self.rachford_rice_equation(v)
            df = self.rachford_rice_derivation(v)

            v_new = v - f / df
            if abs(v_new - v) < tol:
                v = v_new
                break
            v = v_new
        self.V = v
        return v

    def update_compostion(self):
        self.x_matrix = self.z_matrix / (1 + self.V * (self.K_matrix - 1))
        self.y_matrix = self.K_matrix * self.x_matrix
        return self.x_matrix, self.y_matrix


    def update_K(self):
        """
        Nessa funcao, altera-se a composicao dos componentes da mistura para x e y, para obter os coeficientes de fuga-
        cidade, posteriormente, retornamos com a composicao global z.
        :return:
        """
        for i, comp in enumerate(self.mixture.compounds):
            print(comp.name)
            comp.z = self.x_matrix[i]
        self.mixture.update_mixture()
        phi_L_matrix = self.pr_eos.solve_fugacity()

        for i, comp in enumerate(self.mixture.compounds):
            comp.z = self.y_matrix[i]
        self.mixture.update_mixture()
        phi_G_matrix = self.pr_eos.solve_fugacity()

        # retorna a composicao z inicial
        for i, comp in enumerate(self.mixture.compounds):
            comp.z = self.z_matrix[i]

        self.K_matrix = np.array(phi_L_matrix) / np.array(phi_G_matrix)
        print('atualiza K ')
        print(self.K_matrix)
        return self.K_matrix

    def flash_solve(self, tol=1e-4, max_iter=250):
        for i in range(max_iter):
            v_old = self.V

            self.vapor_solve(tol=tol, max_iter=max_iter)
            #
            x_old = self.x_matrix.copy() if self.x_matrix is None else None
            self.update_compostion()

            K_old = self.K_matrix.copy()
            self.update_K()

            if np.linalg.norm(K_old - self.K_matrix) < tol:
                break
            # if abs(v_old - self.V) < tol:
            #     break


        return self.V, self.x_matrix, self.y_matrix

class ELV:
    def __init__(self, mixture: Mixture, T: float, p: float):
        self.liquid_phase = PR_EoS(mixture=Mixture([copy.deepcopy(comp) for comp in mixture.compounds]),
                                   T=T,
                                   p=p,
                                   vapor=False)
        self.vapor_phase = PR_EoS(mixture=Mixture([copy.deepcopy(comp) for comp in mixture.compounds]),
                                   T=T,
                                   p=p)


        self.T = T
        self.p = p

        # array para gerar grafico
        self.x_space = []
        self.y_space = []
        self.p_space = []

    def solve_bubble_p(self, tol=1e-4, max_iter=250):
        x_space = np.linspace(0.0, 0.7, 1500)
        p_old = 3*10**5 # isso eh uma estimativa inicial
        y_old = None
        for x in x_space:
            if y_old:
                y = y_old
            else:
                y = x
            # atualiza as fracoes molares
            self.liquid_phase.mixture.compounds[0].update_molar_fraction(x)
            self.vapor_phase.mixture.compounds[0].update_molar_fraction(y)
            self.liquid_phase.mixture.compounds[1].update_molar_fraction(1 - x)
            self.vapor_phase.mixture.compounds[1].update_molar_fraction(1 - y)

            # atualiza a pressao do problema
            self.liquid_phase.update_pression(p_old)
            self.liquid_phase.mixture.update_mixture()
            self.vapor_phase.update_pression(p_old)
            self.vapor_phase.mixture.update_mixture()

            # calcula a fugacidade

            self.liquid_phase.solve_fugacity()
            self.vapor_phase.solve_fugacity()

            # os Ks
            K1 = self.liquid_phase.mixture.compounds[0].fugacity  / self.vapor_phase.mixture.compounds[0].fugacity
            K2 = self.liquid_phase.mixture.compounds[1].fugacity  / self.vapor_phase.mixture.compounds[1].fugacity
            K = K1*self.liquid_phase.mixture.compounds[0].z + K2*self.liquid_phase.mixture.compounds[1].z
            y = K1*self.liquid_phase.mixture.compounds[0].z / K

            for i in range(max_iter):
                if K > 1:
                    p_old = p_old * 1.0002
                else:
                    p_old = p_old * 0.9998
                # atualiza a composicao
                self.vapor_phase.mixture.compounds[0].update_molar_fraction(y)
                self.vapor_phase.mixture.compounds[1].update_molar_fraction(1 - y)
                # atualiza a pressao
                self.liquid_phase.update_pression(p=p_old)
                self.liquid_phase.mixture.update_mixture()
                self.vapor_phase.update_pression(p=p_old)
                self.vapor_phase.mixture.update_mixture()
                # calcula a fugacidade
                self.liquid_phase.solve_fugacity()
                self.vapor_phase.solve_fugacity()
                # os Ks
                K1 = self.liquid_phase.mixture.compounds[0].fugacity / self.vapor_phase.mixture.compounds[0].fugacity
                K2 = self.liquid_phase.mixture.compounds[1].fugacity / self.vapor_phase.mixture.compounds[1].fugacity
                K = K1 * self.liquid_phase.mixture.compounds[0].z + K2 * self.liquid_phase.mixture.compounds[1].z
                y = K1 * self.liquid_phase.mixture.compounds[0].z / K

                if abs(1 - K) < tol:
                    y_old = y
            print(x, y, K, p_old)
            print('----')
            self.x_space.append(x)
            self.y_space.append(y)
            self.p_space.append(p_old)

    def graph_generator(self):
        plt.figure(figsize=(5, 6))
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12
        plt.rcParams['text.color'] = 'black'
        plt.plot(self.x_space, self.p_space, color='#333333', linewidth=0.3)
        plt.plot(self.y_space, self.p_space, color='#333333', linewidth=0.3)

        plt.xlabel(r'$x\;\;y$')
        plt.ylabel(r'$P\;(bar)$')
        plt.xlim(left=0, right=1.0)

        plt.grid(False)
        plt.show()

if __name__ == '__main__':
    # teste
    T_ = 273.15+50.0
    P = 2*10**5
    propano = Compound(name='propano',
                      z=0.3,
                      Tc=369.8,
                      Pc=42.5 * 10 ** 5,
                      omega=0.152
                      , T=T_)
    n_butano = Compound(name='n_butano',
                        z=0.1,
                        Tc=425.1,
                        Pc=37.96 * 10 ** 5,
                        omega=0.20
                        , T=T_)
    n_pentano = Compound(name='n_pentano',
                      z=0.15,
                      Tc=469.7,
                      Pc=33.7 * 10 ** 5,
                      omega=0.251
                      , T=T_)
    n_hexano = Compound(name='n_hexano',
                        z=0.45,
                        Tc=507.6,
                        Pc=30.3 * 10 ** 5,
                        omega=0.301
                        , T=T_)

    Mix = Mixture([propano, n_butano, n_pentano, n_hexano])
    flash_ = Flash(mixture=Mix, pr_eos=None, T=T_, p=P)
    print(flash_.flash_solve())