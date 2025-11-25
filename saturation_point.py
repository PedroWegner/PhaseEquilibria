from CubicEoS_module import *
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as pyxl
import os
from time import time

class InitialPointFinder():
    def __init__(self, mixture_template: Mixture):
        self.mixture_template = mixture_template
        self.Tc_array = np.array([c.Tc for c in self.mixture_template.components])
        self.Pc_array = np.array([c.Pc for c in self.mixture_template.components])
        self.omega_array = np.array([c.omega for c in self.mixture_template.components])
    
    def _calculate_Wilson_K_factor(self, T: float, P: float):
        Pr = P / self.Pc_array
        Tr = T / self.Tc_array

        ln_K = np.log(1 / Pr) + 5.373 * (1 + self.omega_array) * (1 - 1 / Tr)
        return np.exp(ln_K)

    def _calculate_residual_and_derivative(self, T: float, P: float, z: np.array) -> float:
        K = self._calculate_Wilson_K_factor(T=T, P=P)
        f_K_T = np.sum(K * z) - 1

        dK_dT = K * (5.373 * (1 + self.omega_array) * (self.Tc_array / T**2))

        df_dT = np.sum(dK_dT * z)

        return f_K_T, df_dT
    
    def find_bubble_point_guess(self, z_feed: np.array, P_spec: float) -> np.array:
        T_guess = np.sum(z_feed * self.Tc_array)

        for _ in range(25):
            f, df = self._calculate_residual_and_derivative(T=T_guess, P=P_spec, z=z_feed)

            step = - f / df
            T_guess += step

            if abs(step) < 1e-9:
                break
 
        T_0 = T_guess
        K_0 = self._calculate_Wilson_K_factor(T=T_0, P=P_spec)
        X_0 = np.r_[np.log(K_0), np.log(T_0), np.log(P_spec)]

        return X_0


class PressureSatEngine:
    def __init__(self, mixture: Mixture, EoSEgine: ModeloPengRobinson, max_iter: int=1500, tol: float=1e-6):
        # 1. Inicializa a misutra testada, a Engine Termodinamica (Equacao de Estado)
        # A ideia é poder deixar generalizada para que entre qualquer EoS com o metodo 'calculate_state' implementado
        self.mixture = mixture
        self.eos_engine = EoSEgine()
        self.max_iter = max_iter
        self.tol = tol
        
    def _successive_substitution(self, T: float, P: float, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Calcula os passos sucessivos para o processo iterativo
        """
        # 1. cria duas fases, uma pressuposta e outra incipiente **
        state_liq = State(mixture=self.mixture, T=T, P=P, z=x, is_vapor=False)
        state_vap = State(mixture=self.mixture, T=T, P=P, z=y, is_vapor=True)

        # 2. Usa a Engine termodinamica para calcular parametros essenciais (coeficiente de fugacidade e fator de compressibilidade)
        self.eos_engine.calculate_state(state=state_liq)
        self.eos_engine.calculate_state(state=state_vap)

        # 3. Desempacota os parametros necessarios
        lnphi_x = state_liq.fugacity_dict['lnphi']
        Vm_x = state_liq.Vm
        lnphi_y = state_vap.fugacity_dict['lnphi']
        Vm_y = state_vap.Vm

        # 4. Aplica os limites para evitar overflow
        diff_lnphi = np.clip(lnphi_x - lnphi_y, -50.0, 50.0)

        # 5. Calcular a substituicao sucessiva
        K = np.exp(diff_lnphi) 
        fugacity_x = lnphi_x + np.log(abs(P)) + np.log(abs(x))
        fugacity_y = lnphi_y + np.log(abs(P)) + np.log(abs(y))
        aux_y = x * K

        Q2 = 1.0 - np.sum(aux_y)

        return Q2, aux_y, [Vm_x, Vm_y], [fugacity_x, fugacity_y]
    
    def _calculate_dQ_dP(self, T, P, x, y) -> float:
        """
        Metodo implementado para o calculo numerico da derivada Q2 em relacao ao P
        """
        delta = max(self.tol * abs(P), self.tol)
        
        Q_left, _, _, _ = self._successive_substitution(T=T, P=P+delta, x=x, y=y)
        Q_right, _, _, _ = self._successive_substitution(T=T, P=P-delta, x=x, y=y)

        return (Q_left - Q_right) / (2.0 * delta)
    
    
    def calculate_pressure_saturation(self, T: float, x: np.ndarray, P_guess: float, y_guess: np.ndarray) -> Dict[str, any]:
        """
        Aqui eh para implementar a logica melhor
        """
        P = P_guess
        y = np.copy(y_guess)
        x = x
        
        for _ in range(self.max_iter):
            # 1. Obtem os valores do problema
            Q2, aux_y, Vm_list, fugacity_list = self._successive_substitution(T=T, P=P, x=x, y=y)
            dQ2_dP = self._calculate_dQ_dP(T=T, P=P, x=x, y=y)

            # 2. Analisa se o problema eh instavel
            # if abs(dQ2_dP) < 1e-20:
            #     print("Derivada nula! Calculo instável")
            #     return 
            
            # 3. Calcula a nova pressao e composicao da fase incipiente
            P_new = P - Q2 / dQ2_dP

            if abs(P_new - P) > 0.5 * P:
                P_new = P + 0.5 * P * np.sign(P_new - P)
            if P_new <= 0:
                P_new = P / 2.0

            y_new = aux_y / np.sum(aux_y)

            # 4. Determina o erro para analise de convergencia
            P_error = abs(P_new - P)
            y_error = np.linalg.norm(y_new - y)

            if P_error < self.tol and y_error < self.tol:
                print("Metodo convergiu")
                # print(T)
                print(P_new, y_new)
                return P_new, y_new
        
            P, y = P_new, y_new
        return P, y


if __name__ == '__main__':
    # 1. Cria os componentes para teste
    metano = Component(name='CH4', Tc=190.6, Pc=45.99e5, omega=0.012)    
    isooctano = Component(name='C8H18', Tc=543.9, Pc=25.68e5, omega=0.304)    
    dioxide = Component(name='CO2', Tc=304.2, Pc=73.83e5, omega=0.224)
    sulfeto = Component(name='H2S', Tc=373.5, Pc=89.63e5, omega=0.094)

    kij = 0.102
    k_ij = np.array([[0, kij],[kij, 0]])
    
    # 2. Cria a mistura
    mixture = Mixture([dioxide, isooctano], k_ij=k_ij, l_ij=0.0)
    
    # 3. Instancia uma calculadora
    Pressure_calculator = PressureSatEngine(mixture=mixture, EoSEgine=ModeloPengRobinson)

    T = 303
    P = 97300
    x_space = np.linspace(0.00001, 0.995, 500)
    y = np.array([0.00002, 0.88])
    y_space = []
    P_space = []

    t0 = time()    
    for x in x_space:
        print(x)
        x_ = np.array([x, 1 - x])
        P, y = Pressure_calculator.calculate_pressure_saturation(T=T, x=x_, P_guess=P, y_guess=y)
        y_space.append(y[0])
        P_space.append(P / 10**5)
    tf = time()
    print('tempo final ', tf-t0)
    # Pontos experimentais, Donnelly e Katz, 1954, 271K [CH4, CO2]
    # P_exp = [50.53778268, 55.91560949, 59.98345284, 68.11913955, 68.39492554, 72.53171539, 76.39271925]
    # x_exp = [0.0675, 0.084, 0.103, 0.16, 0.157, 0.165, 0.191]
    # y_exp = [0.253, 0.3, 0.329, 0.367, 0.369, 0.387, 0.39]

    # Pontos experimentais, Donnelly e Katz, 1954, 259.82K [CH4, CO2]
    # P_exp = [31.92222835, 34.68008825, 36.88637617, 40.19580805, 50.53778268, 60.32818533, 68.11913955, 68.46387204, 70.80805295]
    # x_exp = [0.0315, 0.036, 0.051, 0.053, 0.1095, 0.164, 0.224, 0.233, 0.235]
    # y_exp = [0.1885, 0.235, 0.266, 0.306, 0.425, 0.485, 0.505, 0.509, 0.495]
   
    # Pontos experimentais, Donnelly e Katz, 1954, 241.50 [CH4, CO2]
    # P_exp = [23.92443464, 31.02592388, 40.74738003, 47.02151131, 52.60617761, 62.67236624, 66.7402096, 68.39492554, 75.70325427, 79.01268616]
    # x_exp = [0.0413, 0.086, 0.137, 0.166, 0.191, 0.286, 0.273, 0.322, 0.426, 0.501]
    # y_exp = [0.404, 0.521, 0.605, 0.629, 0.652, 0.676, 0.679, 0.686, 0.68, 0.672]

    # Pontos experimentais, Reamer, Sagen e Lacey, 1951, 277.6 [CH4, H2S]
    # P_exp = [13.78952, 17.2369, 20.68428, 24.13166, 27.57904, 31.02642, 34.4738, 41.36856, 48.26332, 55.15808, 62.05284, 
    #          68.9476, 75.84236, 82.73712001, 86.18450001, 89.63188001, 96.52664001, 103.4214, 110.31616, 117.21092, 120.6583, 124.10568, 
    #          131.00044, 134.3788724]
    # y_exp = [0.1371, 0.2783, 0.3896, 0.4604, 0.5126, 0.5551, 0.5879, 0.6394, 0.6755, 0.6989, 0.7141, 0.7242, 0.7299, 0.7321,
    #          0.7319, 0.7306, 0.7262, 0.7185, 0.7075, 0.6931, 0.6828, 0.6686, 0.613, 0.55 ]
    # x_exp = [0.0057, 0.0132, 0.0212, 0.0284, 0.0354, 0.0424, 0.0493, 0.0636, 0.0783, 0.093, 0.1083, 0.125, 0.1433, 0.1635, 
    #          0.1752, 0.1868, 0.2137, 0.245, 0.2798, 0.324, 0.3492, 0.3758, 0.4401, 0.55, ]
    

    # Pontos experimentais, Reamer, Sagen e Lacey, 1951, 310.9 [CH4, H2S]
    # P_exp = [27.57904, 31.02642, 34.4738, 37.92118, 41.36856, 48.26332, 55.15808, 62.05284, 68.9476, 75.84236,
    # 82.73712001, 86.18450001, 89.63188001, 96.52664001, 103.4214, 110.31616, 117.21092, 120.6583, 124.10568,
    # 127.55306, 131.00044, 131.4830732]
    # y_exp = [0.0117, 0.0963, 0.1642, 0.2203, 0.2688, 0.3416, 0.3976, 0.4396, 0.4707, 0.4923, 0.5079, 0.513, 0.5182,
    # 0.524, 0.5255, 0.5195, 0.5058, 0.4947, 0.4797, 0.458, 0.419, 0.388]
    # x_exp = [0.0007, 0.0067, 0.0128, 0.019, 0.0255, 0.0385, 0.0523, 0.067, 0.0828, 0.0996, 0.1182, 0.1282, 0.139, 0.162,
    # 0.1885, 0.2192, 0.2532, 0.2725, 0.294, 0.3185, 0.3578, 0.388]

    # Pontos experimentais, Reamer, Sagen e Lacey, 1951, 344.3 [CH4, H2S]
    # P_exp = [55.15808, 58.60546, 62.05284, 68.9476, 75.84236, 82.73712001, 86.18450001, 89.63188001, 96.52664001, 
    #          103.4214, 110.31616, 113.76354, 114.453016]
    # x_exp = [0.0031, 0.0098, 0.0167, 0.0309, 0.0459, 0.0622, 0.072, 0.0814, 0.1021, 0.1245, 0.1547, 0.183, 0.209]
    # y_exp = [0.0196, 0.0592, 0.0946, 0.1553, 0.2021, 0.2367, 0.2534, 0.2646, 0.2811, 0.2775, 0.258, 0.2295, 0.209]

    # plt.scatter(x_exp, P_exp, marker='x', color='brown')
    # plt.scatter(y_exp, P_exp, marker='x', color='goldenrod')
    # plt.scatter(-100, -100, marker='x', color='k', label='Reamer, Sagen e Lacey, 1951')
    x_exp = [0.5, 0.6, 0.7, 0.8, 0.9]
    P_exp = [37.49, 42.95, 48.71, 51.04, 57.32]
    plt.plot(x_space, P_space, color='k', linewidth=1.25)
    plt.scatter(x_exp, P_exp, color='red')
    plt.plot(y_space, P_space, color='k', linewidth=1.25, label=r'$k_{ij}=$'+f'{kij}')
    plt.ylabel(ylabel=r'$P\;/\;bar$')
    plt.xlabel(xlabel=r'$x_{1}\;/\;y_{1}$')
    plt.xlim(left=0.0, right=max(y_space)*1.10)
    plt.ylim(bottom=P_space[0]*0.8)
    plt.legend()
    plt.show()

    # wb = pyxl.Workbook()
    # st = wb.active
    # st['A1'] = 'T'
    # st['B1'] = 'P_cal'
    # st['C1'] = 'x_cal'
    # st['D1'] = 'y_cal'
    # for i in range(len(x_space)):
    #     st[f'A{i + 2}'] = T
    #     st[f'B{i + 2}'] = P_space[i]
    #     st[f'C{i + 2}'] = x_space[i]
    #     st[f'D{i + 2}'] = y_space[i]

    # comps = ','.join([c.name for c in mixture.components])
    # file_name = comps + '_T=' + str(T) + '_k_ij=' + str(kij)
    # # print(file_name)
    # wb.save((os.path.dirname(os.path.abspath(__file__)) + f'\\data\\{file_name}.xlsx'))

    # plt.show()
