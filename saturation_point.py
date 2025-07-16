from new_code import *
import numpy as np
import matplotlib.pyplot as plt

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
            if abs(dQ2_dP) < 1e-20:
                print("Derivada nula! Calculo instável")
                return 
            
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
    metano = Component(name='Methane', Tc=190.6, Pc=45.99e5, omega=0.012)    
    dioxide = Component(name='Carbon Dioxide', Tc=304.2, Pc=73.83e5, omega=0.224)
    # sulfeto = Component(name='Sulfeto de hidrogenio', Tc=373.5, Pc=89.63e5, omega=0.094)
    k_ij = np.array([[0, 0.093],[0.093, 0]])
    
    # 2. Cria a mistura
    mixture = Mixture([metano, dioxide], k_ij=k_ij, l_ij=0.0)

    # 3. Instancia uma calculadora
    Pressure_calculator = PressureSatEngine(mixture=mixture, EoSEgine=ModeloPengRobinson)

    T = 260
    P = 26e5
    x_space = np.linspace(0.0, 0.45, 100)
    y = np.array([0.02, 0.98])
    y_space = []
    P_space = []
    for x in x_space:
        print(x)
        x_ = np.array([x, 1 - x])
        P, y = Pressure_calculator.calculate_pressure_saturation(T=T, x=x_, P_guess=P, y_guess=y)
        y_space.append(y[0])
        P_space.append(P / 10**5)

    plt.plot(x_space, P_space)
    plt.plot(y_space, P_space)
    plt.show()