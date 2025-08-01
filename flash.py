from CubicEoS_module import *
import numpy as np

class FlashCalculatorEngine:
    def __init__(self, EoS_Engine: ModeloPengRobinson):
        self.eos_engine = EoS_Engine()
     
    def _rachford_rice(self, beta: float, K: np.ndarray, z: np.ndarray) -> float:
        """
        
        """
        g_beta_0 = np.sum(z * K) - 1
        g_beta_1 = 1 - np.sum(z / K)

        if g_beta_0 < 0:
            print('liquido')
            return 0.0
        if g_beta_1 > 0:
            print('vapor')
            return 1.0
        
        max_iter = 200
        for _ in range(max_iter):
            g_beta = np.sum(z * (K - 1) / (1 - beta + beta * K))
            dg_dbeta = - np.sum(z * (K - 1)**2 / (1 - beta + beta * K)**2)

            if abs(g_beta) < 1e-5:
                return beta
            
            beta += - (g_beta / dg_dbeta)
    
        return beta
    
    def _initial_K_Wilson(self, T: float, P: float, Tc: np.ndarray, Pc: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Obtem estimativas iniciais de constante de equilibrio pela aproximacao de Wilson
        """
        Pr = P / Pc
        Tr = T / Tc
        lnK = np.log(1 / Pr) + 5.373 * (1 + omega) * (1 - 1 / Tr)
        return np.exp(lnK) 
        
    def _beta_estimated(self, K: np.ndarray, z:np.ndarray) -> float:
        beta_lower = 999
        beta_upper = -999
        for i, k in enumerate(K):
            if k > 1:
                beta_aux  = (k * z[i] - 1) / (k - 1)
                beta_lower = beta_aux if (beta_aux<beta_lower) else beta_lower
            else:
                beta_aux = (1 - z[i]) / (1 - k)
                beta_upper = beta_aux if (beta_aux>beta_upper) else beta_upper
        
        beta_lower = beta_lower if beta_lower>0 else 0
        beta_upper = beta_upper if beta_upper<1 else 1

        beta = 0.5 * (beta_lower + beta_upper)
        return beta
                
    def _updadte_composition(self, beta: float, K:np.ndarray, z:np.ndarray) -> tuple:
        x_new = z / (1 - beta + beta * K)
        y_new = x_new * K

        return x_new, y_new


    def flash_calculate(self, mixture: Mixture, z: np.ndarray, T: float, P: float, max_iter: int=150, tol: float=1e-4):
        # 1. Desempacota propriedades criticas para estimativa inicial
        Tc = np.array([comp.Tc for comp in mixture.components])
        Pc = np.array([comp.Pc for comp in mixture.components])
        omega = np.array([comp.omega for comp in mixture.components])
        
        # 2. Instancia dois objetos de State, um liquido e outro vapor - sem composicao definida
        state_liq = State(mixture=mixture, T=T, P=P, z=None, is_vapor=False)
        state_vap = State(mixture=mixture, T=T, P=P, z=None, is_vapor=True)
        
        # 3. Estimativas iniciais
        K = self._initial_K_Wilson(T=T, P=P, Tc=Tc, Pc=Pc, omega=omega)
        self._beta_estimated(K=K, z=z)
        beta = self._beta_estimated(K=K, z=z)    


        # 4. Calculo iterativo para analisar a mistura
        for _ in range(max_iter):
            beta = self._rachford_rice(beta=beta, K=K, z=z)
            x, y = self._updadte_composition(beta=beta, K=K, z=z)
            state_liq.z = x
            state_vap.z = y
            self.eos_engine.calculate_state(state=state_liq)
            self.eos_engine.calculate_state(state=state_vap)

            phi_liq = state_liq.fugacity_dict['phi']
            phi_vap = state_vap.fugacity_dict['phi']
            K = phi_liq / phi_vap

            equilibria = np.sum(((state_liq.z * phi_liq / state_vap.z * phi_vap) - 1)**2)

            if equilibria < tol:
                print('flash resolvido')
                return state_liq, state_vap, beta

        return state_liq, state_vap, beta 


if __name__ == '__main__':
    # 1. Instancia os componentes teste
    benzeno = Component(name='Benzeno', Tc=562.05, Pc=48.95e5, omega=0.21030)
    tolueno = Component(name='Tolueno', Tc=591.75, Pc=41.08e5, omega=0.26401)
    # 2. Instancia a mistura
    mixture = Mixture([benzeno, tolueno], k_ij=0.0, l_ij=0.0)

    # 3. Condicoes do flash
    T = 368.5 # K
    P = 101325 # Pa
    z = np.array([0.5, 0.5])

    flash_calculator = FlashCalculatorEngine(EoS_Engine=ModeloPengRobinson)
    
    state_liq, state_vap, beta = flash_calculator.flash_calculate(mixture=mixture, z=z, T=T, P=P)
    print(state_liq.z)
    print(state_vap.z)
    print(beta)