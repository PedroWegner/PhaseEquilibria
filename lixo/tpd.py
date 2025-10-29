from termolit import *
import numpy as np
from dataclasses import dataclass

@dataclass
class StabilityResult:
    is_stable: bool
    tm_vapor: float
    tm_liquid: float
    W_vapor: np.ndarray
    W_liquid: np.ndarray
    x: np.ndarray
    y: np.ndarray

class SolverTMhessiana():
    def __init__(self, Eos_Class, max_newton_iter: int=20, tol: float=1e-9):
        self.EoS_Classe = Eos_Class
        self.max_newton_iter = max_newton_iter
        self.tol = tol
        
    def check_stability(self, phase: Phase) -> StabilityResult:
        eos_calculator = self.EoS_Classe()
        worker = self._Worker(phase=phase, eos_calculator=eos_calculator)

        W_liq, tm_liq = worker.find_minimum(trial_phase_vapor=False)
        W_vap, tm_vap = worker.find_minimum(trial_phase_vapor=True)



        result = StabilityResult(
            is_stable=(tm_liq >= 0 and tm_liq >=0),
            tm_vapor=tm_vap,
            tm_liquid=tm_liq,
            W_vapor=W_vap,
            W_liquid=W_liq,
            x=W_liq / np.sum(W_liq),
            y=W_vap / np.sum(W_vap)
            )
        return result

    class _Worker():
        def __init__(self, phase: Phase, eos_calculator): # aqui devo fazer um construction falando o que é uma classe de eos
            self.eos_calculator = eos_calculator
            self.phase = phase
            self.phase_props = self.eos_calculator.calculate_properties(phase=self.phase)

            self.S_matrix = np.identity(len(phase.z))
            self.d_array = self._calculate_d_array()
            # Criacao da fase a ser testada
            self.trial_phase = phase.copy()
       
        def _calculate_d_array(self):
            return np.log(self.phase.z) + np.log(self.phase_props['phi_i'])


        def _calculate_initial_guess(self) -> np.ndarray:
            # Aqui tenho que analisar melhores estimativas iniciais
            phi_z = self.phase_props['phi_i']
            z = self.phase.z
            W0 = z / phi_z if self.trial_phase else z * phi_z
            # if self.trial_phase.vapor:
            #     Kw = 1 / phi_z
            #     Wo = z * Kw 
            # else:
            #     Wo = z * phi_z
            return W0

        def _calculate_tm_function(self, W: np.ndarray, phi_W: np.ndarray) -> float:
            W_safe = np.maximum(W, 1e-25) # evitar log de zero, rever se faz sentido...
            # Eq. 12, cap.9
            return 1.0 + np.sum(W *(np.log(W_safe) + np.log(phi_W) - self.d_array - 1))
        
        def _calculate_g_hessian(self, W: np.ndarray) -> tuple:
            self.trial_phase.z = W / np.sum(W)
            trial_props = self.eos_calculator.calculate_properties(phase=self.trial_phase)
            phi_W = trial_props['phi_i']

            W_safe = np.maximum(W, 1e-25)
            # Eq. 17, cap.9
            g = np.log(W_safe) + np.log(phi_W) - self.d_array


            dlnphi_dW = trial_props['dlnphi_dnj']
            # Eq. 18, cap.9
            H = np.diag(1 / W_safe) + dlnphi_dW
            return (g, H, phi_W)
        
        def _calculte_alpha_g_hessina(self, W: np.ndarray) -> tuple:
            self.trial_phase.z = W / np.sum(W)
            trial_props = self.eos_calculator.calculate_properties(phase=self.trial_phase)
            phi_W = trial_props['phi_i']
            dlnphi_dW = trial_props['dlnphi_dnj']

            # Gradiente g_i
            g_alpha = np.sqrt(W) * (np.log(W) + np.log(phi_W) - self.d_array)
            
            # Hesiana Hij
            sqrt_W_matrix = np.sqrt(np.outer(W, W))
            H_alpha = np.identity(len(W)) + sqrt_W_matrix * dlnphi_dW

            return (g_alpha, H_alpha)

        def _get_properties_trial_phase(self, W: np.ndarray) -> dict:
            self.trial_phase.z = W / np.sum(W)
            return self.eos_calculator.calculate_properties(self.trial_phase)
        
        def find_minimum(self, trial_phase_vapor: bool) -> tuple:
            self.trial_phase.vapor = trial_phase_vapor
            W_k = self._calculate_initial_guess()
            alpha_K = 2 * np.sqrt(W_k)

            for i in range(25):
                W_k = (alpha_K / 2)**2
                g_alpha, H_alpha = self._calculte_alpha_g_hessina(W=W_k)

                if np.linalg.norm(g_alpha) < 1e-9:
                    break
                
                eta = 0.0
                delta_alpha = None
                for _ in range(15):
                    H_mod = H_alpha + eta * self.S_matrix
                    try:
                        delta_alpha = np.linalg.solve(H_mod, -g_alpha)
                        break
                    except np.linalg.LinAlgError:
                        eta = eta*10 if eta > 0 else 1e-6
                if delta_alpha is None:
                    print("deu ruim")

                alpha_K += delta_alpha
            
            W_f = (alpha_K / 2)**2
            tm = self._calculate_tm_function(W=W_f, phi_W=self._get_properties_trial_phase(W=W_f)['phi_i'])
            return (W_f, tm)
            
class SolverTM():
    class _CalculationWorker():
        def __init__(self, phase: Phase, EoS_Class):
            self.phase = phase
            self.mixture_template = self.phase.mixture_template
            self.T = self.phase.T
            self.P = self.phase.P
            self.eos_calculator = EoS_Class()
            
            self.phase_props = self.eos_calculator.calculate_properties(phase=self.phase)
            self.di = None
            self._calculate_di()

            
        
        def _calculate_di(self):
            self.di = np.log(self.phase.z) + np.log(self.phase_props['phi_i'])
            
        def _initial_point(self, vapor: bool):
            phi_z = self.phase_props['phi_i']
            z = self.phase.z
            if vapor:
                Kw = 1 / phi_z
                Wi = z * Kw 
            else:
                Wi = z * phi_z
            return Wi
            
        def calculate_tm(self, Wk_phik: tuple) -> float:
            Wk, phi_w = Wk_phik
            di = self.di
            tm = 1 + np.sum(Wk * (np.log(Wk) + np.log(phi_w) - di - 1))
            return tm

        def find_stationary_point(self, vapor: bool) -> tuple[np.ndarray, np.ndarray]:
            W0 = np.array(self._initial_point(vapor=vapor))
            z0 = W0 / np.sum(W0)
            incipient_phase = Phase(mixture_template=self.mixture_template, T=self.T, P=self.P, frac_z=z0, vapor=vapor)
            di = self.di
            for _ in range(30):
                W_props = self.eos_calculator.calculate_properties(phase=incipient_phase)
                phi_W = W_props['phi_i']
                Wk = np.exp(di - np.log(phi_W))
                err = np.sum(np.abs(W0 - Wk))
                if err <= 1e-6:
                    break
                incipient_phase.z = Wk / np.sum(Wk)
                W0 = Wk
            
            return (Wk, phi_W)

    def check_stability(self, phase: Phase, Eos_class) -> dict:
        worker = self._CalculationWorker(phase=phase, EoS_Class=Eos_class)

        # Vapor
        Wk_vap, phik_vap = worker.find_stationary_point(vapor=True)
        tm_vap = worker.calculate_tm(Wk_phik=(Wk_vap, phik_vap))

        Wk_liq, phik_liq = worker.find_stationary_point(vapor=False)
        tm_liq = worker.calculate_tm(Wk_phik=(Wk_liq, phik_liq))
        result = StabilityResult(
            is_stable=(tm_liq >= 0 and tm_liq >=0),
            tm_vapor=tm_vap,
            tm_liquid=tm_liq,
            W_vapor=Wk_vap,
            W_liquid=Wk_liq,
            x=Wk_liq / np.sum(Wk_liq),
            y=Wk_vap / np.sum(Wk_vap)
            )

        return result


if __name__ == '__main__':
    metano = Component(name='Methane', Tc=190.6, Pc=45.99e5, omega=0.012)
    butano = Component(name='butane', Tc=425.1, Pc=37.96e5, omega=0.200)
    mistura = Mixture(components=[metano, butano])
    phase_test = Phase(mixture_template=mistura, T=311, P=30e5, frac_z=[0.2, 0.8], vapor=True)

    tpd_calculator = SolverTM()
    stability = tpd_calculator.check_stability(phase=phase_test, Eos_class=PengRobinson)
    tm_calc = SolverTMhessiana(Eos_Class=PengRobinson)
    x = tm_calc.check_stability(phase=phase_test)
    
    print(stability)
    print(x)