from ...state import State
from ...constants import RGAS_SI

import numpy as np
from typing import Dict

class CubicParametersWorker:
    def __init__(self, omega1: float, omega2: float, m):
        self.omega1 = omega1
        self.omega2 = omega2
        self.m = m

        self.mixture = None
        self.components = None
        self.z = None
        self.n = None
        self.Tc = None
        self.Pc = None
        self.omega = None
        self.T = None
        self.Tr = None
        self.params_dict = {}

    def _calculate_pure_params(self) -> None:
        m = self.m(self.omega)
        alpha = (1 + m * (1 - np.sqrt(self.Tr)))**2
        ac = self.omega1 * (RGAS_SI * self.Tc)**2 / self.Pc
        ai = ac * alpha
        bi = self.omega2 * (RGAS_SI * self.Tc) / self.Pc 
        
        self.params_dict['m']= m
        self.params_dict['alpha']= alpha
        self.params_dict['ac']= ac
        self.params_dict['ai']= ai
        self.params_dict['bi']= bi

    def _calculate_binary_mixture_params(self):
        ai = self.params_dict['ai']
        bi = self.params_dict['bi']
        aij_matrix = (np.sqrt(np.outer(ai, ai))) * (1 - self.mixture.k_ij)
        bij_matrix = 0.5 * (np.add.outer(bi,bi)) * (1 - self.mixture.l_ij)
        a_mix = self.z @ aij_matrix @ self.z
        b_mix = self.z @ bij_matrix @ self.z

        self.params_dict['aij_matrix'] =  aij_matrix
        self.params_dict['bij_matrix'] =  bij_matrix
        self.params_dict['a_mix'] =  a_mix
        self.params_dict['b_mix'] =  b_mix

    def _calculate_B_and_derivatives(self):
        ni = self.z * self.n
        bij_matrix = self.params_dict['bij_matrix']
        b_mix = self.params_dict['b_mix']
        B = self.n * b_mix
        Bi = np.array((2 * bij_matrix @ ni - B) / self.n)
        soma_BiBj = Bi.reshape(-1, 1) + Bi.reshape(1, -1)
        Bij = (2 * bij_matrix - soma_BiBj) / self.n

        self.params_dict['B'] =  B
        self.params_dict['Bi'] =  Bi
        self.params_dict['Bij'] =  Bij
        
    def _calculate_D_and_derivatives(self):
        ni = np.array(self.z * self.n)
        ai = self.params_dict['ai']
        aij_matrix = self.params_dict['aij_matrix']
        alpha = self.params_dict['alpha']
        m = self.params_dict['m']
        ac = self.params_dict['ac']
        a_mix = self.params_dict['a_mix']
        D = self.n**2 * a_mix
        Di = 2 * (ni @ aij_matrix)
        Dij = 2 * aij_matrix

        alphaij_T = ac * (- m * (alpha * self.Tr)**0.5) / self.T
        aii_ajj = np.outer(ai, ai)
        aii_dajj = np.outer(ai, alphaij_T)
        ajj_daii = np.outer(alphaij_T, ai)
        aij_T = (1 - self.mixture.k_ij) *(aii_dajj + ajj_daii) / (2 * aii_ajj**0.5)

        DiT = 2 * ni @ aij_T
        DT = (1/2) * ni @ DiT

        alphaii_TT = ac * m * (1 + m) * self.Tr**0.5 / (2 * self.T**2)
        # Eq. 105
        delh_delT = - (1 / (2 * (aii_ajj)**(3 / 2))) * (aii_dajj + ajj_daii)**2
        daii_dajj = np.outer(alphaij_T, alphaij_T)
        aii_ddajj = np.outer(ai, alphaii_TT)
        ajj_ddaii = np.outer(alphaii_TT, ai)
        delg_delT = (2 * daii_dajj + aii_ddajj + ajj_ddaii) * (1 / aii_ajj**0.5)
        daij_TT = ((1 - self.mixture.k_ij) / 2) * (delh_delT + delg_delT)
        DTT = ni @ daij_TT @ ni

        self.params_dict['D'] =  D
        self.params_dict['Di'] =  Di
        self.params_dict['DiT'] =  DiT
        self.params_dict['Dij'] =  Dij
        self.params_dict['DT'] =  DT
        self.params_dict['DTT'] =  DTT

    def _allocate_variables(self, state: State) -> None:
        self.mixture = state.mixture
        self.components = state.mixture.components
        self.z = state.z
        self.n = state.n
        self.Tc = np.array([c.Tc for c in self.components])
        self.Pc = np.array([c.Pc for c in self.components])
        self.omega = np.array([c.omega for c in self.components])
        self.T = state.T 
        self.Tr = self.T / self.Tc

    def _deallocate_variables(self) -> None:
        self.mixture = None
        self.components = None
        self.z = None
        self.n = None
        self.Tc = None
        self.Pc = None
        self.omega = None
        self.T = None
        self.Tr = None
        self.params_dict = {}

    def params_to_dict(self, state: State):
        # Alloca variaveis necessaria para os calculos
        self._deallocate_variables()
        self._allocate_variables(state=state)
        
        self._calculate_pure_params()
        self._calculate_binary_mixture_params()
        self._calculate_B_and_derivatives()
        self._calculate_D_and_derivatives()
        """Empacota todos os calculos do worker para enviar para o strategy"""
        return self.params_dict