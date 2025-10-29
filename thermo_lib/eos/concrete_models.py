from .cubic_eos import CubicEquationOfState
import numpy as np

class PengRobinson(CubicEquationOfState): 
    def __init__(self):
        super().__init__(
            delta1 = 1 + np.sqrt(2),
            delta2 = 1 - np.sqrt(2),
            omega1 = 0.45724,
            omega2 = 0.07780,
            m_func = lambda omega: 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        )

class SoaveRedlichKwong(CubicEquationOfState): 
    def __init__(self):
        super().__init__(
            delta1 = 1,
            delta2 = 0,
            omega1 = 0.42748,
            omega2 = 0.08664,
            m_func = lambda omega: 0.480 + 1.574 * omega - 0.176 * omega**2
        )

class VanDerWaals(CubicEquationOfState): 
    def __init__(self):
        super().__init__(
            delta1 = 0,
            delta2 = 0,
            omega1 = 27/64,
            omega2 = 1/8,
            m_func = lambda omega: 0
        )

class RedlichKwong(CubicEquationOfState): 
    """
    PARA IMPLEMENTAR REDLICH KWONG, PRECISO ALTERAR AS FUNCOES DE DERIVADAS DOS PARAMETROS BINARIOS
    """
    def __init__(self):
        super().__init__(
            delta1 = 1,
            delta2 = 0,
            omega1 = 0.42748,
            omega2 = 0.08664,
            m_func = lambda omega: 0 # PONTO DE APOIO, implementado errado!!!!!
        )