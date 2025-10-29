from thermo_lib.eos.eos_abc import EquationOfState
from thermo_lib.state import FugacityResults, BaseState
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from thermo_lib.constants import RGAS_SI, KBOLTZMANN, NAVOGRADO
from thermo_lib.components import Component, Mixture
from abc import ABC
# DATACLASSES


@dataclass
class PCSaftParametersResults:
    # O que depende so de T, sigma, segment e epsilon
    d: np.ndarray
    m_mean: float
    sij :np.ndarray
    eij :np.ndarray
    
    # faz sentido deixar de passar?
    z: np.ndarray
    m: np.ndarray

    # O que depene de um eta
    rho: Optional[float] = None
    drho_deta: Optional[float] = None
    zeta: Optional[np.ndarray] = None

    # Derivada para o calculo da fugacidade
    zeta_xk: Optional[np.ndarray] = None
    dzeta_dT: Optional[np.ndarray] = None
    ddi_dT: Optional[np.ndarray] = None


@dataclass 
class PCSaftCoeffResult:
    am: float
    bm: float

    ai_xk: Optional[np.ndarray] = None
    bi_xk: Optional[np.ndarray] = None

    ai_xjxk: Optional[np.ndarray] = None
    bi_xjxk: Optional[np.ndarray] = None


@dataclass
class PCSaftHardChainDerivatives:
    dZhs_deta: float
    dgij_detaeta: float
    dZhc_deta: float

    #
    dgji_dx: Optional[np.ndarray] = None 
    dahs_dx: Optional[np.ndarray] = None
    dahc_dx: Optional[np.ndarray] = None 

    

@dataclass
class PCSaftHardChainResults:
    gij_hs: np.ndarray
    ar_hs: float
    ar_hc: float
    Z_hs: float
    rho_dgij_drho: np.ndarray
    Z_hc: float
    derivatives: Optional[PCSaftHardChainDerivatives] = None


@dataclass
class PCSaftDispersionDerivatives:
    # Derivadas para o calculo da pressoa via Newton-Raphson
    detaI1_deta: float
    detaI2_deta: float 
    dI2_deta: float
    dC2_deta: float
    dZdisp_deta: float

    #
    I1_xk: Optional[np.ndarray] = None
    I2_xk: Optional[np.ndarray] = None
    C1_xk: Optional[np.ndarray] = None
    m2es3_xk: Optional[np.ndarray] = None
    m2e2s3_xk: Optional[np.ndarray] = None
    dadisp_dx: Optional[np.ndarray] = None

@dataclass 
class PCSaftDispersionResults:
    ar_disp: float
    Z_disp: float
    # 
    C1: float
    C2: float
    I1: float
    I2: float
    detaI1_eta: float
    detaI2_eta: float
    m2es3: float
    m2e2s3: float
    derivatives: Optional[PCSaftDispersionDerivatives] = None


@dataclass
class PCSaftPressureResult:
    Z: float
    P: float
    dP_deta: Optional[float] = None
    dZ_eta: Optional[float] = None




@dataclass
class CoreModel(ABC):
    pass

@dataclass
class PCCoreModel(CoreModel):
    params: Optional[PCSaftParametersResults] = None
    coeff: Optional[PCSaftCoeffResult] = None
    hc_results: Optional[PCSaftHardChainResults] = None
    disp_results: Optional[PCSaftDispersionResults] = None

@dataclass
class State(BaseState):
    core_model: Optional[PCCoreModel] = None
    rho: Optional[float] = None
    eta: Optional[float] = None
    # Aqui eh para 'polimorfizar' as proximas etapas
    fugacity_result: Optional[FugacityResults] = None
    pressure_result: Optional[PCSaftPressureResult] = None
    # helmholtz_results: Optional[HelmholtzResults] = None
    # residual_props_results: Optional[ResidualPropertiesResults] = None


class PCSaftParametersWorker:
    def __init__(self):
        pass
    
    def calculate_base_results(self, T: float, state: State) -> PCSaftParametersResults:
        z = state.z
        sigma = np.array([c.sigma for c in mixture.components])
        epsilon = np.array([c.epsilon for c in mixture.components])
        segment = np.array([c.segment for c in mixture.components])
        k_ij = mixture.k_ij
        
        d = self._compute_d(s=sigma, e=epsilon, T=T)
        m_mean = self._compute_m_mean(z=z, m=segment)
        sij, eij = self._compute_combining_rule(s=sigma, e=epsilon, kij=k_ij)
        ddi_dT = self._compute_ddi_dT(s=sigma, e=epsilon, T=T)
        
        return PCSaftParametersResults(
            d=d,
            m_mean=m_mean,
            sij=sij,
            eij=eij,
            z=z,
            m=segment,
            ddi_dT=ddi_dT
        )

    def update_results(self, eta: float, params: PCSaftParametersResults) -> None:
        z, d, m, ddi_dT = params.z, params.d, params.m, params.ddi_dT

        rho = self._compute_rho(eta=eta, z=z, m=m, d=d)

        drho_deta = self._compute_drho_deta(z=z, d=d, m=m)
        zeta = self._compute_zeta(z=z, m=m, d=d, rho=rho)
        dzeta_dT = self._compute_dzeta_dT(z=z, m=m, ddi_dT=ddi_dT, di=d, rho=rho)
        
        params.rho = rho
        params.drho_deta = drho_deta
        params.zeta = zeta
        params.dzeta_dT = dzeta_dT

    def calculate_derivatives_for_fugacity(self, params: PCSaftParametersResults) -> None:
        rho, d, m = params.rho, params.d, params.m
        zeta_xk = self._compute_zeta_xk(rho=rho, d=d, m=m)
        params.zeta_xk = zeta_xk

    @staticmethod
    def _compute_d(s: np.ndarray, e: np.ndarray, T: float) -> np.ndarray:
        return s * (1.0 - 0.12 * np.exp(-3.0 * e / T))

    @staticmethod
    def _compute_rho(eta: float, z: np.ndarray, m: np.ndarray, d: np.ndarray) -> float:
        rho = (6 * eta / np.pi) / np.sum(z * m * d**3)
        return rho
    
    @staticmethod
    def _compute_m_mean(z: np.ndarray, m: np.ndarray) -> float:
        return np.sum(m * z)

    @staticmethod
    def _compute_combining_rule(s: np.ndarray, e: np.ndarray, kij: np.ndarray) -> tuple:
        sij = (s[:, np.newaxis] + s[np.newaxis, :]) / 2
        eij = (e[:, np.newaxis] * e[np.newaxis, :])**0.5 * (1 - kij)
        
        return sij, eij
    
    @staticmethod
    def _compute_zeta(z: np.ndarray, m: np.ndarray, d: np.ndarray, rho: float) -> np.ndarray:
        zeta_aux = np.pi * rho / 6
        exp = np.arange(4).reshape(4, 1)

        return zeta_aux * np.sum(z * m * d**exp, axis = 1)

    @staticmethod
    def _compute_drho_deta(z: np.ndarray, m: np.ndarray, d: np.ndarray) -> float:
        drho_deta = (6 / np.pi) * (np.sum(z * m * d**3))**-1
        return drho_deta

    # --- Derivadas para o calculo da fugacidade
    @staticmethod
    def _compute_zeta_xk(rho: float, m: np.ndarray, d: np.ndarray) -> np.ndarray:
        exp_aux = np.arange(4).reshape(4, 1)
        # EQ (A.34)
        zeta_xk = (np.pi * rho / 6) * (m * d**exp_aux)
        return zeta_xk

    @staticmethod
    def _compute_ddi_dT(s:np.ndarray, e:np.ndarray, T:float) -> np.ndarray:
        ddi_dT = s * (3 * e / T) * (-0.12 * np.exp(-3 * e / T))
        return ddi_dT
    
    @staticmethod
    def _compute_dzeta_dT(z:np.ndarray, m:np.ndarray, ddi_dT:np.ndarray, rho:float, di:np.ndarray) -> np.ndarray:
        n = np.arange(1, 4).reshape(3, 1)
        sum_aux = np.sum(z * m * ddi_dT * n * di**(n - 1), axis=1)
        dzeta_dT = (np.pi * rho / 6) * sum_aux
        return dzeta_dT
    
class PCSaftCoeffWorker:
    def __init__(self):
        self.a0 = np.array([0.910563145, 0.636128145, 2.686134789, -26.54736249, 97.75920878, -159.5915409, 91.29777408])
        self.a1 = np.array([-0.308401692, 0.186053116, -2.503004726, 21.41979363, -65.25588533, 83.31868048, -33.74692293])
        self.a2 = np.array([-0.090614835, 0.452784281, 0.596270073, -1.724182913, -4.130211253, 13.77663187, -8.672847037])
        self.b0 = np.array([0.724094694, 2.238279186, -4.002584949, -21.00357682, 26.85564136, 206.5513384, -355.6023561])
        self.b1 = np.array([-0.575549808, 0.699509552, 3.892567339, -17.21547165, 192.6722645, -161.8264617, -165.2076935])
        self.b2 = np.array([0.097688312, -0.255757498, -9.155856153, 20.64207597, -38.80443005, 93.62677408, -29.66690559])

    def calculate(self, params: PCSaftParametersResults) -> PCSaftCoeffResult:
        am, bm = self._compute_ab_m(m_mean=params.m_mean)

        return PCSaftCoeffResult(
            am=am,
            bm=bm,
        )

    def calculate_derivatives_for_fugacity(self, coeff: PCSaftCoeffResult, params: PCSaftParametersResults) -> None:
        m, m_mean = params.m, params.m_mean
        ai_xk, bi_xk = self._compute_ab_xk(m=m, m_mean=m_mean)
        ai_xjxk, bi_xjxk  = self._compute_ab_xjxk(m=m, m_mean=m_mean)
        coeff.ai_xk = ai_xk
        coeff.bi_xk = bi_xk
        coeff.ai_xjxk = ai_xjxk
        coeff.bi_xjxk = bi_xjxk

    def _compute_ab_m(self, m_mean: float):        
        m_mean_1 = (m_mean - 1.0) / m_mean
        m_mean_2 = m_mean_1 * (m_mean - 2.0) / m_mean

        am = self.a0 + self.a1 * m_mean_1 + self.a2 * m_mean_2
        bm = self.b0 + self.b1 * m_mean_1 + self.b2 * m_mean_2

        return am, bm

    def _compute_ab_xk(self, m: float, m_mean: float):
        # EQ (A.44)
        m_x = (m / m_mean**2)[:, None]
        ai_xk =  m_x * self.a1 + m_x*(3 - 4 / m_mean) * self.a2
        # EQ (A.45)
        bi_xk =  m_x * self.b1 + m_x*(3 - 4 / m_mean) * self.b2

        return ai_xk, bi_xk
    
    def _compute_ab_xjxk(self, m: float, m_mean: float):
        scalar1 = - 2 / m_mean**3
        scalar2 = (12 / m_mean**4 - 6 / m_mean**3)
        vec_ai = scalar1 * self.a1 + scalar2 * self.a2
        vec_bi = scalar1 * self.b1 + scalar2 * self.b2
        
        mjmk = np.outer(m, m)

        dai_xjxk = np.einsum('i,jk->ijk', vec_ai, mjmk)
        dbi_xjxk = np.einsum('i,jk->ijk', vec_bi, mjmk)

        return dai_xjxk, dbi_xjxk
    


class PCSaftHardChainWorker:
    def __init__(self):
        pass
    
    def calculate(self, params: PCSaftParametersResults, calc_deriv: bool=True, teste:bool=False) -> PCSaftHardChainResults:
        d, zeta, m_mean, z, m = params.d, params.zeta, params.m_mean, params.z, params.m
        gij_hs = self._compute_RDF_hardsphere(d=d, zeta=zeta)
        ar_hs = self._compute_ar_hardsphere(zeta=zeta, teste=teste)
        ar_hc = self._compute_ar_hardchain(m_mean=m_mean, ar_hs=ar_hs, z=z, m=m, gij_hs=gij_hs)
        Z_hs = self._compute_Z_hardsphere(zeta=zeta)
        rho_dgij_drho = self._compute_rho_dgij_drho(d=d, zeta=zeta)
        Z_hc = self._compute_Z_hardchain(m_mean=m_mean, Z_hs=Z_hs, z=z, m=m, gij_hs=gij_hs, rho_dgij_drho=rho_dgij_drho)

        results = PCSaftHardChainResults(
            gij_hs=gij_hs,
            ar_hs=ar_hs,
            ar_hc=ar_hc,
            Z_hs=Z_hs,
            Z_hc=Z_hc,
            rho_dgij_drho=rho_dgij_drho
        )

        if calc_deriv:
            derivatives = self._calculate_pressure_derivatives(params=params, hc_result=results, teste=teste)
            results.derivatives = derivatives
        
        return results

    def _calculate_pressure_derivatives(self, params: PCSaftParametersResults, hc_result: PCSaftHardChainResults, teste:bool=False) -> PCSaftHardChainDerivatives:

        gij_hs, rho_dgij_drho = hc_result.gij_hs, hc_result.rho_dgij_drho
        d, zeta, m_mean, z, m = params.d, params.zeta, params.m_mean, params.z, params.m
        dzeta_deta = zeta / zeta[3]

        dZhs_deta = self._compute_dZhs_deta(zeta=zeta)
        dgij_detaeta = self._compute_dgij_detaeta(d=d, zeta=zeta, teste=teste)
        dgij_deta = self._compute_dgij_deta(zeta=zeta, dzeta_deta=dzeta_deta, d=d)
        dZhc_deta = self._compute_dZhc_deta(z=z, m=m, d=d, gij=gij_hs, zeta=zeta, m_mean=m_mean, 
                                            dZhs_deta=dZhs_deta, dgij_deta=dgij_deta, rho_dgij_drho=rho_dgij_drho, dgij_detaeta=dgij_detaeta)
    
        return PCSaftHardChainDerivatives(
            dZhs_deta=dZhs_deta,
            dgij_detaeta=dgij_detaeta,
            dZhc_deta=dZhc_deta
        )
    
    def calculate_derivatives_for_fugacity(self, params: PCSaftParametersResults, hc_result: PCSaftHardChainResults, teste:bool=False):
        z, m, d, m_mean = params.z, params.m, params.d, params.m_mean
        zeta_xk, zeta = params.zeta_xk, params.zeta
        ar_hs, gij=  hc_result.ar_hs, hc_result.gij_hs

        dgji_dx = self._compute_dgij_dx(d=d, zeta_xk=zeta_xk, zeta=zeta, teste=teste)
        dahs_dx = self._compute_dahs_dx(ar_hs=ar_hs, zeta=zeta, zeta_xk=zeta_xk)
        dahc_dx = self._compute_dahc_dx(z=z, m=m, gij=gij, m_mean=m_mean, ar_hs=ar_hs, dahs_dx=dahs_dx, dgij_xk=dgji_dx, teste=teste)

        hc_result.derivatives.dgji_dx= dgji_dx
        hc_result.derivatives.dahs_dx= dahs_dx
        hc_result.derivatives.dahc_dx= dahc_dx
        
    @staticmethod
    def _compute_RDF_hardsphere(d: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
        aux_1 = 1 / (1 - zeta[3])
        aux_2 = 3 * zeta[2] / (1 - zeta[3])**2
        aux_3 = 2 * zeta[2]**2 / (1 - zeta[3])**3

        gij_hs = aux_1 + dij * aux_2 + dij**2 * aux_3
        return gij_hs
    
    @staticmethod
    def _compute_ar_hardsphere(zeta: np.ndarray, teste: bool=False) -> float:
        zeta_aux = 1 - zeta[3]
        aux_1 = 3 * zeta[1] * zeta[2] / zeta_aux
        aux_2 = zeta[2]**3 / (zeta[3] * zeta_aux**2)
        aux_3 = (zeta[2]**3 / zeta[3]**2 - zeta[0])*np.log(zeta_aux)
        ar_hs = (1 / zeta[0]) * (aux_1 + aux_2 + aux_3)

        if teste:
            pass
        return ar_hs
    
    @staticmethod
    def _compute_ar_hardchain(m_mean: float, ar_hs: float, z: np.ndarray, m: np.ndarray, gij_hs: np.ndarray) -> float:
        sum_aux = np.sum(z * (m - 1) * np.log(np.diagonal(gij_hs))) 
        ar_hc = m_mean * ar_hs - sum_aux
        return ar_hc
    
    @staticmethod
    def _compute_Z_hardsphere(zeta: np.ndarray) -> float:
        zeta_aux = 1 - zeta[3]
        aux_1 = zeta[3] / zeta_aux
        aux_2 = 3 * zeta[1] * zeta[2] / (zeta[0] * zeta_aux**2)
        aux_3 = (3 * zeta[2]**3 - zeta[3] * zeta[2]**3) / (zeta[0] * zeta_aux**3)
        Z_hs = aux_1 + aux_2 + aux_3
        return Z_hs
    
    @staticmethod
    def _compute_rho_dgij_drho(d: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
        # EQ (A.27)
        zeta_aux = 1 - zeta[3]
        aux_1 = zeta[3] / zeta_aux**2
        aux_2 = 3 * zeta[2] / zeta_aux**2 + 6 * zeta[2] * zeta[3] / zeta_aux**3
        aux_3 = 4 * zeta[2]**2 / zeta_aux**3 + 6 * zeta[2]**2 * zeta[3] / zeta_aux**4e0
        rho_dgij_drho = aux_1 + dij * aux_2 + dij**2 * aux_3

        return rho_dgij_drho
    
    @staticmethod
    def _compute_Z_hardchain(m_mean: float, Z_hs: float, z:np.ndarray, m: np.ndarray, gij_hs: np.ndarray, rho_dgij_drho: np.ndarray) -> float:
        sum_aux = np.sum(z * (m - 1) * (np.diagonal(gij_hs))**-1 * np.diagonal(rho_dgij_drho))
        Z_hc = m_mean * Z_hs - sum_aux
        return Z_hc


    # ---- Derivadas para o calculo da pressao por newton-raphson
    @staticmethod
    def _compute_dZhs_deta(zeta: np.ndarray):
        zeta_aux = (1 - zeta[3])
        aux_1 = 1 / zeta_aux**2
        aux_2 = 3 * zeta[1] * zeta[2] * (1 + zeta[3]) / (zeta[0] * zeta[3] * zeta_aux**3)
        aux_3 = 6 * zeta[2]**3 / (zeta[0] * zeta[3] * zeta_aux**4)
        dZhs_deta = aux_1 + aux_2 + aux_3

        return dZhs_deta
    
    @staticmethod
    def _compute_dgij_detaeta(d: np.ndarray, zeta: np.ndarray, teste:bool=False):
        dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
        zeta_aux = (1 - zeta[3])
        # Grad_grad_g_ij_hs * rho
        aux_1 = (1 + zeta[3]) / zeta_aux**3
        aux_2 = (3 * zeta[2] * (1 + zeta[3])) / (zeta[3] * zeta_aux**3)
        aux_2 += (6 * zeta[2] * (2 + zeta[3])) / (zeta_aux**4)
        aux_3 = (4 * zeta[2]**2 * (2 + zeta[3])) / (zeta[3] * zeta_aux**4)
        aux_3 += (6 * zeta[2]**2 * (3 + zeta[3])) / zeta_aux**5
        dgij_detaeta = aux_1 + dij * aux_2 + dij**2 * aux_3

        return dgij_detaeta

    @staticmethod
    def _compute_dgij_deta(zeta:np.ndarray, dzeta_deta:np.ndarray, d:np.ndarray, ):
        dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])

        zeta_aux = 1 - zeta[3]
        term1 = dzeta_deta[3] / zeta_aux**2
        term2 = ((3 * dzeta_deta[2]) * zeta_aux**2 - 3 * zeta[2] * (- 2 * zeta_aux * dzeta_deta[3])) / zeta_aux**4
        term3 = ((4 * zeta[2] * dzeta_deta[2]) * zeta_aux**3 - (2 * zeta[2]**2) * (-3 * zeta_aux**2 * dzeta_deta[3])) / zeta_aux**6
        dgij_deta = term1 + dij * term2 + dij**2 * term3
        return dgij_deta


    @staticmethod
    def _compute_dZhc_deta(z: np.ndarray, m:np.ndarray, d: np.ndarray, gij: np.ndarray, zeta: np.ndarray,
                           m_mean: float, dZhs_deta: float, dgij_deta: np.ndarray, rho_dgij_drho: np.ndarray, dgij_detaeta: np.ndarray):
        f = (gij)**-1
        dg_deta = - f**2 * dgij_deta
        # Derivada do Z_hc
        sum_aux = np.sum(z * (m - 1) * (rho_dgij_drho.diagonal() * dg_deta.diagonal() + (np.diagonal(f)) * dgij_detaeta.diagonal()))
        dZhc_deta = m_mean * dZhs_deta - sum_aux
        return dZhc_deta

    # --- Derivadas para o calculo da fugacidade
    @staticmethod
    def _compute_dgij_dx(d: float, zeta_xk: np.ndarray, zeta: np.ndarray, teste:bool=False):
        """
        PONTO DE APOIO, SE OUTRO LUGAR COMEÇAR A DAR ERRADO NA FUGACIDADE É AQUI!!!!!!!!!!!
        """
        dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
        zeta_aux = 1 - zeta[3]
        aux_1 = (zeta_xk[3, :] / zeta_aux**2)
        aux_2 = ((3 * zeta_xk[2, :] / zeta_aux**2 + 6 * zeta[2] * zeta_xk[3, :] / zeta_aux**3))
        aux_3 = ((4 * zeta[2] * zeta_xk[2, :] / zeta_aux**3 + 6 * zeta[2]**2 * zeta_xk[3, :] / zeta_aux**4))
        # EQ (A.37) ****
        dgij_dx = aux_1[None, None, :] + dij[:, :, None] * aux_2[None, None, :] + dij[:, :, None]**2 * aux_3[None, None, :]  

        if teste:
            pass
        return dgij_dx

    @staticmethod
    def _compute_dahs_dx(ar_hs: float, zeta: np.ndarray, zeta_xk: np.ndarray):
        zeta_aux = 1 - zeta[3]
        aux_1 = - zeta_xk[0, :] * ar_hs / zeta[0]
        aux_2 = (3 * zeta[2]**2 * zeta_xk[2, :] * zeta[3] - 2 * zeta[2]**3 * zeta_xk[3, :]) / zeta[3]**3 - zeta_xk[0, :]
        aux_2 = aux_2 * np.log(zeta_aux)
        aux_2 += 3 * (zeta_xk[1, :] * zeta[2] + zeta[1] * zeta_xk[2, :]) / zeta_aux
        aux_2 += 3 * zeta[1] * zeta[2] * zeta_xk[3, :] / zeta_aux**2
        aux_2 += 3 * zeta[2]**2 * zeta_xk[2, :] / (zeta[3] * zeta_aux**2)
        aux_2 += zeta[2]**3 * zeta_xk[3, :] * (3 * zeta[3] - 1) / (zeta[3]**2 * zeta_aux**3)
        aux_2 += (zeta[0] - zeta[2]**3 / zeta[3]**2) * (zeta_xk[3, :] / zeta_aux)
        # EQ (A.36)
        dahs_dx = aux_1 + (1 / zeta[0]) * aux_2

        return dahs_dx

    @staticmethod
    def _compute_dahc_dx(z: np.ndarray, m:float, gij: np.ndarray, m_mean: np.ndarray, ar_hs: float, dahs_dx: np.ndarray,
                         dgij_xk: np.ndarray, teste:bool=False):
        gii = np.diagonal(gij)
        dgii_xk = np.diagonal(dgij_xk, axis1=0, axis2=1).T
        sum_vector = z * (m - 1) / gii
        # aux_1 = np.sum(z * (m - 1) * (1/np.diagonal(gij)) * dgij_xk, axis=1) 
        aux_1 = np.einsum('i,ik->k', sum_vector, dgii_xk)
        # EQ (A.35)
        aux_2 = - (m - 1)*np.log(gii)
        dahc_dx = m * ar_hs + m_mean * dahs_dx - aux_1 + aux_2

        if teste: 
            pass
            # termo_1 = m * ar_hs
            # print('termo_1 = ', termo_1[0])
            # termo_2 = m_mean * dahs_dx
            # print('termo_2 = ', termo_2[0])
            # termo_3 = aux_2
            # print('termo_3 = ', termo_3[0])
            # termo_4 = - aux_1
            # print('termo_4 = ', termo_4[0])
        return dahc_dx

class PCSaftDispersionWorker:
    def __init__(self):
        pass
    
    def calculate(self, eta: float, T: float, coeff: PCSaftCoeffResult, params: PCSaftParametersResults, calc_deriv: bool=True, teste:bool=False) -> PCSaftDispersionResults:
        
        C1, C2 = self._compute_C1_C2(eta=eta, m_mean=params.m_mean)
        m2es3, m2e2s3 = self._compute_m2es3_m2e2s3(z=params.z, m=params.m, eij=params.eij, sij=params.sij, T=T)
        I1, I2 = self._compute_I1_I2(eta=eta, am=coeff.am, bm=coeff.bm)
        ar_disp = self._compute_ar_disp(rho=params.rho, I1=I1, I2=I2, m2es3=m2es3, m2e2s3=m2e2s3, C1=C1, m_mean=params.m_mean)
        
        detaI1_eta, detaI2_eta = self._compute_deta_I1I2_deta(eta=eta, am=coeff.am, bm=coeff.bm)
        Z_disp = self._compute_Z_disp(rho=params.rho, eta=eta, detaI1_eta=detaI1_eta, detaI2_eta=detaI2_eta, m2es3=m2es3, m2e2s3=m2e2s3, C1=C1, C2=C2, I2=I2, m_mean=params.m_mean)
        results = PCSaftDispersionResults(
            ar_disp=ar_disp,
            Z_disp=Z_disp,
            C1=C1,
            C2=C2,
            I1=I1,
            I2=I2,
            detaI1_eta=detaI1_eta,
            detaI2_eta=detaI2_eta,
            m2es3=m2es3,
            m2e2s3=m2e2s3
        )
        
        if calc_deriv:
            derivatives = self.calculate_derivatives(eta=eta, coeff=coeff, params=params, disp_results=results, teste=teste)

            results.derivatives = derivatives
            
        return results

    def calculate_derivatives(self, eta: float,  coeff: PCSaftCoeffResult,params: PCSaftParametersResults,
                               disp_results: PCSaftDispersionResults, teste:bool=False) -> PCSaftDispersionDerivatives:
       
        am, bm = coeff.am, coeff.bm
        m_mean, rho, drho_deta, = params.m_mean, params.rho, params.drho_deta
        m2es3, m2e2s3, C1, C2  = disp_results.m2es3, disp_results.m2e2s3, disp_results.C1, disp_results.C2
        I2, detaI1_eta, detaI2_deta = disp_results.I2, disp_results.detaI1_eta, disp_results.detaI2_eta

        detaI1_detaeta, detaI2_detaeta, dI2_deta = self._compute_dI1I2_detaeta(eta=eta, am=am, bm=bm)
        
        dC2_deta = self._compute_dC2_deta(eta=eta, m_mean=m_mean, C1=C1, C2=C2)

        dZdisp_deta = self._compute_dZdisp_deta(eta=eta, rho=rho, m_mean=m_mean, C2=C2, C1=C1, I2=I2, dC2_deta=dC2_deta,
                                                m2es3=m2es3, m2e2s3=m2e2s3, drho_deta=drho_deta, detaI1_eta=detaI1_eta, detaI2_deta=detaI2_deta,
                                                dI2_deta=dI2_deta, detaI1_detaeta=detaI1_detaeta, detaI2_detaeta=detaI2_detaeta, teste=teste)
        
        return PCSaftDispersionDerivatives(
            detaI1_deta=detaI1_detaeta,
            detaI2_deta=detaI2_detaeta,
            dI2_deta=dI2_deta,
            dC2_deta=dC2_deta,
            dZdisp_deta=dZdisp_deta
        )   

    def calculate_derviatives_for_fugacity(self, T:float, eta: float, params: PCSaftParametersResults, disp_result: PCSaftDispersionResults,
                                            coeff: PCSaftCoeffResult, teste: bool=False) -> None:
        am, bm, ai_xk, bi_xk = coeff.am, coeff.bm, coeff.ai_xk, coeff.bi_xk
        z, m, eij, sij, rho, m_mean = params.z, params.m, params.eij, params.sij, params.rho, params.m_mean
        zeta, zeta_xk = params.zeta, params.zeta_xk 
        C1, C2, I1, I2 = disp_result.C1, disp_result.C2, disp_result.I1, disp_result.I2
        m2es3, m2e2s3 = disp_result.m2es3, disp_result.m2e2s3

        I1_xk, I2_xk = self._compute_I12_xk(eta=eta, am=am, bm=bm, zeta_xk=zeta_xk, ai_xk=ai_xk, bi_xk=bi_xk, teste=teste)
        C1_xk = self._compute_C1_xk(m=m, eta=eta, C1=C1, C2=C2, zeta_xk=zeta_xk)
        m2es3_xk, m2e2s3_xk = self._compute_m2es3_m2e2s3_xk(T=T, z=z, m=m, eij=eij, sij=sij)
        dadisp_dx = self._compute_dadisp_dx(rho=rho, m=m, C1=C1, I1=I1, I2=I2, m_mean=m_mean, m2es3=m2es3, m2e2s3=m2e2s3, C1_xk=C1_xk, I1_xk=I1_xk,
                                            I2_xk=I2_xk, m2es3_xk=m2es3_xk, m2e2s3_xk=m2e2s3_xk)
        
        disp_result.derivatives.I1_xk = I1_xk
        disp_result.derivatives.I2_xk = I2_xk
        disp_result.derivatives.C1_xk = C1_xk
        disp_result.derivatives.m2es3_xk = m2es3_xk
        disp_result.derivatives.m2e2s3_xk = m2e2s3_xk
        disp_result.derivatives.dadisp_dx = dadisp_dx

    @staticmethod
    def _compute_C1_C2(eta: float, m_mean: float) -> tuple:
        aux_1 = 1 + m_mean * ((8 * eta - 2 * eta**2) / (1 - eta)**4)
        aux_2 = (1 - m_mean) * (20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4)
        aux_2 = aux_2 / ((1 - eta) * (2 - eta))**2
        C1 = 1 / (aux_1 + aux_2)

        aux_1 = m_mean * (-4 * eta**2 + 20 * eta + 8) / (1 - eta)**5
        aux_2 = (1 - m_mean) * (2 * eta**3 + 12 * eta**2 - 48 * eta + 40) / ((1 - eta) * (2 - eta))**3
        C2 = - C1**2 * (aux_1 + aux_2)
        
        return C1, C2
    
    @staticmethod
    def _compute_m2es3_m2e2s3(z: np.ndarray, m: np.ndarray, eij: np.ndarray, sij: np.ndarray, T: float) -> tuple:
        zij = z[:, np.newaxis] * z[np.newaxis, :]
        mij = m[:, np.newaxis] * m[np.newaxis, :]
        eij_T = eij / T
        # EQ (A.12)
        m2es3 = np.sum(zij * mij * eij_T * sij**3)
        # EQ (A.13)
        m2e2s3 = np.sum(zij * mij * eij_T**2 * sij**3)
        
        return m2es3, m2e2s3
        
    @staticmethod
    def _compute_I1_I2(eta: float, am: np.ndarray, bm: np.ndarray) -> tuple:
        exp = np.arange(7)
        # EQ (A.16)
        I1 = np.sum(am * eta**exp)
        # EQ (A.17)
        I2 = np.sum(bm * eta**exp)
        
        return I1, I2
    
    @staticmethod
    def _compute_ar_disp(rho: float, I1: float, I2: float, m2es3: float, m2e2s3: float, C1: float, m_mean: float) -> float:
        aux_1 = - 2 * np.pi * rho * I1 * m2es3
        aux_2 = - np.pi * rho * m_mean * C1 * I2 * m2e2s3
        ar_disp = aux_1 + aux_2
        return ar_disp

    @staticmethod
    def _compute_deta_I1I2_deta(eta: float, am: np.ndarray, bm: np.ndarray):
        exp = np.arange(7)
        # EQ (A.29)
        detaI1_eta = np.sum(am * (exp + 1) * eta**exp)
        # EQ (A.30)
        detaI2_eta = np.sum(bm * (exp + 1) * eta**exp)

        return detaI1_eta, detaI2_eta

    @staticmethod
    def _compute_Z_disp(rho: float, eta:float, detaI1_eta: float, detaI2_eta: float, m2es3: float,
                        m2e2s3: float, C1: float, C2: float, I2: float, m_mean: float) -> float:
        aux_1 = - 2 * np.pi * rho * detaI1_eta * m2es3
        aux_2 = - np.pi * rho * m_mean * m2e2s3
        aux_2 *= (C1 * detaI2_eta + C2 * eta * I2)
        Z_disp = aux_1 + aux_2
        return Z_disp

    @staticmethod
    def _compute_dI1I2_detaeta(eta: float, am: np.ndarray, bm: np.ndarray):
        j = np.arange(7)
        detaI1_detaeta = np.sum(am * (j + 1) * j * eta**(j-1))
        detaI2_detaeta = np.sum(bm * (j + 1) * j * eta**(j-1))
        dI2_deta = np.sum(bm * j * eta**(j-1))

        return detaI1_detaeta, detaI2_detaeta, dI2_deta
    
    @staticmethod
    def _compute_dC2_deta(eta: float, m_mean: float, C1: float, C2: float):
        u = - 4 * eta**2 + 20 * eta + 8
        du = -8 * eta + 20
        v = (1 - eta)**5
        dv = - 5 * (1 - eta)**4
        f1 = u / v
        df1_deta = (du * v - u * dv) / v**2

        u = 2 * eta**3 + 12 * eta**2 - 48* eta + 40
        du = 6 * eta**2 + 24 * eta - 48
        v = (2 - 3 * eta + eta**2)**3
        dv = 3 * (2 - 3 *eta + eta**2)**2 * (2 * eta - 3)
        f2 = u / v
        df2_deta = (du * v - u * dv) / v**2
        f = m_mean * f1 + (1 - m_mean) * f2
        df_deta = m_mean * df1_deta + (1 - m_mean) * df2_deta
        dC2_deta = - 2 * C1 * C2 * f - C1**2 * df_deta

        return dC2_deta
    
    @staticmethod
    def _compute_dZdisp_deta(eta: float, rho: float, m_mean: float, C2: float, C1: float,
                             I2: float, m2es3: float, m2e2s3: float, drho_deta: float,
                             detaI1_eta: float, detaI2_deta:float, dI2_deta: float, detaI1_detaeta: float, detaI2_detaeta: float,
                             dC2_deta: float, teste:bool=False):
        
        prime_Z_disp_1 = - 2 * np.pi * m2es3 * (drho_deta * detaI1_eta + rho * detaI1_detaeta)
        prime_Z_disp_2_1 = drho_deta * (C1 * detaI2_deta) + rho * (C2 * detaI2_deta + C1 * detaI2_detaeta)
        prime_Z_disp_2_2 = drho_deta * 2 * eta * (C2 * I2) + rho * eta * (dC2_deta * I2 + C2 * dI2_deta)
        prime_Z_disp_2 = - np.pi * m_mean * m2e2s3 * (prime_Z_disp_2_1 + prime_Z_disp_2_2)
        dZdisp_deta = prime_Z_disp_1 + prime_Z_disp_2

        if teste:
            pass
           
        return dZdisp_deta

    @staticmethod
    def _compute_I12_xk(eta: float, am: np.ndarray, bm: np.ndarray, zeta_xk: np.ndarray, ai_xk: np.ndarray, bi_xk: np.ndarray,
                        teste:bool=False) -> tuple:
        e_aux = (np.arange(7))[:, None]
        # EQ (A.42)
        I1_xk = np.sum(am[:, None] * e_aux * zeta_xk[3, :] * eta**(e_aux - 1) + ai_xk.T * eta**e_aux, axis=0)
        # EQ (A.43)
        I2_xk = np.sum(bm[:, None] * e_aux * zeta_xk[3, :] * eta**(e_aux - 1) + bi_xk.T * eta**e_aux, axis=0)

        if teste:
            pass
        return I1_xk, I2_xk
    
    @staticmethod
    def _compute_C1_xk(m: float, eta: float, C1: float, C2: float, zeta_xk: np.ndarray):
        aux_1 = m * (8 * eta - 2 * eta**2) / (1 - eta)**4
        aux_2 = - m * (20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4) / ((1 - eta) * (2 - eta))**2
        # EQ (A.41)
        C1_xk = C2 * zeta_xk[3, :] - C1**2 * (aux_1 + aux_2)

        return C1_xk
    
    @staticmethod
    def _compute_m2es3_m2e2s3_xk(T: float, z: np.ndarray, m: np.ndarray, eij: np.ndarray, sij: np.ndarray):
        # Auxiliares da EQ (A.39) & (A.40)
        sum_aux_1 = np.sum(z * m * (eij/ T) * sij**3, axis=1)
        sum_aux_2 = np.sum(z * m * (eij/ T)**2 * sij**3, axis=1)

        # EQ (A.39)
        m2es3_xk = 2 * m * sum_aux_1    
        # EQ (A.40)
        m2e2s3_xk = 2 * m * sum_aux_2

        return m2es3_xk, m2e2s3_xk

    @staticmethod
    def _compute_dadisp_dx(rho: float, m: np.ndarray, C1: float, I1: float, I2: float, m_mean: float, m2es3: float, m2e2s3: float,
                         C1_xk: np.ndarray, I1_xk: np.ndarray, I2_xk: np.ndarray, m2es3_xk: np.ndarray, m2e2s3_xk: np.ndarray):
        aux_1 = - 2 * np.pi * rho * (I1_xk * m2es3 + I1 * m2es3_xk)
        aux_2 = (m * C1 * I2 + m_mean * C1_xk * I2 + m_mean * C1 * I2_xk)
        aux_3 = - np.pi * rho * ( aux_2 * m2e2s3 + m_mean * C1 * I2 * m2e2s3_xk)
        dadisp_dx = aux_1 + aux_3

        return dadisp_dx





class PCSaftPressureWorker:
    def __init__(self):
        pass

    def calculate(self, T: float, params: PCSaftParametersResults, hc_results: PCSaftHardChainResults,
                   disp_results: PCSaftDispersionResults, calc_dervi: bool=True):

        rho, drho_deta = params.rho, params.drho_deta
        Z_hc, dZhc_deta = hc_results.Z_hc, hc_results.derivatives.dZhc_deta
        Z_disp, dZdisp_deta = disp_results.Z_disp, disp_results.derivatives.dZdisp_deta

        Z, P = self._compute_P_Z(Z_hc=Z_hc, Z_disp=Z_disp, T=T, rho=rho)

        result = PCSaftPressureResult(
            Z=Z,
            P=P
        )

        if calc_dervi:
            dP_deta, dZ_eta = self._compute_dP_deta(dZhc_deta=dZhc_deta, dZdisp_deta=dZdisp_deta, T=T, Z=Z, rho=rho, drho_deta=drho_deta)
            result.dP_deta = dP_deta
            result.dZ_eta = dZ_eta
        
        return result

    @staticmethod
    def _compute_P_Z(Z_hc: float, Z_disp: float, T: float, rho: float):
        Z = 1 + Z_hc + Z_disp
        P = Z * KBOLTZMANN * T * rho * 1.0e10**3
        return Z, P
        pass

    @staticmethod
    def _compute_dP_deta(dZhc_deta: float, dZdisp_deta: float, T: float, drho_deta: float, Z: float, rho: float) -> float:
        """
        Implementa a derivada da pressao em relacao aa variavel ehta;
        Essa implementacao eh usada para aplicar metodo de Newton-Raphson sem derivada numerica e
        para evitar a utilizacao de metodo de otimizacao sem derivada """

        dZ_eta = dZhc_deta + dZdisp_deta
        
        dP_deta = KBOLTZMANN * T * 1.0e10**3 * (drho_deta * Z + rho * dZ_eta)

        return dP_deta, dZ_eta




class PCSaftFugacityWorker:
    def __init__(self):
        pass

    def calculate(self, z: np.ndarray, Z: float, hc_result: PCSaftHardChainResults, disp_result: PCSaftDispersionResults) -> None:
        dach_dx, ar_hc = hc_result.derivatives.dahc_dx, hc_result.ar_hc
        dadisp_dx, ar_disp = disp_result.derivatives.dadisp_dx, disp_result.ar_disp

        dares_dx = self._compute_dares_dx(dahc_dx=dach_dx, dadisp_dx=dadisp_dx)
        ares = self._compute_ares(ar_hc=ar_hc, ar_disp=ar_disp)
        ln_phi, phi, mu = self._compute_fugacity(z=z, Z=Z, ares=ares, dares_dx=dares_dx)
        return FugacityResults(
            ln_phi=ln_phi,
            phi=phi,
            deletar_depois=dares_dx,
            mu=mu
        )

    
    @staticmethod
    def _compute_dares_dx(dahc_dx: np.ndarray, dadisp_dx: np.ndarray) -> np.ndarray:
        dares_dx = dahc_dx + dadisp_dx
        return dares_dx
    
    @staticmethod
    def _compute_ares(ar_hc: float, ar_disp: float) -> float:
        ares = ar_hc + ar_disp
        return ares

    @staticmethod
    def _compute_fugacity(z: np.ndarray, Z: float, ares: float, dares_dx: np.ndarray) -> tuple:
         # Auxiliar d EQ (A.33)
        sum_aux = - np.sum(z * dares_dx)
        # EQ (A.33)
        chemical_pow = ares + (Z - 1) + dares_dx + sum_aux
        # EQ (A.32)
        ln_phi = chemical_pow - np.log(Z)

        return ln_phi, np.exp(ln_phi), chemical_pow
    

@dataclass
class PCSaftWorkerSet():
    coeff: PCSaftCoeffWorker
    params: PCSaftParametersWorker
    hard_chain: PCSaftHardChainWorker
    dispersion: PCSaftDispersionWorker

class PCSaft(EquationOfState):
    def __init__(self, workers: PCSaftWorkerSet):
        print("PC-Saft module init")
        # AQUI VOU MUDAR
        self.parameter_worker = PCSaftParametersWorker()
        self.coeff_worker = PCSaftCoeffWorker()
        self.hc_worker = PCSaftHardChainWorker()
        self.disp_worker = PCSaftDispersionWorker()
        self.pressure_worker = PCSaftPressureWorker()
        self.fugacity_worker = PCSaftFugacityWorker()
        pass

    def calculate_from_TP(self, state: State, is_vapor: bool) -> None:
            if state.T is None or state.P is None:
                raise ValueError('Temperature and Pressure must be inputed to use this method')
            
            state.eta = 1.0e-10 if is_vapor else 0.5

            params = self.parameter_worker.calculate_base_results(T=state.T, state=state)
            coeff_results = self.coeff_worker.calculate(params=params)

            # Aplica um Newton-Raphson basicao para obter o eta
            for i in range(250):
                # Atualiza os parametros rho, drho_deta e zeta(rho)
                self.parameter_worker.update_results(eta=state.eta, params=params)

                hc_result = self.hc_worker.calculate(params=params)
                disp_result = self.disp_worker.calculate(eta=state.eta, T=state.T, coeff=coeff_results, params=params)
                pressure_result = self.pressure_worker.calculate(T=T, params=params, hc_results=hc_result, disp_results=disp_result)
                P = pressure_result.P
                dP_deta = pressure_result.dP_deta

                FO = 1 - P / state.P
                dFO = - dP_deta / state.P
                eta_old = state.eta
                state.eta = state.eta - FO / dFO

                if np.abs(eta_old - state.eta) < 1e-10:
                    break
            
            # Atualizacao final
            params = self.parameter_worker.calculate_base_results(T=state.T, state=state)
            coeff_results = self.coeff_worker.calculate(params=params)
            self.parameter_worker.update_results(eta=state.eta, params=params)
            hc_result = self.hc_worker.calculate(params=params)
            disp_result = self.disp_worker.calculate(eta=state.eta, T=state.T, coeff=coeff_results, params=params)
            pressure_result = self.pressure_worker.calculate(T=T, params=params, hc_results=hc_result, disp_results=disp_result)
            state.P = pressure_result.P
            state.Z = pressure_result.Z
            state.Vm = (params.rho / NAVOGRADO *(1e10)**3 *(1e-3) * 1000)**-1
            state.rho = params.rho
            state.pressure_result = pressure_result
            state.core_model = PCCoreModel(params=params,
                                           coeff=coeff_results,
                                           hc_results=hc_result,
                                           disp_results=disp_result)


    def calculate_fugacity(self, state: State, teste:bool=False):
        z, Z, T, eta = state.z, state.Z, state.T, state.eta
        params, coeff = state.core_model.params, state.core_model.coeff
        hc_result, disp_results = state.core_model.hc_results, state.core_model.disp_results

        # Da update nas coisas
        self.parameter_worker.calculate_derivatives_for_fugacity(params=params)
        self.coeff_worker.calculate_derivatives_for_fugacity(coeff=coeff, params=params)
        self.hc_worker.calculate_derivatives_for_fugacity(params=params, hc_result=hc_result, teste=teste)
        self.disp_worker.calculate_derviatives_for_fugacity(T=T, eta=eta, params=params, disp_result=disp_results, coeff=coeff, teste=teste)

        x = self.fugacity_worker.calculate(z=z, Z=Z, hc_result=hc_result, disp_result=disp_results)
        state.fugacity_result = x
        pass
    
    def calculate_mixture_parameters(self, state):
        pass

    def calculate_pressure(self, state):
        pass

    def calculate_from_TVm(self, state):
        pass

    def calculate_full_state(self, state):
        pass

    def update_parameters(self, state: State, teste:bool=False):
        params = self.parameter_worker.calculate_base_results(T=state.T, state=state)
        coeff_results = self.coeff_worker.calculate(params=params)
        # Aqui preciso recalcular o eta, porque ele muda enquanto o rho é constante (isso?)
        state.eta = (np.pi * state.rho / 6) * np.sum(state.z * params.m * params.d**3)
        self.parameter_worker.update_results(eta=state.eta, params=params)
        params.rho = state.rho
        hc_result = self.hc_worker.calculate(params=params)
        disp_result = self.disp_worker.calculate(eta=state.eta, T=state.T, coeff=coeff_results, params=params)
        pressure_result = self.pressure_worker.calculate(T=T, params=params, hc_results=hc_result, disp_results=disp_result)
        state.P = pressure_result.P
        state.Z = pressure_result.Z
        state.pressure_result = pressure_result
        state.core_model = PCCoreModel(params=params,
                                           coeff=coeff_results,
                                           hc_results=hc_result,
                                           disp_results=disp_result)

    def update_parameters_rho(self, state: State, teste:bool=False):
        params = self.parameter_worker.calculate_base_results(T=state.T, state=state)
        coeff_results = self.coeff_worker.calculate(params=params)
        # Aqui preciso recalcular o eta, porque ele muda enquanto o rho é constante (isso?)
        state.eta = (np.pi * state.rho / 6) * np.sum(state.z * params.m * params.d**3)
        self.parameter_worker.update_results(eta=state.eta, params=params)
        params.rho = state.rho
        hc_result = self.hc_worker.calculate(params=params, teste=teste)
        disp_result = self.disp_worker.calculate(eta=state.eta, T=state.T, coeff=coeff_results, params=params, teste=teste)
        pressure_result = self.pressure_worker.calculate(T=T, params=params, hc_results=hc_result, disp_results=disp_result)
        state.P = pressure_result.P
        state.Z = pressure_result.Z
        state.pressure_result = pressure_result
        state.core_model = PCCoreModel(params=params,
                                           coeff=coeff_results,
                                           hc_results=hc_result,
                                           disp_results=disp_result)

# coisasa para o hard chain
def _compute_dzhs_dxk(zeta_xk: np.ndarray, zeta: np.ndarray):
        zeta_aux = 1 - zeta[3]

        termo_1 = zeta_xk[3, :] / zeta_aux**2

        u = zeta[1] * zeta[2]
        du = zeta_xk[1,:] * zeta[2] + zeta[1] * zeta_xk[2,:]
        v = zeta[0] * zeta_aux**2
        dv = zeta_xk[0,:] * zeta_aux**2 - 2 * zeta[0] * zeta_aux * zeta_xk[3,:]
        termo_2 = 3 * (du * v - dv * u) / v**2

        u = 3 * zeta[2]**3 - zeta[3] * zeta[2]**3
        du = 9 * zeta[2]**2 * zeta_xk[2,:] - (zeta_xk[3,:] * zeta[2]**3 + 3 * zeta[3] * zeta[2]**2 * zeta_xk[2,:])
        v = zeta[0] * zeta_aux**3
        dv = zeta_xk[0,:] * zeta_aux**3 - 3 * zeta[0] * zeta_aux**2 * zeta_xk[3,:]
        termo_3 = (du * v - u * dv) / v**2

        dzhs_dxk = termo_1 + termo_2 + termo_3 

        return dzhs_dxk
        # print('dzhs_dxi: ', dzhs_dxi)


def _compute_dgij_dxk(d: np.ndarray, zeta_xk: np.ndarray, zeta: np.ndarray):
    """
    eu acho que ja tenho..?
    Returns: 
        Tensor (N, N, N): drhodhji_dxk, which [i, j, k] = ∂Y_ij/∂x_k.
    """
    Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
    zeta_aux = 1 - zeta[3]

    termo_1 = zeta_xk[3,:] / zeta_aux**2

    u = 3 * zeta[2]
    du = 3 * zeta_xk[2,:]
    v = zeta_aux**2
    dv = - 2 * zeta_aux * zeta_xk[3,:]
    termo_2 = (du * v - u * dv) / v**2

    u = 2 * zeta[2]**2
    du = 4 * zeta[2] * zeta_xk[2,:]
    v = zeta_aux**3
    dv = - 3 * zeta_aux**2 * zeta_xk[3,:]
    termo_3 = (du * v - u * dv) / v**2

    dgij_dxk = termo_1[None, None, :] + Dij[:, :, None] * termo_2[None, None, :] + Dij[:, :, None]**2 * termo_3[None, None, :]
    return dgij_dxk

def _compute_drhodhji_dxk(d: np.ndarray, zeta_xk: np.ndarray, zeta: np.ndarray):
    """
    
    Returns: 
        Tensor (N, N, N): drhodhji_dxk, which [i, j, k] = ∂Y_ij/∂x_k.
    """
    N = d.shape[0] # número de comps

    # zeta_xk shape (4, N)
    # zeta shape (4,)

    Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
    zeta_aux = 1 - zeta[3]

    u = zeta[3]
    du = zeta_xk[3, :]
    v = zeta_aux**2
    dv = - 2 * zeta_aux * zeta_xk[3, :]
    termo_1 = (du * v - u * dv) / v**2

    u = 3 * zeta[2]
    du = 3 *zeta_xk[2, :]
    termo_2 = (du * v - u * dv) / v**2
    
    u = 6 * zeta[2] * zeta[3]
    du = 6 * (zeta_xk[2, :] * zeta[3] + zeta[2] * zeta_xk[3, :])
    v = zeta_aux**3
    dv = - 3 *zeta_aux**2 * zeta_xk[3, :]
    termo_2 += (du * v - u * dv) / v**2

    u = 4 * zeta[2]**2
    du = 8 * zeta[2] * zeta_xk[2]
    termo_3 = (du * v - u * dv) / v**2

    u = 6 * zeta[2]**2 * zeta[3]
    du = 6 * (2 * zeta[2] * zeta[3] * zeta_xk[2] + zeta[2]**2 * zeta_xk[3, :])
    v = zeta_aux**4
    dv = - 4 * zeta_aux**3 * zeta_xk[3, :]
    termo_3 += (du * v - u * dv) / v**2

    drhodgij_dxk = termo_1[None, None, :] + Dij[:, :, None] * termo_2[None, None, :] + Dij[:, :, None]**2 * termo_3[None, None, :]
    return drhodgij_dxk


def _compute_dZhc_dxk(z: np.ndarray, m_mean: float, m:np.ndarray, Zhs: float, dZhs_dk: np.ndarray, gij: np.ndarray, rho_dgij_drho: np.ndarray,
                      dgij_dxk: np.ndarray, drhodhji_dxk: np.ndarray):

    termo_1_vec = m * Zhs + m_mean * dZhs_dk

    # Termos do somatorio
    gkk = np.diagonal(gij)
    rho_dgkk_drho = np.diagonal(rho_dgij_drho)
    sum_1_factor = (m - 1) * rho_dgkk_drho / gkk
    
    dgii_dxk = np.diagonal(dgij_dxk, axis1=0, axis2=1).T
    drhodhii_dxk = np.diagonal(drhodhji_dxk, axis1=0, axis2=1).T
    gii = gkk
    rho_dgii_drho = rho_dgkk_drho

    factor_aux = (m - 1) * z * rho_dgii_drho / (-gii**2)
    sum_2_factor = np.sum(factor_aux[:, None] * dgii_dxk, axis=0)

    factor_aux = (m - 1) * z / gii
    sum_3_factor = np.sum(factor_aux[:, None] * drhodhii_dxk, axis=0)
    
    termo_2_vec = sum_1_factor + sum_2_factor + sum_3_factor

    dZhc_dxk = termo_1_vec - termo_2_vec

    
    return dZhc_dxk

# coisas para o dispersion
def _compute_detaI1I2_dxk(eta: float, am: np.ndarray, bm: np.ndarray, am_xk: np.ndarray, bm_xk: np.ndarray, zeta3_xk: np.ndarray):
    j = np.arange(7)[:, None]
    detaI1_deta_xk = np.sum((j + 1) * (am_xk.T * eta**j + j * eta**(j - 1) * am[:, None] * zeta3_xk.T), axis=0)
    detaI2_deta_xk = np.sum((j + 1) * (bm_xk.T * eta**j + j * eta**(j - 1) * bm[:, None] * zeta3_xk.T), axis=0)
   
    return detaI1_deta_xk, detaI2_deta_xk

def _compute_C2_xk(m: np.ndarray, m_mean: float, eta: float, zeta3_xk: np.ndarray, C1: float, C1_xk: np.ndarray):

    u = m_mean
    u_xk = m
    
    o = - 4 * eta**2 + 20 * eta + 8
    o_xk = -8 * eta * zeta3_xk + 20 * zeta3_xk
    p = (1 - eta)**5
    p_xk = - 5 * (1 - eta)**4 * zeta3_xk
    v = o / p
    v_xk = (o_xk * p - o * p_xk) / p**2


    a = (1 - m_mean)
    a_xk = - m
    
    o = 2 * eta**3 + 12 *eta**2 - 48 * eta + 40
    o_xk = 6 * eta**2 * zeta3_xk + 24 * eta * zeta3_xk - 48 * zeta3_xk
    p = (eta**2 - 3 * eta + 2)**3
    p_xk = 3 * (2 * eta * zeta3_xk - 3 * zeta3_xk) * (eta**2 - 3 * eta + 2)**2

    b = o / p
    b_xk = (o_xk * p - o * p_xk) / p**2

    s = - C1**2
    s_xk = - 2 * C1 * C1_xk

    t = u * v + a * b
    t_xk = (u_xk * v + u * v_xk) + (a_xk * b + a * b_xk)

    C2_xk = s_xk * t + s * t_xk
    
    return C2_xk

def _compute_dZdisp_dxk(rho: float, eta: float, detaI1_eta: float, detaI2_eta: float, detaI1_xk: np.ndarray, detaI2_xk: np.ndarray,
                        m: np.ndarray, m_mean: float, C1: float, C2: float, C1_xk: np.ndarray, C2_xk: np.ndarray,
                        m2es3: float, m2es3_xk: np.ndarray, zeta3_xk: np.ndarray, I2: float, I2_xk: np.ndarray,
                        m2e2s3: float, m2e2s3_xk: np.ndarray):
    
    termo_1 = - 2 * np.pi * rho * (detaI1_xk * m2es3 + detaI1_eta * m2es3_xk)

    u = m_mean * C1
    du = (m * C1 + m_mean * C1_xk)
    v = detaI2_eta * m2e2s3
    dv = (detaI2_xk * m2e2s3 + detaI2_eta * m2e2s3_xk)
    termo_2 = - np.pi * rho * (du * v + u * dv)
    u = m_mean * C2 * eta
    du = m * (C2 * eta) + m_mean * (C2_xk * eta + C2 * zeta3_xk)
    v = I2 * m2e2s3
    dv = I2_xk * m2e2s3 + I2 * m2e2s3_xk
    termo_3 = - np.pi * rho * (du * v + u * dv)

    dZdips_xk = termo_1 + termo_2 + termo_3

    

    
    return dZdips_xk

def _compute_dash_xjxk(zeta: np.ndarray, zeta_xk: np.ndarray, ahs: float, dahs_xk: np.ndarray):
    zeta_aux = 1 - zeta[3]
    termo_1 = np.outer(zeta_xk[0,:], zeta_xk[0,:]) * ahs / zeta[0]**2 - np.outer(dahs_xk, zeta_xk[0,:]) / zeta[0]

    # T1
    u = zeta_xk[1,:] * zeta[2] + zeta[1] * zeta_xk[2,:]
    du = np.outer(zeta_xk[2,:], zeta_xk[1,:]) + np.outer(zeta_xk[1,:], zeta_xk[2,:])
    v = zeta_aux
    dv = - zeta_xk[3,:]
    T1 = 3 * u / v
    T1_xj = 3 * (du * v - np.outer(dv, u)) / v**2

    # T2
    u = 3 * zeta[1] * zeta[2] * zeta_xk[3,:]
    v = zeta_aux**2
    du = 3 * (np.outer(zeta_xk[1,:], zeta_xk[3,:]) * zeta[2] + np.outer(zeta_xk[2,:],zeta_xk[3,:]) * zeta[1])
    dv = - 2 * zeta_aux * zeta_xk[3,:]
    T2 = u / v
    T2_xj = (du * v - np.outer(dv, u)) / v**2

    # T3
    u = 3 * zeta[2]**2 * zeta_xk[2,:]
    du = 6 * zeta[2] * np.outer(zeta_xk[2,:], zeta_xk[2,:])
    v = zeta[3] * zeta_aux**2
    dv = zeta_xk[3,:] * zeta_aux**2 - 2 * zeta[3] * zeta_aux * zeta_xk[3,:]
    T3 = u / v
    T3_xj = (du * v - np.outer(dv, u)) / v**2

    # T4
    u = zeta[2]**3 * zeta_xk[3,:] * (3 * zeta[3] - 1)
    du = 3 * zeta[2]**2 * (3 * zeta[3] - 1) * np.outer(zeta_xk[2,:], zeta_xk[3,:]) + 3 * zeta[2]**3 * np.outer(zeta_xk[3,:], zeta_xk[3,:])
    v = zeta[3]**2 * zeta_aux**3
    dv = 2 * zeta[3] * zeta_aux**3 * zeta_xk[3,:] - 3 * zeta[3]**2 * zeta_aux**2 * zeta_xk[3,:]
    T4 = u / v
    T4_xj = (du * v - np.outer(dv, u)) / v**2

    # T5
    u = 3 * zeta[2]**2 * zeta[3] * zeta_xk[2,:] - 2 * zeta[2]**3 * zeta_xk[3,:]
    v = zeta[3]**3
    du = 6 * zeta[2] * zeta[3] * np.outer(zeta_xk[2,:], zeta_xk[2,:]) + 3 * zeta[2]**2 * np.outer(zeta_xk[3,:], zeta_xk[2,:]) - 6 * zeta[2]**2 * np.outer(zeta_xk[2,:], zeta_xk[3,:])
    dv = 3 * zeta[3]**2 * zeta_xk[3,:]

    T5_1 = (u / v) - zeta_xk[0,:]
    T5_1xj = (du * v - np.outer(dv, u)) / v**2
    T5_2 = np.log(zeta_aux)
    T5_2xj = - zeta_xk[3,:] / zeta_aux
    T5 = T5_1 * T5_2
    T5_xj = T5_1xj * T5_2 + np.outer(T5_2xj, T5_1)

    # T6
    T6_1 = zeta[0] - zeta[2]**3 / zeta[3]**2
    T6_1xj = zeta_xk[0,:] - (3 * zeta[2]**2 * zeta[3]**2 * zeta_xk[2,:] - 2 * zeta[2]**3 * zeta[3] * zeta_xk[3,:]) / zeta[3]**4
    T6_2 = zeta_xk[3,:] / zeta_aux
    T6_2xj = np.outer(zeta_xk[3,:], zeta_xk[3,:]) / zeta_aux**2
    T6 = T6_1 * T6_2
    T6_xj = np.outer(T6_1xj, T6_2) + T6_1 * T6_2xj

    T = T1 + T2 + T3 + T4 + T5 + T6
    T_xj = T1_xj + T2_xj + T3_xj + T4_xj + T5_xj + T6_xj
   
    termo_2 = - np.outer(zeta_xk[0,:], T) / zeta[0]**2 + T_xj / zeta[0]

    dahs_xjxk = termo_1 + termo_2
    
    return dahs_xjxk


def _compute_dgij_xjxk(d: np.ndarray, zeta: np.ndarray, zeta_xk: np.ndarray):
    Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
    zeta_aux = 1 - zeta[3]
    termo_1 = 2 * np.outer(zeta_xk[3,:], zeta_xk[3,:]) / zeta_aux**3


    termo_21 = (6 / zeta_aux**3) * np.outer(zeta_xk[3,:], zeta_xk[2,:])

    u = 6 * zeta[2] * zeta_xk[3,:]
    du = 6 * np.outer(zeta_xk[2,:], zeta_xk[3,:])
    v = zeta_aux**3
    dv = - 3 * zeta_aux**2 * zeta_xk[3,:]
    termo_22 = (du * v - np.outer(dv, u)) / v**2

    termo_2 = termo_21 + termo_22

    u = 4 * zeta[2] * zeta_xk[2,:]
    du = 4 * np.outer(zeta_xk[2,:], zeta_xk[2,:])
    termo_31 = (du * v - np.outer(dv, u)) / v**2
    
    u = 6 * zeta[2]**2 * zeta_xk[3,:]
    du = 12 * zeta[2] * np.outer(zeta_xk[2,:], zeta_xk[3,:])
    v = zeta_aux**4
    dv = - 4 * zeta_aux**3 *zeta_xk[3,:]
    termo_32 = (du * v - np.outer(dv, u)) / v**2

    termo_3 = termo_31 + termo_32
    dgij_xjxk = termo_1[None, None, :, :] + Dij[:, :, None, None] * termo_2[None, None, :, :] + Dij[:, :, None, None]**2 * termo_3[None, None, :, :]
    # AQUI TEM QUE VER MESMO SE GERA UM TENSOR (n,n,n,n)
    return dgij_xjxk

def _dahc_xjxk(z: np.ndarray, m: np.ndarray, m_mean: float, gij: np.ndarray, dgij_xk: np.ndarray, dgij_xjxk: np.ndarray,
               dahs_xk: np.ndarray, dahs_xjxk: np.ndarray):
    gii = np.diagonal(gij)
    gii_inv = 1 / gii
    gii_inv_sq = gii_inv**2

    dgii_xk = np.diagonal(dgij_xk, axis1=0, axis2=1).T
    # dgii_xjxk = np.einsum('iikj->jk', dgij_xjxk)

    # Termo 1: mₖ (∂ãʰˢ/∂xⱼ) -> Matriz [j, k] = mₖ * (∂ãʰˢ/∂xⱼ)
    term1 = np.outer(dahs_xk, m)

    # Termo 2: mₖ (∂ãʰˢ/∂xₖ) -> Matriz [k, j] = mⱼ * (∂ãʰˢ/∂xₖ)
    term2 = np.outer(m, dahs_xk)

    # Termo 3: m̄ (∂²ãʰˢ/∂xⱼ∂xₖ) -> Escalar * Matriz
    term3 = m_mean * dahs_xjxk

    # Termo 4a: - (mⱼ-1)(gⱼⱼ)⁻¹ (∂gⱼⱼ/∂xₖ) -> Matriz [j, k]
    term4a = - (m - 1.0)[:, None] * gii_inv[:, None] * dgii_xk

    # Termo 4b: + Σᵢ xᵢ(mᵢ-1)(gᵢᵢ)⁻² (∂gᵢᵢ/∂xⱼ) (∂gᵢᵢ/∂xₖ) -> Matriz [j, k]
    sum_4b = z * (m - 1.0) * gii_inv_sq # Vetor (N,)
    term4b = np.einsum('i,ij,ik->jk', sum_4b, dgii_xk, dgii_xk)

    # Termo 4c: - Σᵢ xᵢ(mᵢ-1)(gᵢᵢ)⁻¹ (∂²gᵢᵢ/∂xⱼ∂xₖ) -> Matriz [j, k]
    dgii_xjxk = np.diagonal(dgij_xjxk, axis1=0, axis2=1).transpose(2, 0, 1)
    sum_4c = z * (m - 1.0) * gii_inv # Vetor (N,)
    term4c = -np.einsum('i,ijk->jk', sum_4c, dgii_xjxk)

    # Termo 5: - (mₖ-1)(gₖₖ)⁻¹ (∂gₖₖ/∂xⱼ) -> Matriz [j, k]
    # dgii_dxk.T[k, j] = ∂gₖₖ/∂xⱼ
    term5 = - (m - 1.0)[None, :] * gii_inv[None, :] * dgii_xk.T

    # 3. Soma Final -> Matriz NxN
    dahc_xjxk = term1 + term2 + term3 + term4a + term4b + term4c + term5
    return dahc_xjxk

def _compute_I1I1_xjxk(eta: float, a: np.ndarray, b: np.ndarray, ai_xk:np.ndarray, ai_xjxk: np.ndarray, 
                       bi_xk: np.ndarray, bi_xjxk: np.ndarray, zeta3_xk: np.ndarray):
    
    i_vec = np.arange(7)
    i_minus1 = i_vec - 1
    i_minus2 = i_vec - 2
    eta_pow_i = np.power(eta, i_vec)
    eta_pow_i_minus1 = np.power(eta, i_minus1)
    eta_pow_i_minus2 = np.power(eta, i_minus2)
    zeta3_xjxk = np.outer(zeta3_xk, zeta3_xk)

    # Construção do I1_xjxk
    term1 = np.einsum('ijk,i->jk', ai_xjxk, eta_pow_i)
    aux1 = np.einsum('ji,k->ijk', ai_xk, zeta3_xk)
    aux2 = np.einsum('ki,j->ijk', ai_xk, zeta3_xk)
    term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
    term3 = np.einsum('i,jk->jk', (a * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

    I1_xjxk = term1 + term2 + term3

    # Construção do I12_xjxk
    term1 = np.einsum('ijk,i->jk', bi_xjxk, eta_pow_i)
    aux1 = np.einsum('ji,k->ijk', bi_xk, zeta3_xk)
    aux2 = np.einsum('ki,j->ijk', bi_xk, zeta3_xk)
    term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
    term3 = np.einsum('i,jk->jk', (b * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

    I2_xjxk = term1 + term2 + term3

    return I1_xjxk, I2_xjxk


def _compute_m2es3_m2e2s3_xjxk(m: np.ndarray, eij: np.ndarray, sij: np.ndarray, T: float):
    mjmk = np.outer(m, m)
    m2es3_xjxk = 2 * mjmk * (eij / T) * sij**3
    m2e2s3_xjxk = 2 * mjmk * (eij / T)**2 * sij**3

    return m2es3_xjxk, m2e2s3_xjxk

def _compute_C1_xjxk(eta: float, m:np.ndarray,  C1: float, C1_xk:np.ndarray, C2_xk: np.ndarray, zeta3_xk: np.ndarray):

    C2_xj_zeta3_xk = np.outer(C2_xk, zeta3_xk)
    term1 = C2_xj_zeta3_xk

    # as funcoes de eta dentro do parenteses
    u = 8 * eta - 2 * eta**2
    du = 8 * zeta3_xk - 4 * eta * zeta3_xk
    v = (1 - eta)**4
    dv = - 4 * (1 - eta)**3 * zeta3_xk
    s = 20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4
    ds = 20 * zeta3_xk - 54 * eta * zeta3_xk + 36 * eta**2 * zeta3_xk - 8 * eta**3 * zeta3_xk
    t = (2 - 3 * eta + eta**2)**2
    dt = 2 * t**0.5 * (2 * eta * zeta3_xk - 3 * zeta3_xk)
    aux1_xj = (du * v - u * dv) / v**2
    aux2_xj = (ds * t - s * dt) / t**2
    func_aux = u/v - s/t
    term_aux_xj= 2 * C1 * C1_xk * func_aux + C1**2 * (aux1_xj - aux2_xj)
    term2 = np.outer(term_aux_xj, m)
    C1_xjxk = term1 - term2

    return C1_xjxk

def _compute_dadisp_xjxk(rho: float, m:np.ndarray, m_mean:float, I1:float, I2:float, I1_xk:np.ndarray, I2_xk:np.ndarray, 
                         I1_xjxk:np.ndarray, I2_xjxk:np.ndarray,m2es3:float, m2e2s3:float, m2es3_xk:np.ndarray, m2e2s3_xk:np.ndarray,
                           m2es3_xjxk:np.ndarray, m2e2s3_xjxk:np.ndarray,
                         C1:float, C1_xk:np.ndarray, C1_xjxk:np.ndarray):
    
    m2es3_xj_I1_xk = np.outer(m2es3_xk, I1_xk)
    
    term1_xj = - 2 * np.pi * rho * (I1_xjxk * m2es3 + m2es3_xj_I1_xk + m2es3_xj_I1_xk.T + m2es3_xjxk * I1)

    aux = I2 * m2e2s3
    aux_xj = (I2_xk * m2e2s3 + I2 * m2e2s3_xk)
    aux1_xj = C1_xk * aux + C1 * aux_xj
    aux1_xj = np.outer(aux1_xj, m)

    u = m_mean * C1_xk
    du = np.outer(m, C1_xk) + m_mean * C1_xjxk
    v = aux
    dv = aux_xj
    aux2_xj = du * v + np.outer(dv, u)

    s = m_mean * C1
    ds = m * C1 + m_mean * C1_xk
    t = I2_xk * m2e2s3
    dt = I2_xjxk * m2e2s3 + np.outer(m2e2s3_xk, I2_xk)
    aux_3_xj = np.outer(ds, t) + s * dt

    m = I2 * m2e2s3_xk
    dm = np.outer(I2_xk, m2e2s3_xk) + I2 * m2e2s3_xjxk
    aux_4_xj = np.outer(ds, m) + s * dm

    term2_xj = - np.pi * rho * (aux1_xj + aux2_xj + aux_3_xj + aux_4_xj)
    
    dadisp_xjxk = term1_xj + term2_xj
    return dadisp_xjxk


def _compute_dlnphik_xj_unc(z:np.ndarray, Z:float, dZ_xk:np.ndarray, dares_xjxk:np.ndarray):
    term1 = ((1 - 1 / Z) * dZ_xk)[:, None]
    
    term3 = (- np.einsum('ji,i->j', dares_xjxk, z))[:, None]

    dlnphik_xj_unc = term1 + dares_xjxk + term3

    return dlnphik_xj_unc
 

def _compute_dlnphik_xj_cons(z:np.ndarray, dlnphik_xj_unc:np.ndarray):
    N = len(z)

    A_ij = np.eye(N) - z[:, None]
    M_ik = dlnphik_xj_unc
    dlnphik_xj_cons = np.einsum('ij,ik->jk', A_ij, M_ik)
    return dlnphik_xj_cons

def _compute_dlnphik_nj(z:np.ndarray, dlnphik_xj_cons:np.ndarray, n:float=100.0):
    sum_term = - np.einsum('i,ki->k', z, dlnphik_xj_cons)[:, None]
    n_dlnphik_nj = dlnphik_xj_cons + sum_term
    dlnphik_nj = n_dlnphik_nj / n

    # print(n_dlnphik_nj)
    # print(dlnphik_nj)

    ni = z * n




from time import time
from copy import deepcopy
if __name__ == '__main__':
    T_0 = time()
    T = 200 # K
    P = 30e5 # Pa
    # T = 350 # K
    # P = 9.4573e5 # Pa
    butano = Component(
        name='Butano',
        Tc=None,
        Pc=None,
        omega=None,
        sigma=3.7086,
        epsilon=222.88,
        segment=2.3316
    )

    nitrogenio = Component(
        name='N2',
        Tc=None,
        Pc=None,
        omega=None,
        sigma=3.3130,
        epsilon=90.96,
        segment=1.2053
    )

    metano = Component(
        name='CH4',
        Tc=None,
        Pc=None,
        omega=None,
        sigma=3.7039,
        epsilon=150.03,
        segment=1.000
    )

    mixture = Mixture(
        components=[nitrogenio, metano],
        k_ij=0.0,
        l_ij=0.0
        )
    
    state_trial = State(
        mixture=mixture,
        z=np.array([0.4, 0.6]),
        T=T,
        P=P
    )


    # seta a calc
    pc_saft_engine = PCSaft(workers=None)
    pc_saft_engine.calculate_from_TP(state=state_trial, is_vapor=True)
    pc_saft_engine.calculate_fugacity(state=state_trial)
    t_F = time()
    # print(t_F - T_0)
    

    # Testando numericamente
    # print(160*'*')
    # print('----Comparação numérica com analítica----')
    h_x = 0.00001
    t_num_0 = time()
    z_pos = np.array([state_trial.z[0] + h_x, state_trial.z[1]])
    state_xpos = deepcopy(state_trial)
    state_xpos.z = z_pos
    pc_saft_engine.update_parameters(state=state_xpos)
    pc_saft_engine.calculate_fugacity(state=state_xpos)


    z_neg = np.array([state_trial.z[0] - h_x, state_trial.z[1]])
    state_xneg = deepcopy(state_trial)
    state_xneg.z = z_neg
    pc_saft_engine.update_parameters(state=state_xneg)
    pc_saft_engine.calculate_fugacity(state=state_xneg)

    # dxhs_dxi
    
    # Calculos numericos (estimar tempo)
    
    dzhs_dxk_num = (state_xpos.core_model.hc_results.Z_hs - state_xneg.core_model.hc_results.Z_hs) / (2 * h_x)
    dgij_dxk_num = (state_xpos.core_model.hc_results.gij_hs - state_xneg.core_model.hc_results.gij_hs) / (2 * h_x)
    drhodhji_dxk_num = (state_xpos.core_model.hc_results.rho_dgij_drho - state_xneg.core_model.hc_results.rho_dgij_drho) / (2 * h_x)
    dZhc_dxk_num = (state_xpos.core_model.hc_results.Z_hc - state_xneg.core_model.hc_results.Z_hc) / (2 * h_x)
    detaI1_deta_xk_num = (state_xpos.core_model.disp_results.detaI1_eta - state_xneg.core_model.disp_results.detaI1_eta) / (2 * h_x)
    detaI2_deta_xk_num = (state_xpos.core_model.disp_results.detaI2_eta - state_xneg.core_model.disp_results.detaI2_eta) / (2 * h_x)
    C2_xk_num = (state_xpos.core_model.disp_results.C2 - state_xneg.core_model.disp_results.C2) / (2 * h_x)
    dZdisp_xk_num = (state_xpos.core_model.disp_results.Z_disp - state_xneg.core_model.disp_results.Z_disp) / (2 * h_x)
    dZ_xk_num = (state_xpos.Z - state_xneg.Z) / (2 * h_x)



   
    

    
    # Calculos analiticos (estimar tempo)
    zeta_xk = state_trial.core_model.params.zeta_xk
    zeta = state_trial.core_model.params.zeta
    z = state_trial.z
    d = state_trial.core_model.params.d
    m_mean = state_trial.core_model.params.m_mean
    m = state_trial.core_model.params.m
    Zhs = state_trial.core_model.hc_results.Z_hs
    eta=state_trial.eta
    am=state_trial.core_model.coeff.am
    bm=state_trial.core_model.coeff.bm
    am_xk=state_trial.core_model.coeff.ai_xk
    bm_xk=state_trial.core_model.coeff.bi_xk
    gij = state_trial.core_model.hc_results.gij_hs
    dahs_xk = state_trial.core_model.hc_results.derivatives.dahs_dx

    ai_xjxk = state_trial.core_model.coeff.ai_xjxk
    bi_xjxk = state_trial.core_model.coeff.bi_xjxk


    detaI1_eta=state_trial.core_model.disp_results.detaI1_eta
    detaI2_eta=state_trial.core_model.disp_results.detaI2_eta

    m2es3=state_trial.core_model.disp_results.m2es3
    m2es3_xk=state_trial.core_model.disp_results.derivatives.m2es3_xk

    C1=state_trial.core_model.disp_results.C1
    C1_xk=state_trial.core_model.disp_results.derivatives.C1_xk
    C2 = state_trial.core_model.disp_results.C2
    I1 = state_trial.core_model.disp_results.I1 
    I2 = state_trial.core_model.disp_results.I2 
    I2_xk = state_trial.core_model.disp_results.derivatives.I2_xk

    m2e2s3=state_trial.core_model.disp_results.m2e2s3
    m2e2s3_xk=state_trial.core_model.disp_results.derivatives.m2e2s3_xk

    dzhs_dxk_anal = _compute_dzhs_dxk(zeta_xk=zeta_xk,
                      zeta=zeta)
    dgij_dxk_anal = _compute_dgij_dxk(d=d,
                          zeta_xk=zeta_xk,
                      zeta=zeta)
    drhodhji_dxk_anal = _compute_drhodhji_dxk(d=d,
                          zeta_xk=zeta_xk,
                      zeta=zeta)
    dZhc_dxk_anal = _compute_dZhc_dxk(z=state_trial.z,
                                m_mean=m_mean,
                                m=m,
                                Zhs=Zhs,
                                dZhs_dk=dzhs_dxk_anal,
                                gij=state_trial.core_model.hc_results.gij_hs,
                                rho_dgij_drho=state_trial.core_model.hc_results.rho_dgij_drho,
                                dgij_dxk=dgij_dxk_anal,
                                drhodhji_dxk=drhodhji_dxk_anal)
    detaI1_deta_xk_anal, detaI2_deta_xk_anal = _compute_detaI1I2_dxk(eta=eta,
                                                                     am=am,
                                                                     bm=bm,
                                                                     am_xk=am_xk,
                                                                     bm_xk=bm_xk,
                                                                     zeta3_xk=zeta_xk[3,:])
    
    C2_xk_anal = _compute_C2_xk(m=m,
                                m_mean=m_mean,
                                eta=state_trial.eta,
                                zeta3_xk=zeta_xk[3,:],
                                C1=C1,
                                C1_xk=C1_xk,
                                )
    t_anal_f = time()
    dZdisp_xk_anal = _compute_dZdisp_dxk(rho=state_trial.rho,
                                         eta=state_trial.eta,
                                         detaI1_eta=detaI1_eta,
                                         detaI2_eta=detaI2_eta,
                                         detaI1_xk=detaI1_deta_xk_anal,
                                         detaI2_xk=detaI2_deta_xk_anal,
                                         m2es3=m2es3,
                                         m2es3_xk=m2es3_xk,
                                         m=m,
                                         m_mean=m_mean,
                                         C1=C1,
                                         C1_xk=C1_xk,
                                         C2=C2,
                                         C2_xk=C2_xk_anal,
                                         zeta3_xk=zeta_xk[3,:],
                                         I2=I2,
                                         I2_xk=I2_xk,
                                         m2e2s3=m2e2s3,
                                         m2e2s3_xk=m2e2s3_xk
                                         )
    dZ_xk_anal = dZdisp_xk_anal + dZhc_dxk_anal

    # print("tempo analitico: ", (t_anal_f - t_anal_0))
    # print("tempo numerico: ", (t_num_f - t_num_0))
    # print('--- dzhs_dxk ---')
    # print('dzhs_dxi_num: ', dzhs_dxk_num)
    # print('dzhs_dxi_anal: ', dzhs_dxk_anal[0])
    # print('--- dgij_dxk ---')
    # print('dgij_dxk_num: ', dgij_dxk_num)
    # print('dgij_dxk_anal: ', dgij_dxk_anal[:, :, 0])
    # print('--- drhodhji_dxk ---')
    # print('drhodhji_dxk_num: ', drhodhji_dxk_num)
    # print('drhodhji_dxk_anal: ', drhodhji_dxk_anal[:, :, 0])
    # print('--- dZhc_dxk ---')
    # print('dZhc_dxk_num: ', dZhc_dxk_num)
    # print('dZhc_dxk_anal: ', dZhc_dxk_anal[0])
    # print('--- detaI1_deta_xk ---')
    # print('detaI1_deta_xk_num: ', detaI1_deta_xk_num)
    # print('detaI1_deta_xk_anal: ', detaI1_deta_xk_anal[0])
    # print('--- detaI2_deta_xk ---')
    # print('detaI2_deta_xk_num: ', detaI2_deta_xk_num)
    # print('detaI2_deta_xk_anal: ', detaI2_deta_xk_anal[0])
    # print('--- C2_xk ---')
    # print('C2_xk_num: ', C2_xk_num)
    # print('C2_xk_anal: ', C2_xk_anal[0])
    # print('--- dZdisp_xk ---')
    # print('dZdisp_xk_num: ', dZdisp_xk_num)
    # print('dZdisp_xk_anal: ', dZdisp_xk_anal[0])
    # print('--- dZ_xk ---')
    # print('dZ_xk_num: ', dZ_xk_num)
    # print('dZ_xk_anal: ', dZ_xk_anal)

    # teste para a segundas derivadas de helmhotz
    a = time()
    h = 0.00001
    
    j = 1
    k = 0
    state_j_pos = deepcopy(state_trial)
    state_j_pos.z[j] += h
    
    pc_saft_engine.update_parameters(state=state_j_pos, teste=True)
    pc_saft_engine.calculate_fugacity(state=state_j_pos, teste=True)

    state_j_neg = deepcopy(state_trial)
    state_j_neg.z[j] -= h
    pc_saft_engine.update_parameters(state=state_j_neg, teste=True)
    pc_saft_engine.calculate_fugacity(state=state_j_neg, teste=True)


    dahs_xk_pos = state_j_pos.core_model.hc_results.derivatives.dahs_dx
    dahs_xk_neg = state_j_neg.core_model.hc_results.derivatives.dahs_dx
    dahs_xjxk_num = (dahs_xk_pos[0] - dahs_xk_neg[0]) / (2 * h)


    dgij_xk_pos = state_j_pos.core_model.hc_results.derivatives.dgji_dx
    dgij_xk_neg = state_j_neg.core_model.hc_results.derivatives.dgji_dx
    dgij_xjxk_num = (dgij_xk_pos[0,1] - dgij_xk_neg[0,1]) / (2 * h)

    dahc_xk_pos = state_j_pos.core_model.hc_results.derivatives.dahc_dx
    dahc_xk_neg = state_j_neg.core_model.hc_results.derivatives.dahc_dx
    dahc_xjxk_num = (dahc_xk_pos - dahc_xk_neg) / (2 * h)

    ai_xjxk_pos = state_j_pos.core_model.coeff.ai_xk
    ai_xjxk_neg = state_j_neg.core_model.coeff.ai_xk
    ai_xjxk_num = (ai_xjxk_pos - ai_xjxk_neg) / (2 * h)

    bi_xjxk_pos = state_j_pos.core_model.coeff.bi_xk
    bi_xjxk_neg = state_j_neg.core_model.coeff.bi_xk
    bi_xjxk_num = (bi_xjxk_pos - bi_xjxk_neg) / (2 * h)

    I1_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.I1_xk
    I1_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.I1_xk
    I1_xjxk_num = (I1_xjxk_pos - I1_xjxk_neg) / (2 * h)

    I2_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.I2_xk
    I2_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.I2_xk
    I2_xjxk_num = (I2_xjxk_pos - I2_xjxk_neg) / (2 * h)

    m2es3_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.m2es3_xk
    m2es3_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.m2es3_xk
    m2es3_xjxk_num = (m2es3_xjxk_pos - m2es3_xjxk_neg) / (2 * h)

    m2e2s3_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.m2e2s3_xk
    m2e2s3_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.m2e2s3_xk
    m2e2s3_xjxk_num = (m2e2s3_xjxk_pos - m2e2s3_xjxk_neg) / (2 * h)

    C1_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.C1_xk
    C1_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.C1_xk
    C1_xjxk_num = (C1_xjxk_pos - C1_xjxk_neg) / (2 * h)


    dadisp_xjxk_pos = state_j_pos.core_model.disp_results.derivatives.dadisp_dx
    dadisp_xjxk_neg = state_j_neg.core_model.disp_results.derivatives.dadisp_dx
    dadisp_xjxk_num = (dadisp_xjxk_pos - dadisp_xjxk_neg) / (2 * h)

    dares_xjxk_pos = state_j_pos.fugacity_result.deletar_depois
    dares_xjxk_neg = state_j_neg.fugacity_result.deletar_depois
    dares_xjxk_num = (dares_xjxk_pos - dares_xjxk_neg) / (2 * h)

    dlnphik_xj_pos = state_j_pos.fugacity_result.ln_phi
    dlnphik_xj_neg = state_j_neg.fugacity_result.ln_phi
    dlnphik_xj_num = (dlnphik_xj_pos - dlnphik_xj_neg) / (2 * h)


    dmuk_xj_pos = state_j_pos.fugacity_result.mu
    dmuk_xj_neg = state_j_neg.fugacity_result.mu
    dmuk_xj_num = (dmuk_xj_pos - dmuk_xj_neg) / (2 * h)

    dash_xjxk_anal = _compute_dash_xjxk(zeta=zeta,
                                        zeta_xk=zeta_xk,
                                        ahs=state_trial.core_model.hc_results.ar_hs,
                                        dahs_xk=state_trial.core_model.hc_results.derivatives.dahs_dx)


    dgij_xjxk_anal = _compute_dgij_xjxk(d=d,
                                   zeta=zeta,
                                   zeta_xk=zeta_xk
                                   )

    dahc_xjxk_anal = _dahc_xjxk(z=z, m=m, m_mean=m_mean, gij=gij, dgij_xk=dgij_dxk_anal, dgij_xjxk=dgij_xjxk_anal,
                          dahs_xk=dahs_xk, dahs_xjxk=dash_xjxk_anal)
    
    I1_xjxk, I2_xjxk = _compute_I1I1_xjxk(eta=eta, a=am, b=bm, ai_xk=am_xk, bi_xk=bm_xk, ai_xjxk=ai_xjxk, bi_xjxk=bi_xjxk, zeta3_xk=zeta_xk[3,:])
    
    I1_xk = state_trial.core_model.disp_results.derivatives.I1_xk
    I2_xk = state_trial.core_model.disp_results.derivatives.I2_xk

   
    
    eij = state_trial.core_model.params.eij
    sij = state_trial.core_model.params.sij
    T = state_trial.T
    m2es3_xjxk, m2e2s3_xjxk = _compute_m2es3_m2e2s3_xjxk(m=m, eij=eij, sij=sij, T=T)
    
    C1_xjxk = _compute_C1_xjxk(eta=eta, m=m, C1=C1, C1_xk=C1_xk, C2_xk=C2_xk_anal, zeta3_xk=zeta_xk[3,:])


    dadisp_xjxk = _compute_dadisp_xjxk(rho=state_trial.rho, m=m, m_mean=m_mean, I1=I1, I2=I2, I1_xk=I1_xk, I2_xk=I2_xk, 
                         I1_xjxk=I1_xjxk, I2_xjxk=I2_xjxk, m2es3=m2es3, m2e2s3=m2e2s3, m2es3_xk=m2es3_xk, m2e2s3_xk=m2e2s3_xk,
                           m2es3_xjxk=m2es3_xjxk, m2e2s3_xjxk=m2e2s3_xjxk,
                         C1=C1, C1_xk=C1_xk, C1_xjxk=C1_xjxk)

    dares_xjxk = dahc_xjxk_anal + dadisp_xjxk



    # Quero obter o del rho la

    h = 0.000001
    
    state_rho_pos = deepcopy(state_trial)
    state_rho_pos.rho += h  
    pc_saft_engine.update_parameters_rho(state=state_rho_pos, teste=True)
    pc_saft_engine.calculate_fugacity(state=state_rho_pos, teste=True)


    state_rho_neg = deepcopy(state_trial)
    state_rho_neg.rho -= h  
    pc_saft_engine.update_parameters_rho(state=state_rho_neg, teste=True)
    pc_saft_engine.calculate_fugacity(state=state_rho_neg, teste=True)

    dares_rhoxk_pos =  state_rho_pos.fugacity_result.deletar_depois
    dares_rhoxk_neg =  state_rho_neg.fugacity_result.deletar_depois
    dares_rhoxk_num = (dares_rhoxk_pos - dares_rhoxk_neg) / (2 * h)

    Z_rho_pos =  state_rho_pos.pressure_result.Z
    Z_rho_neg =  state_rho_neg.pressure_result.Z
    Z_rho_num = (Z_rho_pos - Z_rho_neg) / (2 * h)

    
    Zhc_rho_pos =  state_rho_pos.core_model.hc_results.Z_hc
    Zhc_rho_neg =  state_rho_neg.core_model.hc_results.Z_hc
    Zhc_rho_num = (Zhc_rho_pos - Zhc_rho_neg) / (2 * h)

    Zdisp_rho_pos =  state_rho_pos.core_model.disp_results.Z_disp
    Zdisp_rho_neg =  state_rho_neg.core_model.disp_results.Z_disp
    Zdisp_rho_num = (Zdisp_rho_pos - Zdisp_rho_neg) / (2 * h)


    muk_rho_pos =  state_rho_pos.fugacity_result.mu
    muk_rho_neg =  state_rho_neg.fugacity_result.mu
    muk_rho_num = (muk_rho_pos - muk_rho_neg) / (2 * h)

    # x = np.sum(state_trial.z * dZ_xk_anal)
    # print(state_trial.rho * (1 - 1 / state_trial.Z)*Z_rho_num + (state_trial.Z - 1) + dZ_xk_anal - x)
    # print(state_trial.rho * ln_phi_rho_num)

    z = state_trial.z
    Z = state_trial.Z

    dlnphik_xj_unc = _compute_dlnphik_xj_unc(z=z, Z=Z, dZ_xk=dZ_xk_anal, dares_xjxk=dares_xjxk)

    dlnphik_xj_cons = _compute_dlnphik_xj_cons(z=z, dlnphik_xj_unc=dlnphik_xj_unc)
    _compute_dlnphik_nj(z=z, dlnphik_xj_cons=dlnphik_xj_cons)

   
    def _compute_dF_dV_Tn(rho_mol:float, Z:float):
        dF_dV_Tn = - rho_mol * (Z - 1)
        return dF_dV_Tn
    
    rho_mol = (state_trial.rho * (1e10)**3 / NAVOGRADO)
    V = 100 / rho_mol
    Z = state_trial.Z


    def _compute_dF_dVV_Tn(rho_mol:float, Z:float, eta:float, dZ_eta:float, n:float=100.0):
        dF_dVV = rho_mol**2 * (eta * dZ_eta + (Z - 1)) / n
        return dF_dVV


    eta = state_trial.eta
    dZ_eta = state_trial.pressure_result.dZ_eta

    z = state_trial.z
    dZ_xk = dZ_xk_anal
    def _compute_dF_dVnk_Tn(rho_mol:float, z:np.ndarray, Z:float, eta:float, dZ_eta:float, dZ_xk:np.ndarray, n:float=100.0):
        sum_xidZxi = - np.sum(z * dZ_xk)

        dF_dVnk = - rho_mol * ((Z-1) + eta * dZ_eta + dZ_xk + sum_xidZxi) / n
        return dF_dVnk

    dF_dVnk = _compute_dF_dVnk_Tn(rho_mol=rho_mol, z=z, Z=Z, eta=eta, dZ_eta=dZ_eta, dZ_xk=dZ_xk) 


    def _compute_rho_dmuk_rho(z:np.ndarray, Z:float, eta:float, dZ_eta:float, dZ_xk:np.ndarray):
        sum_xi_dZ_xi = - np.sum(z * dZ_xk)
        rho_dmuk_rho = (Z - 1) + eta * dZ_eta + dZ_xk + sum_xi_dZ_xi
        return rho_dmuk_rho
   
    def _compute_dmuk_xj(z:np.ndarray, dZ_xk:np.ndarray, dares_xjxk:np.ndarray):
        sum_xi_dares_xjxi = - np.einsum('i,ji->j', z, dares_xjxk)
        dmuk_xj = dZ_xk[:,None] + dares_xjxk + sum_xi_dares_xjxi[:,None]
        return dmuk_xj
    
    def _compute_dF_njnk(z:np.ndarray, rho_dmuk_rho:np.ndarray, dmuk_xj:np.ndarray, n:float=100.00):
        sum_xi_dmuk_xi = - np.einsum('i,ik->k', z, dmuk_xj)

        dF_njnk = (1.0 / n) * (rho_dmuk_rho + dmuk_xj + sum_xi_dmuk_xi)
        return dF_njnk

    
    rho_dmuk_rho = _compute_rho_dmuk_rho(z=z, Z=Z, eta=eta, dZ_eta=dZ_eta, dZ_xk=dZ_xk)
    dF_dV = _compute_dF_dV_Tn(rho_mol=rho_mol, Z=Z)
    dF_dVV = _compute_dF_dVV_Tn(rho_mol=rho_mol, Z=Z, eta=eta, dZ_eta=dZ_eta)

    dmuk_xj = _compute_dmuk_xj(z=z, dZ_xk=dZ_xk, dares_xjxk=dares_xjxk)
    dF_njnk = _compute_dF_njnk(z=z, rho_dmuk_rho=rho_dmuk_rho, dmuk_xj=dmuk_xj)

    def _teste_final(dF_njnk:np.ndarray, dF_dV: float, dF_dVV: float, dF_dVnk:np.ndarray, V:float, T:float, n:float=100.00):
        dP_dV = - RGAS_SI * T *dF_dVV - n * RGAS_SI * T / V**2
        dP_dnk = - RGAS_SI * T  * dF_dVnk + RGAS_SI * T / V
        dP_dnk_dP_dnj = np.outer(dP_dnk, dP_dnk)

        n_dlnphik_dnj = n * dF_njnk + 1  + (n / (RGAS_SI * T)) * dP_dnk_dP_dnj / dP_dV
        
        return n_dlnphik_dnj, n_dlnphik_dnj / n  


    # 
    print(30*'#', ' Obtencao do dZ/dT ', 30*'#')
    s = state_trial.core_model.params.sij
    e = state_trial.core_model.params.eij
    T = state_trial.T
    ddi_dT = state_trial.core_model.params.ddi_dT
    dzeta_dT = state_trial.core_model.params.dzeta_dT

    def _compute_dZhs_dT(zeta:np.ndarray, dzeta_dT:np.ndarray):
        zeta0, zeta1, zeta2, zeta3 = zeta
        dzeta1_dT, dzeta2_dT, dzeta3_dT = dzeta_dT
        zeta_aux = 1 - zeta3
        u = zeta3
        du = dzeta3_dT
        v = zeta_aux
        dv = -dzeta3_dT
        f1 = u / v
        df1_dT = (du * v - u * dv) / v**2

        u = 3 * zeta1 * zeta2
        du = 3 * (dzeta1_dT * zeta2 + zeta1 * dzeta1_dT)
        v = zeta0 * zeta_aux**2
        dv = - 2 * zeta0 * zeta_aux * dzeta3_dT
        f2 = u / v
        df2_dT = (du * v - u * dv) / v**2

        u = 3 * zeta2**3 - zeta3 * zeta2**3
        du = 3 * zeta2 * dzeta2_dT - (dzeta3_dT * zeta2**3 + 3 * zeta3 * zeta2**2 * dzeta2_dT)
        v = zeta0 * zeta_aux**3
        dv = - 3 * zeta0 * zeta_aux**2 * dzeta3_dT
        f3 = u / v
        df3_dT = (du * v - u * dv) / v**2

        dZhs_dT = df1_dT + df2_dT + df3_dT
        pass