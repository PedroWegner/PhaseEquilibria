from thermo_lib.eos.eos_abc import EquationOfState
from thermo_lib.state import FugacityResults, BaseState, HelmholtzResult, PCSAFTHelmholtzResult
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from thermo_lib.constants import RGAS_SI, KBOLTZMANN, NAVOGRADO
from thermo_lib.components import Component, Mixture
from abc import ABC
from copy import deepcopy
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

    # Derivadas com relação a composição
    dgij_dxk: Optional[np.ndarray] = None 
    dahs_dxk: Optional[np.ndarray] = None
    dahc_dxk: Optional[np.ndarray] = None 

    dZhs_dxk: Optional[np.ndarray] = None
    dgij_dxk: Optional[np.ndarray] = None
    drhodgij_dxk: Optional[np.ndarray] = None
    dZhc_dxk: Optional[np.ndarray] = None
    
    dahs_dxjxk: Optional[np.ndarray] = None
    dgij_dxjxk: Optional[np.ndarray] = None
    dahc_dxjxk: Optional[np.ndarray] = None

    # Derivadas com relacao a temperatura
    dZhs_dT: Optional[float] = None
    dgij_dT: Optional[float] = None
    drho_dgij_drho_dT: Optional[float] = None
    dZhc_dT: Optional[float] = None

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

    # Derivadas com relacao a composição
    dI1_dxk: Optional[np.ndarray] = None
    dI2_dxk: Optional[np.ndarray] = None
    dC1_dxk: Optional[np.ndarray] = None
    dm2es3_dxk: Optional[np.ndarray] = None
    dm2e2s3_dxk: Optional[np.ndarray] = None
    dadisp_dxk: Optional[np.ndarray] = None

    detaI1_deta_dxk: Optional[np.ndarray] = None
    detaI2_deta_dxk: Optional[np.ndarray] = None
    dC2_dxk: Optional[np.ndarray] = None
    dZdisp_dxk: Optional[np.ndarray] = None
    dI1_dxjxk: Optional[np.ndarray] = None
    dI2_dxjxk: Optional[np.ndarray] = None
    dm2es3_dxjxk: Optional[np.ndarray] = None
    dm2e2s3_dxjxk: Optional[np.ndarray] = None
    dC1_dxjxk: Optional[np.ndarray] = None
    dadisp_dxjxk: Optional[np.ndarray] = None
    # dC2_dxjxk: Optional[np.ndarray] = None


    # Derivadas com relacao a temperatura
    ddetaI1_deta_dT: Optional[float] = None
    ddetaI2_deta_dT: Optional[float] = None
    dC2_dT: Optional[float] = None
    dZdisp_dT: Optional[float] = None

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

    dP_dV: Optional[float] = None
    dP_dni: Optional[float] = None


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
    helmholtz_result: Optional[PCSAFTHelmholtzResult] = None
    fugacity_result: Optional[FugacityResults] = None
    pressure_result: Optional[PCSaftPressureResult] = None
    # residual_props_results: Optional[ResidualPropertiesResults] = None





class PCSaftParametersWorker:
    def __init__(self):
        pass
    
    def calculate_base_results(self, T: float, state: State) -> PCSaftParametersResults:
        z = state.z
        mixture = state.mixture
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
        ddi_dT = s * (3 * e / T**2) * (-0.12 * np.exp(-3 * e / T))
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
        ar_hs = self._compute_ar_hardsphere(zeta=zeta)
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
    

    def calculate_dxk(self, params: PCSaftParametersResults, hc_result: PCSaftHardChainResults) -> None:
        z, d, zeta, dzeta_dxk = params.z, params.d, params.zeta, params.zeta_xk
        m, m_mean = params.m, params.m_mean
        ahs, gij, rho_dgij_drho = hc_result.ar_hs, hc_result.gij_hs, hc_result.rho_dgij_drho
        Zhs = hc_result.Z_hs
        Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])


        dgij_dxk = self._compute_dgij_dxk(Dij=Dij, dzeta_dxk=dzeta_dxk, zeta=zeta)
        dahs_dxk = self._compute_dahs_dxk(ar_hs=ahs, zeta=zeta, dzeta_dxk=dzeta_dxk)
        dahc_dxk = self._compute_dahc_dxk(z=z, m=m, m_mean=m_mean, ar_hs=ahs, gij=gij, dahs_dxk=dahs_dxk, dgij_dxk=dgij_dxk)

        dZhs_dxk = self._compute_dZhs_dxk(dzeta_dxk=dzeta_dxk, zeta=zeta)
        drhodgij_dxk = self._compute_drhodgji_dxk(Dij=Dij, dzeta_dxk=dzeta_dxk, zeta=zeta)
        dZhc_dxk = self._compute_dZhc_dxk(z=z, m_mean=m_mean, m=m, Zhs=Zhs, dZhs_dxk=dZhs_dxk, gij=gij, rho_dgij_drho=rho_dgij_drho,
                                          dgij_dxk=dgij_dxk, drhodgij_dxk=drhodgij_dxk)
        
        dahs_dxjxk = self._compute_dash_dxjxk(zeta=zeta, dzeta_dxk=dzeta_dxk, ahs=ahs, dahs_dxk=dahs_dxk)
        dgij_dxjxk = self._compute_dgij_dxjxk(Dij=Dij, zeta=zeta, dzeta_dxk=dzeta_dxk)
        dahc_dxjxk = self._dahc_dxjxk(z=z, m=m, m_mean=m_mean, gij=gij, dgij_dxk=dgij_dxk, dgij_dxjxk=dgij_dxjxk, dahs_dxk=dahs_dxk,
                                      dahs_dxjxk=dahs_dxjxk)
        

        hc_result.derivatives.dgij_dxk= dgij_dxk
        hc_result.derivatives.dahs_dxk= dahs_dxk
        hc_result.derivatives.dahc_dxk= dahc_dxk
        hc_result.derivatives.dZhs_dxk = dZhs_dxk
        hc_result.derivatives.dgij_dxk = dgij_dxk
        hc_result.derivatives.drhodgij_dxk = drhodgij_dxk
        hc_result.derivatives.dZhc_dxk = dZhc_dxk
        
        hc_result.derivatives.dahs_dxjxk = dahs_dxjxk
        hc_result.derivatives.dgij_dxjxk = dgij_dxjxk
        hc_result.derivatives.dahc_dxjxk = dahc_dxjxk
        
    # ---------------------------------------------------------------------------------------------
    # ---------------------------EQUACOES PARA CALCULAR A PRESSAO----------------------------------
    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_RDF_hardsphere(d: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
        aux_1 = 1 / (1 - zeta[3])
        aux_2 = 3 * zeta[2] / (1 - zeta[3])**2
        aux_3 = 2 * zeta[2]**2 / (1 - zeta[3])**3

        gij_hs = aux_1 + dij * aux_2 + dij**2 * aux_3
        return gij_hs
    
    @staticmethod
    def _compute_ar_hardsphere(zeta: np.ndarray) -> float:
        zeta_aux = 1 - zeta[3]
        aux_1 = 3 * zeta[1] * zeta[2] / zeta_aux
        aux_2 = zeta[2]**3 / (zeta[3] * zeta_aux**2)
        aux_3 = (zeta[2]**3 / zeta[3]**2 - zeta[0])*np.log(zeta_aux)
        ar_hs = (1 / zeta[0]) * (aux_1 + aux_2 + aux_3)

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


    # ---------------------------------------------------------------------------------------------
    # -----------------------------DERIVADAS COM RELACAO A ETA-------------------------------------
    # ---------------------------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------------------------
    # -------------------------DERIVADAS COM RELACAO A COMPOSICAO----------------------------------
    # --------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_dahs_dxk(ar_hs: float, zeta: np.ndarray, dzeta_dxk: np.ndarray):
        zeta_aux = 1 - zeta[3]
        aux_1 = - dzeta_dxk[0, :] * ar_hs / zeta[0]
        aux_2 = (3 * zeta[2]**2 * dzeta_dxk[2, :] * zeta[3] - 2 * zeta[2]**3 * dzeta_dxk[3, :]) / zeta[3]**3 - dzeta_dxk[0, :]
        aux_2 = aux_2 * np.log(zeta_aux)
        aux_2 += 3 * (dzeta_dxk[1, :] * zeta[2] + zeta[1] * dzeta_dxk[2, :]) / zeta_aux
        aux_2 += 3 * zeta[1] * zeta[2] * dzeta_dxk[3, :] / zeta_aux**2
        aux_2 += 3 * zeta[2]**2 * dzeta_dxk[2, :] / (zeta[3] * zeta_aux**2)
        aux_2 += zeta[2]**3 * dzeta_dxk[3, :] * (3 * zeta[3] - 1) / (zeta[3]**2 * zeta_aux**3)
        aux_2 += (zeta[0] - zeta[2]**3 / zeta[3]**2) * (dzeta_dxk[3, :] / zeta_aux)
        # EQ (A.36)
        dahs_dx = aux_1 + (1 / zeta[0]) * aux_2

        return dahs_dx

    @staticmethod
    def _compute_dahc_dxk(z: np.ndarray, m:float, gij: np.ndarray, m_mean: np.ndarray, ar_hs: float, dahs_dxk: np.ndarray,
                         dgij_dxk: np.ndarray):
        gii = np.diagonal(gij)
        dgii_xk = np.diagonal(dgij_dxk, axis1=0, axis2=1).T
        sum_vector = z * (m - 1) / gii
        # aux_1 = np.sum(z * (m - 1) * (1/np.diagonal(gij)) * dgij_xk, axis=1) 
        aux_1 = np.einsum('i,ik->k', sum_vector, dgii_xk)
        # EQ (A.35)
        aux_2 = - (m - 1)*np.log(gii)
        dahc_dx = m * ar_hs + m_mean * dahs_dxk - aux_1 + aux_2

        return dahc_dx

    @staticmethod
    def _compute_dZhs_dxk(dzeta_dxk:np.ndarray, zeta:np.ndarray) -> np.ndarray:
        zeta_aux = 1 - zeta[3]

        termo_1 = dzeta_dxk[3, :] / zeta_aux**2

        u = zeta[1] * zeta[2]
        du = dzeta_dxk[1,:] * zeta[2] + zeta[1] * dzeta_dxk[2,:]
        v = zeta[0] * zeta_aux**2
        dv = dzeta_dxk[0,:] * zeta_aux**2 - 2 * zeta[0] * zeta_aux * dzeta_dxk[3,:]
        termo_2 = 3 * (du * v - dv * u) / v**2

        u = 3 * zeta[2]**3 - zeta[3] * zeta[2]**3
        du = 9 * zeta[2]**2 * dzeta_dxk[2,:] - (dzeta_dxk[3,:] * zeta[2]**3 + 3 * zeta[3] * zeta[2]**2 * dzeta_dxk[2,:])
        v = zeta[0] * zeta_aux**3
        dv = dzeta_dxk[0,:] * zeta_aux**3 - 3 * zeta[0] * zeta_aux**2 * dzeta_dxk[3,:]
        termo_3 = (du * v - u * dv) / v**2

        dZhs_dxk = termo_1 + termo_2 + termo_3 

        return dZhs_dxk
    
    @staticmethod
    def _compute_dgij_dxk(Dij:np.ndarray, dzeta_dxk:np.ndarray, zeta:np.ndarray):
        """
        eu acho que ja tenho..?
        Returns: 
            Tensor (N, N, N): drhodhji_dxk, which [i, j, k] = ∂Y_ij/∂x_k.
        """
        # Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
        zeta_aux = 1 - zeta[3]

        termo_1 = dzeta_dxk[3,:] / zeta_aux**2

        u = 3 * zeta[2]
        du = 3 * dzeta_dxk[2,:]
        v = zeta_aux**2
        dv = - 2 * zeta_aux * dzeta_dxk[3,:]
        termo_2 = (du * v - u * dv) / v**2

        u = 2 * zeta[2]**2
        du = 4 * zeta[2] * dzeta_dxk[2,:]
        v = zeta_aux**3
        dv = - 3 * zeta_aux**2 * dzeta_dxk[3,:]
        termo_3 = (du * v - u * dv) / v**2

        dgij_dxk = termo_1[None, None, :] + Dij[:, :, None] * termo_2[None, None, :] + Dij[:, :, None]**2 * termo_3[None, None, :]
        return dgij_dxk
    
    @staticmethod
    def _compute_drhodgji_dxk(Dij:np.ndarray, dzeta_dxk:np.ndarray, zeta:np.ndarray):
        """
        
        Returns: 
            Tensor (N, N, N): drhodhji_dxk, which [i, j, k] = ∂Y_ij/∂x_k.
        """
        _, _, zeta2, zeta3 = zeta
        _, _, zeta2_xk, zeta3_xk = dzeta_dxk

        # Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
        zeta_aux = 1 - zeta3

        termo_1 = (zeta3_xk * zeta_aux**2 - (zeta3) * (- 2 * zeta_aux * zeta3_xk)) / zeta_aux**4

        termo_2 = ((3 * zeta2_xk) * zeta_aux**2 - (3 * zeta2) * (- 2 * zeta_aux * zeta3_xk)) / zeta_aux**4
        
        termo_2 += ((6 * (zeta2_xk * zeta3 + zeta2 * zeta3_xk)) * zeta_aux**3 - (6 * zeta2 * zeta3) * (- 3 *zeta_aux**2 * zeta3_xk)) / zeta_aux**6

        termo_3 = ((8 * zeta2 * zeta2_xk) * zeta_aux**3 - (4 * zeta2**2) * ((- 3 *zeta_aux**2 * zeta3_xk))) / zeta_aux**6

        u = 6 * zeta2**2 * zeta3
        du = 6 * (2 * zeta2 * zeta3 * zeta2_xk + zeta2**2 * zeta3_xk)
        v = zeta_aux**4
        dv = - 4 * zeta_aux**3 * zeta3_xk
        termo_3 += (du * v - u * dv) / v**2

        drhodgij_dxk = termo_1[None, None, :] + Dij[:, :, None] * termo_2[None, None, :] + Dij[:, :, None]**2 * termo_3[None, None, :]
        return drhodgij_dxk
    
    @staticmethod
    def _compute_dZhc_dxk(z: np.ndarray, m_mean: float, m:np.ndarray, Zhs: float, dZhs_dxk: np.ndarray, gij: np.ndarray, rho_dgij_drho: np.ndarray,
                      dgij_dxk: np.ndarray, drhodgij_dxk: np.ndarray):

        termo_1_vec = m * Zhs + m_mean * dZhs_dxk

        # Termos do somatorio
        gkk = np.diagonal(gij)
        rho_dgkk_drho = np.diagonal(rho_dgij_drho)
        sum_1_factor = (m - 1) * rho_dgkk_drho / gkk
        
        dgii_dxk = np.diagonal(dgij_dxk, axis1=0, axis2=1).T
        drhodhii_dxk = np.diagonal(drhodgij_dxk, axis1=0, axis2=1).T
        gii = gkk
        rho_dgii_drho = rho_dgkk_drho

        factor_aux = (m - 1) * z * rho_dgii_drho / (-gii**2)
        sum_2_factor = np.sum(factor_aux[:, None] * dgii_dxk, axis=0)

        factor_aux = (m - 1) * z / gii
        sum_3_factor = np.sum(factor_aux[:, None] * drhodhii_dxk, axis=0)
        
        termo_2_vec = sum_1_factor + sum_2_factor + sum_3_factor

        dZhc_dxk = termo_1_vec - termo_2_vec

        
        return dZhc_dxk
    
    @staticmethod
    def _compute_dash_dxjxk(zeta: np.ndarray, dzeta_dxk: np.ndarray, ahs: float, dahs_dxk: np.ndarray):
        zeta_aux = 1 - zeta[3]
        termo_1 = np.outer(dzeta_dxk[0,:], dzeta_dxk[0,:]) * ahs / zeta[0]**2 - np.outer(dahs_dxk, dzeta_dxk[0,:]) / zeta[0]

        # T1
        u = dzeta_dxk[1,:] * zeta[2] + zeta[1] * dzeta_dxk[2,:]
        du = np.outer(dzeta_dxk[2,:], dzeta_dxk[1,:]) + np.outer(dzeta_dxk[1,:], dzeta_dxk[2,:])
        v = zeta_aux
        dv = - dzeta_dxk[3,:]
        T1 = 3 * u / v
        T1_xj = 3 * (du * v - np.outer(dv, u)) / v**2

        # T2
        u = 3 * zeta[1] * zeta[2] * dzeta_dxk[3,:]
        v = zeta_aux**2
        du = 3 * (np.outer(dzeta_dxk[1,:], dzeta_dxk[3,:]) * zeta[2] + np.outer(dzeta_dxk[2,:],dzeta_dxk[3,:]) * zeta[1])
        dv = - 2 * zeta_aux * dzeta_dxk[3,:]
        T2 = u / v
        T2_xj = (du * v - np.outer(dv, u)) / v**2

        # T3
        u = 3 * zeta[2]**2 * dzeta_dxk[2,:]
        du = 6 * zeta[2] * np.outer(dzeta_dxk[2,:], dzeta_dxk[2,:])
        v = zeta[3] * zeta_aux**2
        dv = dzeta_dxk[3,:] * zeta_aux**2 - 2 * zeta[3] * zeta_aux * dzeta_dxk[3,:]
        T3 = u / v
        T3_xj = (du * v - np.outer(dv, u)) / v**2

        # T4
        u = zeta[2]**3 * dzeta_dxk[3,:] * (3 * zeta[3] - 1)
        du = 3 * zeta[2]**2 * (3 * zeta[3] - 1) * np.outer(dzeta_dxk[2,:], dzeta_dxk[3,:]) + 3 * zeta[2]**3 * np.outer(dzeta_dxk[3,:], dzeta_dxk[3,:])
        v = zeta[3]**2 * zeta_aux**3
        dv = 2 * zeta[3] * zeta_aux**3 * dzeta_dxk[3,:] - 3 * zeta[3]**2 * zeta_aux**2 * dzeta_dxk[3,:]
        T4 = u / v
        T4_xj = (du * v - np.outer(dv, u)) / v**2

        # T5
        u = 3 * zeta[2]**2 * zeta[3] * dzeta_dxk[2,:] - 2 * zeta[2]**3 * dzeta_dxk[3,:]
        v = zeta[3]**3
        du = 6 * zeta[2] * zeta[3] * np.outer(dzeta_dxk[2,:], dzeta_dxk[2,:]) + 3 * zeta[2]**2 * np.outer(dzeta_dxk[3,:], dzeta_dxk[2,:]) - 6 * zeta[2]**2 * np.outer(dzeta_dxk[2,:], dzeta_dxk[3,:])
        dv = 3 * zeta[3]**2 * dzeta_dxk[3,:]

        T5_1 = (u / v) - dzeta_dxk[0,:]
        T5_1xj = (du * v - np.outer(dv, u)) / v**2
        T5_2 = np.log(zeta_aux)
        T5_2xj = - dzeta_dxk[3,:] / zeta_aux
        T5 = T5_1 * T5_2
        T5_xj = T5_1xj * T5_2 + np.outer(T5_2xj, T5_1)

        # T6
        T6_1 = zeta[0] - zeta[2]**3 / zeta[3]**2
        T6_1xj = dzeta_dxk[0,:] - (3 * zeta[2]**2 * zeta[3]**2 * dzeta_dxk[2,:] - 2 * zeta[2]**3 * zeta[3] * dzeta_dxk[3,:]) / zeta[3]**4
        T6_2 = dzeta_dxk[3,:] / zeta_aux
        T6_2xj = np.outer(dzeta_dxk[3,:], dzeta_dxk[3,:]) / zeta_aux**2
        T6 = T6_1 * T6_2
        T6_xj = np.outer(T6_1xj, T6_2) + T6_1 * T6_2xj

        T = T1 + T2 + T3 + T4 + T5 + T6
        T_xj = T1_xj + T2_xj + T3_xj + T4_xj + T5_xj + T6_xj
    
        termo_2 = - np.outer(dzeta_dxk[0,:], T) / zeta[0]**2 + T_xj / zeta[0]

        dahs_xjxk = termo_1 + termo_2
        
        return dahs_xjxk

    @staticmethod
    def _compute_dgij_dxjxk(Dij:np.ndarray, zeta: np.ndarray, dzeta_dxk: np.ndarray):
        # Dij = d[:, np.newaxis] * d[np.newaxis, :] / (d[:, np.newaxis] + d[np.newaxis, :])
        zeta_aux = 1 - zeta[3]
        termo_1 = 2 * np.outer(dzeta_dxk[3,:], dzeta_dxk[3,:]) / zeta_aux**3


        termo_21 = (6 / zeta_aux**3) * np.outer(dzeta_dxk[3,:], dzeta_dxk[2,:])

        u = 6 * zeta[2] * dzeta_dxk[3,:]
        du = 6 * np.outer(dzeta_dxk[2,:], dzeta_dxk[3,:])
        v = zeta_aux**3
        dv = - 3 * zeta_aux**2 * dzeta_dxk[3,:]
        termo_22 = (du * v - np.outer(dv, u)) / v**2

        termo_2 = termo_21 + termo_22

        u = 4 * zeta[2] * dzeta_dxk[2,:]
        du = 4 * np.outer(dzeta_dxk[2,:], dzeta_dxk[2,:])
        termo_31 = (du * v - np.outer(dv, u)) / v**2
        
        u = 6 * zeta[2]**2 * dzeta_dxk[3,:]
        du = 12 * zeta[2] * np.outer(dzeta_dxk[2,:], dzeta_dxk[3,:])
        v = zeta_aux**4
        dv = - 4 * zeta_aux**3 *dzeta_dxk[3,:]
        termo_32 = (du * v - np.outer(dv, u)) / v**2

        termo_3 = termo_31 + termo_32
        dgij_xjxk = termo_1[None, None, :, :] + Dij[:, :, None, None] * termo_2[None, None, :, :] + Dij[:, :, None, None]**2 * termo_3[None, None, :, :]
        # AQUI TEM QUE VER MESMO SE GERA UM TENSOR (n,n,n,n)
        return dgij_xjxk

    @staticmethod
    def _dahc_dxjxk(z: np.ndarray, m: np.ndarray, m_mean: float, gij: np.ndarray, dgij_dxk: np.ndarray, dgij_dxjxk: np.ndarray,
                dahs_dxk: np.ndarray, dahs_dxjxk: np.ndarray):
        gii = np.diagonal(gij)
        gii_inv = 1 / gii
        gii_inv_sq = gii_inv**2

        dgii_xk = np.diagonal(dgij_dxk, axis1=0, axis2=1).T
        # dgii_xjxk = np.einsum('iikj->jk', dgij_xjxk)

        # Termo 1: mₖ (∂ãʰˢ/∂xⱼ) -> Matriz [j, k] = mₖ * (∂ãʰˢ/∂xⱼ)
        term1 = np.outer(dahs_dxk, m)

        # Termo 2: mₖ (∂ãʰˢ/∂xₖ) -> Matriz [k, j] = mⱼ * (∂ãʰˢ/∂xₖ)
        term2 = np.outer(m, dahs_dxk)

        # Termo 3: m̄ (∂²ãʰˢ/∂xⱼ∂xₖ) -> Escalar * Matriz
        term3 = m_mean * dahs_dxjxk

        # Termo 4a: - (mⱼ-1)(gⱼⱼ)⁻¹ (∂gⱼⱼ/∂xₖ) -> Matriz [j, k]
        term4a = - (m - 1.0)[:, None] * gii_inv[:, None] * dgii_xk

        # Termo 4b: + Σᵢ xᵢ(mᵢ-1)(gᵢᵢ)⁻² (∂gᵢᵢ/∂xⱼ) (∂gᵢᵢ/∂xₖ) -> Matriz [j, k]
        sum_4b = z * (m - 1.0) * gii_inv_sq # Vetor (N,)
        term4b = np.einsum('i,ij,ik->jk', sum_4b, dgii_xk, dgii_xk)

        # Termo 4c: - Σᵢ xᵢ(mᵢ-1)(gᵢᵢ)⁻¹ (∂²gᵢᵢ/∂xⱼ∂xₖ) -> Matriz [j, k]
        dgii_xjxk = np.diagonal(dgij_dxjxk, axis1=0, axis2=1).transpose(2, 0, 1)
        sum_4c = z * (m - 1.0) * gii_inv # Vetor (N,)
        term4c = -np.einsum('i,ijk->jk', sum_4c, dgii_xjxk)

        # Termo 5: - (mₖ-1)(gₖₖ)⁻¹ (∂gₖₖ/∂xⱼ) -> Matriz [j, k]
        # dgii_dxk.T[k, j] = ∂gₖₖ/∂xⱼ
        term5 = - (m - 1.0)[None, :] * gii_inv[None, :] * dgii_xk.T

        # 3. Soma Final -> Matriz NxN
        dahc_xjxk = term1 + term2 + term3 + term4a + term4b + term4c + term5
        return dahc_xjxk
    
    # ---------------------------------------------------------------------------------------------
    # -------------------------DERIVADAS COM RELACAO A TEMPERATURA---------------------------------
    # ---------------------------------------------------------------------------------------------
    @staticmethod
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
        du = 3 * (dzeta1_dT * zeta2 + zeta1 * dzeta2_dT)
        v = zeta0 * zeta_aux**2
        dv = 0.0 - 2 * zeta0 * zeta_aux * dzeta3_dT
        f2 = u / v
        df2_dT = (du * v - u * dv) / v**2

        u = 3 * zeta2**3 - zeta3 * zeta2**3
        du = 9 * zeta2**2 * dzeta2_dT - (dzeta3_dT * zeta2**3 + 3 * zeta3 * zeta2**2 * dzeta2_dT)
        v = zeta0 * zeta_aux**3
        dv = 0.0 - 3 * zeta0 * zeta_aux**2 * dzeta3_dT
        f3 = u / v
        df3_dT = (du * v - u * dv) / v**2

        dZhs_dT = df1_dT + df2_dT + df3_dT
        return dZhs_dT

    @staticmethod
    def _compute_dgij_dT(zeta:np.ndarray, dzeta_dT:np.ndarray, Dij:np.ndarray, dDij_dT:np.ndarray):
        _, _, zeta2, zeta3 = zeta
        _, dzeta2_dT, dzeta3_dT = dzeta_dT

        zeta_aux = 1 - zeta3
        
        df1_dT = dzeta3_dT / zeta_aux**2

        u =  3 * zeta2
        v = zeta_aux**2
        f2 = u /v
        df2_dT = ((3 * dzeta2_dT) * v - u * (- 2 * zeta_aux * dzeta3_dT)) / v**2

        u = 2 * zeta2**2
        v = zeta_aux**3
        f3 = u / v
        df3_dT = (4 * zeta2 * dzeta2_dT * v - u * (- 3 * zeta_aux**2 * dzeta3_dT)) / v**2

        dgij_dT = df1_dT + dDij_dT * f2 + Dij * df2_dT + 2 * Dij * dDij_dT * f3 + Dij**2 * df3_dT
        return dgij_dT

    @staticmethod
    def _compute_drho_dgij_rho_dT(zeta:np.ndarray, dzeta_dT:np.ndarray, Dij:np.ndarray, dDij_dT:np.ndarray):
        _, _, zeta2, zeta3 = zeta
        _, dzeta2_dT, dzeta3_dT = dzeta_dT

        zeta_aux = 1 - zeta3

        f1 = zeta3 / zeta_aux**2
        df1_dT = (dzeta3_dT * zeta_aux**2 - zeta3 * (- 2 * zeta_aux * dzeta3_dT)) / zeta_aux**4

        f21 = 3 * zeta2 / zeta_aux**2
        df21_dT = (3 * dzeta2_dT * zeta_aux**2 - 3 * zeta2 * (- 2 * zeta_aux * dzeta3_dT)) / zeta_aux**4

        f22 = 6 * zeta2 * zeta3 / zeta_aux**3
        df22_dT = (6 * (dzeta2_dT * zeta3 + zeta2 * dzeta3_dT) * zeta_aux**3 - 6 * zeta2 * zeta3 * (- 3 * zeta_aux**2 * dzeta3_dT)) / zeta_aux**6

        f2 = f21 + f22
        df2_dT = df21_dT + df22_dT

        f31 = 4 * zeta2**2 / zeta_aux**3
        df31_dT = (8 * zeta2 * dzeta2_dT * zeta_aux**3 - 4 * zeta2**2 * (- 3 * zeta_aux**2 * dzeta3_dT)) / zeta_aux**6
        
        f32 = 6 * zeta2**2 * zeta3 / zeta_aux**4
        df32_dT = (6 * (2 * zeta2 * dzeta2_dT * zeta3 + zeta2**2 * dzeta3_dT) * zeta_aux**4 - 6 * zeta2**2 * zeta3 * (-4 * zeta_aux**3 * dzeta3_dT)) / zeta_aux**8

        f3 = f31 + f32
        df3_dT = df31_dT + df32_dT

        drho_dgij_rho_dT = df1_dT + dDij_dT * f2 + Dij * df2_dT + 2 * Dij * dDij_dT * f3 + Dij**2 * df3_dT
        return drho_dgij_rho_dT

    @staticmethod
    def _compute_dZhc_dT(z:np.ndarray, m:np.ndarray, m_mean:float, dZhs_dT:float, gij:np.ndarray, rho_dgij_drho:np.ndarray,
                            dgij_dT:np.ndarray, drho_dgij_drho_dT:np.ndarray):
        gii = np.diagonal(gij)
        dgii_dT = np.diagonal(dgij_dT)
        rho_dgii_drho = np.diagonal(rho_dgij_drho)
        drho_dgii_drho_dT = np.diagonal(drho_dgij_drho_dT)

        term1 = m_mean * dZhs_dT

        f1 = - gii**-2 * dgii_dT * rho_dgii_drho
        f2 = gii**-1 * drho_dgii_drho_dT
        term2 = - np.sum(z * (m - 1) * (f1 + f2))

        dZhc_dT = term1 + term2
        return dZhc_dT


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

    def calculate_dxk(self, eta:float, T:float, coeff:PCSaftCoeffResult, params:PCSaftParametersResults,
                      disp_results:PCSaftDispersionResults):
        
        a, b = coeff.am, coeff.bm
        da_dxk, db_dxk = coeff.ai_xk, coeff.bi_xk
        da_dxjxk, db_dxjxk = coeff.ai_xjxk, coeff.bi_xjxk
        m, eij, sij, z = params.m, params.eij, params.sij, params.z
        rho, m_mean = params.rho, params.m_mean
        dzeta_dxk = params.zeta_xk
        C1, C2, I1, I2 = disp_results.C1, disp_results.C2, disp_results.I1, disp_results.I2
        m2es3, m2e2s3 = disp_results.m2es3, disp_results.m2e2s3
        detaI1_deta, detaI2_deta = disp_results.detaI1_eta, disp_results.detaI2_eta

        dI1_dxk, dI2_dxk = self._compute_dI12_dxk(eta=eta, a=a, b=b, dzeta_dxk=dzeta_dxk, da_dxk=da_dxk, db_dxk=db_dxk)
        dC1_dxk = self._compute_dC1_dxk(m=m, eta=eta, C1=C1, C2=C2, dzeta_dxk=dzeta_dxk)
        dm2es3_dxk, dm2e2s3_dxk = self._compute_dm2es3_dm2e2s3_dxk(T=T, z=z, m=m, eij=eij, sij=sij)
        dadisp_dxk = self._compute_dadisp_dx(rho=rho, m=m, C1=C1, I1=I1, I2=I2, m_mean=m_mean, m2es3=m2es3,
                                             m2e2s3=m2e2s3, dC1_dxk=dC1_dxk, dI1_dxk=dI1_dxk, dI2_dxk=dI2_dxk, dm2es3_dxk=dm2es3_dxk,
                                             dm2e2s3_dxk=dm2e2s3_dxk)

        detaI1_deta_dxk, detaI2_deta_dxk = self._compute_detaI1I2_deta_dxk(eta=eta, a=a, b=b, da_dxk=da_dxk, db_dxk=db_dxk, dzeta3_dxk=dzeta_dxk[3,:])
        dC2_dxk = self._compute_dC2_dxk(m=m, m_mean=m_mean, eta=eta, dzeta3_dxk=dzeta_dxk[3,:], C1=C1, dC1_dxk=dC1_dxk)

        dZdisp_dxk = self._compute_dZdisp_dxk(rho=rho, eta=eta, detaI1_deta=detaI1_deta, detaI2_deta=detaI2_deta, detaI1_deta_dxk=detaI1_deta_dxk,
                                              detaI2_deta_dxk=detaI2_deta_dxk, m=m, m_mean=m_mean, C1=C1, C2=C2, dC1_dxk=dC1_dxk, dC2_dxk=dC2_dxk,
                                              m2es3=m2es3, m2e2s3=m2e2s3, dm2es3_dxk=dm2es3_dxk, dm2e2s3_dxk=dm2e2s3_dxk, I2=I2, dI2_dxk=dI2_dxk,
                                              dzeta3_dxk=dzeta_dxk[3,:])
        
        dI1_dxjxk, dI2_dxjxk = self._compute_dI1I2_dxjxk(eta=eta, a=a, b=b, da_dxk=da_dxk, db_dxk=db_dxk,
                                                         da_dxjxk=da_dxjxk, db_dxjxk=db_dxjxk, dzeta3_dxk=dzeta_dxk[3,:])
        

        dm2es3_dxjxk, dm2e2s3_dxjxk = self._compute_dm2es3_dm2e2s3_dxjxk(m=m, eij=eij, sij=sij, T=T)

        dC1_dxjxk = self._compute_dC1_dxjxk(eta=eta, m=m, C1=C1, dC1_dxk=dC1_dxk, dC2_dxk=dC2_dxk, dzeta3_dxk=dzeta_dxk[3,:])

        dadisp_dxjxk = self._compute_dadisp_dxjxk(rho=rho, m=m, m_mean=m_mean, I1=I1, I2=I2, dI1_dxk=dI1_dxk, dI2_dxk=dI2_dxk, dI1_dxjxk=dI1_dxjxk, dI2_dxjxk=dI2_dxjxk,
                                                  m2es3=m2es3, m2e2s3=m2e2s3, dm2es3_dxk=dm2es3_dxk, dm2e2s3_dxk=dm2e2s3_dxk, dm2es3_dxjxk=dm2es3_dxjxk,
                                                  dm2e2s3_dxjxk=dm2e2s3_dxjxk, C1=C1, dC1_dxk=dC1_dxk, dC1_dxjxk=dC1_dxjxk)


        disp_results.derivatives.dI1_dxk = dI1_dxk
        disp_results.derivatives.dI2_dxk = dI2_dxk
        disp_results.derivatives.dC1_dxk = dC1_dxk
        disp_results.derivatives.dm2es3_dxk = dm2es3_dxk
        disp_results.derivatives.dm2e2s3_dxk = dm2e2s3_dxk
        disp_results.derivatives.dadisp_dxk = dadisp_dxk
        disp_results.derivatives.detaI1_deta_dxk  = detaI1_deta_dxk
        disp_results.derivatives.detaI2_deta_dxk = detaI2_deta_dxk
        disp_results.derivatives.dC2_dxk = dC2_dxk
        disp_results.derivatives.dZdisp_dxk = dZdisp_dxk
        disp_results.derivatives.dI1_dxjxk = dI1_dxjxk
        disp_results.derivatives.dI2_dxjxk = dI2_dxjxk 
        disp_results.derivatives.dm2es3_dxjxk = dm2es3_dxjxk
        disp_results.derivatives.dm2e2s3_dxjxk = dm2e2s3_dxjxk 
        disp_results.derivatives.dC1_dxjxk = dC1_dxjxk 
        disp_results.derivatives.dadisp_dxjxk = dadisp_dxjxk 


    def calculte_dT(self, eta:float, T:float, params:PCSaftParametersResults, coeff:PCSaftCoeffResult, disp_results:PCSaftDispersionResults):
        m_mean, rho = params.m_mean, params.rho
        dzeta_dT = params.dzeta_dT
        a, b = coeff.am, coeff.bm
        C1, C2 = disp_results.C1, disp_results.C2
        I1, I2 = disp_results.I1, disp_results.I2
        detaI1_deta, detaI2_deta = disp_results.detaI1_eta, disp_results.detaI2_eta
        m2es3, m2e2s3 = disp_results.m2es3, disp_results.m2e2s3

        ddetaI1_deta_dT, ddetaI2_deta_dT = self._compute_ddetaI1I2_deta_dT(dzeta3_dT=dzeta_dT[2,:], eta=eta, a=a, b=b)
        dI1_dT, dI2_dT = self._compute_dI12_dT(a=a, b=b, dzeta3_dT=dzeta_dT[2,:], eta=eta)
        dC2_dT = self._compute_dC2_dT(eta=eta, dzeta3_dT=dzeta_dT[2,:], C1=C1, C2=C2, m_mean=m_mean)
        dZdisp_dT = self._compute_dZdisp_dT(rho=rho, eta=eta, detaI1_deta=detaI1_deta, detaI2_deta=detaI2_deta, ddetaI1_deta_dT=ddetaI1_deta_dT,
                                            ddetaI2_deta_dT=ddetaI2_deta_dT, I2=I2, C1=C1, C2=C2, dC2_dT=dC2_dT, m2es3=m2es3, m2e2s3=m2e2s3, m_mean=m_mean,
                                            T=T, dzeta3_dT=dzeta_dT[2,:], dI2_dT=dI2_dT)
        
        disp_results.derivatives.ddetaI1_deta_dT = ddetaI1_deta_dT
        disp_results.derivatives.ddetaI2_deta_dT = ddetaI2_deta_dT
        disp_results.derivatives.dC2_dT = dC2_dT
        disp_results.derivatives.dZdisp_dT = dZdisp_dT
        

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

    # ---------------------------------------------------------------------------------------------
    # ---------------------------EQUACOES PARA CALCULAR A PRESSAO----------------------------------
    # ---------------------------------------------------------------------------------------------
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


    # ---------------------------------------------------------------------------------------------
    # -----------------------------DERIVADAS COM RELACAO A ETA-------------------------------------
    # ---------------------------------------------------------------------------------------------
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


    # ---------------------------------------------------------------------------------------------
    # -------------------------DERIVADAS COM RELACAO A COMPOSICAO----------------------------------
    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_dI12_dxk(eta: float, a: np.ndarray, b: np.ndarray, dzeta_dxk: np.ndarray, da_dxk: np.ndarray, db_dxk: np.ndarray) -> tuple:
        e_aux = (np.arange(7))[:, None]
        # EQ (A.42)
        I1_xk = np.sum(a[:, None] * e_aux * dzeta_dxk[3, :] * eta**(e_aux - 1) + da_dxk.T * eta**e_aux, axis=0)
        # EQ (A.43)
        I2_xk = np.sum(b[:, None] * e_aux * dzeta_dxk[3, :] * eta**(e_aux - 1) + db_dxk.T * eta**e_aux, axis=0)
        return I1_xk, I2_xk
    
    @staticmethod
    def _compute_dC1_dxk(m: float, eta: float, C1: float, C2: float, dzeta_dxk: np.ndarray):
        aux_1 = m * (8 * eta - 2 * eta**2) / (1 - eta)**4
        aux_2 = - m * (20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4) / ((1 - eta) * (2 - eta))**2
        # EQ (A.41)
        C1_xk = C2 * dzeta_dxk[3, :] - C1**2 * (aux_1 + aux_2)

        return C1_xk
    
    @staticmethod
    def _compute_dm2es3_dm2e2s3_dxk(T: float, z: np.ndarray, m: np.ndarray, eij: np.ndarray, sij: np.ndarray):
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
                         dC1_dxk: np.ndarray, dI1_dxk: np.ndarray, dI2_dxk: np.ndarray, dm2es3_dxk: np.ndarray, dm2e2s3_dxk: np.ndarray):
        aux_1 = - 2 * np.pi * rho * (dI1_dxk * m2es3 + I1 * dm2es3_dxk)
        aux_2 = (m * C1 * I2 + m_mean * dC1_dxk * I2 + m_mean * C1 * dI2_dxk)
        aux_3 = - np.pi * rho * ( aux_2 * m2e2s3 + m_mean * C1 * I2 * dm2e2s3_dxk)
        dadisp_dx = aux_1 + aux_3

        return dadisp_dx

    @staticmethod
    def _compute_detaI1I2_deta_dxk(eta:float, a:np.ndarray, b:np.ndarray, da_dxk:np.ndarray, db_dxk:np.ndarray, dzeta3_dxk:np.ndarray):
        j = np.arange(7)[:, None]
        detaI1_deta_xk = np.sum((j + 1) * (da_dxk.T * eta**j + j * eta**(j - 1) * a[:, None] * dzeta3_dxk.T), axis=0)
        detaI2_deta_xk = np.sum((j + 1) * (db_dxk.T * eta**j + j * eta**(j - 1) * b[:, None] * dzeta3_dxk.T), axis=0)
    
        return detaI1_deta_xk, detaI2_deta_xk
    
    @staticmethod
    def _compute_dC2_dxk(m:np.ndarray, m_mean:float, eta:float, dzeta3_dxk:np.ndarray, C1:float, dC1_dxk:np.ndarray):

        u = m_mean
        u_xk = m
        
        o = - 4 * eta**2 + 20 * eta + 8
        o_xk = -8 * eta * dzeta3_dxk + 20 * dzeta3_dxk
        p = (1 - eta)**5
        p_xk = - 5 * (1 - eta)**4 * dzeta3_dxk
        v = o / p
        v_xk = (o_xk * p - o * p_xk) / p**2


        a = (1 - m_mean)
        a_xk = - m
        
        o = 2 * eta**3 + 12 *eta**2 - 48 * eta + 40
        o_xk = 6 * eta**2 * dzeta3_dxk + 24 * eta * dzeta3_dxk - 48 * dzeta3_dxk
        p = (eta**2 - 3 * eta + 2)**3
        p_xk = 3 * (2 * eta * dzeta3_dxk - 3 * dzeta3_dxk) * (eta**2 - 3 * eta + 2)**2

        b = o / p
        b_xk = (o_xk * p - o * p_xk) / p**2

        s = - C1**2
        s_xk = - 2 * C1 * dC1_dxk

        t = u * v + a * b
        t_xk = (u_xk * v + u * v_xk) + (a_xk * b + a * b_xk)

        dC2_dxk = s_xk * t + s * t_xk
        
        return dC2_dxk

    @staticmethod
    def _compute_dZdisp_dxk(rho:float, eta: float, detaI1_deta: float, detaI2_deta: float, detaI1_deta_dxk: np.ndarray, detaI2_deta_dxk: np.ndarray,
                        m: np.ndarray, m_mean: float, C1: float, C2: float, dC1_dxk: np.ndarray, dC2_dxk: np.ndarray,
                        m2es3: float, dm2es3_dxk: np.ndarray, dzeta3_dxk: np.ndarray, I2: float, dI2_dxk: np.ndarray,
                        m2e2s3: float, dm2e2s3_dxk: np.ndarray):
    
        term1 = - 2 * np.pi * rho * (detaI1_deta_dxk * m2es3 + detaI1_deta * dm2es3_dxk)

        u = m_mean * C1
        du = (m * C1 + m_mean * dC1_dxk)
        v = detaI2_deta * m2e2s3
        dv = (detaI2_deta_dxk * m2e2s3 + detaI2_deta * dm2e2s3_dxk)
        term2 = - np.pi * rho * (du * v + u * dv)
        u = m_mean * C2 * eta
        du = m * (C2 * eta) + m_mean * (dC2_dxk * eta + C2 * dzeta3_dxk)
        v = I2 * m2e2s3
        dv = dI2_dxk * m2e2s3 + I2 * dm2e2s3_dxk
        term3 = - np.pi * rho * (du * v + u * dv)

        dZdips_dxk = term1 + term2 + term3

        return dZdips_dxk

    @staticmethod
    def _compute_dI1I2_dxjxk(eta:float, a:np.ndarray, b:np.ndarray, da_dxk:np.ndarray, da_dxjxk:np.ndarray, db_dxk:np.ndarray, 
                             db_dxjxk: np.ndarray, dzeta3_dxk:np.ndarray):
    
        i_vec = np.arange(7)
        i_minus1 = i_vec - 1
        i_minus2 = i_vec - 2
        eta_pow_i = np.power(eta, i_vec)
        eta_pow_i_minus1 = np.power(eta, i_minus1)
        eta_pow_i_minus2 = np.power(eta, i_minus2)
        zeta3_xjxk = np.outer(dzeta3_dxk, dzeta3_dxk)

        # Construção do I1_xjxk
        term1 = np.einsum('ijk,i->jk', da_dxjxk, eta_pow_i)
        aux1 = np.einsum('ji,k->ijk', da_dxk, dzeta3_dxk)
        aux2 = np.einsum('ki,j->ijk', da_dxk, dzeta3_dxk)
        term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
        term3 = np.einsum('i,jk->jk', (a * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

        dI1_dxjxk = term1 + term2 + term3

        # Construção do I12_xjxk
        term1 = np.einsum('ijk,i->jk', db_dxjxk, eta_pow_i)
        aux1 = np.einsum('ji,k->ijk', db_dxk, dzeta3_dxk)
        aux2 = np.einsum('ki,j->ijk', db_dxk, dzeta3_dxk)
        term2 = np.einsum('i,ijk->jk', (i_vec * eta_pow_i_minus1), (aux1 + aux2))
        term3 = np.einsum('i,jk->jk', (b * i_vec * i_minus1 * eta_pow_i_minus2), zeta3_xjxk)

        dI2_dxjxk = term1 + term2 + term3

        return dI1_dxjxk, dI2_dxjxk

    @staticmethod
    def _compute_dm2es3_dm2e2s3_dxjxk(m:np.ndarray, eij:np.ndarray, sij:np.ndarray, T:float):
        mjmk = np.outer(m, m)
        m2es3_xjxk = 2 * mjmk * (eij / T) * sij**3
        m2e2s3_xjxk = 2 * mjmk * (eij / T)**2 * sij**3

        return m2es3_xjxk, m2e2s3_xjxk

    @staticmethod
    def _compute_dC1_dxjxk(eta: float, m:np.ndarray,  C1: float, dC1_dxk:np.ndarray, dC2_dxk: np.ndarray, dzeta3_dxk: np.ndarray):

        C2_xj_zeta3_xk = np.outer(dC2_dxk, dzeta3_dxk)
        term1 = C2_xj_zeta3_xk

        # as funcoes de eta dentro do parenteses
        u = 8 * eta - 2 * eta**2
        du = 8 * dzeta3_dxk - 4 * eta * dzeta3_dxk
        v = (1 - eta)**4
        dv = - 4 * (1 - eta)**3 * dzeta3_dxk
        s = 20 * eta - 27 * eta**2 + 12 * eta**3 - 2 * eta**4
        ds = 20 * dzeta3_dxk - 54 * eta * dzeta3_dxk + 36 * eta**2 * dzeta3_dxk - 8 * eta**3 * dzeta3_dxk
        t = (2 - 3 * eta + eta**2)**2
        dt = 2 * t**0.5 * (2 * eta * dzeta3_dxk - 3 * dzeta3_dxk)
        aux1_xj = (du * v - u * dv) / v**2
        aux2_xj = (ds * t - s * dt) / t**2
        func_aux = u/v - s/t
        term_aux_xj= 2 * C1 * dC1_dxk * func_aux + C1**2 * (aux1_xj - aux2_xj)
        term2 = np.outer(term_aux_xj, m)
        dC1_dxjxk = term1 - term2

        return dC1_dxjxk

    @staticmethod
    def _compute_dadisp_dxjxk(rho: float, m:np.ndarray, m_mean:float, I1:float, I2:float, dI1_dxk:np.ndarray, dI2_dxk:np.ndarray, 
                            dI1_dxjxk:np.ndarray, dI2_dxjxk:np.ndarray,m2es3:float, m2e2s3:float, dm2es3_dxk:np.ndarray, dm2e2s3_dxk:np.ndarray,
                            dm2es3_dxjxk:np.ndarray, dm2e2s3_dxjxk:np.ndarray, C1:float, dC1_dxk:np.ndarray, dC1_dxjxk:np.ndarray):
        
        m2es3_xj_I1_xk = np.outer(dm2es3_dxk, dI1_dxk)
        
        term1_xj = - 2 * np.pi * rho * (dI1_dxjxk * m2es3 + m2es3_xj_I1_xk + m2es3_xj_I1_xk.T + dm2es3_dxjxk * I1)

        aux = I2 * m2e2s3
        aux_xj = (dI2_dxk * m2e2s3 + I2 * dm2e2s3_dxk)
        aux1_xj = dC1_dxk * aux + C1 * aux_xj
        aux1_xj = np.outer(aux1_xj, m)

        u = m_mean * dC1_dxk
        du = np.outer(m, dC1_dxk) + m_mean * dC1_dxjxk
        v = aux
        dv = aux_xj
        aux2_xj = du * v + np.outer(dv, u)

        s = m_mean * C1
        ds = m * C1 + m_mean * dC1_dxk
        t = dI2_dxk * m2e2s3
        dt = dI2_dxjxk * m2e2s3 + np.outer(dm2e2s3_dxk, dI2_dxk)
        aux_3_xj = np.outer(ds, t) + s * dt

        m = I2 * dm2e2s3_dxk
        dm = np.outer(dI2_dxk, dm2e2s3_dxk) + I2 * dm2e2s3_dxjxk
        aux_4_xj = np.outer(ds, m) + s * dm

        term2_xj = - np.pi * rho * (aux1_xj + aux2_xj + aux_3_xj + aux_4_xj)
        
        dadisp_dxjxk = term1_xj + term2_xj
        return dadisp_dxjxk

    # ---------------------------------------------------------------------------------------------
    # -------------------------DERIVADAS COM RELACAO A TEMPERATURA---------------------------------
    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_ddetaI12_deta_dT(dzeta3_dT:float, eta:float, a:np.ndarray, b:np.ndarray):
        j = np.arange(7)
        ddetaI1_deta_dT = np.sum(a * (j + 1) * j * dzeta3_dT * eta**(j - 1))
        ddetaI2_deta_dT = np.sum(b * (j + 1) * j * dzeta3_dT * eta**(j - 1))

        
        return ddetaI1_deta_dT, ddetaI2_deta_dT

    @staticmethod
    def _compute_dI12_dT(a:np.ndarray, b:np.ndarray, dzeta3_dT:float, eta:float):
        j = np.arange(7)
        dI1_dT = np.sum(a * j * dzeta3_dT * eta**(j - 1))
        dI2_dT = np.sum(b * j * dzeta3_dT * eta**(j - 1))

        return dI1_dT, dI2_dT

    @staticmethod
    def _compute_dC2_dT(eta:float, dzeta3_dT:float, C1:float, C2:float, m_mean:float):
        
        dC1_dT = dzeta3_dT * C2

        u = - 4 * eta**2 + 20 * eta + 8
        du = - 8 * eta * dzeta3_dT + 20 * dzeta3_dT
        v = (1 - eta)**5
        dv = - 5 * (1 - eta)**4 * dzeta3_dT
        f1 = u / v
        df1_dT = (du * v - u * dv) / v**2

        u = 2 * eta**3 + 12 * eta**2 - 48 * eta + 40
        du = 6 * eta**2 * dzeta3_dT + 24 * eta * dzeta3_dT - 48 * dzeta3_dT
        v = (2 - 3 * eta + eta**2)**3
        dv = 3 * (2 - 3 * eta + eta**2)**2 * (- 3 * dzeta3_dT + 2 * eta * dzeta3_dT)
        f2 = u / v
        df2_dT = (du * v - u * dv) / v**2

        f = m_mean * f1 + (1 - m_mean) * f2
        df_dT = m_mean * df1_dT + (1 - m_mean) * df2_dT

        dC2_dT = - 2 * C1 * dC1_dT * f - C1**2 * df_dT

        return dC2_dT

    @staticmethod
    def _compute_dZdisp_dT(rho:float, eta:float, detaI1_deta:float, detaI2_deta:float, ddetaI1_deta_dT:float, ddetaI2_deta_dT:float,
                           I2:float, C1:float, C2:float, dC2_dT:float, m2es3:float, m2e2s3:float, m_mean:float, T:float, dzeta3_dT:float,
                           dI2_dT:float):
        
        dC1_dT = dzeta3_dT * C2

        term1 = - 2 * np.pi * rho * (ddetaI1_deta_dT - detaI1_deta / T) * m2es3

        f1 = C1 * detaI2_deta
        df1_dT = dC1_dT * detaI2_deta + C1 * ddetaI2_deta_dT
        f2 = C2 * eta * I2
        df2_dT = (dC2_dT * eta + C2 * dzeta3_dT) * I2 + C2 * eta * dI2_dT

        f = f1 + f2
        df_dT = df1_dT + df2_dT

        term2 = - rho * np.pi * m_mean * (df_dT - 2 * f / T) * m2e2s3

        dZdisp_dT = term1 + term2
        return dZdisp_dT


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

    def calculate_derivatives(self, T:float, V:float, helmholtz_results:PCSAFTHelmholtzResult, result:PCSaftPressureResult, n:float=100.0):
        dF_dVV = helmholtz_results.dF_dVV
        dF_dVnk = helmholtz_results.dF_dniV
        
        dP_dV = - RGAS_SI * T * dF_dVV - n * RGAS_SI * T / V**2
        dP_dnk = - RGAS_SI * T * dF_dVnk + RGAS_SI * T / V
        
        result.dP_dV = dP_dV
        result.dP_dni = dP_dnk

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


class HelmholtzWorker:
    def __init__(self):
        pass

    def calculate(self, z:np.ndarray, eta:float, Z:float, V:float, n:float,
                  hc_result:PCSaftHardChainResults, disp_results:PCSaftDispersionResults):
        
        ahc, dahc_dxk, dahc_dxjxk = hc_result.ar_hc, hc_result.derivatives.dahc_dxk, hc_result.derivatives.dahc_dxjxk
        adips, dadisp_dxk, dadisp_dxjxk = disp_results.ar_disp, disp_results.derivatives.dadisp_dxk, disp_results.derivatives.dadisp_dxjxk
        dZhc_dxk, dZdisp_dxk = hc_result.derivatives.dZhc_dxk, disp_results.derivatives.dZdisp_dxk
        dZhc_deta, dZdisp_deta = hc_result.derivatives.dZhc_deta, disp_results.derivatives.dZdisp_deta
        dZ_deta = dZhc_deta + dZdisp_deta
        dZ_dxk = dZhc_dxk + dZdisp_dxk
        # as derivadas de helmhotz (T, rho, x)
        ares = ahc + adips
        dares_dxk = dahc_dxk + dadisp_dxk
        dares_dxjxk = dahc_dxjxk + dadisp_dxjxk
        



        rho_dmuk_drho = self._compute_rho_dmuk_drho(z=z, Z=Z, eta=eta, dZ_deta=dZ_deta, dZ_dxk=dZ_dxk)
        dmuk_dxj = self._compute_dmuk_xj(z=z, dZ_dxk=dZ_dxk, dares_dxjxk=dares_dxjxk)

        
        dF_dV = self._compute_dF_dV(V=V, Z=Z, n=n)
        dF_dnk = self._compute_dF_dnk(z=z, ares=ares, dares_dxk=dares_dxk, Z=Z, n=n)
        dF_dVV = self._compute_dF_dVV(V=V, Z=Z, eta=eta, dZ_deta=dZ_deta, n=n)
        dF_dVnk = self._compute_dF_dVnk(V=V, z=z, Z=Z, eta=eta, dZ_deta=dZ_deta, dZ_dxk=dZ_dxk, n=n)
        dF_dnjnk = self._compute_dF_dnjnk(z=z, rho_dmuk_drho=rho_dmuk_drho, dmuk_dxj=dmuk_dxj, n=n)

        return PCSAFTHelmholtzResult(
            dF_dT = None,
            dF_dni=dF_dnk,
            dF_dninj=dF_dnjnk,
            dF_dV=dF_dV,
            dF_dVV=dF_dVV,
            dF_dniV=dF_dVnk
        )
        

    @staticmethod
    def _compute_dF_dV(V:float, Z:float, n:float=100.0):
        dF_dV = - n * (Z - 1) / V
        return dF_dV
    
    @staticmethod
    def _compute_dF_dnk(z:np.ndarray, ares:float, dares_dxk:np.ndarray, Z:float, n:float=100.0):
        sum_aux = - np.sum(z * dares_dxk)
        # EQ (A.33)
        chemical_pow = ares + (Z - 1) + dares_dxk + sum_aux
        dF_dnk = chemical_pow
        return dF_dnk

    @staticmethod
    def _compute_dF_dVV(V:float, Z:float, eta:float, dZ_deta:float, n:float=100.0):
        dF_dVV = n * (eta * dZ_deta + (Z - 1)) / V**2
        return dF_dVV
    
    @staticmethod
    def _compute_dF_dVnk(V:float, z:np.ndarray, Z:float, eta:float, dZ_deta:float, dZ_dxk:np.ndarray, n:float=100.0):
        sum_xidZxi = - np.sum(z * dZ_dxk)

        dF_dVnk = - ((Z-1) + eta * dZ_deta + dZ_dxk + sum_xidZxi) / V
        return dF_dVnk

    @staticmethod
    def _compute_rho_dmuk_drho(z:np.ndarray, Z:float, eta:float, dZ_deta:float, dZ_dxk:np.ndarray):
        sum_xi_dZ_xi = - np.sum(z * dZ_dxk)
        rho_dmuk_rho = (Z - 1) + eta * dZ_deta + dZ_dxk + sum_xi_dZ_xi
        return rho_dmuk_rho
    
    @staticmethod
    def _compute_dmuk_xj(z:np.ndarray, dZ_dxk:np.ndarray, dares_dxjxk:np.ndarray):
        sum_xi_dares_xjxi = - np.einsum('i,ji->j', z, dares_dxjxk)
        dmuk_xj = dZ_dxk[:,None] + dares_dxjxk + sum_xi_dares_xjxi[:,None]
        return dmuk_xj
    
    @staticmethod    
    def _compute_dF_dnjnk(z:np.ndarray, rho_dmuk_drho:np.ndarray, dmuk_dxj:np.ndarray, n:float=100.00):
        sum_xi_dmuk_xi = - np.einsum('i,ik->k', z, dmuk_dxj)

        dF_dnjnk = (1.0 / n) * (rho_dmuk_drho + dmuk_dxj + sum_xi_dmuk_xi)
        return dF_dnjnk


class PCSaftFugacityWorker:
    def __init__(self):
        pass
    
    def calculate(self, T:float, Z:float, helmholtz_results:HelmholtzResult,
                   pressure_results:PCSaftPressureResult, n:float=100.0) -> FugacityResults:
        
        dF_dnk, dF_dnjnk = helmholtz_results.dF_dni, helmholtz_results.dF_dninj
        dP_dnk, dP_dV = pressure_results.dP_dni, pressure_results.dP_dV
        P = pressure_results.P

        ln_phi, phi = self._compute_fugacity_coefficient(dF_dnk=dF_dnk, Z=Z)
        n_dln_phik_dnj, dlnphik_dnj = self._compute_n_dlnphik_dnj(T=T, dF_dnjnk=dF_dnjnk, dP_dnk=dP_dnk, dP_dV=dP_dV, n=n)
        dlnphik_dP = self._compute_dlnphik_dP(T=T, P=P, dP_dnk=dP_dnk, dP_dV=dP_dV)

        return FugacityResults(
            ln_phi=ln_phi,
            phi=phi,
            dlnphi_dni=dlnphik_dnj,
            dlnphi_dP=dlnphik_dP,
        )
    
    

    @staticmethod
    def _compute_fugacity_coefficient(dF_dnk:np.ndarray, Z:float):
        ln_phi = dF_dnk - np.log(Z)
        phi = np.exp(ln_phi)

        return ln_phi, phi

    @staticmethod
    def _compute_n_dlnphik_dnj(T:float, dF_dnjnk:np.ndarray, dP_dnk:np.ndarray, dP_dV:float, n:float=100.00):
        dP_dnk_dP_dnj = np.outer(dP_dnk, dP_dnk)
        n_dlnphik_dnj = n * dF_dnjnk + 1  + (n / (RGAS_SI * T)) * dP_dnk_dP_dnj / dP_dV

        return n_dlnphik_dnj, n_dlnphik_dnj / n 


    @staticmethod
    def _compute_dlnphik_dP(T:float, P:float, dP_dnk:np.ndarray, dP_dV:float):
        parcial_Vk = - dP_dnk / dP_dV

        dlnphik_dP = parcial_Vk / (RGAS_SI * T) - 1 / P
        return dlnphik_dP

    def _calculate(self, z: np.ndarray, Z: float, hc_result: PCSaftHardChainResults, disp_result: PCSaftDispersionResults) -> None:
        dach_dx, ar_hc = hc_result.derivatives.dahc_dxk, hc_result.ar_hc
        dadisp_dx, ar_disp = disp_result.derivatives.dadisp_dxk, disp_result.ar_disp

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
        self.helmholtz_worker = HelmholtzWorker()
        pass

    def calculate_from_TP(self, state: State, is_vapor: bool) -> None:
            if state.T is None or state.P is None:
                raise ValueError('Temperature and Pressure must be inputed to use this method')
            
            T = state.T
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
            state.V = state.Vm * state.n
            state.rho = params.rho
            state.pressure_result = pressure_result
            state.core_model = PCCoreModel(params=params,
                                           coeff=coeff_results,
                                           hc_results=hc_result,
                                           disp_results=disp_result)


    def calculate_fugacity(self, state: State, teste:bool=False):
        z, Z, T, eta, V, n = state.z, state.Z, state.T, state.eta, state.V, state.n
        params, coeff = state.core_model.params, state.core_model.coeff
        hc_result, disp_results = state.core_model.hc_results, state.core_model.disp_results
        pressure_results = state.pressure_result
        # Da update nas coisas
        self.parameter_worker.calculate_derivatives_for_fugacity(params=params)
        self.coeff_worker.calculate_derivatives_for_fugacity(coeff=coeff, params=params)
        self.hc_worker.calculate_dxk(params=params, hc_result=hc_result)

        self.disp_worker.calculate_dxk(eta=eta, T=T, coeff=coeff, params=params, disp_results=disp_results)

        helmholtz_results = self.helmholtz_worker.calculate(z=z, eta=eta, Z=Z, V=V, n=n, hc_result=hc_result, disp_results=disp_results)
        state.helmholtz_result = helmholtz_results
        self.pressure_worker.calculate_derivatives(T=T, V=V, helmholtz_results=helmholtz_results, result=pressure_results, n=n)
        
        state.fugacity_result = self.fugacity_worker.calculate(T=T, Z=Z, helmholtz_results=helmholtz_results, pressure_results=pressure_results, n=n)

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
        pressure_result = self.pressure_worker.calculate(T=state.T, params=params, hc_results=hc_result, disp_results=disp_result)
        state.P = pressure_result.P
        state.Z = pressure_result.Z
        state.pressure_result = pressure_result
        state.core_model = PCCoreModel(params=params,
                                            coeff=coeff_results,
                                            hc_results=hc_result,
                                            disp_results=disp_result)


if __name__ == "__main__":
    T = 200 # K
    P = 30e5 # Pa

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

    state = State(
        mixture=mixture,
        z=np.array([0.4, 0.6]),
        T=T,
        P=P
    )

    pc_saft_engine = PCSaft(workers=None)
    pc_saft_engine.calculate_from_TP(state=state, is_vapor=True)
    pc_saft_engine.calculate_fugacity(state=state)

    dZhc_dxk_anal = state.core_model.hc_results.derivatives.dahc_dxk


    # Testando numericamente
    # print(160*'*')
    # print('----Comparação numérica com analítica----')
    h_x = 0.00001
    state_xpos = deepcopy(state)
    state_xpos.z[0] += h_x
    pc_saft_engine.update_parameters(state=state_xpos)
    pc_saft_engine.calculate_fugacity(state=state_xpos)

    state_xneg = deepcopy(state)
    state_xneg.z[0] -= h_x
    pc_saft_engine.update_parameters(state=state_xneg)
    pc_saft_engine.calculate_fugacity(state=state_xneg)

