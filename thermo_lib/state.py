from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from abc import ABC

from .components import Mixture, Component

@dataclass
class FugacityResults:
    ln_phi: Optional[np.ndarray[float]] = None
    phi: Optional[np.ndarray[float]] = None
    dlnphi_dT: Optional[np.ndarray[float]] = None
    dlnphi_dP: Optional[np.ndarray[float]] = None
    dlnphi_dni: Optional[np.ndarray[float]] = None 
    n_dlnphi_dni: Optional[np.ndarray[float]] = None

@dataclass
class HelmholtzResult:
    dF_dT: Optional[float] = None
    dF_dV: Optional[float] = None
    dF_dP: Optional[float] = None # Isso aqui, acredito que posso deletar
    dF_dni: Optional[np.ndarray[float]] = None
    dF_dninj: Optional[np.ndarray[float]] = None
    dF_dniT: Optional[np.ndarray[float]] = None
    dF_dniV: Optional[np.ndarray[float]] = None
    dF_dTT: Optional[float] = None
    dF_dTV: Optional[float] = None
    dF_dVV: Optional[float] = None 

@dataclass
class PressureResult:
    dP_dV: Optional[float] = None 
    dP_dT: Optional[float] = None
    dP_dni: Optional[np.ndarray] = None

@dataclass
class PCSAFTHelmholtzResult(HelmholtzResult):
    a_res: Optional[float] = None
    dares_dxk: Optional[np.ndarray] = None
    dares_dxjxk: Optional[np.ndarray] = None
    # Aqui tenho que incluir as paradas de 'a_hs, a_hc...'
    pass

@dataclass
class ResidualPropertiesResults:
    # Preciso checar se esses valores sao adimensionalizados!!!!!!!
    Sr: Optional[float] = None
    Hr: Optional[float] = None
    Gr: Optional[float] = None
    F: Optional[float] = None # Here, the definition F = Ar / RT


@dataclass
class BaseState(ABC):
    mixture: Mixture
    z: np.ndarray
    T: float # Kelvin
    P: Optional[float] = None # Pascal
    Z: Optional[float] = None # Adm
    Vm: Optional[float] = None # m3 mol-1
    V: Optional[float] = None # m3
    n: float = 100 # mol

@dataclass
class State(BaseState):
    core_model: Optional[Dict[str, any]] = None
    helmholtz_derivatives: Optional[Dict[str, any]] = None
    P_derivatives: Optional[Dict[str, any]] = None
    fugacity_dict: Optional[Dict[str, any]] = None
    residual_props: Optional[Dict[str, float]] = None
    params: Optional[Dict[str, any]] = None
    props: Optional[Dict[str, any]] = None
    coeff: Optional[Dict[str, any]] = None
    rho: Optional[float] = None
    eta: Optional[float] = None
    # Aqui eh para 'polimorfizar' as proximas etapas
    fugacity_result: Optional[FugacityResults] = None
    helmholtz_results: Optional[HelmholtzResult] = None
    pressure_results: Optional[PressureResult] = None
    residual_props_results: Optional[ResidualPropertiesResults] = None



@dataclass
class CoreModel(ABC):
    pass

@dataclass
class PCCoreModel(CoreModel):
    pass