from ...state import State, HelmholtzResult
from typing import Dict
import numpy as np

class CubicHelmholtzDerivativesWorker:
    def __init__(self):
        # self.derivatives = {}
        pass

    def _calculate_F_parcial_derivatives(self, params: Dict[str, any], core_model: Dict[str, float]) -> HelmholtzResult:
        Bi = np.array(params['Bi'])
        Di = np.array(params['Di'])
        DiT = params['DiT']
        DT = params['DT']
        DTT = params['DTT']
        Bij = np.array(params['Bij'])
        Dij = np.array(params['Dij'])
        Fn = core_model['Fn']
        FB = core_model['FB']
        FD = core_model['FD']
        FT = core_model['FT']
        FV = core_model['FV']
        FnB = core_model['FnB']
        FBD = core_model['FBD']
        FBB = core_model['FBB']
        FBT = core_model['FBT']
        FDT = core_model['FDT']
        FnV = core_model['FnV']
        FBV = core_model['FBV']
        FDV = core_model['FDV']
        FTT = core_model['FTT']
        FTV = core_model['FTV']
        FVV = core_model['FVV']

        # First derivatives
        dF_dni = Fn + FB * Bi + FD * Di
        dF_dT = FT + FD * DT
        dF_dV = FV
        # Second derivatives
        t_FnB = FnB * (np.add.outer(Bi, Bi))
        t_FBD = FBD * (np.outer(Bi, Di) + np.outer(Di, Bi))
        dF_dninj = t_FnB + t_FBD + FB * Bij + FBB * np.outer(Bi, Bi) + FD * Dij
        dF_dniT = (FBT + FBD * DT) * Bi + FDT * Di + FD * DiT
        dF_dniV = FnV + FBV * Bi + FDV * Di 
        dF_dTT = FTT + 2 * FDT * DT + FD * DTT
        dF_dTV = FTV + FDV * DT
        dF_dVV = FVV

        return HelmholtzResult(
            dF_dT=dF_dT,
            dF_dV=dF_dV,
            dF_dP=None,
            dF_dni=dF_dni,
            dF_dninj=dF_dninj,
            dF_dniT=dF_dniT,
            dF_dniV=dF_dniV,
            dF_dTT=dF_dTT,
            dF_dTV=dF_dTV,
            dF_dVV = dF_dVV
        )
    
    def helmholtz_derivatives_to_dict(self, state: State) -> any:
        params = state.params
        core_model = state.core_model
        # self.derivatives = {}
        helmhotz_results = self._calculate_F_parcial_derivatives(params=params, core_model=core_model)
        return helmhotz_results


    