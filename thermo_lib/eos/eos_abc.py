from abc import ABC, abstractmethod
from ..state import State
from ..constants import RGAS_SI

class EquationOfState(ABC):
    """
    This is a abstract class which implements a equation of state. It is a stateless abstract class
    """
    # @abstractmethod
    # def get_eos_params(self, state: State): pass PONTO DE APOIO, posso deletar isso, porque eu implementarei nos workers

    @abstractmethod
    def calculate_pressure(self, state: State) -> None: pass

    @abstractmethod
    def calculate_from_TP(self, state: State, is_vapor: bool) -> None: pass

    @abstractmethod
    def calculate_fugacity(self, state: State) -> None: pass

    @abstractmethod
    def calculate_mixture_parameters(self, state: State) -> None: pass

    def calculate_from_TVm(self, state: State) -> None:
        """
        This method is the same for any equation of state, the only thing would change is the pressure calculate method
        For using this function, the user must input temperature end molar volume
        """
        if state.T is None or state.Vm is None:
            raise ValueError('Temperature (T) or Molar Volume (Vm) is None')
        
        # It utilises the abstract method to calculate the parameters of eos
        # state.params = self.get_eos_params(state=State)

        self.calculate_pressure(state=state) # -> Teoricamente aqui esta calculando a pressão....  
        state.Z = state.P * state.Vm / (RGAS_SI * state.T) # PONTO DE APOIO, talvez seja necessario criar uma outra logica do R_nSI
        state.V = state.Vm * state.n

    def calculate_full_state(self, state: State) -> None:
        if state.T is None or state.Vm is None or state.Z is None or state.P is None:
            raise ValueError('Thermodynamic state must be setted')

    def get_Z(self, state: State) -> float:
        if state.Z is None:
            raise ValueError('You must run a method to calculate the compressibility before')
        return state.Z