from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

@dataclass
class Component:
    name: str
    Tc: float
    Pc: float
    omega: float
    sigma: Optional[float] = None
    epsilon: Optional[float] = None
    segment: Optional[float] = None

@dataclass
class Mixture:
    components: list[Component]
    k_ij: np.ndarray
    l_ij: np.ndarray
