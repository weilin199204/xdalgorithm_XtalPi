from typing import List
from dataclasses import dataclass
import math

@dataclass
class ComponentParameters:
    component_type: str
    name: str
    weight: float
    smiles: List[str]
    model_path: str
    specific_parameters: dict = None
    up_limit: tuple = ('<',math.inf)
    low_limit: tuple = ('>',-math.inf)
