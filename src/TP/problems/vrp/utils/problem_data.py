from typing import Dict

import numpy as np
from pydantic.dataclasses import dataclass


@dataclass
class CVRPProblemData:
    n_vehicles: int
    vehicle_cap: int
    depot_node: int
    node_demand: Dict
    dist_mtx: np.ndarray
