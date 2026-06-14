import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class VRPProblemData:
    n_vehicles: int
    depot_node: int
    dist_mtx: np.ndarray
