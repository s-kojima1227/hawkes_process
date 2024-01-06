from ...base import Parameters as BaseP, ParametersFactory as BasePF
from typing import Tuple, Dict, Union
import numpy as np

class Parameters(BaseP):
    def __init__(self, unpacked: np.ndarray, dim: int):
        self._unpacked = unpacked
        self._dim = dim

    @property
    def unpacked(self) -> np.ndarray:
        return self._unpacked

    @property
    def dict(self) -> Dict:
        return {
            'baselines': self.baselines,
            'adjacencies': self.adjacencies,
            'decays': self.decays,
        }

    @property
    def baselines(self) -> np.ndarray:
        return self._unpacked[:self._dim]

    @property
    def adjacencies(self) -> np.ndarray:
        dim = self._dim
        return self._unpacked[dim:dim * (dim + 1)].reshape(dim, dim)

    @property
    def decays(self) -> np.ndarray:
        dim = self._dim
        return self._unpacked[dim * (dim + 1):].reshape(dim, dim)

    @property
    def kernel_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.adjacencies, self.decays

class ParametersFactory(BasePF):
    def __init__(self, dim: Union[int, None]=None, num_exps: Union[int, None]=None):
        self._dim = dim
        self._num_exps = num_exps

    def build_from_packed(self, baselines, kernel_params) -> Parameters:
        adjacencies, decays = kernel_params
        if isinstance(baselines, (int, float)):
            baselines, adjacencies, decays = (np.array([baselines]), np.array([[adjacencies]]), np.array([[decays]]))
        else:
            baselines, adjacencies, decays = (np.array(baselines), np.array(adjacencies), np.array(decays))

        dim = baselines.shape[0]
        unpacked = np.concatenate([baselines, adjacencies.flatten(), decays.flatten()])

        return Parameters(unpacked, dim)

    def build_from_unpacked(self, params: np.ndarray) -> Parameters:
        if self._dim is None:
            raise ValueError('dim is not set')
        return Parameters(params, self._dim)

    def build_from_dict(self, params: Dict) -> Parameters:
        baselines = params.get('baselines')
        adjacencies = params.get('adjacencies')
        decays = params.get('decays')
        return self.build_from_packed(baselines, (adjacencies, decays))