import numpy as np
from .kernel import PowLawKernel
from .loglik import PowLawKernelLogLik
from ...simulators import ThinningSimulator
from ...optimizers import OptimizerBuilder
from ..intensity_fn import IntensityFunction
from ..dto.estimation import EstimationOutput
from ..dto.simutation import SimulationOutput

class PowLawKernelModel:
    def __init__(self):
        self._is_fitted = False

    def fit(self, events, T, optimizer_settings=None, delta=1):
        """
        args:
            events: list of np.ndarray(多次元) or np.ndarray(1次元)
                各ノードのイベント時刻のリスト
            T: float
                シミュレーションの終了時刻
        """
        # 1次元の場合の対応
        if isinstance(events, np.ndarray):
            events = [events]
        self._is_fitted = True
        dim = len(events)
        search_space = {
            'mu': np.array([[0, 10] for _ in range(dim)]),
            'K': np.array([[[0, 10] for _ in range(dim)] for _ in range(dim)]),
            'p': np.array([[[1, 10] for _ in range(dim)] for _ in range(dim)]),
            'c': np.array([[[1, 10] for _ in range(dim)] for _ in range(dim)]),
        }
        init_params = np.array([0.1, 0.1, 2, 1])
        params_order = ['mu', 'K', 'p', 'c']
        optimizer = OptimizerBuilder(optimizer_settings, dim, search_space, init_params, params_order)()
        params, score = optimizer(PowLawKernelLogLik(events, T))
        if isinstance(params, dict):
            mu = params['mu']
            K = params['K']
            p = params['p']
            c = params['c']
        else:
            mu, K, p, c = params
        kernel = np.array([[PowLawKernel(K[i, j], p[i, j], c[i, j]) for j in range(dim)] for i in range(dim)], dtype=object)
        events = ThinningSimulator(mu, kernel)(T)
        t_vals = np.arange(0, T + delta, delta)
        intensity = IntensityFunction(mu, kernel, events)(t_vals)
        return EstimationOutput(events, T, intensity, params={'mu': mu, 'K': K, 'p': p, 'c': c}, kernel_type='pow_law_kernel', loglik=score)

    def score(self, mu, K, p, c, events, T):
        # 1次元の場合の対応
        if isinstance(mu, (int, float)):
            mu = np.array([mu])
            K = np.array([[K]])
            p = np.array([[p]])
            c = np.array([[c]])
            events = [events]

        return PowLawKernelLogLik(events, T)(mu, K, p, c)

    def simulate(self, mu, K, p, c, T, delta=1):
        # 1次元の場合の対応
        if isinstance(mu, (int, float)):
            mu = np.array([mu])
            K = np.array([[K]])
            p = np.array([[p]])
            c = np.array([[c]])

        if not (K.shape == p.shape == c.shape):
            raise ValueError('パラメーターの次元が不適切です')

        dim = mu.shape[0]
        kernel = np.array([[PowLawKernel(K[i, j], p[i, j], c[i, j]) for j in range(dim)] for i in range(dim)], dtype=object)
        events = ThinningSimulator(mu, kernel)(T)
        t_vals = np.arange(0, T + delta, delta)
        intensity = IntensityFunction(mu, kernel, events)(t_vals)

        return SimulationOutput(events, T, intensity, params={'mu': mu.tolist(), 'K': K.tolist(), 'p': p.tolist(), 'c': c.tolist()}, kernel_type='pow_law_kernel')
