import numpy as np

class Kernel:
    def __init__(self, K: float, p: float, c: float):
        self._K = float(K)
        self._p = float(p)
        self._c = float(c)

    def __call__(self, t):
        return self._K * np.power(t + self._c, -self._p)

def build_kernel_matrix(K: np.ndarray, p: np.ndarray, c: np.ndarray):
    if not (K.shape == p.shape == c.shape):
        raise ValueError('パラメーターの次元が不適切です')
    dim = K.shape[0]
    return np.array([[Kernel(K[i, j], p[i, j], c[i, j]) for j in range(dim)] for i in range(dim)], dtype=object)