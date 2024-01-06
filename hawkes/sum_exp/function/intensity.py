from ...base import Intensities as Base, Events
from .kernel import Kernels
from ..converter import ParamsConverter as PC

class Intensities(Base):
    def __init__(self, params, events: Events, num_exps: int):
        baselines, adjacencies, decays = PC.unpack(params, events.dim, num_exps)
        super().__init__(baselines, Kernels(adjacencies, decays), events)
