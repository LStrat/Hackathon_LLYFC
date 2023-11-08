from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import OutPort
import numpy as np


class InputReader(AbstractProcess):
    """reads input file and sends a frame of spikes

    Parameters
    ----------
    shape: tuple
        defines the dimensionality of the generated spikes per timestep
    path: string
        defines the path to the data to be read
    """
    def __init__(self, shape: tuple, num_steps: int) -> None:        
        super().__init__()
        self.flat_out = OutPort(shape=shape)
        self.num_steps = Var(shape=(1,), init=num_steps)
