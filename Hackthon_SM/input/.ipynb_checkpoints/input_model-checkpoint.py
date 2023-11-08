from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort
from input.input_process import InputReader
import numpy as np
from data_laoder import read_events
@implements(proc=InputReader, protocol=LoihiProtocol)
@requires(CPU)
class InputReaderProcessModel(PyLoihiProcessModel):
    flat_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    num_steps: int = LavaPyType(int, int, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.run_id = 0
        self.input = read_events()
        # print(self.input.shape)
        self.in_data = self.input[self.run_id,:,:]
        self.out_data = np.reshape(self.in_data, (6400,))
        # print(self.in_data.shape)


    def post_guard(self):
        """Guard function for PostManagement phase. 
        """
        return True

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
        print("Iteration {}".format(self.run_id))
        self.in_data = self.input[self.run_id,:,:]
        self.out_data = np.reshape(self.in_data, (6400,))
        self.run_id += 1

    def run_spk(self) -> None:
        self.flat_out.send(self.out_data)