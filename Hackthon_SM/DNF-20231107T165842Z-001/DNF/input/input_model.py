from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort
from input.input_process import InputReader
import numpy as np

def read_events():
    file_name='2.aedat4'
    path ='data/'+file_name
    with AedatFile(path) as f:
            events = np.hstack([packet for packet in f['events'].numpy()])

    x = events['x']
    y = events['y']
    p = events['polarity']
    t = events['timestamp']
    data_time = t[-1]-t[0]
    print("The total time of the recording is {} us".format(data_time))
    data = np.stack((x,y,t,p))
    data = np.array(data)
    data = np.transpose(data)
    

    dt = np.dtype([("x","int"),("y","int"),("t","int"),("p","int")])
    data = rfn.unstructured_to_structured(data, dt)

    transform = tonic.transforms.ToFrame(
        sensor_size=(640,480,2),
        time_window=10000,
    )

    frames = transform(data)

    # Uncomment if needed to plot the input
    # animation = tonic.utils.plot_animation(frames=frames, file_name=file_name)

    event_bin = np.logical_or(frames[:,0,:,:],frames[:,1,:,:])
    # event_bin = frames[:,1,:,:]

    max_pool_data = skimage.measure.block_reduce(event_bin, (1,3,4), np.mean)
    max_pool_data = skimage.measure.block_reduce(max_pool_data, (1,2,2), np.max)
    # np.save("ori_line_edge_1ms", max_pool_data)
    print(max_pool_data.shape)
    return max_pool_data


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
