# Copyright (C) 2023 Priya Kannan, fortiss - Neuromorphic Computing group
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

'''
Data loader to preprocess the input data from event camera - downsampling and binning
'''
import numpy as np
from numpy.lib import recfunctions as rfn
import tonic 
from dv import AedatFile
import skimage.measure

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

# read_events()