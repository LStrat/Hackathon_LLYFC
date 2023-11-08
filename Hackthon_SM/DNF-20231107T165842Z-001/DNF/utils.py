# Copyright (C) 2023 Priya Kannan, fortiss - Neuromorphic Computing group
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

'''
Utils helps to plot an animation of the output 
plot_out - plots output alone
plot_combined - plots the output overlayyed on top of the input
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def plot_out(edge: np.ndarray, name: str) -> None:

        edge = edge.astype(float)
        edge = np.transpose(edge,(0,2,1))
        num_time_steps = np.size(edge, axis=0)

        fig, ax = plt.subplots()
        ims = []
        for i in range(num_time_steps):

                im = ax.imshow(edge[i,:,:], animated=True)
                if i == 0:
                        ax.imshow(edge[i,:,:])
                ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
        ani.save(name+".gif")
        return ani
