#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:45:28 2022

@author: djs522
"""

#%%
import os
import numpy as np
import OrientationTools
import SGODAnalysis
import matplotlib.pyplot as plt

#%%
size = 1000
grain_rod = np.random.uniform(low=-0.02, high=0.02, size=(size, 3))
grain_odf = np.ones(size)

#%%
fig, ax = SGODAnalysis.plot_grain_dsgod(grain_rod, grain_odf=grain_odf, scatter_size=50)
plt.show()

#%%
SGODAnalysis.grain_dsgod_to_vtk(vtk_save_dir=os.path.join(os.getcwd(), 'example_dsgod'), grain_rod=grain_rod, grain_odf=grain_odf)

#%%
grain_quat = OrientationTools.rod2quat(grain_rod)
[grain_mis_quat, mis_ang, avg_quat] = SGODAnalysis.calc_misorientation_quats(grain_quat, avg_quat=None, disp_stats=False)

#%%
[sn, gn, kn, s, g, k] = SGODAnalysis.calc_misorient_moments(grain_mis_quat, grain_odf=grain_odf, norm_regularizer=0)
print(sn, gn, kn)

#%%
grain_rod_list = []
for i in range(5):
    grain_rod_list.append(np.random.uniform(low=-0.02, high=0.02, size=(size, 3)))

SGODAnalysis.animate_dsgods_rod(grain_rod_list, grain_odf_list=None, labels_list=None, 
                 interval=1000, save_gif_dir=os.path.join(os.getcwd(), 'example_dsgod_anim.gif'))

