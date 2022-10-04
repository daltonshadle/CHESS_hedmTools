#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:01:28 2018

@author: chess_f2
"""

import matplotlib.pyplot as plt
import numpy as np

from hexrd.xrd import rotations
from hexrd import instrument

#%%-------------------- Reading grains.out --------------------------------#

fname='/nfs/chess/aux/cycles/2018-1/f2/hurley-698-3/ff_data_processing/conccube3/conccube3_11/grains.out'
tmp_data_0=np.loadtxt(fname)

#--------------------

grain_id=tmp_data_0[:,0]
comp_0=tmp_data_0[:,1]
chi_0=tmp_data_0[:,2]
strain_0=tmp_data_0[:,15:]
ori_0=tmp_data_0[:,3:6]
pos_0=tmp_data_0[:,6:9]
v_strain_0=np.sum(strain_0[:,0:3],axis=1)/3.

strain_sample = np.loadtxt(fname, skiprows=1, usecols=(15,20,19,20,16,18,19,18,17))

#%%------------------Center Data to Beam #FIRST TIME ONLY ONCE CORRECTED------------------------------------#

pos_0ri=pos_0
pos_0[:,1]-=(np.amax(pos_0ri[:,1])+np.amin(pos_0ri[:,1]))/2

#%%---------- Correcting for vertical variation in beam energy ------------#

good_grains=np.where(np.logical_and(comp_0>0.7,chi_0<0.01))[0] #Grains selected based on good completeness and Chi2 values
good_grain_pos=np.where(np.abs(pos_0[good_grains,1])<0.6)[0] #Grain selected based on positions close to 0

x = pos_0[good_grains[good_grain_pos],1]
y = v_strain_0[good_grains[good_grain_pos]]
p =  np.polyfit(x, y, 1) #polynomial fit of gradient p[0] will be the slope p[1] the intercept

#plot of old values, polyfit
fig, ax = plt.subplots()
ax.plot(x,y,'gx', label='old')
ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),'g-',label='polyfit')
plt.xlabel ('t_vec_c [y]')
plt.ylabel ('avg_vol_strain')

for i in range (0, len(good_grain_pos)):
        plt.text(x[i]+.015,y[i],good_grains[good_grain_pos[i]])

#updating v_strain_0 to corrected values
v_strain_0 = v_strain_0-p[0]*pos_0[:,1]-p[1]

#add corrected values to plot
ax.plot(pos_0[good_grains[good_grain_pos],1],v_strain_0[good_grains[good_grain_pos]],'bo', label='corrected')
legend = ax.legend(loc='lower right')

#%%------------------Re-write Grains.out with Center correction -------------#

grain_params = np.hstack([ori_0, pos_0, tmp_data_0[:,9:15]])

new_dir = '/nfs/chess/user/ken38/Ti7_project/ff_data_all/new-build/ti7-05-new/all_grains_initial/'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

gw = instrument.GrainDataWriter(os.path.join(new_dir + 'grains.out'))
grain_params_list = []
for i in range(0,len(grain_id)):
    grain_params = np.hstack([ori_0[i], pos_0[i], tmp_data_0[i,9:15]])
    gw.dump_grain(i, comp_0[i], chi_0[i], grain_params)
    grain_params_list.append(grain_params)
gw.close()
