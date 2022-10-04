#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:48:29 2018

@author: sriramya nair, kelly nygren
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from hexrd import rotations
from hexrd import instrument

#%%-------------------- Reading grains.out --------------------------------#

fname = sys.argv[0] + 'analysis/combined_grains_c0_0_ungripped.out'
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

#%%---------- Correcting for vertical variation in beam energy ------------#

good_grains=np.where(np.logical_and(np.logical_and(comp_0>0.7,chi_0<1e-2),np.abs(v_strain_0)<0.001))[0] #Grains selected based on good completeness and Chi2 values
good_grain_pos=np.where(np.abs(pos_0[good_grains,1])<0.8)[0] #Grain selected based on positions close to 0

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


#%%------------------Correcting strain for vertical variation in beam ---------------#

strain_sample[:,0] -= (p[0]*pos_0[:,1])
strain_sample[:,4] -= (p[0]*pos_0[:,1])
strain_sample[:,8] -= (p[0]*pos_0[:,1])

#--------- Converting strain from sample frame to crystal frame --------#
num_grains = len(good_grain_pos)

strain_a = np.zeros(num_grains)
strain_b = np.zeros(num_grains)
strain_c = np.zeros(num_grains)

for j in range (0,num_grains):
    i = good_grains[good_grain_pos[j]]
    exp_map_rot = rotations.rotMatOfExpMap_opt(ori_0[i,:]) #from angle-axis to rotation matrix
    strain_sample_grain = np.reshape(strain_sample[i,:], (3,3))
    strain_crystal_grain = np.dot(exp_map_rot.T,np.dot(strain_sample_grain,exp_map_rot))
    strain_a[j] = strain_crystal_grain[0,0]
    strain_b[j] = strain_crystal_grain[1,1]
    strain_c[j] = strain_crystal_grain[2,2]

fig, ax = plt.subplots()
ax.plot(strain_a,'x')

fig, ax = plt.subplots()
ax.plot(strain_b,'x')

fig, ax = plt.subplots()
ax.plot(strain_c,'x')

#%%------------- Correcting for the lattice parameter -----------#

#FOR HEXAGONAL
a0=4.91442513779
c0=5.40708158119

a=(1.+(np.mean(strain_a)+np.mean(strain_b))/2.)*a0
c=(1.+(np.mean(strain_c)))*c0

print(a)
print(c)


#%%------------- Correcting for the lattice parameter -----------#

#FOR CUBIC
a0 = 3.6#original lattice parameter
a=(1.+(np.mean(strain_a)+np.mean(strain_b)+np.mean(strain_c))/3.)*a0

print(a)
