#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:32:06 2018

@author: djs522
"""


#%%
#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from hexrd.xrd import rotations as rot
from hexrd import rotations as rot
import scipy.io as sio    

#USER INPUT
IN_PATH = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/c0_0_gripped/c0_0_gripped_sc32/'
FILENAME = 'grains.out'
OUT_PATH = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/neper/'
SAVENAME = 'ss718_sc32_neper_centroid.npos'
ORI_SAVENAME = 'ss718_sc32_neper_ori.npy'
SCAN_BOUNDS = [[-0.495, 0.495], [-0.13, 0.04], [-0.495, 0.495]]
SCAN_DIMEN = [np.round(np.abs(SCAN_BOUNDS[0][0] - SCAN_BOUNDS[0][1]), decimals=4),
              np.round(np.abs(SCAN_BOUNDS[1][0] - SCAN_BOUNDS[1][1]), decimals=4),
              np.round(np.abs(SCAN_BOUNDS[2][0] - SCAN_BOUNDS[2][1]), decimals=4)]
COMP_THRESH = 0.75
CHI2_THRESH = 1e-2
VOXEL_SIZE = 0.005

#%%

# load grain mat
grain_mat = np.loadtxt(IN_PATH + FILENAME)
print(grain_mat.shape)

# threshold completeness and chi^2 from fitting of grain mat
good_grain_mat = grain_mat[((grain_mat[:, 1] >= COMP_THRESH) & (grain_mat[:, 2] <= CHI2_THRESH)), :]
print(good_grain_mat.shape)

# threshold position of grains in grain mat
xyz = good_grain_mat[:, 6:9]
good_pos_ind = ( (xyz[:, 0] > SCAN_BOUNDS[0][0]) & (xyz[:, 0] < SCAN_BOUNDS[0][1]) 
               & (xyz[:, 1] > SCAN_BOUNDS[1][0]) & (xyz[:, 1] < SCAN_BOUNDS[1][1]) 
               & (xyz[:, 2] > SCAN_BOUNDS[2][0]) & (xyz[:, 2] < SCAN_BOUNDS[2][1]))
good_pos_grain_mat = good_grain_mat[good_pos_ind, :]
print(good_pos_grain_mat.shape)


#%%

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(good_grain_mat[:, 6], good_grain_mat[:, 8], good_grain_mat[:, 7])

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(good_pos_grain_mat[:, 6], good_pos_grain_mat[:, 8], good_pos_grain_mat[:, 7])
plt.show()

#%%

np.savetxt(OUT_PATH + SAVENAME, good_pos_grain_mat[:, [6,8,7]])

np.save(OUT_PATH + ORI_SAVENAME, good_pos_grain_mat[:, 3:6])

print('NOTE: UPDATE THESE COMMANDS IN THE NEPER TESSLATION SCRIPT')
print('   NUMBER:   -n "%i"' %(good_pos_grain_mat.shape[0]))
print('   DOMAIN:   -domain "cube(%.3f,%.3f,%.3f)"' %(SCAN_DIMEN[0], SCAN_DIMEN[2], SCAN_DIMEN[1]))
print('   POS_LOAD:   -loadpoint "file("%s"):dim"' %(SAVENAME))
print('   TESR_SIZE:   -tesrsize "%i:%i:%i"' %(SCAN_DIMEN[0] / VOXEL_SIZE, SCAN_DIMEN[2] / VOXEL_SIZE, SCAN_DIMEN[1] / VOXEL_SIZE))

print('neper -T -n %i -dim 3 -domain "cube(%.3f,%.3f,%.3f)" -loadpoint "file("%s"):dim" \
-reg 1 -mloop 10 -morpho voronoi -morphooptistop itermax=5000 -format tess,tesr \
-tesrsize "%i:%i:%i" -tesrformat "ascii" -o $MICRO_FN_OUT' 
      %(good_pos_grain_mat.shape[0],
        SCAN_DIMEN[0], 
        SCAN_DIMEN[2], 
        SCAN_DIMEN[1],
        SAVENAME,
        SCAN_DIMEN[0] / VOXEL_SIZE, 
        SCAN_DIMEN[2] / VOXEL_SIZE, 
        SCAN_DIMEN[1] / VOXEL_SIZE))

