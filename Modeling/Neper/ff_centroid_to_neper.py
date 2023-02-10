#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:32:06 2018

@author: djs522
"""


#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from hexrd.xrd import rotations as rot
from hexrd import rotations as rot
import scipy.io as sio    

#%% FUNCTIONS
def quat2rod(quat):
    """
    % quat2rod - Rodrigues parameterization from quaternion.
    %
    %   USAGE:
    %
    %   rod = quat2rod(quat)
    %
    %   INPUT:
    %
    %   quat is n x 3,
    %        an array whose columns are quaternion paramters;
    %        it is assumed that there are no binary rotations
    %        (through 180 degrees) represented in the input list
    %
    %   OUTPUT:
    %
    %  rod is n x 3,
    %      an array whose columns form the Rodrigues parameterization
    %      of the same rotations as quat
    %
    """
    return np.true_divide(quat[:, 1:4], np.tile(quat[:, 0], (3, 1)).T)

#%% USER INPUT
IN_PATH = '/home/djs522/Downloads/'
FILENAME = 'Ti7al-4a1-4-ff-1100grains.out'
OUT_PATH = '/home/djs522/Downloads/temp/'
SAVENAME = 'cTi_pos.npos'
ORI_SAVENAME = 'cTi_rod.txt'
ORI_SYM = 'hexagonal'
SCAN_BOUNDS = [[-0.5, 0.5], [-0.2, 0.4], [-0.5, 0.5]]
SCAN_BOUNDS = [[-1.5, 1.0], [-0.3, 0.4], [-1.5, 1.0]]
SCAN_DIMEN = [np.round(np.abs(SCAN_BOUNDS[0][0] - SCAN_BOUNDS[0][1]), decimals=4),
              np.round(np.abs(SCAN_BOUNDS[1][0] - SCAN_BOUNDS[1][1]), decimals=4),
              np.round(np.abs(SCAN_BOUNDS[2][0] - SCAN_BOUNDS[2][1]), decimals=4)]
COMP_THRESH = 0.75
CHI2_THRESH = 1e0
VOXEL_SIZE = 0.005

#%% LOAD DATA AND TREHSOLD GRAINS

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


#%% PLOT CENTER OF MASS POSITIONS OF RAW DATA AND THRESHOLD GRAINS

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(good_grain_mat[:, 6], good_grain_mat[:, 7], good_grain_mat[:, 8])
ax.set_xlabel('X (mm)')
ax.set_ylabel('Z (mm)')
ax.set_zlabel('Y (mm)')

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(good_pos_grain_mat[:, 6], good_pos_grain_mat[:, 7], good_pos_grain_mat[:, 8])
ax.set_xlabel('X (mm)')
ax.set_ylabel('Z (mm)')
ax.set_zlabel('Y (mm)')
plt.show()

#%% SAVE POSITIONS AND ORIENTATIONS, WRITE NEPER COMMAND

np.savetxt(OUT_PATH + SAVENAME, good_pos_grain_mat[:, 6:9])


exp_maps = good_pos_grain_mat[:, 3:6]
quat = rot.quatOfExpMap(exp_maps.T).T
rod = quat2rod(quat)
np.savetxt(OUT_PATH + ORI_SAVENAME, rod)

print('NOTE: UPDATE THESE COMMANDS IN THE NEPER TESSLATION SCRIPT')
print('   NUMBER:   -n "%i"' %(good_pos_grain_mat.shape[0]))
print('   DOMAIN:   -domain "cube(%.3f,%.3f,%.3f)"' %(SCAN_DIMEN[0], SCAN_DIMEN[1], SCAN_DIMEN[2]))
print('   POS_LOAD:   -loadpoint "file("%s"):dim"' %(SAVENAME))
print('   TESR_SIZE:   -tesrsize "%i:%i:%i"' %(SCAN_DIMEN[0] / VOXEL_SIZE, SCAN_DIMEN[1] / VOXEL_SIZE, SCAN_DIMEN[2] / VOXEL_SIZE))

print('neper -T -n %i -dim 3 -domain "cube(%.3f,%.3f,%.3f)" -loadpoint "msfile(%s)" \
-ori "file(%s [,des=rodrigues:passive])" -oridescriptor rodrigues:passive -oricrysym %s \
-reg 1 -mloop 10 -morpho voronoi -morphooptistop itermax=5000 -format tess,tesr \
-tesrsize "%i:%i:%i" -tesrformat "ascii" -o $MICRO_FN_OUT' 
      %(good_pos_grain_mat.shape[0],
        SCAN_DIMEN[0], 
        SCAN_DIMEN[1], 
        SCAN_DIMEN[2],
        OUT_PATH+SAVENAME,
        OUT_PATH + ORI_SAVENAME,
        ORI_SYM,
        SCAN_DIMEN[0] / VOXEL_SIZE, 
        SCAN_DIMEN[1] / VOXEL_SIZE, 
        SCAN_DIMEN[2] / VOXEL_SIZE))

