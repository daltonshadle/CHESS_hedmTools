#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:32:06 2018

@author: dcp99
"""


#%%
#IMPORTS
import numpy as np
import matplotlib.pyplot as plt

import sys
if sys.version_info[0] < 3:
    from hexrd.xrd import rotations as rot
else:
    from hexrd import rotations as rot
import scipy.io as sio    

#KEYS
GRAIN_MAP = 'grain_map'
X_COORD = 'Xs'
Y_COORD = 'Ys'
Z_COORD = 'Zs'
ORI_LIST = 'ori_list'
OLD_IDS = 'old_ids'
NEW_IDS = 'new_ids'

#EXTRA FUNCTIONS
def calc_coord(gm_shape):
    # Create meshed grid for data points [0,1] (centers of voxels)
    n_x = gm_shape[0]
    n_y = gm_shape[1]
    n_z = gm_shape[2]
    
    x = np.linspace(1/(2*n_x), 1-1/(2*n_x), num=n_x, endpoint=True)
    y = np.linspace(1/(2*n_y), 1-1/(2*n_y), num=n_y, endpoint=True)
    z = np.linspace(1/(2*n_z), 1-1/(2*n_z), num=n_z, endpoint=True)
    
    [X,Y,Z] = np.meshgrid(x,y,z)
    
    return [X, Y, Z]

def reorder_ids(gm):
    old_ids = np.unique(gm)
    
    ret_gm = np.copy(gm)
    
    for i in range(old_ids.size):
        ret_gm[gm == old_ids[i]] = i + 1
    
    new_ids = np.arange(old_ids.size) + 1
    
    return ret_gm, new_ids.astype(int), old_ids.astype(int)

#%%
# NOTES: 
#  - Make sure data is modified to be ORTHOGONAL TO THE ARRAY from sample coord sys
#  - The example_HEDM_map file is saved in the order ['x_coord', 'z_coord', 'grain_id_map', 'y_coord']
#  - Any file saved with this program will have the order ['grain_id_map', 'x_coord', 'y_coord', 'z_coord']
#  - HAVE_COORD = True if coordinates are attached, False if coordinates need to be calculated
#  - EXAMPLE = True if using example_HEDM_map
#  - SAVE_NPZ = True if user wants to save .npz of data (for instance, if coordinates are calculated)

#USER INPUT
MY_PATH = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/neper/'
FILENAME = 'total_nf_with_ff_buffer_map.npz'
SAVENAME = 'ss718_total_nf_with_ff_tesr'
voxel_spacing = 0.005

#USER OPTIONS
HAVE_COORD = False
EXAMPLE = False
SAVE_NPZ = True
DEBUG = False
HAVE_ORI = True

#LOAD DATA
data=np.load(open(MY_PATH + FILENAME,'rb'))  

if HAVE_COORD:
    if EXAMPLE:
        grain_map = data['example_grain_map']
        Xs = data['example_Xs']
        Ys = data['example_Ys']
        Zs = data['example_Zs']
    else:
        grain_map = data[GRAIN_MAP]
        Xs = data[X_COORD]
        Ys = data[Y_COORD]
        Zs = data[Z_COORD] 
else:
    grain_map = data[GRAIN_MAP]
    #grain_map = grain_map[20:70, 85:115, 85:115] # DJS added for debug
    grain_map, new_ids, old_ids = reorder_ids(grain_map)
    if HAVE_ORI:
        new_exp_maps = data[ORI_LIST][old_ids, :]
    [Xs, Ys, Zs] = calc_coord(grain_map.shape)

# check shape of grain map, get list of grain id with removed 0
gm_list = np.unique(grain_map)
gm_list = np.trim_zeros(gm_list)
gm_shape = grain_map.shape
print('Grain map shape: %i, %i, %i' %(gm_shape[0], gm_shape[1], gm_shape[2]))
IS_CUBE = (gm_shape[0] == gm_shape[1] == gm_shape[2])

#%%
#CREATE ASSEMBLED DATA -- LIST OF [VOXEL COORDINATES (X,Y,Z),GRAIN ID]
coordinate_list=np.vstack((Xs.ravel(),Ys.ravel(),Zs.ravel()))
assembled_data=np.hstack((coordinate_list.T,np.atleast_2d(grain_map.ravel()).T))


#%%
print('Preparing strings...')
np.set_printoptions(threshold=np.inf)
l1  = '***tesr'
l2  = ' **format'
l3  = '   2.0 ascii'
l4  = ' **general'
l5  = '   3'
# l6  = '   ' + str(grain_map.shape[1]) + ' ' + str(grain_map.shape[0])  + ' ' + str(grain_map.shape[2]) 
l6  = '   ' + str(gm_shape[2]) + ' ' + str(gm_shape[1])  + ' ' + str(gm_shape[0]) 
l7  = '   ' + str(voxel_spacing) + ' ' + str(voxel_spacing) + ' ' + str(voxel_spacing)
l8  = ' **cell';
l9  = '   ' + str(len(gm_list))
l10 = '  *id';
# l11 = '   ' + str(np.arange(1,np.max(grain_map)+1).astype('int').T)[1:-1]
l11 = '   ' + str(gm_list.astype('int').T)[1:-1]
l12 = ' **data'
#l13 = '   ' + str(assembled_data[:,3].astype('int'))[1:-1]
l14 = '***end'


#%%
print('Writing to tesr...')
output = open('%s.tesr'%(SAVENAME),'w');
output.write('%s\n' % l1)
output.write('%s\n' % l2)
output.write('%s\n' % l3)
output.write('%s\n' % l4)
output.write('%s\n' % l5)
output.write('%s\n' % l6)
output.write('%s\n' % l7)
output.write('%s\n' % l8)
output.write('%s\n' % l9)
output.write('%s\n' % l10)
output.write('%s\n' % l11)
output.write('%s\n' % l12)
output.write('   ')
np.savetxt(output,np.atleast_2d(assembled_data[:,3]).T,fmt='%d')
#output.write('%s\n' % l13)
output.write('%s\n' % l14)

output.close()

if not IS_CUBE:
    print('NOTE: THIS GRAIN MAP DOES NOT HAVE A CUBE SHAPE')
print('NOTE: DOMAIN FOR THIS MAP IS NOW')
print('   FROM: -domain "cube(0.5,0.5,0.5)"')
print('   TO:   -domain "cube(%.3f,%.3f,%.3f)"' %(gm_shape[2]*voxel_spacing, gm_shape[1]*voxel_spacing, gm_shape[0]*voxel_spacing))

if SAVE_NPZ:
    print('Writing to .npz...')
    if HAVE_ORI:
        np.savez(SAVENAME+'.npz', GRAIN_MAP=grain_map, X_COORD=Xs, Y_COORD=Ys, Z_COORD=Zs, ORI_LIST=new_exp_maps, OLD_IDS=old_ids, NEW_IDS=new_ids)
    else:
        np.savez(SAVENAME+'.npz', GRAIN_MAP=grain_map, X_COORD=Xs, Y_COORD=Ys, Z_COORD=Zs, OLD_IDS=old_ids, NEW_IDS=new_ids)
    
if DEBUG:
    my_n = 25
    print(grain_map[0:my_n, 0, 0])
    print(grain_map[0, 0:my_n, 0])
    print(grain_map[0, 0, 0:my_n])
    print(grain_map.ravel()[0:my_n])



print('Done!')
