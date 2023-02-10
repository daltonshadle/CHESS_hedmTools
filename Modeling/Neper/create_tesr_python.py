#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:32:06 2018

@author: dcp99
"""


# *****************************************************************************
#%% IMPORTS
# *****************************************************************************
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
if sys.version_info[0] < 3:
    from hexrd.xrd import rotations as rot
    from hexrd.grainmap import nfutil
    from hexrd.grainmap import vtkutil
    from hexrd.xrd import symmetry as sym
else:
    from hexrd import rotations as rot
import scipy.io as sio    
import cc3d

# *****************************************************************************
#%% CONSTANTS
# *****************************************************************************
GRAIN_MAP = 'grain_map'
GRAIN_MAP_ORI = 'grain_map_ori'
CONF_MAP = 'confidence_map'
X_COORD = 'Xs'
Y_COORD = 'Ys'
Z_COORD = 'Zs'
ORI_LIST = 'ori_list'
OLD_IDS = 'old_ids'
NEW_IDS = 'new_ids'

# *****************************************************************************
#%% FUNCTIONS
# *****************************************************************************
def calc_coord(gm_shape):
    # Create meshed grid for data points [0,1] (centers of voxels)
    n_x = gm_shape[1]
    n_y = gm_shape[0]
    n_z = gm_shape[2]
    
    x = np.linspace(1/(2*n_x), 1-1/(2*n_x), num=n_x, endpoint=True)
    y = np.linspace(1/(2*n_y), 1-1/(2*n_y), num=n_y, endpoint=True)
    z = np.linspace(1/(2*n_z), 1-1/(2*n_z), num=n_z, endpoint=True)
    
    [X,Y,Z] = np.meshgrid(x,y,z,indexing='ij')
    
    return [X, Y, Z]

def reorder_ids(gm, voxel_threshold=0, do_cc=False, conf_map=None, connectivity=18):
    
    old_ids = np.unique(gm)
    ret_gm = np.zeros(gm.shape)
    new_old_ids = []
    old_grain_size_distrib = []
    new_grain_size_distrib = []
    
    
    if conf_map is not None:
        gm[~conf_map] = -1
    
    j = 1
    if do_cc:
        #connectivity = 18 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
        labels_out = cc3d.connected_components(gm.astype(int), connectivity=connectivity)
        labels_out += 1
        labels_out[gm == -1] = 0
        N = np.max(labels_out)
        print(N)
        
        new_old_ids = []
        uni_label, label_count = np.unique(labels_out, return_counts=True)
        uni_label = uni_label[1:]
        label_count = label_count[1:]
        
        print(np.sum(label_count > voxel_threshold))
        
        for i, label in enumerate(uni_label):
            if label <= 0:
                continue
            else:
                old_grain_size_distrib.append(label_count[i])
                
                if label_count[i] >= voxel_threshold:
                    ind = (labels_out == label)
                
                    ret_gm[ind] = j
                    
                    new_grain_size_distrib.append(np.sum(ind))
                    temp_old_id = np.unique(gm[ind])
                    if temp_old_id.size == 1:
                        new_old_ids.append(temp_old_id)
                    else:
                        raise Exception(temp_old_id)
                    j = j+1
        
        
    else:
        for i in range(old_ids.size):
            ind = (gm == old_ids[i])
            old_grain_size_distrib.append(np.sum(ind))
            
            if np.sum(ind) >= voxel_threshold:
                ret_gm[ind] = j
                new_old_ids.append(old_ids[i])
                new_grain_size_distrib.append(np.sum(ind))
                j = j+1
            else:
                ret_gm[ind] = 0 # cell_id = 0 means ignore/fill-in in Neper
                
        
    new_ids = np.arange(j-1) + 1
    
    return [ret_gm, new_ids.astype(int), np.array(new_old_ids).astype(int), 
            np.array(new_grain_size_distrib).astype(int), np.array(old_grain_size_distrib).astype(int)]


def volume_fraction_conf(grain_map, conf_map):
    old_ids = np.unique(grain_map)
    vol_frac_conf = np.zeros(old_ids.shape)
    
    for i in range(old_ids.size):
        ind = (grain_map == old_ids[i])
        vol_frac_conf[i] = np.sum(conf_map[ind]) / float(np.sum(ind))
        
    return vol_frac_conf
    
    


#%%
# NOTES: 
#  - Make sure data is modified to be ORTHOGONAL TO THE ARRAY from sample coord sys
#  - The example_HEDM_map file is saved in the order ['x_coord', 'z_coord', 'grain_id_map', 'y_coord']
#  - Any file saved with this program will have the order ['grain_id_map', 'x_coord', 'y_coord', 'z_coord']
#  - HAVE_COORD = True if coordinates are attached, False if coordinates need to be calculated
#  - EXAMPLE = True if using example_HEDM_map
#  - SAVE_NPZ = True if user wants to save .npz of data (for instance, if coordinates are calculated)

#USER INPUT
voxel_spacing = 0.003 # mm
conf_thresh = 0.0 # minimum confidence / completeness threshold for voxels to use in grain map
voxel_threshold = 0 # minimum number of voxels to be considered a grain
voxel_cc3d = True # bool flag True = do connected components analysis to segment grains that have the same grain ID but are not connected, False = ignore
connectivity = 26 # extra variable for cc3d analysis to describe connectivity type, can be 6, 18, 26

REORDER_GRAIN_ID = True # bool flag True = reorder grain ids in grain map to sequentially range from 0-n needed for Mech-Suite, False = ignore
HAVE_COORD = True # bool flag True = grain map has coordinates data, False = grain map does not have coordinates, need to generate in this script
HAVE_ORI = True # bool flag True = grain map has orientation data, False = grain map does not have orientations
EXAMPLE = False # bool flag True = use example data, False = use user input
SAVE_NPZ = True # bool flag True = save new grain map used for neper input as npz, False = ignore
DEBUG = False # bool flag True = show some debug statements, False = ignore


MY_PATH = '/media/djs522/djs522_nov2020/chess_2020_11/djs522_nov2020_in718/dp718/'
FILENAME = 'final_dp718_total_3micron_grain_map_data.npz'
SAVENAME = 'final_dp718_total_3micron'

if voxel_cc3d:
    UPDATED_EXT = '_conf%i_voxel%i_cc3d%i_tesr' %(int(conf_thresh*100), voxel_threshold, connectivity)
else:
    UPDATED_EXT = '_conf%i_voxel%i_tesr' %(int(conf_thresh*100), voxel_threshold)
SAVENAME = SAVENAME + UPDATED_EXT

#LOAD DATA
data=np.load(open(MY_PATH + FILENAME,'rb'))  

if EXAMPLE:
    grain_map = data['example_grain_map']  
    Xs = data['example_Xs']
    Ys = data['example_Ys']
    Zs = data['example_Zs']
else:
    grain_map = data[GRAIN_MAP]
    conf_map = data[CONF_MAP]
    if HAVE_COORD:
        Xs = data[X_COORD]
        Ys = data[Y_COORD]
        Zs = data[Z_COORD]
    else:
        [Xs, Ys, Zs] = calc_coord(grain_map.shape)

#vol_frac_conf_orig = volume_fraction_conf(grain_map, conf_map) 

#%%

if REORDER_GRAIN_ID:
    grain_map_ori = np.copy(grain_map)
    grain_map_id, new_ids, old_ids, new_gsd, old_gsd = reorder_ids(grain_map, voxel_threshold=voxel_threshold, 
                                          do_cc=voxel_cc3d, conf_map=(conf_map >= conf_thresh), connectivity=connectivity)

#vol_frac_conf_new = volume_fraction_conf(grain_map, conf_map) 


#%%
if HAVE_ORI:
    old_exp_maps = data[ORI_LIST]
    old_quat = rot.quatOfExpMap(old_exp_maps.T).T
    '''
    if sys.version_info[0] < 3:
        old_quat = sym.toFundamentalRegion(old_quat).T
    else:
        old_quat = rot.toFundamentalRegion(old_quat).T
    '''
    old_rod = rot.quat2rod(old_quat)
    
    
    new_exp_maps = old_exp_maps[old_ids.flatten(), :]
    new_quat = old_quat[old_ids.flatten(), :]
    new_rod = old_rod[old_ids.flatten(), :]

#%%

# check shape of grain map, get list of grain id with removed 0
gm_list = np.unique(grain_map_id)
gm_list = np.trim_zeros(gm_list)
gm_shape = grain_map_id.shape
print('Grain map shape: %i, %i, %i' %(gm_shape[0], gm_shape[1], gm_shape[2]))
IS_CUBE = (gm_shape[0] == gm_shape[1] == gm_shape[2])
print('   TO:   -domain "cube(%.3f,%.3f,%.3f)"' %(gm_shape[0]*voxel_spacing, gm_shape[1]*voxel_spacing, gm_shape[2]*voxel_spacing))

#%%
#CREATE ASSEMBLED DATA -- LIST OF [VOXEL COORDINATES (X,Y,Z),GRAIN ID]
coordinate_list=np.vstack((Xs.ravel(),Ys.ravel(),Zs.ravel()))

#temp = np.copy(grain_map)
#assembled_data=np.hstack((coordinate_list.T,np.atleast_2d(temp.transpose(2, 0, 1).ravel()).T))
assembled_data=np.hstack((coordinate_list.T,np.atleast_2d(grain_map_id.ravel()).T))

#%%
print('Preparing strings...')
np.set_printoptions(threshold=np.inf)
l1  = '***tesr'
l2  = ' **format'
l3  = '   2.0 ascii'
l4  = ' **general'
l5  = '   3'
# l6  = '   ' + str(grain_map.shape[1]) + ' ' + str(grain_map.shape[0])  + ' ' + str(grain_map.shape[2]) 
l6  = '   ' + str(gm_shape[0]) + ' ' + str(gm_shape[1])  + ' ' + str(gm_shape[2]) 
l7  = '   ' + str(voxel_spacing) + ' ' + str(voxel_spacing) + ' ' + str(voxel_spacing)
#l7_5  = '  *hasvoid 1'
l8  = ' **cell';
l9  = '   ' + str(len(gm_list))
l10 = '  *id';
# l11 = '   ' + str(np.arange(1,np.max(grain_map)+1).astype('int').T)[1:-1]
l11 = '   ' + str(gm_list.astype('int').T)[1:-1] # [1:-1] to remove brackets
l11_25 = '  *ori \n   rodrigues:passive'
l11_5 = '  *crysym \n   cubic'
l12 = ' **data'
#l13 = '   ' + str(assembled_data[:,3].astype('int'))[1:-1]
l14 = '***end'




#%%
print('Writing to tesr...')
output = open('%s.tesr'%(MY_PATH + SAVENAME),'w');
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
if HAVE_ORI:
    output.write('%s\n' % l11_25)
    np.savetxt(output,new_rod,fmt='%0.4f')
    output.write('%s\n' % l11_5)
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
print('   TO:   -domain "cube(%.3f,%.3f,%.3f)"' %(gm_shape[0]*voxel_spacing, gm_shape[1]*voxel_spacing, gm_shape[2]*voxel_spacing))


#%%

#bunge_eulers = rot.exp_map_2_bunge_euler(new_exp_maps.T) * 180 / np.pi # degrees

if SAVE_NPZ:
    print('Writing to .npz...')
    if HAVE_ORI:
        #np.savez(MY_PATH + SAVENAME+'.npz', GRAIN_MAP=grain_map, CONFIDENCE_MAP=conf_map, X_COORD=Xs, Y_COORD=Ys, Z_COORD=Zs, RODRIGUES_VECTORS=new_rod, BUNGE_EULER_ANGLES=bunge_eulers, OLD_IDS=old_ids, NEW_IDS=new_ids)
        #nfutil.save_nf_data(MY_PATH,SAVENAME,grain_map,conf_map,Xs,Ys,Zs,new_exp_maps,id_remap=None,misorientation_map=None)
        #vtkutil.output_grain_map_vtk(MY_PATH,[SAVENAME],SAVENAME+'_vtk',voxel_spacing,top_down=True)
        np.savez(MY_PATH + SAVENAME + '.npz', 
                 GRAIN_MAP=grain_map_id, CONFIDENCE_MAP=conf_map, GRAIN_MAP_ORI=grain_map_ori,
                 X_COORD=Xs, Y_COORD=Ys, Z_COORD=Zs, 
                 RODRIGUES_VECTORS=old_rod, EXP_MAPS=old_exp_maps)
        
    else:
        np.savez(MY_PATH + SAVENAME+'.npz', GRAIN_MAP=grain_map, X_COORD=Xs, Y_COORD=Ys, Z_COORD=Zs, OLD_IDS=old_ids, NEW_IDS=new_ids)


print('Done!')


#%%

'''

from NF.nf_vtk_with_ipf import add_ipf_to_nf_vtk

path_to_nf_npz = MY_PATH + SAVENAME + '.npz'
add_ipf_to_nf_vtk(path_to_nf_npz, ipf_axis=np.array([0.0, 1.0, 0.0]),
                      GRAIN_MAP_ID_KEY='GRAIN_MAP', ORIENTATION_KEY='EXP_MAPS',
                      GRAIN_MAP_ORI_KEY='GRAIN_MAP_ORI', CONFIDENCE_KEY='CONFIDENCE_MAP',
                      X_KEY='X_COORD', Y_KEY='Y_COORD', Z_KEY='Z_COORD')

path_to_nf_npz = '/media/djs522/djs522_nov2020/chess_2020_11/djs522_nov2020_in718/ss718/final_ss718_total_3micron_xyz_grain_map_data.npz'
# add_ipf_to_nf_vtk(path_to_nf_npz, ipf_axis=np.array([0.0, 1.0, 0.0]),
#                       GRAIN_MAP_ID_KEY='grain_map', ORIENTATION_KEY='ori_list',
#                       GRAIN_MAP_ORI_KEY='grain_map', CONFIDENCE_KEY='confidence_map',
#                       X_KEY='Xs', Y_KEY='Ys', Z_KEY='Zs')





#%%

def calc_nf_centroids(grain_map_id, Xs, Ys, Zs):
    grain_ids = np.unique(grain_map_id)
    centroids = np.zeros([grain_ids.size, 3])
    print(grain_ids.size)
    
    for i, grain_id in enumerate(grain_ids):
        if i % 100 == 0:
            print(i)
        ind = np.where(grain_map_id == grain_id)[0]
        centroids[i, :] = np.hstack([np.mean(Xs[ind]), np.mean(Ys[ind]), np.mean(Zs[ind])])
    return centroids

def run_euc(list_a,list_b):
    return np.array([[ np.linalg.norm(i-j) for j in list_b] for i in list_a])

def matrix_to_list(matrix):
    graph = {}
    for i, node in enumerate(matrix):
        adj = []
        for j, connected in enumerate(node):
            if connected:
                adj.append(j)
        graph[i] = adj
    return graph

def combine_similar_nf_grains(grain_ids, centroids, quats, centroid_thresh=0.05, ori_thresh=3.0):
    print('centroid diff')
    centroid_diff = run_euc(centroids, centroids)
    print('quat diff')
    quats_diff = run_euc(quats, quats)
    
    print('adj')
    similar_adj_mat = (centroid_diff < centroid_thresh) & (quats_diff < ori_thresh)
    
    
    print('graph')
    similar_graph = matrix_to_list(similar_adj_mat)
    
    print(len(similar_graph))
    
    
    
    return similar_graph

grain_ids = np.unique(grain_map_ori)
print('centroids')
centroids = calc_nf_centroids(grain_map_ori, Xs, Ys, Zs)
print('combine')

#%%
sim_graph = combine_similar_nf_grains(grain_ids, centroids, old_quat[grain_ids.astype(int), :], centroid_thresh=0.1, ori_thresh=0.052)

#%%
grain_ids = grain_ids.astype(int)

new_combined_ids = np.zeros(grain_ids.max().astype(int)+1)
for key in sim_graph:
    island = sim_graph[key]
    if len(island) > 1:
        for island_id in island:
            new_combined_ids[grain_ids[island_id]] = grain_ids[island[0]]
    else:
        new_combined_ids[grain_ids[island[0]]] = grain_ids[island[0]]
print(np.unique(new_combined_ids).size)

#%%

reorder_grain_map = np.copy(grain_map_ori)
for i, grain in enumerate(grain_ids):
    reorder_grain_map[reorder_grain_map == grain] = new_combined_ids[grain]


#%%

from NF.nf_vtk_with_ipf import add_ipf_to_nf_vtk


path_to_nf_npz = MY_PATH + SAVENAME + '_100_combined.npz'

np.savez(path_to_nf_npz, 
         GRAIN_MAP=reorder_grain_map, CONFIDENCE_MAP=conf_map, GRAIN_MAP_ORI=reorder_grain_map,
         X_COORD=Xs, Y_COORD=Ys, Z_COORD=Zs, 
         RODRIGUES_VECTORS=old_rod, EXP_MAPS=old_exp_maps)

add_ipf_to_nf_vtk(path_to_nf_npz, ipf_axis=np.array([0.0, 1.0, 0.0]),
                      GRAIN_MAP_ID_KEY='GRAIN_MAP', ORIENTATION_KEY='EXP_MAPS',
                      GRAIN_MAP_ORI_KEY='GRAIN_MAP_ORI', CONFIDENCE_KEY='CONFIDENCE_MAP',
                      X_KEY='X_COORD', Y_KEY='Y_COORD', Z_KEY='Z_COORD')
'''

