#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:35:18 2021

@author: djs522
"""

# *****************************************************************************
#%% IMPORTS
# *****************************************************************************

from pymicro.crystal.microstructure import Orientation

import numpy as np

import hexrd.rotations as hexrd_rot


'''
in grain map to tesr, save original grain map ids to access orientations,
new grain ids can be saved as a grain map for tesr (ids 1-n, no 0),
can evnetually get smart and do a check or have a key to transfer old to new
'''

# *****************************************************************************
#%% FUNCTION
# *****************************************************************************

def add_ipf_to_nf_vtk(path_to_nf_npz, ipf_axis=np.array([0.0, 1.0, 0.0]),
                      GRAIN_MAP_ID_KEY='grain_map', ORIENTATION_KEY='ori_list',
                      GRAIN_MAP_ORI_KEY='grain_map', CONFIDENCE_KEY='confidence_map',
                      X_KEY='Xs', Y_KEY='Ys', Z_KEY='Zs'):
    '''

    Parameters
    ----------
    path_to_nf_npz : TYPE
        DESCRIPTION.
    ipf_axis : TYPE, optional
        DESCRIPTION. The default is np.array([0.0, 1.0, 0.0]).
    GRAIN_MAP_ID_KEY : TYPE, optional
        DESCRIPTION. The default is 'grain_map'.
    ORIENTATION_KEY : TYPE, optional
        DESCRIPTION. The default is 'ori_list'.
    GRAIN_MAP_ORI_KEY : TYPE, optional
        DESCRIPTION. The default is 'grain_map'.
    CONFIDENCE_KEY : TYPE, optional
        DESCRIPTION. The default is 'confidence_map'.
    X_KEY : TYPE, optional
        DESCRIPTION. The default is 'Xs'.
    Y_KEY : TYPE, optional
        DESCRIPTION. The default is 'Ys'.
    Z_KEY : TYPE, optional
        DESCRIPTION. The default is 'Zs'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.
    
    Notes
    -----
    Assumes position indexing [X, Y, Z] or meshgrid(indexing='ij')

    '''
    # prepare output stem
    output_stem = path_to_nf_npz.replace('.npz', '_with_ipf.vtk')
    
    # get data, find path stem
    grain_map_data = np.load(path_to_nf_npz)
    exp_maps = grain_map_data[ORIENTATION_KEY]
    grain_map_ori = grain_map_data[GRAIN_MAP_ORI_KEY]
    grain_map_shape = grain_map_ori.shape
    
    # extract grain ids, used to index orientations from ori_list
    grain_ori_ids = np.unique(grain_map_ori)
    
    # check if GRAIN_MAP_ID_KEY and GRAIN_MAP_ORI_KEY are the same
    diff_grain_map_key_check = GRAIN_MAP_ID_KEY != GRAIN_MAP_ORI_KEY
    
    # check if we're going to have an index error max_id_check == True is index error
    max_id_check = (grain_ori_ids.max() >= exp_maps.shape[0])
    if max_id_check:
        raise ValueError('Grain IDs cannot be used to index orientations!')
    
    # get ipf colors for each orientation using pymicro
    rot_mats = hexrd_rot.rotMatOfExpMap_opt(exp_maps.T)
    ipf_colors = []

    for ori in range(rot_mats.shape[0]):
        temp_ori = Orientation(rot_mats[ori, :, :])
        ipf_colors.append(temp_ori.get_ipf_colour(axis=ipf_axis))
        
    ipf_color_arr = np.array(ipf_colors)

    # prepare data to write to vtk file
    Xslist = grain_map_data[X_KEY].ravel()
    Yslist = grain_map_data[Y_KEY].ravel()
    Zslist = grain_map_data[Z_KEY].ravel() # negative to match Neper for now TODO
    #print(np.vstack([Xslist, Yslist, Zslist]).T)
    #print(Xslist.size)
    
    grainorilist = grain_map_ori.ravel()
    conflist = grain_map_data[CONFIDENCE_KEY].ravel()
    if diff_grain_map_key_check:
        grainidlist = grain_map_data[GRAIN_MAP_ID_KEY].ravel()
        
    num_x = grain_map_shape[0]
    num_y = grain_map_shape[1]
    num_z = grain_map_shape[2]
    #print(grain_map_ori.shape)
    
    num_pts = Xslist.shape[0]
    num_cells = (num_x-1)*(num_y-1)*(num_z-1)
    
    print('Writing VTK data...')
    # VTK Dump 
    with open(output_stem, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('grainmap Data\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write('POINTS %d double\n' % (num_pts))
    
        for i in np.arange(num_pts):
            f.write('%e %e %e \n' %(Xslist[i], Yslist[i], Zslist[i]))
    
        scale2 = num_z*num_y
        scale1 = num_z
        #print(scale1, scale2)
        
        f.write('CELLS %d %d\n' % (num_cells, 9*num_cells))   
        # need to check indexing for these guys, old [Y,X,Z] vs new [X,Y,Z] 
        for k in np.arange(num_x-1):
            for j in np.arange(num_y-1):
                for i in np.arange(num_z-1):
                    base=scale2*k+scale1*j+i  
                    '''
                    p1=base
                    p2=base+1
                    p3=base+1+scale1
                    p4=base+scale1
                    p5=base+scale2
                    p6=base+scale2+1
                    p7=base+scale2+scale1+1
                    p8=base+scale2+scale1
                    '''
                    
                    p1 = base
                    p2 = base+scale2
                    p3 = base+scale2+scale1
                    p4 = base+scale1
                    p5 = base + 1
                    p6 = base+scale2+1
                    p7 = base+scale2+scale1+1
                    p8 = base+1+scale1
                    
                    
                    
                    if p7 >= Xslist.size:
                        print("INDEX ERROR: %i -- %i, %i, %i" %(p7, i, j, k))
                    
                    # if k == 0 and j == 0 and i == 0:
                    #     print(p1,p2,p3,p4,p5,p6,p7,p8)
                    #     l = [p1,p2,p3,p4,p5,p6,p7,p8]
                    #     for t in l:
                    #         print(Xslist[t.astype(int)], Yslist[t.astype(int)], Zslist[t.astype(int)])
                    f.write('8 %d %d %d %d %d %d %d %d \n' %(p1,p2,p3,p4,p5,p6,p7,p8))    
            
            
        f.write('CELL_TYPES %d \n' % (num_cells))    
        for i in np.arange(num_cells):
            f.write('12 \n')  
        
        
        f.write('POINT_DATA %d \n' % (num_pts))
        f.write('SCALARS grain_ori_id int \n')  
        f.write('LOOKUP_TABLE default \n')      
        for i in np.arange(num_pts):
            f.write('%d \n' %(grainorilist[i]))    
        
        if diff_grain_map_key_check:
            f.write('SCALARS grain_new_id int \n')  
            f.write('LOOKUP_TABLE default \n')      
            for i in np.arange(num_pts):
                f.write('%d \n' %(grainidlist[i])) 
        
        f.write('SCALARS ipf_color float 3 \n')  
        f.write('LOOKUP_TABLE default \n')    
        f.write('COLOR_SCALARS ipf_color 3 \n') 
        for i in np.arange(num_pts):
            if grainorilist[i] < 0:
                temp_ipf = [1.0, 1.0, 1.0]
            else:
                temp_ipf = ipf_color_arr[grainorilist[i].astype(int), :]
            f.write('%e %e %e \n' %(temp_ipf[0], temp_ipf[1], temp_ipf[2]))  
    
        f.write('FIELD FieldData 1 \n' )
        f.write('confidence 1 %d float \n' % (num_pts))       
        for i in np.arange(num_pts):
            f.write('%e \n' %(conflist[i]))  





