#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:35:18 2021

@author: djs522
"""

import pymicro

from pymicro.crystal.microstructure import Orientation, Grain, Microstructure

import sys

import time

import numpy as np

import matplotlib.pyplot as plt

import os

import hexrd.rotations as hexrd_rot

from scipy import ndimage


#%%
scan_num = 30
path = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf_analysis/output/'
stem = 'new_ss718_scan%i_out' %(scan_num)
stem = 'ss718_total_stitch_with_greyclose'
output_stem = stem + '_grain_map_data.npz'

# path = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf_analysis/output/packaged_nf_final/'
# output_stem = 'ff_final_dp718_scan18_1.70_5.00_grain_map_data.npz'

grain_map_data = np.load(os.path.join(path, output_stem))
exp_maps = grain_map_data['ori_list']
grain_map = grain_map_data['grain_map']
#grain_ids = grain_map_data['id_remap']

grain_ids = np.unique(grain_map)
grain_ids_to_idx = np.vstack([grain_ids, np.arange(grain_ids.size)]).T #np.arange(grain_ids.size)]).T


#grain_ids_to_idx = np.vstack([grain_ids, np.arange(exp_maps.shape[0])]).T


#%%
rot_mats = hexrd_rot.rotMatOfExpMap_opt(exp_maps.T)

pymicro_obj = Microstructure(name='test')
print(pymicro_obj.get_number_of_grains())


print(rot_mats.shape)

#%%
pymicro_oris = []
ipf_colors = []

for ori in range(rot_mats.shape[0]):
    temp_ori = Orientation(rot_mats[ori, :, :])
    
    pymicro_oris.append(temp_ori)
    ipf_colors.append(temp_ori.get_ipf_colour(axis=np.array([0., 1., 0.])))
    
ipf_color_arr = np.array(ipf_colors)
print(ipf_color_arr.shape)

#%%
data_location = path
data_stems = [stem]
output_stem = data_stems[0]
top_down=True
vol_spacing = 0.005

num_scans=len(data_stems)
    
confidence_maps=[None]*num_scans
grain_maps=[None]*num_scans
Xss=[None]*num_scans
Yss=[None]*num_scans
Zss=[None]*num_scans


for ii in np.arange(num_scans):
    print('Loading Volume %d ....'%(ii))
    conf_data = np.load(os.path.join(data_location,data_stems[ii]+'_grain_map_data.npz'))
    
    confidence_maps[ii]=conf_data['confidence_map']
    grain_maps[ii]=conf_data['grain_map']
    Xss[ii]=conf_data['Xs']
    Yss[ii]=conf_data['Ys']
    Zss[ii]=conf_data['Zs']
     
#assumes all volumes to be the same size
num_layers=grain_maps[0].shape[0]

total_layers=num_layers*num_scans

num_rows=grain_maps[0].shape[1]
num_cols=grain_maps[0].shape[2]

grain_map_stitched=np.zeros((total_layers,num_rows,num_cols))
confidence_stitched=np.zeros((total_layers,num_rows,num_cols))
Xs_stitched=np.zeros((total_layers,num_rows,num_cols))
Ys_stitched=np.zeros((total_layers,num_rows,num_cols))
Zs_stitched=np.zeros((total_layers,num_rows,num_cols))


for i in np.arange(num_scans):
    if top_down==True:
        grain_map_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=grain_maps[num_scans-1-i]
        confidence_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=confidence_maps[num_scans-1-i]
        Xs_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Xss[num_scans-1-i]
        Zs_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Zss[num_scans-1-i]
        Ys_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Yss[num_scans-1-i]+vol_spacing*i    
    else:
        
        grain_map_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=grain_maps[i]
        confidence_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=confidence_maps[i]
        Xs_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Xss[i]
        Zs_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Zss[i]
        Ys_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Yss[i]+vol_spacing*i

                


print('Writing VTK data...')
# VTK Dump 
Xslist=Xs_stitched[:,:,:].ravel()
Yslist=Ys_stitched[:,:,:].ravel()
Zslist=Zs_stitched[:,:,:].ravel()

grainlist=grain_map_stitched[:,:,:].ravel()
conflist=confidence_stitched[:,:,:].ravel()

num_pts=Xslist.shape[0]
num_cells=(total_layers-1)*(num_rows-1)*(num_cols-1)

f = open(os.path.join(data_location, output_stem +'_stitch_with_ipf.vtk'), 'w')


f.write('# vtk DataFile Version 3.0\n')
f.write('grainmap Data\n')
f.write('ASCII\n')
f.write('DATASET UNSTRUCTURED_GRID\n')
f.write('POINTS %d double\n' % (num_pts))

for i in np.arange(num_pts):
    f.write('%e %e %e \n' %(Xslist[i],Yslist[i],Zslist[i]))

scale2=num_cols*num_rows
scale1=num_cols    
    
f.write('CELLS %d %d\n' % (num_cells, 9*num_cells))   
for k in np.arange(Xs_stitched.shape[0]-1):
    for j in np.arange(Xs_stitched.shape[1]-1):
        for i in np.arange(Xs_stitched.shape[2]-1):
            base=scale2*k+scale1*j+i    
            p1=base
            p2=base+1
            p3=base+1+scale1
            p4=base+scale1
            p5=base+scale2
            p6=base+scale2+1
            p7=base+scale2+scale1+1
            p8=base+scale2+scale1
            
            f.write('8 %d %d %d %d %d %d %d %d \n' %(p1,p2,p3,p4,p5,p6,p7,p8))    
    
    
f.write('CELL_TYPES %d \n' % (num_cells))    
for i in np.arange(num_cells):
    f.write('12 \n')  

f.write('POINT_DATA %d \n' % (num_pts))
f.write('SCALARS grain_id int \n')  
f.write('LOOKUP_TABLE default \n')      
for i in np.arange(num_pts):
    f.write('%d \n' %(grainlist[i]))    
    
#f.write('POINT_DATA %d \n' % (num_pts))
f.write('SCALARS ipf_color float 3 \n')  
f.write('LOOKUP_TABLE default \n')    
f.write('COLOR_SCALARS ipf_color 3 \n') 
for i in np.arange(num_pts):
    if grainlist[i] < 0:
        temp_ipf = [1.0, 1.0, 1.0]
    else:
        ipf_idx = grain_ids_to_idx[(grain_ids_to_idx[:, 0] == grainlist[i].astype(int)), 1][0].astype(int)
        temp_ipf = ipf_color_arr[ipf_idx, :]
    f.write('%e %e %e \n' %(temp_ipf[0], temp_ipf[1], temp_ipf[2]))  

f.write('FIELD FieldData 1 \n' )
f.write('confidence 1 %d float \n' % (num_pts))       
for i in np.arange(num_pts):
    f.write('%e \n' %(conflist[i]))  

f.close()

#%%

data_location = path
data_stems = [stem]
output_stem = data_stems[0]
top_down=True
vol_spacing = 0.005

num_scans=len(data_stems)

confidence_maps=[None]*num_scans
grain_maps=[None]*num_scans
Xss=[None]*num_scans
Yss=[None]*num_scans
Zss=[None]*num_scans
top_n_conf=[None]*num_scans
top_n_id=[None]*num_scans


for ii in np.arange(num_scans):
    print('Loading Volume %d ....'%(ii))
    conf_data=np.load(os.path.join(data_location,data_stems[ii]+'_top_n_grain_map_data.npz'))
    
    confidence_maps[ii]=conf_data['confidence_map']
    grain_maps[ii]=conf_data['grain_map']
    Xss[ii]=conf_data['Xs']
    Yss[ii]=conf_data['Ys']
    Zss[ii]=conf_data['Zs']
    top_n_conf[ii]=conf_data['top_n_conf']
    top_n_id[ii]=conf_data['top_n_id']
     
#assumes all volumes to be the same size
num_layers=grain_maps[0].shape[0]

total_layers=num_layers*num_scans

num_rows=grain_maps[0].shape[1]
num_cols=grain_maps[0].shape[2]

grain_map_stitched=np.zeros((total_layers,num_rows,num_cols))
confidence_stitched=np.zeros((total_layers,num_rows,num_cols))
Xs_stitched=np.zeros((total_layers,num_rows,num_cols))
Ys_stitched=np.zeros((total_layers,num_rows,num_cols))
Zs_stitched=np.zeros((total_layers,num_rows,num_cols))
top_n_conf_stitched=np.zeros((3,total_layers,num_rows,num_cols))
top_n_id_stitched=np.zeros((3,total_layers,num_rows,num_cols))


for i in np.arange(num_scans):
    if top_down==True:
        grain_map_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=grain_maps[num_scans-1-i]
        confidence_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=confidence_maps[num_scans-1-i]
        Xs_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Xss[num_scans-1-i]
        Zs_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Zss[num_scans-1-i]
        Ys_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Yss[num_scans-1-i]+vol_spacing*i  
        
        top_n_conf_stitched[:,((i)*num_layers):((i)*num_layers+num_layers),:,:]=top_n_conf[num_scans-1-i]
        top_n_id_stitched[:,((i)*num_layers):((i)*num_layers+num_layers),:,:]=top_n_id[num_scans-1-i]
    else:
        
        grain_map_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=grain_maps[i]
        confidence_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=confidence_maps[i]
        Xs_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Xss[i]
        Zs_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Zss[i]
        Ys_stitched[((i)*num_layers):((i)*num_layers+num_layers),:,:]=Yss[i]+vol_spacing*i
        
        top_n_conf_stitched[:,((i)*num_layers):((i)*num_layers+num_layers),:,:]=top_n_conf[i]
        top_n_id_stitched[:,((i)*num_layers):((i)*num_layers+num_layers),:,:]=top_n_id[i]

                


print('Writing VTK data...')
# VTK Dump 
Xslist=Xs_stitched[:,:,:].ravel()
Yslist=Ys_stitched[:,:,:].ravel()
Zslist=Zs_stitched[:,:,:].ravel()

grainlist=grain_map_stitched[:,:,:].ravel()
conflist=confidence_stitched[:,:,:].ravel()

top_n_conf_stitched = np.reshape(top_n_conf_stitched, [3, total_layers*num_rows*num_cols])
top_n_id_stitched = np.reshape(top_n_id_stitched, [3, total_layers*num_rows*num_cols])

num_pts=Xslist.shape[0]
num_cells=(total_layers-1)*(num_rows-1)*(num_cols-1)

f = open(os.path.join(data_location, output_stem +'_top_n_stitch_with_ipf.vtk'), 'w')


f.write('# vtk DataFile Version 3.0\n')
f.write('grainmap Data\n')
f.write('ASCII\n')
f.write('DATASET UNSTRUCTURED_GRID\n')
f.write('POINTS %d double\n' % (num_pts))

for i in np.arange(num_pts):
    f.write('%e %e %e \n' %(Xslist[i],Yslist[i],Zslist[i]))

scale2=num_cols*num_rows
scale1=num_cols    
    
f.write('CELLS %d %d\n' % (num_cells, 9*num_cells))   
for k in np.arange(Xs_stitched.shape[0]-1):
    for j in np.arange(Xs_stitched.shape[1]-1):
        for i in np.arange(Xs_stitched.shape[2]-1):
            base=scale2*k+scale1*j+i    
            p1=base
            p2=base+1
            p3=base+1+scale1
            p4=base+scale1
            p5=base+scale2
            p6=base+scale2+1
            p7=base+scale2+scale1+1
            p8=base+scale2+scale1
            
            f.write('8 %d %d %d %d %d %d %d %d \n' %(p1,p2,p3,p4,p5,p6,p7,p8))    
    
    
f.write('CELL_TYPES %d \n' % (num_cells))    
for i in np.arange(num_cells):
    f.write('12 \n')    

f.write('POINT_DATA %d \n' % (num_pts))
f.write('SCALARS grain_id int \n')  
f.write('LOOKUP_TABLE default \n')      
for i in np.arange(num_pts):
    f.write('%d \n' %(grainlist[i]))    

f.write('SCALARS grain_id_2 int \n')  
f.write('LOOKUP_TABLE default \n')      
for i in np.arange(num_pts):
    f.write('%d \n' %(top_n_id_stitched[1, i])) 
    
f.write('SCALARS grain_id_3 int \n')  
f.write('LOOKUP_TABLE default \n')      
for i in np.arange(num_pts):
    f.write('%d \n' %(top_n_id_stitched[2, i])) 

f.write('SCALARS confidence float \n')  
f.write('LOOKUP_TABLE default \n')        
for i in np.arange(num_pts):
    f.write('%e \n' %(conflist[i]))  

f.write('SCALARS confidence_2 float \n')  
f.write('LOOKUP_TABLE default \n')       
for i in np.arange(num_pts):
    f.write('%e \n' %(top_n_conf_stitched[1, i])) 

f.write('SCALARS confidence_3 float \n')  
f.write('LOOKUP_TABLE default \n')        
for i in np.arange(num_pts):
    f.write('%e \n' %(top_n_conf_stitched[2, i])) 

f.write('SCALARS ipf_color float 3 \n')  
f.write('LOOKUP_TABLE default \n')    
f.write('COLOR_SCALARS ipf_color 3 \n') 
for i in np.arange(num_pts):
    ipf_idx = grain_ids_to_idx[(grain_ids_to_idx[:, 0] == grainlist[i].astype(int)), 1][0].astype(int)
    temp_ipf = ipf_color_arr[ipf_idx, :]
    f.write('%e %e %e \n' %(temp_ipf[0], temp_ipf[1], temp_ipf[2]))  

f.write('SCALARS ipf_color_2 float 3 \n')  
f.write('LOOKUP_TABLE default \n')    
f.write('COLOR_SCALARS ipf_color_2 3 \n') 
for i in np.arange(num_pts):
    ipf_idx = grain_ids_to_idx[(grain_ids_to_idx[:, 0] == top_n_id_stitched[1, i].astype(int)), 1][0].astype(int)
    temp_ipf = ipf_color_arr[ipf_idx, :]
    f.write('%e %e %e \n' %(temp_ipf[0], temp_ipf[1], temp_ipf[2])) 
    
f.write('SCALARS ipf_color_3 float 3 \n')  
f.write('LOOKUP_TABLE default \n')    
f.write('COLOR_SCALARS ipf_color_3 3 \n') 
for i in np.arange(num_pts):
    ipf_idx = grain_ids_to_idx[(grain_ids_to_idx[:, 0] == top_n_id_stitched[2, i].astype(int)), 1][0].astype(int)
    temp_ipf = ipf_color_arr[ipf_idx, :]
    f.write('%e %e %e \n' %(temp_ipf[0], temp_ipf[1], temp_ipf[2])) 

f.close()






