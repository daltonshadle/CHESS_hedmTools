#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:05:24 2022

@author: djs522
"""

import numpy as np

# read neper mesh
mesh_fname = '/media/djs522/djs522_nov2020/chess_2020_11/djs522_nov2020_in718/dp718/mesh_3micron/final_dp718_total_3micron_conf50_voxel400_cc3d6_1mm_rcl055_mesh.msh'

# Neper flags
NodeFlag = '$Nodes'
ElementFlag = '$Elements'
OrientationFlag = '$ElsetOrientations'
GroupFlag = '$Groups'
FasetsFlag = '$Fasets'

# face set flags
x0 = ['x0', 'cut1', 'f3']
x1 = ['x1', 'cut2', 'f4']
y0 = ['y0', 'cut3', 'f5']
y1 = ['y1', 'cut4', 'f6']
z0 = ['z0', 'cut5', 'f1']
z1 = ['z1', 'cut6', 'f2']

# initialize face set numbers
nse_1 = 0
nse_2 = 0
nse_3 = 0
nse_4 = 0
nse_5 = 0
nse_6 = 0

element_count = 0
phase_count = 1

with open(mesh_fname, 'r+') as mesh_file:
  
    for iline in mesh_file:
        iline = iline.strip()
        
        # find node flag
        if (iline == NodeFlag):
            # grad number of nodes
            num_nodes = int(next(mesh_file).strip())
            
            # extract coordinates for each node
            coord_nodes = np.zeros([num_nodes, 3])
            for i in range(num_nodes):
                i_node, x_n, y_n, z_n = next(mesh_file).strip().split()
                coord_nodes[i, :] = [x_n, y_n, z_n]
        
        # find element flag
        if (iline == ElementFlag):
            # grad number of elements
            num_eles = int(next(mesh_file).strip())
            
            # extract elements and grains
            nodenums_eles = np.zeros([num_eles, 10]) #  assumes tet10
            grain_eles = np.zeros([num_eles, 1]) # this might be 
            for i in range(num_eles):
                i_ele = next(mesh_file).strip().split()
                
                if int(i_ele[1]) == 11: # tet10 code is 11
                    element_count += 1
                    nodenums_eles[i, :] = i_ele[6:]
                    grain_eles[i, :] = int(i_ele[3])   
                    
            
        # find orientation flag
        if (iline == OrientationFlag):
            # grab orientation header info
            num_oris, descrip_oris = next(mesh_file).strip().split()
            num_oris = int(num_oris)
            
            # get all the orientations
            grain_ids = np.zeros([num_oris, 1])
            grain_oris = np.zeros([num_oris, 3])
            for i in range(num_oris):
                i_ori = next(mesh_file).strip().split()
                grain_ids[i, :] = int(i_ori[0])
                for j, j_ori in enumerate(i_ori[1:]):
                    grain_oris[i, j] = float(j_ori)
        
        # find group flag
        if (iline == GroupFlag):
            raise ValueError("Groups have not been implemented yet")
            # TODO: Implement Groups
            '''
            phase_count =2
            tmp_group = fgetl(fid)
            numgrain = fscanf(fid, '#d', 1)
            phases = zeros(2,numgrain)
            tmp_phase = fscanf(fid, '#d #d', 2*numgrain)
            tmp_phase = reshape(tmp_phase, [2, numgrain])
            '''
        
        # find face sets flag
        if (iline == FasetsFlag):
            num_faces = int(next(mesh_file).strip())
            
            for i in range(num_faces):
                i_face = next(mesh_file).strip()
                print(i_face)
                
                # x faces
                if i_face in x0:
                    nse_x0 = int(next(mesh_file).strip())
                    x0_nodes = np.zeros([nse_x0, 7])
                    for i_nodes in range(nse_x0):
                        x0_nodes[i_nodes, :] = next(mesh_file).strip().split() 
                if i_face in x1:
                    nse_x1 = int(next(mesh_file).strip())
                    x1_nodes = np.zeros([nse_x1, 7])
                    for i_nodes in range(nse_x1):
                        x1_nodes[i_nodes, :] = next(mesh_file).strip().split()
                
                # y faces
                if i_face in y0:
                    nse_y0 = int(next(mesh_file).strip())
                    y0_nodes = np.zeros([nse_y0, 7])
                    for i_nodes in range(nse_y0):
                        y0_nodes[i_nodes, :] = next(mesh_file).strip().split()
                if i_face in y1:
                    nse_y1 = int(next(mesh_file).strip())
                    y1_nodes = np.zeros([nse_y1, 7])
                    for i_nodes in range(nse_y1):
                        y1_nodes[i_nodes, :] = next(mesh_file).strip().split()
                
                # z faces
                if i_face in z0:
                    nse_z0 = int(next(mesh_file).strip())
                    z0_nodes = np.zeros([nse_z0, 7])
                    for i_nodes in range(nse_z0):
                        z0_nodes[i_nodes, :] = next(mesh_file).strip().split() 
                if i_face in z1:
                    nse_z1 = int(next(mesh_file).strip())
                    z1_nodes = np.zeros([nse_z1, 7])
                    for i_nodes in range(nse_z1):
                        z1_nodes[i_nodes, :] = next(mesh_file).strip().split()
                
                # remove dummy line from each face block
                #dummy = next(mesh_file)
                #print(dummy)
            
# STOP HERE

'''
numel = element_count
con = zeros(numel,10)
grains=zeros(2,numel)
orientations=zeros(3,numori)
icount = 1
for iele=1:1:tmp_numel
    if(tmp_np(1,iele)~=0)
        con(icount,[1 3 5 10 2 4 6 7 9 8]) = tmp_np([1 2 3 4 5 6 7 8  9 10],iele)
        grainnum = tmp_grains(iele)
        grains(1,icount) = grainnum
        if(phase_count==1)
            grains(2,icount) = 1
        else
            grains(2,icount) = tmp_phase(2,grainnum)
        end
        icount = icount+1
    end  
end
for iori = 1:1:numori
 orientations(:,iori) = tmp_oris(2:4,iori)    
end
quaterions = QuatOfRod(orientations)
rotations  = RMatOfQuat(quaterions)
'''



