#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:58:21 2021

@author: djs522
"""

#IMPORTS
import numpy as np
import meshio
import warnings
from hexrd import rotations as hexrd_rot

import sys
sys.path.append('/home/djs522/additional_sw/hedmTools/')
from CHESS_hedmTools.SingleGrainOrientationDistributions import OrientationTools as OT

'''
MY_PATH  = '/media/djs522/djs522_nov2020/chess_2020_11/djs522_nov2020_in718/dp718/xtal_mesh_3micron/'
XTAL_FN = 'XtalMesh_final_3m_cropped'

mesh = meshio.read(MY_PATH + XTAL_FN + '.inp')
meshio.gmsh.write(MY_PATH + XTAL_FN + '.msh', mesh, fmt_version='2.2', binary=False)

neper_faset = ['x0', 'x1', 'y0', 'y1', 'z0', 'z1']
xtal_faset = ['f_n-1', 'f_n+1', 'f_n-2', 'f_n+2', 'f_n-3', 'f_n+3']
xtal2neper_faset = {'f_n-1':'x0', 'f_n+1':'x1', 
                    'f_n-2':'y0', 'f_n+2':'y1',
                    'f_n-3':'z0', 'f_n+3':'z1'}

fasets = {'x0':[], 'x1':[], 'y0':[], 'y1':[], 'z0':[], 'z1':[]}
for key in mesh.point_sets.keys():
    if key in xtal_faset:
        print(xtal2neper_faset[key])
        face_nodes = mesh.point_sets[key]
        print(face_nodes.shape)
        
        face_ele_node_mask = np.isin(mesh.cells[0].data, face_nodes)
        face_ele_node_sum = np.sum(face_ele_node_mask, axis=1)
        face_elset = np.where(face_ele_node_sum > 5)[0]
        
        print(face_elset.shape)
        
        face_elset_nodes = mesh.cells[0].data[face_elset, :]
        face_elset_nodes_red = face_elset_nodes[face_ele_node_mask[face_elset, :]]
        face_elset_nodes_red = np.reshape(face_elset_nodes_red, [int(face_elset_nodes_red.size / 6), 6])
        
        print(face_elset_nodes_red.shape)
        
        total_face_elset = np.hstack([face_elset.reshape([face_elset.size, 1]), face_elset_nodes_red])
        
        fasets[xtal2neper_faset[key]] = total_face_elset

geo_tags = np.ones([len(mesh.cells[0]), 1])
phys_tags = np.ones([len(mesh.cells[0]), 1]) * 2
id_tags = np.ones([len(mesh.cells[0]), 1]) * 3
mesh.cell_data = {"gmsh:physical":phys_tags.flatten(),
                      "gmsh:geometrical":geo_tags.flatten(),
                      "cell_tags":id_tags.flatten()}
["gmsh:physical", "gmsh:geometrical", "cell_tags"


mesh.cell_data[cell_type]["tags"] = {element id: unique element set id}

meshio.gmsh.write(MY_PATH + XTAL_FN + '.msh', mesh, fmt_version='2.2', binary=False)

'''

#%%
def meshio_2_neper_mesh(mesh_fn, path, orientations=np.array([]), reorder_ids=False):
    '''

    Parameters
    ----------
    mesh : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    '''
    
    # handle basic mesh conversion
    in_mesh_file = path + mesh_fn + '.inp'
    out_mesh_file = path + mesh_fn + '.msh'
    
    # read in mesh
    mesh = meshio.read(in_mesh_file)
    
    # handle tags to the data
    grain_id_tags = np.zeros([len(mesh.cells[0].data)])
    #grain_id_tags = np.zeros([len(mesh.cells[0])])
    print(grain_id_tags.shape)
    cell_nums_list = []
    for cell_set_key in mesh.cell_sets.keys():
        if 'GRAIN' in cell_set_key:
            grain_id = int(cell_set_key.strip('GRAIN_'))
            cells = np.array(mesh.cell_sets[cell_set_key]).flatten()
            cell_nums_list.append(cells.size)
            grain_id_tags[cells] = grain_id
    
    print(np.min(cell_nums_list))
    print(np.max(cell_nums_list))
    print(np.mean(cell_nums_list))
    
    # do a continuous check on grain ids [1-max_grain_id]
    uni_grain_ids = np.unique(grain_id_tags)
    cont_grain_ids = np.arange(uni_grain_ids.size) + 1
    if np.setdiff1d(cont_grain_ids, uni_grain_ids).size > 0:
        warnings.warn("Grain IDs are not continuous: " + np.array2string(np.setdiff1d(cont_grain_ids, uni_grain_ids)))
        
        if reorder_ids:
            print("Reordering IDs from 1 to %i" %(uni_grain_ids.size))
            new_grain_id_tags = np.zeros(grain_id_tags.shape)
            for i, gid in enumerate(uni_grain_ids):
                new_grain_id_tags[grain_id_tags == gid] =  i + 1
            if np.sum(new_grain_id_tags < 1) > 0:
                raise ValueError("Something went wrong when reordering ids...")
            grain_id_tags = new_grain_id_tags  
                
    
    mesh.cell_data = {"gmsh:physical":np.atleast_2d(grain_id_tags),
              "gmsh:geometrical":np.atleast_2d(grain_id_tags),
              "cell_tags":np.atleast_2d(np.zeros(grain_id_tags.shape))}
    
    # write first pass of gmsh file
    meshio.gmsh.write(out_mesh_file, mesh, fmt_version='2.2', binary=False)
    
    # find face element sets and nodes for neper fasets
    neper_faset = ['x0', 'x1', 'y0', 'y1', 'z0', 'z1']
    xtal_faset = ['f_n-1', 'f_n+1', 'f_n-2', 'f_n+2', 'f_n-3', 'f_n+3']
    xtal2neper_faset = {'f_n-1':'x0', 'f_n+1':'x1', 
                        'f_n-2':'y0', 'f_n+2':'y1',
                        'f_n-3':'z0', 'f_n+3':'z1'}
    fasets = {'x0':[], 'x1':[], 'y0':[], 'y1':[], 'z0':[], 'z1':[]}
    print(mesh.point_sets.keys())
    for key in mesh.point_sets.keys():
        if key in xtal_faset:
            print(xtal2neper_faset[key])
            face_nodes = mesh.point_sets[key]
            print(face_nodes.shape)
            
            face_ele_node_mask = np.isin(mesh.cells[0].data, face_nodes)
            face_ele_node_sum = np.sum(face_ele_node_mask, axis=1)
            face_elset = np.where(face_ele_node_sum == 6)[0]
            
            print(face_elset.shape)
            
            face_elset_nodes = mesh.cells[0].data[face_elset, :]
            face_elset_nodes_red = face_elset_nodes[face_ele_node_mask[face_elset, :]]
            
            print(face_elset_nodes_red.shape)
            
            face_elset_nodes_red = np.reshape(face_elset_nodes_red, [int(face_elset_nodes_red.size / 6), 6])
            
            print(face_elset_nodes_red.shape)
            
            face_elset = face_elset + 1 # plus one for gmsh indexing
            face_elset_nodes_red = face_elset_nodes_red + 1
            total_face_elset = np.hstack([face_elset.reshape([face_elset.size, 1]), face_elset_nodes_red])
            fasets[xtal2neper_faset[key]] = total_face_elset
    
    
    with open(out_mesh_file, "a") as out:
        # write fasets element and nodes
        out.write("$Fasets\n")
        out.write("6\n")
        for label in neper_faset:
            out.write(label + "\n")
            out.write("%i\n" %fasets[label].shape[0])
            for elem in fasets[label]:
                elem_line = '%i %i %i %i %i %i %i \n' %(tuple(elem))
                out.write(elem_line)
        out.write("$EndFasets\n")
        
        # write orientation information
        if orientations.size > 0:
            out.write("$ElsetOrientations\n")
            out.write("%i rodrigues:active\n" %(orientations.shape[0]))
            
            for i in range(orientations.shape[0]):
                out.write("%i %0.6f %0.6f %0.6f\n" %(i+1, orientations[i, 0], orientations[i, 1], orientations[i, 2]))
            
            out.write("$EndElsetOrientations\n")

if __name__ == '__main__':
    grain_out_path = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_0_gripped/combined_grains_c0_0.out'
    grain_out = np.loadtxt(grain_out_path)
    
    ori = OT.exp_map2rod(grain_out[:, 3:6].T)
    
    MY_PATH = "C:/Users/Dalton Shadle/Downloads/"
    XTAL_FN = "XtalMesh_2mil_fulldata"
    MY_PATH  = '/media/djs522/djs522_nov2020/chess_2020_11/djs522_nov2020_in718/dp718/xtal_mesh_3micron/'
    XTAL_FN = 'XtalMesh_final_3m_cropped'
    
    meshio_2_neper_mesh(XTAL_FN, MY_PATH, orientations=ori)




