#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:17:05 2021

@author: djs522
"""
# =============================================================================
#%% IMPORTS
# =============================================================================
from __future__ import print_function

import numpy as np
import os

import sys
if sys.version_info[0] < 3:
    from hexrd.xrd import rotations as hexrd_rot
    import hexrd.xrd.symmetry as hexrd_sym
else:
    from hexrd import rotations as hexrd_rot
    from hexrd import symmetry as hexrd_sym

# =============================================================================
#%% FUNCTION DECLARATION AND IMPLEMENTATION
# =============================================================================

# transform euler angles to kocks angles
def euler2kocks(euler):
    # euler angles (rad) to kocks angles (rad)
    kocks = np.vstack([np.mod(np.pi/2 - euler[:, 2], np.pi * 2), np.mod(euler[:, 1], np.pi * 2), np.mod(euler[:, 0] - np.pi/2, np.pi * 2)]).T
    return kocks

# create .kocks file
def create_kocks_file(path, out_f_name_kocks, exp_maps, grain_ids):
    num_grains = exp_maps.shape[0]
    
    # do rotation transformations to go from exp_maps to kocks
    rotmat = hexrd_rot.rotMatOfExpMap_opt(exp_maps.T)
    eulers = np.zeros([num_grains, 3])
    for i in range(num_grains):
        eulers[i] = hexrd_rot.angles_from_rmat_xyz(rotmat[i])
    kocks_rad = euler2kocks(eulers)
    
    # write output file
    out_f_kocks = open(os.path.join(path, out_f_name_kocks), "w")
    out_f_kocks.writelines('grain-orientations\n')
    out_f_kocks.writelines(str(num_grains) + '\n')
    
    kocks_deg = np.rad2deg(kocks_rad)
    
    write_orient = np.hstack([kocks_deg, grain_ids[:, np.newaxis]])
    np.savetxt(out_f_kocks, write_orient, fmt='%.8f \t%.8f \t%.8f \t%i')
    
    out_f_kocks.writelines('EOF')
    out_f_kocks.close()

# create .grain file
def create_grain_file(path, f_name_opt, out_f_name_grain):
    # open opt file, read lines, and remove first line
    with open(os.path.join(path, f_name_opt)) as f_opt:
        lines_opt = f_opt.readlines()
    
    # remove first and last line
    lines_opt.pop(0)
    lines_opt.pop(-1)
    
    # write grain file
    out_f_grain = open(os.path.join(path, out_f_name_grain), "w")
    out_f_grain.writelines(lines_opt)
    out_f_grain.close()

# create .mesh file
def create_mesh_file(path, f_name_params, f_name_mesh, out_f_name_mesh):
    # read params file
    f_params = open(os.path.join(path, f_name_params), "r")
    f_params_lines = f_params.readlines()
    f_params.close()
    
    # read mesh file
    f_mesh = open(os.path.join(path, f_name_mesh), "r")
    f_mesh_lines = f_mesh.readlines()
    f_mesh.close()
    
    f_mesh_lines.remove('  1.0   1.0   1.0\n')
    
    # write mesh file
    out_f_mesh = open(os.path.join(path, out_f_name_mesh), "w")
    out_f_mesh.write(f_params_lines[0])
    out_f_mesh.writelines(f_mesh_lines)
    out_f_mesh.close()

# create .matl file
def create_matl_file(path, out_f_name_matl, comp, c_11, c_12, c_44, crss_1, crss_2):
    # write matl file
    out_f_matl = open(os.path.join(path, out_f_name_matl), "w")
    out_f_matl.write('%i \n' %(comp))
    out_f_matl.write('%0.3e \t %0.3e \t %0.3e \n' %(c_11, c_12, c_44))
    out_f_matl.write('%0.3e \t %0.3e \n' %(crss_1, crss_2))
    out_f_matl.close()


if __main__
# =============================================================================
#%% USER DEFINED VARIABLES
# =============================================================================
    
path = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/neper/'
stem = 'ss718_total_nf_with_ff_mesh'
new_name = 'mechmet_' + stem
grain_map_fname = os.path.join(path, 'ss718_total_nf_with_ff_tesr.npz')

comp = 3
c_11 = 260.0e3 # MPa
c_12 = 180.0e3 # MPa
c_44 = 110.0e3 # MPa
crss_1 = 450.0 # MPa
crss_2 = 450.0 # MPa


#%% preprocessing
f_name_mesh = stem + '.mesh'
f_name_params = stem + '.parms'
f_name_opt = stem + '.opt'

out_f_name_mesh = new_name + '.mesh'
out_f_name_kocks = new_name + '.kocks'
out_f_name_matl = new_name + '.matl'
out_f_name_grain = new_name + '.grain'

grain_map_data = np.load(grain_map_fname)
grain_map = grain_map_data['GRAIN_MAP']
exp_maps = grain_map_data['ORI_LIST']
grain_ids = grain_map_data['NEW_IDS']


#%% create mesh file
create_mesh_file(path, f_name_params, f_name_mesh, out_f_name_mesh)

#%% write matl file
create_matl_file(path, out_f_name_matl, comp, c_11, c_12, c_44, crss_1, crss_2)

#%% create grain file
create_grain_file(path, f_name_opt, out_f_name_grain)

#%% create kocks file
create_kocks_file(path, out_f_name_kocks, exp_maps, grain_ids)





