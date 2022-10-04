#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:54:45 2022

@author: djs522
"""


import numpy as np
import os
from hexrd.matrixutil import stressVecToTen

path = '/media/djs522/djs522_nov2020/chess_2020_11/djs522_nov2020_in718/dp718/xtal_mesh_3micron/model_results/'
stress_1_fname = 'expcomp_allpoints_Tension1_6_S.txt'
stress_data_1 = np.loadtxt(os.path.join(path, stress_1_fname), delimiter=',')
stress_data_1 = stress_data_1[:, [1, 2, 3, 6, 5, 4]]

stress_2_fname = 'expcomp_allpoints_Tension1_11_S.txt'
stress_data_2 = np.loadtxt(os.path.join(path, stress_2_fname), delimiter=',')
stress_data_2 = stress_data_2[:, [1, 2, 3, 6, 5, 4]]

stress_data = stress_data_2 - stress_data_1

print(stress_data.shape)
print(np.mean(stress_data, axis=0))
print(np.max(np.abs(stress_data), axis=0))






#%%

import meshio
mechmonics_vtk_fname = '/home/djs522/additional_sw/mech_suite/dp718_XtalMesh_cropped/dp718_cropped_16m_50w.vtk'
mechmonics_vtk = meshio.read(mechmonics_vtk_fname)

#%%

uni_ids = np.unique(mechmonics_vtk.cell_data['grains-0'][0]).astype(int)
avg_model_stress = np.zeros([uni_ids.size, 6])
for i, g_id in enumerate(uni_ids):
    t = (mechmonics_vtk.cell_data['grains-0'][0][:, 0] == g_id).astype(int)
    avg_model_stress[i] = np.mean(stress_data[t, :], axis=0)

#%%

import scipy.io as io
avg_exp_stress = io.loadmat('/home/djs522/additional_sw/mech_suite/dp718_XtalMesh_cropped/final_dp718_xtalmesh_stress_MPa_c0_2-c0_1_298MPa.mat')['stress_mpa']
avg_exp_stress = avg_exp_stress[uni_ids, :]



#%%
import matplotlib.pyplot as plt
avg_diff_stress = avg_model_stress - avg_exp_stress
print(np.max(avg_diff_stress, axis=0))
print(np.min(avg_diff_stress, axis=0))
print(np.mean(avg_diff_stress, axis=0))

print(np.sum(np.abs(avg_diff_stress) > 150, axis=0))

for i in range(6):
    plt.figure()
    plt.hist(avg_diff_stress[:, i])


#%%
mechmonics_w_model_vtk_fname = '/home/djs522/additional_sw/mech_suite/dp718_XtalMesh_cropped/dp718_cropped_16m_50w_model.vtk'


VERSION = '# vtk DataFile Version 3.0'
FORMAT = 'ASCII'
DESCRIPTION = 'MechMet'
DSET_TYPE = 'DATASET UNSTRUCTURED_GRID'
DTYPE_I = 'int'
DTYPE_R = 'double'
PDATA = 'POINT_DATA'
CDATA = 'CELL_DATA'
POINTS = 'POINTS'
CELLS = 'CELLS'
CTYPES = 'CELL_TYPES'
CTYPE_TET10 = 24 # VTK_QUADRATIC_TETRA 
CON_ORDER = [0, 2, 4, 9, 1, 3, 5, 6, 7, 8] #[1, 3, 5, 10, 2, 4, 6, 7, 8, 9] # corners, then midpoints
DAT_SCA = 'SCALARS'
DAT_VEC = 'VECTORS'
DAT_TEN = 'TENSORS'
LOOKUP_DFLT = 'LOOKUP_TABLE default'
VEC_ORDER = [0, 1, 2]
TEN_ORDER = [0, 5, 3, 5, 1, 4, 3, 4, 2]


def voigt_stress_vt2_3d(v):
    return np.array([[v[:,0], v[:,5], v[:,4]],
                    [v[:,5], v[:,1], v[:,3]],
                    [v[:,4], v[:,3], v[:,2]]]).T


def write_point_data(f, pt_data, label, data_type, data_fmt):
    if data_fmt == DAT_SCA:
        f.write('%s %s %s\n' %(DAT_SCA, label, data_type))
        f.write('%s\n' %(LOOKUP_DFLT))
        np.savetxt(f, pt_data, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='')
    elif data_fmt == DAT_VEC:
        f.write('%s %s %s\n' %(DAT_VEC, label, data_type))
        np.savetxt(f, pt_data, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='')
    elif data_fmt == DAT_TEN:
        f.write('%s %s %s\n' %(DAT_TEN, label, data_type))
        # convert (n, x, y) 3D to 2D
        pt_data = pt_data.reshape((pt_data.shape[0]*pt_data.shape[1]), pt_data.shape[2])
        np.savetxt(f, pt_data, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='')
    
    return f

def write_cell_data(f, c_data, label, data_type, data_fmt):
    if data_fmt == DAT_SCA:
        f.write('%s %s %s\n' %(DAT_SCA, label, data_type))
        f.write('%s\n' %(LOOKUP_DFLT))
        np.savetxt(f, c_data, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='')
    elif data_fmt == DAT_VEC:
        f.write('%s %s %s\n' %(DAT_VEC, label, data_type))
        #f.write('%s\n' %(LOOKUP_DFLT))
        np.savetxt(f, c_data, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='')
    elif data_fmt == DAT_TEN:
        f.write('%s %s %s\n' %(DAT_TEN, label, data_type))
        #f.write('%s\n' %(LOOKUP_DFLT))
        # convert (n, x, y) 3D to 2D
        c_data = c_data.reshape((c_data.shape[0]*c_data.shape[1]), c_data.shape[2])
        np.savetxt(f, c_data, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='')
    
    return f

pt_data_list = [[mechmonics_vtk.point_data['Stress'], 'stress_mechmonics', DTYPE_R, DAT_TEN]]
c_data_list = [[mechmonics_vtk.cell_data['phases-0'][0], 'phases-0', DTYPE_R, DAT_SCA],
               [mechmonics_vtk.cell_data['grains-0'][0], 'grains-0', DTYPE_R, DAT_SCA],
               [voigt_stress_vt2_3d(stress_data), 'stress_model', DTYPE_R, DAT_TEN]]

#%%
with open(mechmonics_w_model_vtk_fname, 'w') as f:
    
    # header
    f.write('%s\n' %(VERSION))
    f.write('%s\n' %(DESCRIPTION))
    f.write('%s\n' %(FORMAT))
    
    # mesh points
    npts = mechmonics_vtk.points.shape[0]
    f.write('%s\n' %(DSET_TYPE))
    f.write('%s %d %s\n' %(POINTS, npts, DTYPE_R))
    np.savetxt(f, mechmonics_vtk.points, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='')
    
    # mesh cells
    nels = mechmonics_vtk.cells[0].data.shape[0]
    f.write('%s %d %d\n' %(CELLS, nels, 11*nels))
    vtk_cells = np.hstack([np.ones([nels, 1]) * 10, mechmonics_vtk.cells[0].data])
    np.savetxt(f, vtk_cells, fmt='%i', delimiter=' ', newline='\n', header='', footer='', comments='')
    
    # mesh cell types
    f.write('%s %d\n' %(CTYPES, nels))
    np.savetxt(f, np.ones([nels, 1]) * CTYPE_TET10, fmt='%i', delimiter=' ', newline='\n')
    
    # point data
    f.write('%s %d\n' %(PDATA, npts))
    for pt_data in pt_data_list:
        # pt_data = [data, label, data_type, data_fmt]
        f = write_point_data(f, pt_data[0], pt_data[1], pt_data[2], pt_data[3])
    
    # cell data
    f.write('%s %d\n' %(CDATA, nels))
    for c_data in c_data_list:
        # c_data = [data, label, data_type, data_fmt]
        f = write_cell_data(f, c_data[0], c_data[1], c_data[2], c_data[3])





