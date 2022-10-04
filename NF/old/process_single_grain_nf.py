#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 17:30:09 2021

@author: djs522
"""


from __future__ import print_function

import sys

import time

import numpy as np

import matplotlib.pyplot as plt

import multiprocessing as mp

import os

from scipy import ndimage
from skimage import measure

from hexrd.grainmap import nfutil
from hexrd.grainmap import tomoutil
from hexrd.grainmap import vtkutil
from hexrd.xrd import rotations  as hexrd_rot
import hexrd.xrd.symmetry as hexrd_sym
from hexrd.xrd import transforms_CAPI as xfcapi

IMPORT_HEXRD_SCRIPT_DIR = '/home/djs522/djs522/hexrd_utils'
sys.path.insert(0, IMPORT_HEXRD_SCRIPT_DIR)
import post_process_stress as pp_stress
import post_process_goe as pp_GOE

# *****************************************************************************
# %% CONSTANTS
# *****************************************************************************
pi = np.pi


#==============================================================================
# %% NF ADDITIONAL FUNCTIONS
#==============================================================================
def quat2exp_map(quat):
    '''
    % quat2exp_map - quaternion to exponential map conversion
    %   
    %   USAGE:
    %
    %   exp_map = quat2exp_map(quat)
    %
    %   INPUT:
    %
    %   quat is n x 4, 
    %        n quaternions for conversion
    %
    %   OUTPUT:
    %
    %   exp_map is n x 3, 
    %        returns an array of n exponential maps
    %
    %   NOTES:  
    %
    %   *  None
    %
    '''
    
    phi = 2 * np.arccos(quat[:, 0])
    norm = xfcapi.unitRowVector(quat[:, 1:])
    
    exp_map = norm * phi[:, None]
    
    return exp_map

#==============================================================================
# %% OUTPUT INFO -CAN BE EDITED
#==============================================================================

my_scan_num = 1077
my_grain_num = 6
ncpus = 66

# set up output directory and filename
output_dir = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf_analysis_post/output/single_grain/'
output_stem = 'ss718-1_sc%i_grain%i' %(my_scan_num, my_grain_num)

det_file='/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf_analysis_pre/retiga.yml'
mat_file='/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf_analysis_pre/materials_dp718_36000.cpl'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# location of grains.out folder
grain_out_file='/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_0_gripped/c0_0_gripped_sc%i_nf/grains.out' %(my_scan_num)

x_ray_energy=61.332 #keV

#name of the material for the reconstruction
mat_name='dp718-2'

#reconstruction with misorientation included, for many grains, this will quickly
#make the reconstruction size unmanagable
misorientation_bnd=0.0 #degrees 
misorientation_spacing=1.0 #degrees

beam_stop_width=0.6#mm, assumed to be in the center of the detector

ome_range_deg=[(0.,360.)] #degrees 

max_tth=12. #degrees, if a negative number is input, all peaks that will hit the detector are calculated

chunk_size=500#chunksize for multiprocessing, don't mess with unless you know what you're doing

#thresholds for grains in reconstructions
comp_thresh=0.8 #only use orientations from grains with completnesses ABOVE this threshold
chi2_thresh=1e-2 #only use orientations from grains BELOW this chi^2

cross_sectional_dim=1.25 #cross sectional to reconstruct (should be at least 20%-30% over sample width)
#voxel spacing for the near field reconstruction
voxel_spacing=0.005#in mm
#vertical (y) reconstruction voxel bounds in mm
v_bnds=[-0.13,0.09]

# location of grains.out folder
grain_out_file='/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/c65_0/c65_0_sc%i/grains.out' %(my_scan_num)

nf_image_stack_file = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf_analysis_post/nf_image_stack_ffsc%i.npy' %my_scan_num
image_stack = np.load(nf_image_stack_file)

num_imgs=1441

show_images = False

#==============================================================================
#%% EXPERIMENT INIT
#==============================================================================

# load original grain map
orig_grain_data = np.load(os.path.join(output_dir, output_stem + '_grain_map_data.npz'))

orig_grain_map = orig_grain_data['grain_map']
orig_confidence_map = orig_grain_data['confidence_map']
orig_Xs = orig_grain_data['Xs']
orig_Ys = orig_grain_data['Ys']
orig_Zs = orig_grain_data['Zs']
orig_exp_maps = orig_grain_data['ori_list']

grain_ind = np.where(orig_grain_map == my_grain_num)
grain_Xs = orig_Xs[grain_ind]
grain_Ys = orig_Ys[grain_ind]
grain_Zs = orig_Zs[grain_ind]

# create a bounding box around grain
grain_buffer = 3 * voxel_spacing
grain_Xs, grain_Ys, grain_Zs = np.meshgrid(np.arange(grain_Xs.min() - grain_buffer, grain_Xs.max() + grain_buffer + voxel_spacing, voxel_spacing),
                                           np.arange(grain_Ys.min() - grain_buffer, grain_Ys.max() + grain_buffer + voxel_spacing, voxel_spacing),
                                           np.arange(grain_Zs.min() - grain_buffer, grain_Zs.max() + grain_buffer + voxel_spacing, voxel_spacing))

# load DSGOD
dsgod_comp_thresh = 0.7
dsgod_path = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/c65_0/dsgod/dsgod/dsgod_ss718-1_%i/' %(my_scan_num)
dsgod_stem = 'grain_%i_dsgod_map_data_inv.npz' %(my_grain_num)
dsgod_npz_dir = os.path.join(dsgod_path, dsgod_stem)
[grain_quat, grain_mis_quat, grain_odf] = pp_GOE.process_dsgod_file(dsgod_npz_dir, 
                                                  scan=my_scan_num, comp_thresh=dsgod_comp_thresh,
                                                  do_avg_ori=True, do_conn_comp=True, save=False,
                                                  connectivity_type=26)
good_grain_quat = grain_quat[grain_odf > 0, :]
good_grain_exp_maps = quat2exp_map(good_grain_quat)


#%%
experiment, nf_to_ff_id_map  = nfutil.gen_trial_exp_data(grain_out_file,det_file,mat_file, 
                                                         x_ray_energy, mat_name, max_tth, comp_thresh, 
                                                         chi2_thresh, misorientation_bnd,
                                                         misorientation_spacing,ome_range_deg, 
                                                         num_imgs, beam_stop_width)

exp_maps = good_grain_exp_maps
experiment.n_grains = exp_maps.shape[0]
experiment.rMat_c = hexrd_rot.rotMatOfExpMap(exp_maps.T)
experiment.exp_maps = exp_maps

#==============================================================================
# %% INIT TEST COORDS
#==============================================================================

test_crds = np.vstack([grain_Xs.flatten(), grain_Ys.flatten(), grain_Zs.flatten()]).T
n_crds = len(test_crds)

#==============================================================================
# %% INSTANTIATE CONTROLLER - RUN BLOCK NO EDITING
#==============================================================================

progress_handler = nfutil.progressbar_progress_observer()
save_handler=nfutil.forgetful_result_handler()

controller = nfutil.ProcessController(save_handler, progress_handler,
                               ncpus=ncpus, chunk_size=chunk_size)

multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

#==============================================================================
# %% TEST ORIENTATIONS - RUN BLOCK NO EDITING
#==============================================================================

raw_confidence = nfutil.test_orientations(image_stack, experiment, test_crds,
                  controller, multiprocessing_start_method)

#==============================================================================
# %% POST PROCESS W WHEN TOMOGRAPHY HAS BEEN USED
#==============================================================================

grain_map, confidence_map = nfutil.process_raw_confidence(raw_confidence, grain_Xs.shape)

#==============================================================================
# %% SAVE PROCESSED GRAIN MAP DATA
#==============================================================================

status = 'SAVING NF MAP'
print('STARTED %s' %(status))
nfutil.save_nf_data(output_dir,output_stem,grain_map,confidence_map,grain_Xs,grain_Ys,grain_Zs,
                    experiment.exp_maps)
print('FINISHED %s' %(status)) 

#==============================================================================
# %% SAVE DATA AS VTK
#==============================================================================

status = 'SAVING NF MAP TO VTK'
print('STARTED %s' %(status))
vtkutil.output_grain_map_vtk(output_dir,[output_stem],output_stem,0.1)
print('FINISHED %s' %(status))

















