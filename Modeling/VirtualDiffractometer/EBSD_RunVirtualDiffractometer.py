#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:17:04 2021

@author: djs522
"""

# -*- coding: utf-8 -*-

#%% Imports
###############################################################################

import os

import matplotlib.pyplot as plt

import numpy as np

import h5py

from hexrd import rotations as hexrd_rot

import sys
sys.path.insert(1, '/home/djs522/additional_sw/hedmTools/CHESS_hedmTools/')

import Modeling.VirtualDiffractometer.VirtualDiffractometer as vd

#%% User Input
###############################################################################

# path variables
base_dir = '/home/djs522/additional_sw/hedmTools/CHESS_hedmTools/Modeling/VirtualDiffractometer/'
cfg_file = os.path.join(base_dir, 'EBSD_VirtDiffConfig.yml')
output_dir = os.path.join(base_dir, 'vd_output/')

# virt diff variables
ome_period = [-180, 180] # since variable is deprecated in hexrd.config, define here
fwhm = 1.2
cts_per_event = 25.0
min_inten = 1.0
max_inten = 65000
ncpus = 50

ebsd_dir = '/home/djs522/djs522/new_tribeam_dp718/2D_EBSD_data/718DP_AfterDef_Cleaned_Final.dream3d'
f = h5py.File(ebsd_dir, 'r')


#%% pull data from file
print("PREPROCESSING *************************************************")
samp_dimen = np.array(f['DataContainers']['ImageDataContainer']['_SIMPL_GEOMETRY']['DIMENSIONS'])
samp_spacing = np.array(f['DataContainers']['ImageDataContainer']['_SIMPL_GEOMETRY']['SPACING'])

tot_feat_ids = np.array([f['DataContainers']['ImageDataContainer']['CellData']['FeatureIds']])
tot_feat_quats = np.array([f['DataContainers']['ImageDataContainer']['CellData']['QUAT']])
tot_feat_x = np.array([f['DataContainers']['ImageDataContainer']['CellData']['X Position']]) / 1000.0 # mm
tot_feat_y = np.array([f['DataContainers']['ImageDataContainer']['CellData']['Y Position']]) / 1000.0 # mm
uni_feat_id, uni_feat_id_cnt = np.unique(tot_feat_ids, return_counts=True)

# reshape arrays
tot_feat_ids.shape = [tot_feat_ids.shape[2] * tot_feat_ids.shape[3], tot_feat_ids.shape[4]]
tot_feat_quats.shape = [tot_feat_quats.shape[2] * tot_feat_quats.shape[3], tot_feat_quats.shape[4]]
tot_feat_x.shape = [tot_feat_x.shape[2] * tot_feat_x.shape[3], tot_feat_x.shape[4]]
tot_feat_y.shape = [tot_feat_y.shape[2] * tot_feat_y.shape[3], tot_feat_y.shape[4]]
tot_feat_exp_maps = np.zeros([tot_feat_ids.size, 3])

num_vox = tot_feat_ids.size

total_tasks = 10
chunk_size = num_vox / total_tasks
for ii in np.arange(total_tasks):
    start_i = int(ii * chunk_size)
    end_i = int((ii+1) * chunk_size)
    if end_i > num_vox:
        end_i = num_vox
    tot_feat_exp_maps[start_i:end_i, :] = hexrd_rot.quat2exp_map(tot_feat_quats[start_i:end_i, :])

#%% assemble into grain_mat
grain_mat = np.zeros([tot_feat_x.size, 21])
grain_mat[:, 0] = tot_feat_ids.flatten()
grain_mat[:, 3:6] = tot_feat_exp_maps
grain_mat[:, 6] = tot_feat_x.flatten()
grain_mat[:, 8] = tot_feat_y.flatten()
grain_mat[:, 9:12] = 1

#%% Calculate Detector Intercepts and Intercept Frame Cache
###############################################################################
print("CALC INTERCEPTS *************************************************")
[pixel_intercepts, intercept_frame_cache] = vd.calc_diffraction_detector_intercepts_and_frame(grain_mat, cfg_file, 
                                               output_dir, ome_period=[-180.0, 180.0], 
                                               ncpus=ncpus, save_pixel_intercepts=True, 
                                               save_frame_cache_ims=True,
                                               chunk_size=None)

#%% Apply Point Spread Filter to Frame Cache
###############################################################################
print("APPLY FILTER *************************************************")
filter_frame_cache = vd.apply_ps_filter_to_frame_cache(cfg_file, intercept_frame_cache, output_dir,
                               det_psf_fwhm=fwhm, cts_per_event=cts_per_event, gauss_or_lorentz=False, 
                               min_intensity=0.01, max_intensity=65000, ncpus=ncpus,
                               save_frame_cache_ims=True)


#%% Plot a Frame to Check Diffraction Spot
###############################################################################
fig = plt.figure()
plt.imshow(intercept_frame_cache[509])

fig = plt.figure()
plt.imshow(filter_frame_cache[509])

plt.show()
    



