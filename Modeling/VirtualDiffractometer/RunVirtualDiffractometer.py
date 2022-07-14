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

import VirtualDiffractometer as vd

#%% User Input
###############################################################################

# path variables
base_dir = '/home/djs522/additional_sw/hedmTools/CHESS_hedmTools/Modeling/VirtualDiffractometer/'
cfg_file = os.path.join(base_dir, 'VirtDiffConfig.yml')
output_dir = os.path.join(base_dir, 'vd_output/')

# virt diff variables
ome_period = [-180, 180] # since variable is deprecated in hexrd.config, define here
fwhm = 0.8 
# NOTE: fwhm values for detectors and filters below (psf_sigma values taken from APEX-RD)
# psf_sigma: GE = 0.55, Dexela = 0.4
# fwhm GAUSS: GE = 0.85, Dexela = 0.62
# fwhm LORENTZ: GE = 1.1, Dexela = 0.8

cts_per_event = 100.0
min_inten = 1.0
max_inten = 65000
ncpus = 2

vox_size = 0.015
x = np.arange(-0.2, 0.2, step=vox_size)
y = np.arange(-0.2, 0.2, step=vox_size)
z = np.arange(-0.2, 0.2, step=vox_size)

xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
pos = np.vstack([xv.flatten(), yv.flatten(), zv.flatten()]).T

num_vox = xv.size
split_idx = int(num_vox / 2)
print("Number of voxels in example microstructure: %i" %(num_vox))
grain_mat = np.zeros([num_vox, 21])
grain_mat[:split_idx, 0] = 0 
grain_mat[:split_idx, 3:6] = np.array([4.2537e-01,   2.0453e-01 ,  5.8538e-01])
grain_mat[:split_idx, 6:9] = pos[:split_idx, :]

grain_mat[split_idx:, 0] = 1
grain_mat[split_idx:, 3:6] = np.array([3.3462e-01,   1.2188e-01 ,  6.0102e-01])
grain_mat[split_idx:, 6:9] = pos[split_idx:, :]

grain_mat[:, 9:12] = 1

#%% Calculate Detector Intercepts and Intercept Frame Cache
###############################################################################
[pixel_intercepts, intercept_frame_cache] = vd.calc_diffraction_detector_intercepts_and_frame(grain_mat, cfg_file, 
                                               output_dir, ome_period=[-180.0, 180.0], 
                                               ncpus=ncpus, save_pixel_intercepts=True, 
                                               save_frame_cache_ims=True,
                                               chunk_size=None)

#%% Apply Point Spread Filter to Frame Cache
###############################################################################
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
    



