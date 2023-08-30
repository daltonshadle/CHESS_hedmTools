#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:16:34 2023

@author: djs522
"""

import numpy as np
from scipy.ndimage import morphology

# Region ID to find neighbors for (replace 3 with the desired region ID)
target_grain_id = 3

# Replace 'connected_regions' with your 3D array of connected region IDs
input_data = inputs[0]
gm = input_data.PointData['grain_id']
gm_shape = (400, 180, 400)
gm = np.reshape(gm, gm_shape)

# perform threshold and dilation
grain_gm_ind = (gm == target_grain_id)
dilated_grain_gm_ind = morphology.binary_dilation(grain_gm_ind, iterations=1)

# get neighboring grain ids
neighboring_grain_ids = np.unique(gm[dilated_grain_gm_ind])
neighboring_grain_gm_ind = np.zeros(grain_gm_ind.shape)
for ngid in neighboring_grain_ids:
    n_ind = (gm == ngid)
    neighboring_grain_gm_ind[n_ind] = 1
    
    if ngid == target_grain_id:
        neighboring_grain_gm_ind[n_ind] = 2

neighboring_grain_gm_ind = neighboring_grain_gm_ind.flatten()

for key in input_data.PointData.keys():
    output.PointData.append(input_data.PointData[key], key)

output.PointData.append(neighboring_grain_gm_ind, 'neighbor_grains_ind')

