#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:26:48 2023

@author: djs522
"""

import numpy as np
ff_data_path = '/media/djs522/djs522_nov2020/chess_2020_11_dataset/n4_sample/ff/c0_0/c0_0_dp718_ff_data.npz'
ff_data = np.load(ff_data_path)

grain_ids = ff_data['GRAIN_IDS']
comp = ff_data['COMPLETENESS']
chi2 = ff_data['CHI2_FIT']
exp_map = ff_data['EXP_MAPS']
com = ff_data['COM_POSITION']

grains_out = np.zeros([grain_ids.size, 21])
grains_out[:, 0] = grain_ids.flatten()
grains_out[:, 1] = comp.flatten()
grains_out[:, 2] = chi2.flatten()
grains_out[:, 3:6] = exp_map
grains_out[:, 6:9] = com

grains_out_path = '/media/djs522/djs522_nov2020/chess_2020_11_dataset/n4_sample/modeling/c0_0_dp718_no_strains_grains.out'
np.savetxt(grains_out_path, grains_out)



script_cmd = 'python /home/djs522/additional_sw/hedmTools/CHESS_hedmTools/Modeling/Neper/ff_centroid_to_neper.py %s \
    --output_dir %s --output_stem %s --orientation_sym cubic --orientation_conv passive --completeness_thresh %0.3f --chi2_thresh %0.3f \
    --voxel_size %0.4f --x_lower %0.3f --x_upper %0.3f --y_lower %0.3f --y_upper %0.3f --z_lower %0.3f --z_upper %0.3f --plot True \
'
output_dir = '/media/djs522/djs522_nov2020/chess_2020_11_dataset/n4_sample/modeling/ff_centroid/'
output_stem = 'c0_0_dp718'
comp_thresh = 0.5
chi2_thresh = 1e0
voxel_size = 0.003
x_lower = -0.5
x_upper = 0.5
y_lower = -0.3
y_upper = 0.2
z_lower = -0.5
z_upper = 0.5
print(script_cmd %(grains_out_path, output_dir, output_stem, comp_thresh, chi2_thresh, voxel_size, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper))