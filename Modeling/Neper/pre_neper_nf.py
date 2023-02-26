#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 21:41:27 2023

@author: djs522
"""


script_cmd = 'python /home/djs522/additional_sw/hedmTools/CHESS_hedmTools/Modeling/Neper/nf_grain_map_to_neper.py %s \
    --output_dir %s --output_stem %s --orientation_sym cubic --orientation_conv passive --completeness_thresh %0.3f \
    --voxel_size %0.4f --voxel_thresh %i --do_cc3d %s --cc3d_connectivity %i --do_reorder_ids %s --tess_max_iter %i --debug True \
'

nf_grain_map_dir = '/media/djs522/djs522_nov2020/chess_2020_11_dataset/n4_sample/nf/c0_0/dp718_nf_global_mg_8_1.00_1.00_rt_gc_ij_grain_map_data.npz'
output_dir = '/media/djs522/djs522_nov2020/chess_2020_11_dataset/n4_sample/modeling/nf_grain_map/'
output_stem = 'c0_0_dp718'
comp_thresh = 0.5
voxel_size = 0.0025
voxel_thresh = 50
do_cc3d = True
cc3d_con = 18
do_reorder_ids = True
tess_max_iter = 1000
print(script_cmd %(nf_grain_map_dir, output_dir, output_stem, comp_thresh, voxel_size, voxel_thresh, do_cc3d, cc3d_con, do_reorder_ids, tess_max_iter))

import numpy as np

data = np.load(nf_grain_map_dir)
print(data['Xs'].shape)