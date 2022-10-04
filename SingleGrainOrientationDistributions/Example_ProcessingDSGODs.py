#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 21:48:13 2022

@author: djs522
"""


import os
import glob
import numpy as np
import SGODAnalysis
import OrientationTools

path_stem = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_1/dsgod/dsgod/'


load_step_stem = ['c0_1', 'c0_2', 'c0_3', 'c1_1']
path_stem = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/%s/dsgod/dsgod/'

comp_thresh = 0.85

i = 0

for ls in load_step_stem:
    temp_path = path_stem %(ls)
    out_folder = temp_path + 'output/'
    #if not os.path.exists(out_folder):
    #    os.mkdir(out_folder)
    
    print(ls)
    for f in glob.glob(temp_path + 'dsgod*/*_data.npz', recursive=True):
        #print(f)
        [grain_quat, grain_mis_quat, grain_odf] = SGODAnalysis.process_dsgod_file(f, comp_thresh=comp_thresh, inten_thresh=0, 
                                        do_avg_ori=True, do_conn_comp=True, 
                                        save=True, connectivity_type=18)
    #     i += 1
    #     if i > 4:
    #         break
    # break


#%%

# /media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_1/dsgod/dsgod/dsgod_dp718-1_38
grain_stats = np.zeros([4, 3, 2125, 5])
load_step_stem = ['c0_1', 'c0_2', 'c0_3', 'c1_1']
for i, ls in enumerate(load_step_stem):
    temp_path = path_stem %(ls)
    out_folder = temp_path + 'output/'
    #if not os.path.exists(out_folder):
    #    os.mkdir(out_folder)
    
    print(ls)
    temp_scan_list = glob.glob(temp_path + 'dsgod*')
    temp_scan_list.sort()
    print(temp_scan_list)
    for j, scan in enumerate(temp_scan_list):
        print(scan)
        for f in glob.glob(scan + '/*0_85_reduced.npz'):
            fname = os.path.basename(f)
            g_id = int(fname.split('_')[1])
            
            dsgod_data = np.load(f)
            mis_quat = dsgod_data['dsgod_box_mis_quat']
            quat = dsgod_data['dsgod_box_quat']
            odf = dsgod_data['dsgod_box_dsgod']
            sum_inten = dsgod_data['dsgod_sum_inten']
            
            [norm_sigma, norm_gamma, norm_kappa, sigma, gamma, kappa] = SGODAnalysis.calc_misorient_moments(grain_mis_quat=mis_quat, grain_odf=odf)
            grain_stats[i, j, g_id, 0] = norm_sigma
            grain_stats[i, j, g_id, 1] = norm_gamma
            grain_stats[i, j, g_id, 2] = norm_kappa
            grain_stats[i, j, g_id, 3] = odf.size
            grain_stats[i, j, g_id, 4] = sum_inten

#%%

i = 3

l_0 =  np.argmax(grain_stats[0, :, :, i], axis=0)
l_1 =  np.argmax(grain_stats[1, :, :, i], axis=0)
l_2 =  np.argmax(grain_stats[2, :, :, i], axis=0)
l_3 =  np.argmax(grain_stats[3, :, :, i], axis=0)


print(np.where(l_0 - l_1 != 0)[0].shape)
print(np.where(l_3 - l_2 != 0)[0].shape)
print(np.where(l_3 - l_1 != 0)[0].shape)

#%%

import shutil



load_step_stem = ['c0_1', 'c0_2', 'c0_3', 'c1_1']
l = [l_0, l_1, l_2, l_3]
for i, ls in enumerate(load_step_stem):
    temp_path = path_stem %(ls)
    out_folder = temp_path + 'output/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    temp_scan_list = glob.glob(temp_path + 'dsgod*')
    temp_scan_list.sort()
    print(temp_scan_list)
    for j, scan in enumerate(temp_scan_list):
        temp_ids = np.where(l[i] == j)[0]
        
        for t_id in temp_ids:
            src = scan + '/grain_%i_dsgod_data0_85_reduced.npz' %(t_id)
            dst = out_folder + '/grain_%i_dsgod_data_0_85_reduced.npz' %(t_id)
            shutil.copyfile(src, dst)

#%%

#/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_2/dsgod/dsgod/output
grain_stats = np.zeros([4, 2125, 5])
load_step_stem = ['c0_1', 'c0_2', 'c0_3', 'c1_1']
for i, ls in enumerate(load_step_stem):
    temp_path = path_stem %(ls)
    out_folder = temp_path + 'output/'
    
    print(ls)
    print(out_folder)
    for f in glob.glob(out_folder + '/*0_85_reduced.npz'):
        fname = os.path.basename(f)
        g_id = int(fname.split('_')[1])
        
        dsgod_data = np.load(f)
        mis_quat = dsgod_data['dsgod_box_mis_quat']
        quat = dsgod_data['dsgod_box_quat']
        odf = dsgod_data['dsgod_box_dsgod']
        sum_inten = dsgod_data['dsgod_sum_inten']
        
        [norm_sigma, norm_gamma, norm_kappa, sigma, gamma, kappa] = SGODAnalysis.calc_misorient_moments(grain_mis_quat=mis_quat, grain_odf=odf)
        grain_stats[i, g_id, 0] = norm_sigma
        grain_stats[i, g_id, 1] = norm_gamma
        grain_stats[i, g_id, 2] = norm_kappa
        grain_stats[i, g_id, 3] = odf.size
        grain_stats[i, g_id, 4] = sum_inten

#%%
import matplotlib.pyplot as plt
fig = plt.figure()
i = 3

ind = grain_stats[i, :, 3] > 25
print(np.sum(ind))
plt.scatter(grain_stats[i, ind, 1], grain_stats[i, ind, 2], c=grain_stats[i, ind, 4])

#%%
print(np.where(grain_stats[i, :, 1] > 1)[0])
print(np.where(grain_stats[i, :, 1] > 1)[0].size)

#%%

grain_rod = OrientationTools.quat2rod(quat)
SGODAnalysis.plot_grain_dsgod(grain_rod, grain_odf=odf, reverse_map=False, 
                      just_faces=False, no_axis=False, 
                      scatter_size=400, fig=None, ori_ax=None)



#%%
load_step_stem = ['c1_1']
for i, ls in enumerate(load_step_stem):
    temp_path = path_stem %(ls)
    out_folder = temp_path + 'output/'
    #if not os.path.exists(out_folder):
    #    os.mkdir(out_folder)
    
    print(ls)
    for f in glob.glob(temp_path + 'dsgod*/*_reduced.npz', recursive=True):
        fname = os.path.basename(f)
        
        dsgod_data = np.load(f)
        mis_quat = dsgod_data['dsgod_box_mis_quat']
        quat = dsgod_data['dsgod_box_quat']
        odf = dsgod_data['dsgod_box_dsgod']
        
        [norm_sigma, norm_gamma, norm_kappa, sigma, gamma, kappa] = SGODAnalysis.calc_misorient_moments(grain_mis_quat=mis_quat, grain_odf=odf)
        if norm_gamma > 1.0:
            grain_rod = OrientationTools.quat2rod(quat)
            SGODAnalysis.plot_grain_dsgod(grain_rod, grain_odf=odf, reverse_map=False, 
                                  just_faces=False, no_axis=False, 
                                  scatter_size=400, fig=None, ori_ax=None)
            

# grain_quat = grain_quat[:, grain_odf > 0].T
# grain_odf = grain_odf[grain_odf > 0]
# grain_rod = OrientationTools.quat2rod(grain_quat)
# SGODAnalysis.plot_grain_dsgod(grain_rod, grain_odf=grain_odf, reverse_map=False, 
#                      just_faces=False, no_axis=False, 
#                      scatter_size=400, fig=None, ori_ax=None)