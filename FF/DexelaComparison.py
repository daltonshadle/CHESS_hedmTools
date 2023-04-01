#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:11:21 2023

@author: djs522
"""

import numpy as np
import os
import yaml
from hexrd import instrument
import matplotlib.pyplot as plt

# shadle_nov_2020
analysis = 'shadle_nov2020'
basepath = '/media/djs522/djs522_nov2020/dexela_distortion/shadle_nov_2020/'
instr_paths = {'chunk':os.path.join(basepath, '1_dexela_nov_2020_61_332_ceo2_instr__chunk.yml'),
             'chunk_tilt':os.path.join(basepath, '2_dexela_nov_2020_61_332_ceo2_instr__tilts.yml'),
             'chunk_pos':os.path.join(basepath, '3_dexela_nov_2020_61_332_ceo2_instr__pos.yml'),
             'chunk_tilt_pos':os.path.join(basepath, '4_dexela_nov_2020_61_332_ceo2_instr__tilts_and_pos.yml')}

# greeley_april_2018
# analysis = 'greeley_april2018'
# basepath = '/media/djs522/djs522_nov2020/dexela_distortion/greeley_april_2018_??/'
# instr_paths = {'chunk':os.path.join(basepath, '1_dexela_ceo2_recali_nodistortion_mpanel.yml'),
#               'chunk_tilt':os.path.join(basepath, '2_dexela_ceo2_recali_nodistortion_mpanel_ceo2refit-tiltonly.yml'),
#               'chunk_pos':os.path.join(basepath, '3_dexela_ceo2_recali_nodistortion_mpanel_ceo2refit-distonly.yml'),
#               'chunk_tilt_pos':os.path.join(basepath, '4_dexela_ceo2_recali_nodistortion_mpanel_ceo2refit-tiltanddist.yml')}

# kirks_dec_2022
analysis = 'kirks_dec2022'
basepath = '/media/djs522/djs522_nov2020/dexela_distortion/kirks_dec_2022/'
instr_paths = {'chunk':os.path.join(basepath, '1_dexela_90-524kev_ceo2_instr_mpanel.yml'),
             'chunk_tilt':os.path.join(basepath, '2_dexela_90-524kev_ceo2_instr_mpanel__tilt.yml'),
             'chunk_pos':os.path.join(basepath, '3_dexela_90-524kev_ceo2_instr_mpanel__pos.yml'),
             'chunk_tilt_pos':os.path.join(basepath, '4_dexela_90-524kev_ceo2_instr_mpanel__tilt_and_pos.yml')}

instr_dict = {'chunk':None,
             'chunk_tilt':None,
             'chunk_pos':None,
             'chunk_tilt_pos':None}

for instr_key, instr_path in instr_paths.items():
    with open(instr_path, "r") as stream:
        try:
            instr_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    instr_dict[instr_key] =  instrument.HEDMInstrument(instrument_config=instr_yaml)
    

#%% 3d quiver plots

from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots(nrows=2, ncols=4, subplot_kw=dict(projection='3d'))
switch_yz = True

for i, det_key in enumerate(instr_dict['chunk'].detectors.keys()):
    for instr_key, instr in instr_dict.items():
        ax[int(i / 4), int(i % 4)].set_title(det_key)
        
        x = instr.detectors[det_key].tvec[0]
        y = instr.detectors[det_key].tvec[1]
        z = instr.detectors[det_key].tvec[2]
        
        normal = instr.detectors[det_key].normal
        u = normal[0]
        v = normal[1]
        w = normal[2]
        
        if switch_yz:
            ax[int(i / 4), int(i % 4)].scatter(x, z, y)
            ax[int(i / 4), int(i % 4)].quiver(x, z, y, u, w, v, length=0.1, normalize=True)
            
            ax[int(i / 4), int(i % 4)].set_xlabel('X_lab')
            ax[int(i / 4), int(i % 4)].set_ylabel('Z_lab')
            ax[int(i / 4), int(i % 4)].set_zlabel('Y_lab')
        else:
            ax[int(i / 4), int(i % 4)].scatter(x, y, z)
            ax[int(i / 4), int(i % 4)].quiver(x, y, z, u, v, w, length=0.1, normalize=True)
            
            ax[int(i / 4), int(i % 4)].set_xlabel('X_lab')
            ax[int(i / 4), int(i % 4)].set_ylabel('Y_lab')
            ax[int(i / 4), int(i % 4)].set_zlabel('Z_lab')
        
plt.show()


#%% coaxiality and distance

import seaborn as sns
from matplotlib.cm import ScalarMappable
cmap = plt.get_cmap("viridis")

dist_range = [-2.0, 2.0]
tilt_range = [0, 0.7]

compare_instr_key = 'chunk'
compare_instr = instr_dict[compare_instr_key]

fig_dist, ax_dist = plt.subplots(nrows=2, ncols=4)
fig_dist.suptitle('%s Detector Tvec Difference (current - %s)' %(analysis, compare_instr_key))

fig_tilt, ax_tilt = plt.subplots(nrows=2, ncols=4)
fig_tilt.suptitle('%s Detector Tilt Difference (current - %s)' %(analysis, compare_instr_key))

for i, det_key in enumerate(instr_dict['chunk'].detectors.keys()):
    print('\n', det_key)
    dist_diff_data = []
    tilt_diff_data = []
    for instr_key, instr in instr_dict.items():
        
        tvec = instr.detectors[det_key].tvec
        normal = instr.detectors[det_key].normal
        
        c_tvec = compare_instr.detectors[det_key].tvec
        c_normal = compare_instr.detectors[det_key].normal
        
        dist_diff = np.linalg.norm(tvec - c_tvec)
        normal_ang_diff = np.arccos(np.dot(normal, c_normal) / (np.linalg.norm(normal) * np.linalg.norm(c_normal)))
        
        print(instr_key, dist_diff, np.degrees(normal_ang_diff))
        
        dist_diff_data.append(np.hstack([tvec - c_tvec, dist_diff]))
        #tilt_diff_data.append(np.hstack([np.degrees(normal_ang_diff), normal - c_normal]))
        tilt_diff_data.append(np.hstack([np.degrees(normal_ang_diff)]))
    
    y_labels = []
    x_dist_labels = []
    x_tilt_labels = []
    if i % 4 == 0:
        y_labels = instr_dict.keys()
    if i / 4 >= 1:
        x_dist_labels = ['x', 'y', 'z', 'total']
        x_tilt_labels = ['normal angle'] #['ang_deg', 'norm_x', 'norm_y', 'norm_z']
    ax_dist[int(i / 4), int(i % 4)].set_title(det_key)
    dist_hm = sns.heatmap(np.vstack(dist_diff_data), vmin=dist_range[0], vmax=dist_range[1],
                          cmap=cmap, annot=True, fmt='.2g', annot_kws=None, 
                          cbar=False,
                          xticklabels=x_dist_labels, yticklabels=y_labels, 
                          ax=ax_dist[int(i / 4), int(i % 4)])
    ax_tilt[int(i / 4), int(i % 4)].set_title(det_key)
    tilt_hm = sns.heatmap(np.vstack(tilt_diff_data), vmin=tilt_range[0], vmax=tilt_range[1], 
                          cmap=cmap, annot=True, fmt='.2g', annot_kws=None, 
                          cbar=False,
                          xticklabels=x_tilt_labels, yticklabels=y_labels, 
                          ax=ax_tilt[int(i / 4), int(i % 4)])


fig_dist.subplots_adjust(right=0.8)
norm_dist = plt.Normalize(dist_range[0], dist_range[1])
sm_dist =  ScalarMappable(norm=norm_dist, cmap=cmap)
sm_dist.set_array([])
cbar_ax = fig_dist.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig_dist.colorbar(sm_dist, cax=cbar_ax)
cbar.ax.set_title("mm")


fig_tilt.subplots_adjust(right=0.8)
norm_tilt = plt.Normalize(tilt_range[0], tilt_range[1])
sm_tilt =  ScalarMappable(norm=norm_tilt, cmap=cmap)
sm_tilt.set_array([])
cbar_ax = fig_tilt.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig_tilt.colorbar(sm_tilt, cax=cbar_ax)
cbar.ax.set_title("deg")

   
plt.show()      