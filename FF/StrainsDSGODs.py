#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:14:18 2022

@author: djs522
"""

import numpy as np
from hexrd import fitgrains
from hexrd import config
from hexrd import rotations
from hexrd import matrixutil as mutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(1, '/home/djs522/additional_sw/hedmTools/CHESS_hedmTools/')
#import SingleGrainOrientationDistributions.SGODAnalysis as SGOD
#import SingleGrainOrientationDistributions.OrientationTools as OT
import FF.PostProcessStress as PPS


#fit_grains_dsgod(cfg,
#               grains_table,
#               show_progress=False,
#               ids_to_refine=None,
#               write_spots_files=True,
#               gFlag=gFlag_com_strain)


#%%
gFlag_com_strain = np.ones(12, dtype=bool)
gFlag_com_strain[:6] = 0

base_path = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/testing/dsgod_testing/'
cfg_path = base_path + 'c0_3_sc76_testing.yml'
grains_out_path = base_path + 'grains.out'
dsgod_path = base_path + 'grain_%i_dsgod_data0_85_reduced.npz'

orig_cfg = config.open(cfg_path)[0]
orig_grains_out = np.loadtxt(grains_out_path)

fit_results = []
quats_list = []
weights_list = []

for i, grain_id in enumerate(orig_grains_out[:, 0]):
    dsgod = np.load(dsgod_path %(grain_id))
    est_path = base_path + 'est_grains_%i.out' %(grain_id)
    
    quats = dsgod['dsgod_box_quat']
    exp_maps = rotations.expMapOfQuat(quats.T).T
    weights = dsgod['dsgod_box_dsgod']
    n_ori = weights.size
    
    quats_list.append(quats)
    weights_list.append(weights)
    
    print(grain_id, n_ori)
    grains_out = np.zeros([n_ori, 21])
    grains_out[:, 0] = np.arange(n_ori)
    grains_out[:, 3:6] = exp_maps
    grains_out[:, 6:9] = orig_grains_out[i, 6:9]
    grains_out[:, 9:] = orig_grains_out[i, 9:]
    
    np.savetxt(est_path, grains_out)
    
    cfg = orig_cfg
    cfg.fit_grains.estiamte = est_path
    
    fit = fitgrains.fit_grains_dsgod(cfg,
              grains_out,
              show_progress=False,
              ids_to_refine=None,
              write_spots_files=False,
              gFlag=gFlag_com_strain)
    
    fit_results.append(fit)
    
#%% 
from scipy.linalg import logm

j = 2
fit0 = np.array(fit_results[j])
stats0 = fit0[:, :3].astype(float)
params0 = np.vstack(fit0[:, 3])
strains0 = np.zeros([params0.shape[0], 6])

for i in range(params0.shape[0]):
    strains0[i, :] = PPS.voigt_strain_t2v(logm(np.linalg.inv(mutil.vecMVToSymm(params0[i, 6:]))))
quats = quats_list[j]
weights = weights_list[j]

grains0 = np.hstack([stats0, params0, strains0])

print(np.average(grains0[:, 15:], axis=0, weights=weights))
print(orig_grains_out[j, 15:])

#%%
ind = (grains0[:, 2] < 1e-2)
grains0 = grains0[ind, :]
print(grains0.shape)
print(np.min(grains0[:, [1, 2]], axis=0), np.mean(grains0[:, [1, 2]], axis=0), np.max(grains0[:, [1, 2]], axis=0))


fig = plt.figure()
plt.hist(grains0[:, 16])

#%%

strains = PPS.voigt_strain_vt2_3d(grains0[:, 15:])

eig = np.linalg.eig(PPS.voigt_strain_v2t(orig_grains_out[j, 15:]))
print(eig[1].T @ PPS.voigt_strain_v2t(orig_grains_out[j, 15:]) @ eig[1])

#%%
fig = plt.figure()
ax = Axes3D(fig)
for i in range(strains.shape[0]):
    t = eig[1].T @ strains[i, :, :] @ eig[1] 
    ax.scatter(t[0, 0], t[1, 1], t[2, 2], c=weights[ind][i], vmin=np.min(weights), vmax=np.max(weights))
    
#%%
fig = plt.figure()
ax = Axes3D(fig)
rod = rotations.quat2rod(quats)[ind, :]
n_comp = 0
ax.scatter(rod[:, 0], rod[:, 1], rod[:, 2], c=strains[:, n_comp, n_comp])


#%%

print(np.linalg.norm(grains_out[:, 3:6] - grains0[:, 3:6], axis=1))








