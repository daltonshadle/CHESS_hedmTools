#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:28:44 2023

@author: djs522
"""
import numpy as np

from hexrd import config
from hexrd import instrument
from hexrd import rotations
from hexrd import constants

from hexrd.transforms.xfcapi import \
    anglesToGVec, \
    angularDifference, \
    detectorXYToGvec, \
    gvecToDetectorXY, \
    anglesToDVec, \
    makeGVector, \
    makeOscillRotMatArray, \
    makeEtaFrameRotMat

from hexrd.matrixutil import vecMVToSymm, strainVecToTen

from hexrd import xrdutil

from scipy import ndimage



#%%
starting_omega = None
ending_omega = None
cfg_file = '/media/djs522/djs522_nov2020/chess_2020_11_dataset/n4_sample/ff/c1_1/scan_94/dp718_scan_94_40.yml'
cfg = config.open(cfg_file)[0]

# initialize grain mat from grains.out
grain_mat = np.loadtxt(cfg.fit_grains.estimate)
select_grain_ids = '1'
if select_grain_ids is not None:
    if ".txt" in select_grain_ids:
        good_ids  = np.loadtxt(select_grain_ids).astype(int)
    elif ".npy" in select_grain_ids:
        good_ids = np.load(select_grain_ids).astype(int)
    elif select_grain_ids.isnumeric():
        select_grain_ids = int(select_grain_ids)
        good_ids = np.array(select_grain_ids).astype(int)
    else:
        raise ValueError("Only .txt or .npy files for select grain ids (%s not supported)" %(select_grain_ids))
        
    good_ind = np.searchsorted(grain_mat[:, 0], good_ids)
    grain_mat = np.atleast_2d(grain_mat[good_ind, :])

# get active hkls for generating eta-omega maps
active_hkls = cfg.find_orientations.orientation_maps.active_hkls
if active_hkls == 'all':
    active_hkls = None

max_tth = cfg.fit_grains.tth_max
if max_tth:
    if type(cfg.fit_grains.tth_max) != bool:
        max_tth = float(max_tth)
else:
    max_tth = None

# load plane data
plane_data = cfg.material.plane_data
plane_data.tThMax = np.radians(max_tth)
plane_data.set_lparms(plane_data.get_lparms())
print(plane_data)

# load instrument
instr = cfg.instrument.hedm
det_keys = instr.detectors.keys()

# threshold on frame cache building eta-omega maps
fit_grains_threshold = cfg.fit_grains.threshold

# grab image series
if cfg.__dict__['_cfg']['image_series']['format'] == 'frame-cache':
    ims_dict = cfg.image_series
else:
    raise ValueError("Only frame-caches supported! (%s not supported)" %(cfg.__dict__['_cfg']['image_series']['format']))

# set omega period
oims0 = next(iter(ims_dict.values()))
ome_ranges = [np.radians([i['ostart'], i['ostop']])
              for i in oims0.omegawedges.wedges]
omegas = ims_dict[next(iter(ims_dict))].omega
if starting_omega is None:
    starting_omega = omegas[0, 0]
if ending_omega is None:
    ending_omega = omegas[-1, 1]
ome_period = np.radians([starting_omega, ending_omega])
# delta omega in DEGREES grabbed from first imageseries in the dict
delta_ome = oims0.omega[0, 1] - oims0.omega[0, 0]

# set eta ranges
eta_ranges=[(-np.pi, np.pi), ]

# set up multiprocessing details
grain_mat_ids = grain_mat[:, 0]
num_grains = grain_mat_ids.size
ncpus = num_grains if num_grains < cfg.multiprocessing else cfg.multiprocessing

def create_exp_map_grid(avg_exp_map, misorientation_bnd=1.0, misorientation_spacing=0.25):
    mis_amt = np.radians(misorientation_bnd)
    mis_spacing = np.radians(misorientation_spacing)
    
    ori_pts = np.arange(-mis_amt, (mis_amt+(mis_spacing*0.999)), mis_spacing)
    
    ori_Xs, ori_Ys, ori_Zs = np.meshgrid(ori_pts, ori_pts, ori_pts)
    ori_grid = np.vstack([ori_Xs.flatten(), ori_Ys.flatten(), ori_Zs.flatten()]).T
    
    all_exp_maps = ori_grid + avg_exp_map
    return all_exp_maps, ori_Xs.shape
    
#%%
misorientation_bnd = 2.5 # deg
misorientation_spacing = 2.5 # deg
threshold = 1
pad_size = 5 # pixels
interp = 'bilinear'

for igrain, grain_id in enumerate(grain_mat_ids):
    print("Processing grain %i / %i | %s-" %(igrain+1, num_grains, cfg.analysis_id))
    
    # get grain info
    grain = grain_mat[grain_mat[:, 0] == grain_id, :]
    
    # get trial orientation grid
    all_exp_maps, grid_shape = create_exp_map_grid(grain[:, 3:6], 
                                       misorientation_bnd=misorientation_bnd, 
                                       misorientation_spacing=misorientation_spacing)
    num_oris = all_exp_maps.shape[0]
    print(num_oris)
    
    curr_grain_mat = np.zeros([num_oris, 12])
    curr_grain_mat[:, :3] = all_exp_maps
    curr_grain_mat[:, 3:] = grain[:, 6:15]

    # simulate rotation series
    # extract simulation results
    # sim_results_p = sim_results[detector_id]
    # hkl_ids = sim_results_p[0][0]
    # hkls_p = sim_results_p[1][0]
    # ang_centers = sim_results_p[2][0]
    # xy_centers = sim_results_p[3][0]
    # ang_pixel_size = sim_results_p[4][0]
    sim_results = instr.simulate_rotation_series(
        plane_data, curr_grain_mat,
        eta_ranges=eta_ranges,
        ome_ranges=ome_ranges,
        ome_period=ome_period)
    
    full_hkl = xrdutil._fetch_hkls_from_planedata(plane_data) # [id, h, k, l]
    full_hkl = np.vstack([full_hkl, full_hkl]) # double for freidel pairs
    full_ori_inten_array = np.ones([full_hkl.shape[0], num_oris]) * -1
    
    # ultiimately wnat for each orientation for each reeflection, what is the intensity? / is reflection on panel?
    # initialize return to be -1's, if its on the panel, it will have something >= 0
    # sort by hkl ID for each grain
    
    debug_xy = []
    
    # for each panel
    for detector_id, panel in instr.detectors.items():
        # sim_results[panel_id][item_key][grain][reflection]
        
        # pull out the OmegaImageSeries for this panel from input dict
        ome_imgser = instrument._parse_imgser_dict(ims_dict,
                                                    detector_id,
                                                    roi=panel.roi)
        
        panel_xy = panel.pixel_coords
        pad_ind = np.meshgrid(np.arange(-pad_size, pad_size+1), np.arange(-pad_size, pad_size+1))
        pad_ind = np.vstack([pad_ind[0].flatten(), pad_ind[1].flatten()]).T
        
        # for each orientation
        for j_ori in range(num_oris):
            # grab the hkl, angs, det_xy for this orientation and this panel
            hkl_ori = sim_results[detector_id][1][j_ori]
            ang_ori = sim_results[detector_id][2][j_ori]
            detxy_ori = sim_results[detector_id][3][j_ori]
            num_panel_refl = hkl_ori.shape[0]
            
            # for each reflection
            for k_refl in range(num_panel_refl):
                # grab full hkl ind
                hkl_refl_ind = np.where(np.all(full_hkl[:, 1:] == hkl_ori[k_refl, :], axis=1))[0]
                if full_ori_inten_array[hkl_refl_ind[0], j_ori] == -1: # for dealing with freidel pairs
                    hkl_refl_ind = hkl_refl_ind[0]
                else:
                    hkl_refl_ind = hkl_refl_ind[1]
                
                # get omega and frame index for reflection
                ome_eval = np.degrees(ang_ori[k_refl, 2])
                frame_ind = ome_imgser.omega_to_frame(ome_eval)[0]
                

                if frame_ind == -1:
                    msg = """
                    window for (%d%d%d) falls outside omega range
                    """ % tuple(hkl_ori[k_refl])
                    print(msg)
                    continue
                else:
                    # grab pixel indices
                    ijs = panel.cartToPixel(detxy_ori[k_refl]).astype(int)
                    pad_ijs = ijs + pad_ind
                    pad_ijs = pad_ijs[~np.any(pad_ijs < 0, axis=1)] # remove any negative indices
                    pad_ijs = pad_ijs[pad_ijs[:, 0] < panel_xy[0].shape[0]] # remove any over the row max
                    pad_ijs = pad_ijs[pad_ijs[:, 1] < panel_xy[0].shape[1]] # remove any over the col max
                    pad_x = panel_xy[1][pad_ijs[:, 0], pad_ijs[:, 1]]
                    pad_y = panel_xy[0][pad_ijs[:, 0], pad_ijs[:, 1]]
                    pad_weights = 1 / np.linalg.norm(np.vstack([pad_x, pad_y]).T - detxy_ori[k_refl], axis=1)
                    pad_weights = pad_weights / np.sum(pad_weights)
                    
                    if detector_id == 'ff1_0_0' and k_refl == 0:
                        debug_panel_xy = panel_xy
                        debug_frame = ome_imgser[frame_ind]
                        debug_xy.append(detxy_ori[k_refl])
                        debug_grad = panel.pixel_eta_gradient()
 
                    # grab intensity from frame
                    inten_ori = np.average(ome_imgser[frame_ind][pad_ijs[:, 0], pad_ijs[:, 1]], weights=pad_weights)
                    
                    full_ori_inten_array[hkl_refl_ind, j_ori] = inten_ori
#%%

print(np.sum(full_ori_inten_array > 0, axis=0), full_ori_inten_array.shape)

#%%
# ret = instr.pull_spots(plane_data, curr,
#                imgser_dict)     
     
import matplotlib.pyplot as plt


plot_xy = np.vstack(debug_xy)
frame_xy = np.vstack([debug_panel_xy[1].flatten(), debug_panel_xy[0].flatten()]).T
frame = debug_frame.flatten()

ind = np.where(np.linalg.norm(frame_xy - plot_xy[int(plot_xy.shape[0]/2), :], axis=1) < 4)[0]
frame_xy = frame_xy[ind, :]
frame = frame[ind]

fig = plt.figure()
plt.scatter(frame_xy[:, 0], frame_xy[:, 1], c=frame, vmax=1000, s=150)
plt.scatter(plot_xy[:, 0], plot_xy[:, 1], c='r')



grad = np.degrees(debug_grad)
fig = plt.figure()
plt.imshow(grad)
plt.show()
