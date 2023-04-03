#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:05:33 2023

@author: djs522
"""

import sys
import os

import numpy as np

import matplotlib.pyplot as plt

import copy

from hexrd import config
from hexrd.fitting.calibration import InstrumentCalibrator, PowderCalibrator
from hexrd import material
from hexrd import rotations

sys.path.append('/home/djs522/additional_sw/hedmTools/')
from CHESS_hedmTools.FF import CalibrateInstrumentFromSX as CalibSX
from CHESS_hedmTools.FF import ChunkDexela

#%% extra functions
def make_matl(mat_name, sgnum, lparms, hkl_ssq_max=50):
    '''
    This is a simple function for creating a ceria material quickly without
    a materials config file

    Parameters
    ----------
    mat_name : TYPE
        DESCRIPTION.
    sgnum : TYPE
        DESCRIPTION.
    lparms : TYPE
        DESCRIPTION.
    hkl_ssq_max : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    matl : TYPE
        DESCRIPTION.

    '''
    
    matl = material.Material(mat_name)
    matl.sgnum = sgnum
    matl.latticeParameters = lparms
    matl.hklMax = hkl_ssq_max

    nhkls = len(matl.planeData.exclusions)
    matl.planeData.set_exclusions(np.zeros(nhkls, dtype=bool))
    return matl
    
def set_instr_calib_flags(init_instr, 
                          instr_flags=np.zeros(7, dtype=bool),
                          panel_tilt_flags=np.array([1, 1, 0], dtype=bool), 
                          panel_pos_flags=np.array([1, 1, 1], dtype=bool),
                          panel_distortion_flags=np.array([], dtype=bool)):
    '''
    Sets the insturment calibration flags from initial instr object and input. 
    The calibration flags for the instrument/ powder calibration function is
    listed below.
    
    -----
    Calibration parameter flags
     for instrument level, len is 7
     [beam energy,
      beam azimuth,
      beam elevation,
      chi,
      tvec[0],
      tvec[1],
      tvec[2],
      ]
     
     for each panel, order is:
     [tilt[0],
      tilt[1],
      tilt[2],
      tvec[0],
      tvec[1],
      tvec[2],
      <dparams>,
      ]
     len is 6 + len(dparams) for each panel
     by default, dparams are not set for refinement
     
     the last element in the powder calibration flag array is reserved for the
     lattice parameter, this is taken care of outside this function
     -----

    Parameters
    ----------
    init_instr : TYPE
        DESCRIPTION.
    instr_flags : TYPE, optional
        DESCRIPTION. The default is np.zeros(7, dtype=bool).
    panel_tilt_flags : TYPE, optional
        DESCRIPTION. The default is np.array([1, 1, 0], dtype=bool).
    panel_pos_flags : TYPE, optional
        DESCRIPTION. The default is np.array([1, 1, 1], dtype=bool).
    panel_distortion_flags : TYPE, optional
        DESCRIPTION. The default is np.array([], dtype=bool).

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    calibration_flags : TYPE
        DESCRIPTION.
    instr : TYPE
        DESCRIPTION.

    '''
    
    # intialize new instr
    instr = copy.copy(init_instr)
    
    # first, add tilt calibration
    # This needs to come from the GUI input somehow
    #rme = rotations.RotMatEuler(np.zeros(3), 'xyz', extrinsic=True)
    #instr.tilt_calibration_mapping = rme
    
    # initialize calibration flags
    calibration_flags = instr_flags
    
    if panel_tilt_flags.ndim != panel_pos_flags.ndim or panel_tilt_flags.ndim != panel_distortion_flags.ndim:
        raise ValueError('panel_tilt_flags, panel_pos_flags, panel_distortion_flags must be same dimension')
        
    panel_calibration_flags = np.hstack([panel_tilt_flags, panel_pos_flags, panel_distortion_flags])
    
    det_keys = list(instr.detectors.keys())
    for i, det_key in enumerate(det_keys):
        if panel_calibration_flags.ndim == 2:
            calibration_flags = np.hstack([calibration_flags, panel_calibration_flags[i, :]])
            instr.detectors[det_key]._calibration_flags = panel_calibration_flags[i, :]
        else:
            calibration_flags = np.hstack([calibration_flags, panel_calibration_flags])
            instr.detectors[det_key]._calibration_flags = panel_calibration_flags

    instr.calibration_flags = calibration_flags

    # last item for calibrating lattice parameter
    calibration_flags = np.hstack([calibration_flags, False]).astype(bool)
    
    return calibration_flags, instr

def visualize_powder_calibration(pc, instr, img_dict, plane_data, eta_tol, vmin=0, vmax=3000):
    # extract powder lines from ceria powder pattern and newly calibrated detector
    pc._extract_powder_lines()
    data_dict = pc._calibration_data

    # plotting results
    if instr.num_panels == 2:
        fig, ax = plt.subplots(1, 2)
        #fig_row, fig_col = np.unravel_index(np.arange(instr.num_panels), (2))
        fig_key = {'ff1':[0],
                 'ff2':[1]}
    elif instr.num_panels == 8:
        fig, ax = plt.subplots(2, 4)
        #fig_row, fig_col = np.unravel_index(np.arange(instr.num_panels), (2, 4))
        fig_key = {'ff1_0_0':[0, 0],
                 'ff1_0_1':[0, 1],
                 'ff1_1_0':[1, 0],
                 'ff1_1_1':[1, 1],
                 'ff2_0_0':[0, 2],
                 'ff2_0_1':[0, 3],
                 'ff2_1_0':[1, 2],
                 'ff2_1_1':[1, 3]}
    
    i = 0
    for det_key, panel in instr.detectors.items():
        i = i + 1
        
        # gather panel image
        pimg = np.array(img_dict[det_key], dtype=float)
        
        # gather ideal virtual diffraction angles
        ideal_angs, ideal_xys, tth_ranges = panel.make_powder_rings(plane_data,
                                                        delta_eta=eta_tol)
        ideal_pixel = panel.cartToPixel(np.vstack(ideal_xys))
        
        # gather measured angles from ceria powder pattern
        all_pts = np.vstack(data_dict[det_key])
        meas_pixels = panel.cartToPixel(all_pts[:, :2])
        
        # plot panel results
        if instr.num_panels == 2:
            fig_col = fig_key[det_key][0]
            
            # plot image
            ax[fig_col].imshow(
                pimg,
                vmin=vmin, #np.percentile(pimg, 10),
                vmax=vmax, #np.percentile(pimg, 90),
                cmap=plt.cm.bone_r
            )
            
            # plot ideal angles
            ax[fig_col].plot(ideal_pixel[:, 1], ideal_pixel[:, 0], 'cx')
            ax[fig_col].set_title(det_key)
            
            # plot measured angles
            ax[fig_col].plot(meas_pixels[:, 1], meas_pixels[:, 0], 'm+')
            ax[fig_col].set_title(det_key)
            
            # legend
            if i == 1:
                ax[fig_col].legend(['Simulated', 'Measured'])
            
        elif instr.num_panels == 8:
            fig_row = fig_key[det_key][0]
            fig_col = fig_key[det_key][1]
            
            # plot image
            ax[fig_row, fig_col].imshow(
                pimg,
                vmin=vmin, #np.percentile(pimg, 10),
                vmax=vmax, #np.percentile(pimg, 90),
                cmap=plt.cm.bone_r
            )
            
            # plot ideal angles
            ax[fig_row, fig_col].plot(ideal_pixel[:, 1], ideal_pixel[:, 0], 'cx')
            ax[fig_row, fig_col].set_title(det_key)
            
            # plot measured angles
            ax[fig_row, fig_col].plot(meas_pixels[:, 1], meas_pixels[:, 0], 'm+')
            ax[fig_row, fig_col].set_title(det_key)
            
            # legend
            if i == 1:
                ax[fig_row, fig_col].legend(['Simulated', 'Measured'])
    plt.show()
    
    return fig, ax

def save_instr_and_config(cfg, instr, config_path, instr_path):
    # write instrument
    instr.write_config(file=instr_path, style='yaml', calibration_dict={})
    
    # write config
    cfg._cfg['instrument'] = instr_path
    cfg.dump(config_path)

#%% user input for powder calibration
'''
powder calibration defaults:
  eta_tol: 2.0
  fit_tth_tol: 0.05
  max_iter: 5
  int_cutoff: 0.0001
  conv_tol: 0.0001
  pk_type: 'pvoigt'
  bg_type: 'linear'
  use_robust_optimization: false
  auto_guess_initial_fwhm: true
  initial_fwhm: 0.5
'''

# path to hexrd config file with 2 panel dexela instrument and ceria powder images
config_path = '/media/djs522/djs522_nov2020/dexela_distortion/kirks_dec_2022/dexela_90-524kev_ceo2_instr_config.yml'
pd_data_exclusions = [0] #[3, 4, 6, 7]

# new paths to hexrd instr and config file with multipanel chunked dexela instrument and ceria powder images
# these will be created below, just a new path for these files is necessary
base_path = '/media/djs522/djs522_nov2020/dexela_distortion/kirks_dec_2022/'
init_chunk_instr_path = os.path.join(base_path, '1_dexela_90-524kev_ceo2_instr_mpanel.yml')
init_chunk_config_path = os.path.join(base_path, '1_dexela_90-524kev_ceo2_instr_mpanel_config.yml')

tilt_chunk_instr_path = os.path.join(base_path, '2_dexela_90-524kev_ceo2_instr_mpanel_tilt_only.yml')
tilt_chunk_config_path = os.path.join(base_path, '2_dexela_90-524kev_ceo2_instr_mpanel_tilt_only_config.yml')

pos_chunk_instr_path = os.path.join(base_path, '3_dexela_90-524kev_ceo2_instr_mpanel_pos_only.yml')
pos_chunk_config_path = os.path.join(base_path, '3_dexela_90-524kev_ceo2_instr_mpanel_pos_only_config.yml')

tilt_and_pos_chunk_instr_path = os.path.join(base_path, '4_dexela_90-524kev_ceo2_instr_mpanel_tilt_and_pos.yml')
tilt_and_pos_chunk_config_path = os.path.join(base_path, '4_dexela_90-524kev_ceo2_instr_mpanel_tilt_and_pos_config.yml')

# additional instr calibration parameters
tth_max = 8.0
visualize_plots = True

# powder calibration parameters (can be compared with GUI)
tth_tol = 0.15 # GUI: Annular sector widths 2_theta
eta_tol = 1. # GUI : Annular sector widths eta
fit_tth_tol = 0.15 # GUI : Max fit |delta 2_theta| / 2_theta_0 error
min_peak_amp_raw_units = 1e-4 # GUI : Min Peak Amplitude (Raw Units)
convergence_tol = 1e-4 # GUI : convergence tolerance (delta r) 
max_iter = 5 # GUI : maximum iterations
background_type = 'quadratic' # GUI : Background type ('constant', 'linear', 'quadratic', 'cubic')
pktype = 'pvoigt' # GUI : peak fitting type ('gaussian', 'pvoigt', 'splpvoigt')
auto_guess_initial_fwhm = False # GUI : Automatically guess initial FWHM? (True, False), this trumps initial_fwhm
initial_fwhm = 0.5 # GUI : Initial FWHM
use_robust_optimization = False # GUI : Use robust optimization? (True, False)

#%% preprocessing inputs

# initial chunking
ChunkDexela.chunk_detector(config_path, base_dim=(1944, 1536), n_rows_cols=(2, 2), 
                  row_col_gap=(0, 0), updated_instr_path=init_chunk_instr_path)

ChunkDexela.chunk_frame_cache(config_path, base_dim=(1944, 1536), n_rows_cols=(2, 2), 
                  row_col_gap=(0, 0), updated_instr_path=init_chunk_instr_path, 
                  updated_config_path=init_chunk_config_path)

# initialize hexrd config object
cfg = config.open(init_chunk_config_path)[0]

# initialize   instrument
init_instr = cfg.instrument.hedm
det_keys = list(init_instr.detectors.keys())

# initialize image series and image dict
ims = cfg.image_series
img_dict = dict.fromkeys(det_keys)
for det_key in img_dict.keys():
    img_dict[det_key] = ims[det_key][0]

# initialize ceria material 
matl = make_matl('ceria', 225, [5.41153, ])
matl.beamEnergy = init_instr.beam_energy
plane_data = matl.planeData
if tth_tol is not None:
    plane_data.tThWidth = np.radians(tth_tol)
if tth_max is not None:
    plane_data.exclusions = None
    plane_data.tThMax = np.radians(tth_max)

curr_exclusions = plane_data.exclusions
for i in pd_data_exclusions:
    if i < curr_exclusions.size:
        curr_exclusions[i] = True
plane_data.exclusions = curr_exclusions

# process initial fwhm
if auto_guess_initial_fwhm:
    initial_fwhm = None

#%% powder calibration 2 (tilt only)
# initialize new instr for calibration with calibration flags
tilt_only_cf, tilt_only_instr = set_instr_calib_flags(init_instr, 
                          instr_flags=np.zeros(7, dtype=bool),
                          panel_tilt_flags=np.array([1, 1, 0], dtype=bool), 
                          panel_pos_flags=np.array([0, 0, 0], dtype=bool),
                          panel_distortion_flags=np.array([], dtype=bool))
# tag a zero on the end for calibration of lattice paramter = False
#tilt_only_cf = np.hstack([tilt_only_cf, 0])

# initialize powder calibrator
tilt_only_pc = PowderCalibrator(tilt_only_instr, plane_data, img_dict, flags=tilt_only_cf,
                      tth_tol=tth_tol, eta_tol=eta_tol, fwhm_estimate=initial_fwhm,
                      min_pk_sep=1e-3, min_ampl=0.,
                      pktype=pktype, bgtype=background_type,
                      tth_distortion=None)

# initialize instrument calibrator
tilt_only_ic = InstrumentCalibrator(tilt_only_pc)

# run powder calibration
tilt_only_ic.run_calibration(fit_tth_tol=fit_tth_tol, int_cutoff=min_peak_amp_raw_units,
                    conv_tol=convergence_tol, max_iter=max_iter,
                    use_robust_optimization=use_robust_optimization)

# write instr and config
save_instr_and_config(cfg, tilt_only_ic.instr, tilt_chunk_config_path, tilt_chunk_instr_path)

# visualize powder calibration
if visualize_plots:
    visualize_powder_calibration(tilt_only_pc, tilt_only_instr, img_dict, plane_data, eta_tol)

#%% powder calibration 3 (pos only)
# initialize new instr for calibration with calibration flags
pos_only_cf, pos_only_instr = set_instr_calib_flags(init_instr, 
                          instr_flags=np.zeros(7, dtype=bool),
                          panel_tilt_flags=np.array([0, 0, 0], dtype=bool), 
                          panel_pos_flags=np.array([1, 1, 1], dtype=bool),
                          panel_distortion_flags=np.array([], dtype=bool))
# tag a zero on the end for calibration of lattice paramter = False
#pos_only_cf = np.hstack([pos_only_cf, 0])

# initialize powder calibrator
pos_only_pc = PowderCalibrator(pos_only_instr, plane_data, img_dict, flags=pos_only_cf,
                      tth_tol=tth_tol, eta_tol=eta_tol, fwhm_estimate=initial_fwhm,
                      min_pk_sep=1e-3, min_ampl=0.,
                      pktype=pktype, bgtype=background_type,
                      tth_distortion=None)

# initialize instrument calibrator
pos_only_ic = InstrumentCalibrator(pos_only_pc)

# run powder calibration
pos_only_ic.run_calibration(fit_tth_tol=fit_tth_tol, int_cutoff=min_peak_amp_raw_units,
                    conv_tol=convergence_tol, max_iter=max_iter,
                    use_robust_optimization=use_robust_optimization)

# write instr and config
save_instr_and_config(cfg, pos_only_ic.instr, pos_chunk_config_path, pos_chunk_instr_path)

# visualize powder calibration
if visualize_plots:
    visualize_powder_calibration(pos_only_pc, pos_only_instr, img_dict, plane_data, eta_tol)

#%% powder calibration 4 (tilt and pos)
# initialize new instr for calibration with calibration flags
tilt_and_pos_cf, tilt_and_pos_instr = set_instr_calib_flags(init_instr, 
                          instr_flags=np.zeros(7, dtype=bool),
                          panel_tilt_flags=np.array([1, 1, 0], dtype=bool), 
                          panel_pos_flags=np.array([1, 1, 1], dtype=bool),
                          panel_distortion_flags=np.array([], dtype=bool))
# tag a zero on the end for calibration of lattice paramter = False
#tilt_and_pos_cf = np.hstack([tilt_and_pos_cf, 0])

# initialize powder calibrator
tilt_and_pos_pc = PowderCalibrator(tilt_and_pos_instr, plane_data, img_dict, flags=tilt_and_pos_cf,
                      tth_tol=tth_tol, eta_tol=eta_tol, fwhm_estimate=initial_fwhm,
                      min_pk_sep=1e-3, min_ampl=0.,
                      pktype=pktype, bgtype=background_type,
                      tth_distortion=None)

# initialize instrument calibrator
tilt_and_pos_ic = InstrumentCalibrator(tilt_and_pos_pc)

# run powder calibration
tilt_and_pos_ic.run_calibration(fit_tth_tol=fit_tth_tol, int_cutoff=min_peak_amp_raw_units,
                    conv_tol=convergence_tol, max_iter=max_iter,
                    use_robust_optimization=use_robust_optimization)

# write instr and config
save_instr_and_config(cfg, tilt_and_pos_ic.instr, tilt_and_pos_chunk_config_path, tilt_and_pos_chunk_instr_path)

# visualize powder calibration
if visualize_plots:
    visualize_powder_calibration(tilt_and_pos_pc, tilt_and_pos_instr, img_dict, plane_data, eta_tol)