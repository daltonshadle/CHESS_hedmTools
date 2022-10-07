#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:30:56 2022

@author: djs522
"""

# =============================================================================
# %% *IMPORTS*
# =============================================================================

import numpy as np
from hexrd import rotations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(1, '/home/djs522/additional_sw/hedmTools/CHESS_hedmTools/')
import FF.PostProcessStress as PPS

from hexrd.transforms.xfcapi import \
    anglesToGVec, \
    angularDifference, \
    detectorXYToGvec, \
    gvecToDetectorXY, \
    anglesToDVec, \
    makeGVector, \
    makeOscillRotMatArray, \
    makeEtaFrameRotMat

from hexrd import config

from hexrd.constants import keVToAngstrom

from hexrd.matrixutil import vecMVToSymm, strainVecToTen

# =============================================================================
# %% *HELPER FUNCTIONS*
# =============================================================================

def calc_gvec_l(angs, beam_energy_kev):
    '''
    

    Parameters
    ----------
    angs : array (n x 3)
        [tth, eta, omega] for n spots.
    beam_energy_kev : float
        energy of beam in keV.

    Returns
    -------
    gvec_l : array (n x 3)
        n scattering vectors in the laboratory frame.
    np.sin(theta) : array (n x 1)
        n sin(theta) scattering angles.
    const : arrray (n x 1)
        n diffraction constants from n scattering angles.

    '''
    theta = angs[:, 0] * 0.5
    eta = angs[:, 1]
    const = 4 * np.pi * np.sin(theta) / keVToAngstrom(beam_energy_kev) # 4 pi sin(tth / 2) / lambda
    const = 2 * np.sin(theta) / keVToAngstrom(beam_energy_kev) # 2 sin(tth / 2) / lambda
    gvec_l = np.array([np.cos(theta) * np.cos(eta), np.cos(theta) * np.sin(eta), np.sin(theta)])
    gvec_l = const * gvec_l
    
    return gvec_l.T, np.sin(theta), const

def d(tth, beam_energy_kev):
    '''
    

    Parameters
    ----------
    tth : array (n x 1)
        n two theta scattering angles.
    beam_energy_kev : float
        beam energy in keV.

    Returns
    -------
    d : array (n x 1)
        n interplanar spacing distances from scattering angle.

    '''
    return keVToAngstrom(beam_energy_kev) / 2.0 * (1 / np.sin(tth / 2.0))

def calc_scattering_const(tth, beam_energy_kev):
    '''
    

    Parameters
    ----------
    tth : array (n x 1)
        n two theta scattering angles.
    beam_energy_kev : float
        beam energy in keV.

    Returns
    -------
    scattering_const : array (n x 1)
        n scattering constants from the scattering angles.

    '''
    return (2 * np.sin(0.5 * tth) / keVToAngstrom(beam_energy_kev))

def strain_rossette(lattice_strain, nvec):
    '''
    

    Parameters
    ----------
    lattice_strain : array (n x 1)
        n lattice strains for rosette.
    nvec : array (n x 3)
        n normal vectors (normalized) that correspond to the lattice strains.

    Returns
    -------
    lstsq_rosette: np.linalg.lstsq object
        least squares answer with strain tensor and residual.

    '''
    NMat = np.array([nvec[:, 0]**2,
    nvec[:, 1]**2,
    nvec[:, 2]**2,
    nvec[:, 1] * nvec[:, 2],
    nvec[:, 0] * nvec[:, 2],
    nvec[:, 0] * nvec[:, 1]]).T
    
    return np.linalg.lstsq(NMat, lattice_strain)

def grab_gvec_strains_from_spots(grains_out_file, cfg_file, spots_file_base, nhkl_fams=4, return_flag=0):
    '''
    

    Parameters
    ----------
    grains_out_file : string
        path to grains.out file.
    cfg_file : string
        path to configuration file.
    spots_file_base : string
        path stem for spots files with "/%s/spots_%i6i".
    nhkl_fams : TYPE, optional
        DESCRIPTION. The default is 4.
    return_flag : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    
    cfg = config.open(cfg_file)[0]
    instr = cfg.instrument
    det = instr.hedm
    pd = cfg.material.plane_data
    
    grains = np.loadtxt(grains_out_file)
    num_grains = grains.shape[0]
    
    all_hkls = []
    for i_fams in range(nhkl_fams):
        all_hkls.append(pd.hklDataList[i_fams]['symHKLs'])
    all_hkls = np.hstack(all_hkls).T

    num_spots_per_panel = all_hkls.shape[0]
    num_spots = num_spots_per_panel * det.num_panels

    # init all strains (crystal coord system) [strain_vec, fit_resid]
    pred_angs_strain = np.zeros([num_grains, 7])
    meas_angs_strain = np.zeros([num_grains, 7])
    pred_xy_strain = np.zeros([num_grains, 7])
    meas_xy_strain = np.zeros([num_grains, 7])
    fit_strain = np.zeros([num_grains, 6])

    # init all gvecs (crystal coord system) [x, y, z, strain_nn, n * strain * n]
    bad_const = -999
    pred_angs_gvec = np.ones([num_grains, num_spots, 5]) * bad_const
    meas_angs_gvec = np.ones([num_grains, num_spots, 5]) * bad_const
    pred_xy_gvec = np.ones([num_grains, num_spots, 5]) * bad_const
    meas_xy_gvec = np.ones([num_grains, num_spots, 5]) * bad_const
    
    for i_grain, grain_id in enumerate(grains[:, 0]):
        # get grain info
        grain = grains[grains[:, 0] == grain_id, :]
        # get sample to lab and crys to sample transformations
        R_sc = rotations.rotMatOfExpMap(grain[:, 3:6].T)
        # elastic strain in sample, crystal
        strain_s_t = strainVecToTen(grain[:, 15:].flatten())
        strain_c_t = np.dot(np.dot(R_sc.T, strain_s_t), R_sc)
        # stretch in sample, crystal
        stretch_s_t = vecMVToSymm(grain[:, 9:15].flatten())
        stretch_c_t = np.dot(np.dot(R_sc.T, stretch_s_t), R_sc)
        # position in sample
        pos_s = grain[:, 6:9].T
        # get recip_lattice_vectors
        bMat = pd.latVecOps['B']
        
        for i_panel, panel_id in enumerate(instr.detector_dict.keys()):
            panel = instr.detector_dict[panel_id]
            
            # get spot info
            # [id, pid, h, k, l, sum, max, pred_tth, pred_eta, pred_ome, meas_tth, meas_eta, meas_ome, pred_x, pred_y, meas_x, meas_y]
            spots_file = spots_file_base %(panel_id, grain[:, 0])
            spots_data = np.loadtxt(spots_file)
            spots_list = spots_data[spots_data[:, 0] != bad_const, :]
            for i_spot in range(spots_list.shape[0]):
                spot = np.atleast_2d(spots_list[i_spot, :])
                # get hkl and recip lattice vectors
                hkl = spot[:, 2:5]
                
                if np.where((all_hkls == hkl.flatten()).all(axis=1))[0].size == 1:
                    spot_id = int(np.where((all_hkls == hkl.flatten()).all(axis=1))[0][0] + (i_panel * num_spots_per_panel))
                    meas_angs = spot[:, 10:13] # meas
                    pred_angs = spot[:, 7:10] # pred
                    meas_xy = spot[:, 15:] # meas
                    pred_xy = spot[:, 13:15] # pred
                    
                    # get lab to sample for this omega
                    R_ls = makeOscillRotMatArray(det.chi, pred_angs[:, 2])[0, :, :]
                    
                    # calc gvec_o (undeformed) (note: not unit vectors)
                    gvec_c_o = np.dot(bMat, hkl.T).T
                    gvec_s_o = np.dot(R_sc, gvec_c_o.T).T 
                    gvec_l_o = np.dot(R_ls, gvec_s_o.T).T
                    const_o = np.linalg.norm(gvec_l_o, axis=1)
                    
                    # calc gvec_d (deform) from stretch (note: not unit vectors)
                    gvec_c_d = np.dot(stretch_c_t, gvec_c_o.T).T
                    gvec_s_d = np.dot(R_sc, gvec_c_d.T).T
                    gvec_l_d = np.dot(R_ls, gvec_s_d.T).T
                    const_d = np.linalg.norm(gvec_l_d, axis=1)
                    
                    # calc gvech from pred_angs in hexrd code (note: all unit vectors)
                    gvech_c_pred_angs = anglesToGVec(pred_angs, bHat_l=det._beam_vector, eHat_l=det._eta_vector, chi=det.chi, rMat_c=R_sc)
                    gvech_s_pred_angs = np.dot(R_sc, gvech_c_pred_angs.T).T
                    gvech_l_pred_angs = np.dot(R_ls, gvech_s_pred_angs.T).T
                    const_pred_angs = calc_scattering_const(pred_angs[:, 0], det._beam_energy)
                    
                    # calc gvech from meas_angs in hexrd code (note: all unit vectors)
                    gvech_c_meas_angs = anglesToGVec(meas_angs, bHat_l=det._beam_vector, eHat_l=det._eta_vector, chi=det.chi, rMat_c=R_sc)
                    gvech_s_meas_angs = np.dot(R_sc, gvech_c_meas_angs.T).T
                    gvech_l_meas_angs = np.dot(R_ls, gvech_s_meas_angs.T).T
                    const_meas_angs = calc_scattering_const(meas_angs[:, 0], det._beam_energy)
                    
                    # calc gvech from pred_xy detector spot position in hexrd code (note: all unit vectors)
                    [tth_eta, gvech_l_pred_xy] = detectorXYToGvec(pred_xy,
                                     rMat_d=panel.rmat, rMat_s=R_ls,
                                     tVec_d=panel._tvec, tVec_s=det.tvec, tVec_c=pos_s,
                                     beamVec=det._beam_vector, etaVec=det.eta_vector)
                    gvech_s_pred_xy = np.dot(R_ls.T, gvech_l_pred_xy.T).T
                    gvech_c_pred_xy = np.dot(R_sc.T, gvech_s_pred_xy.T).T
                    const_pred_xy = calc_scattering_const(tth_eta[0][0], det._beam_energy)
                    
                    # calc gvech from meas_xy detector spot position in hexrd code (note: all unit vectors)
                    [tth_eta, gvech_l_meas_xy] = detectorXYToGvec(meas_xy,
                                     rMat_d=panel.rmat, rMat_s=R_ls,
                                     tVec_d=panel._tvec, tVec_s=det.tvec, tVec_c=pos_s,
                                     beamVec=det._beam_vector, etaVec=det.eta_vector)
                    gvech_s_meas_xy = np.dot(R_ls.T, gvech_l_meas_xy.T).T
                    gvech_c_meas_xy = np.dot(R_sc.T, gvech_s_meas_xy.T).T
                    const_meas_xy = calc_scattering_const(tth_eta[0][0], det._beam_energy)
                    
                    
                    # add to mats (crystal coord system) [x, y, z, strain_nn]
                    pred_angs_gvec[i_grain, spot_id, :] = np.hstack([gvech_c_pred_angs.flatten(), 
                                                                     const_o/const_pred_angs-1, 
                                                                     (gvech_c_pred_angs @ strain_c_t @ gvech_c_pred_angs.T).flatten()])
                    meas_angs_gvec[i_grain, spot_id, :] = np.hstack([gvech_c_meas_angs.flatten(), 
                                                                     const_o/const_meas_angs-1, 
                                                                     (gvech_c_meas_angs @ strain_c_t @ gvech_c_meas_angs.T).flatten()])
                    pred_xy_gvec[i_grain, spot_id, :] = np.hstack([gvech_c_pred_xy.flatten(), 
                                                                   const_o/const_pred_xy-1, 
                                                                   (gvech_c_pred_xy @ strain_c_t @ gvech_c_pred_xy.T).flatten()])
                    meas_xy_gvec[i_grain, spot_id, :] = np.hstack([gvech_c_meas_xy.flatten(), 
                                                                   const_o/const_meas_xy-1, 
                                                                   (gvech_c_meas_xy @ strain_c_t @ gvech_c_meas_xy.T).flatten()])
            
        # calc strain tensors from gvecs
        ind = pred_angs_gvec[i_grain, :, 0] != bad_const
        t = strain_rossette(pred_angs_gvec[i_grain, ind, 3], pred_angs_gvec[i_grain, ind, :3])
        pred_angs_strain[i_grain, :] = np.hstack([t[0], t[1]])
        
        ind = meas_angs_gvec[i_grain, :, 0] != bad_const
        t = strain_rossette(meas_angs_gvec[i_grain, ind, 3], meas_angs_gvec[i_grain, ind, :3])
        meas_angs_strain[i_grain, :] = np.hstack([t[0], t[1]])
        
        ind = pred_xy_gvec[i_grain, :, 0] != bad_const
        t = strain_rossette(pred_xy_gvec[i_grain, ind, 3], pred_xy_gvec[i_grain, ind, :3])
        pred_xy_strain[i_grain, :] = np.hstack([t[0], t[1]])
        
        ind = meas_xy_gvec[i_grain, :, 0] != bad_const
        t = strain_rossette(meas_xy_gvec[i_grain, ind, 3], meas_xy_gvec[i_grain, ind, :3])
        meas_xy_strain[i_grain, :] = np.hstack([t[0], t[1]])
        
        fit_strain[i_grain, :] = PPS.voigt_strain_t2v(strain_c_t)
        
    if return_flag == 0:
        # return everything
        return [pred_angs_strain, meas_angs_strain, pred_xy_strain, meas_xy_strain, 
                fit_strain, 
                pred_angs_gvec, meas_angs_gvec, pred_xy_gvec, meas_xy_gvec]
    elif return_flag == 1:
        # return strains
        return [pred_angs_strain, meas_angs_strain, pred_xy_strain, meas_xy_strain, 
                fit_strain, 
                [], [], [], []]
    elif return_flag == 2:
        # return gvecs
        return [[], [], [], [], 
                fit_strain, 
                pred_angs_gvec, meas_angs_gvec, pred_xy_gvec, meas_xy_gvec]
    else:
        # return fit_strains
        return [[], [], [], [], 
                fit_strain, 
                [], [], [], []]

def analyze_gvecs_and_strain(g, s, f):
    
    print("Residual Analysis")
    print('Mean: %.3e   Max: %.3e' %(np.mean(s[:, 6]), np.max(s[:, 6])))
    
    print("Fit to Strain Analysis")
    print('Min: %.3e   Mean: %.3e   Max: %.3e' %(np.min(s[:, :6] - f), np.mean(s[:, :6] - f), np.max(s[:, :6] - f)))

    
    print("Grain Fit to Strain Stats")
    for i in range(s.shape[0]):
        print('Grain %i   Min: %.3e   Mean: %.3e   Max: %.3e' %(i, np.min(s[i, :6] - f[i, :]), np.mean(s[i, :6] - f[i, :]), np.max(s[i, :6] - f[i, :])))
    
    # print("Diff Normal Strains from Gvecs")
    # print('Mean: %.3e   Max: %.3e' %(np.mean(g[:, :, 3] - g[:, :, 4]), np.max(g[:, :, 3] - g[:, :, 4])))
    
    # print("Grain Gvec Normal Strain Stats")
    # for i in range(g.shape[0]):
    #     print('Grain %i   Mean: %.3e   Max: %.3e' %(i, np.mean(g[i, :, 3] - g[i, :, 4]), np.max(g[i, :, 3] - g[i, :, 4])))
    
    print()


if __name__ == '__main__':
    scan = 37

    cfg_yaml = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/testing/c0_1_sc%i_testing.yml' %(scan)
    spots_file_base = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/testing/c0_1_sc37_testing/%s/spots_%05i.out' 
    grains_file = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/testing/c0_1_sc%i_testing/grains.out' %(scan)
    
    [pred_angs_strain, meas_angs_strain, pred_xy_strain, meas_xy_strain, fit_strain, 
            pred_angs_gvec, meas_angs_gvec, pred_xy_gvec, meas_xy_gvec] = grab_gvec_strains_from_spots(grains_file, cfg_yaml, spots_file_base, nhkl_fams=4, return_flag=0)
    
    print("Pred Ang ******************")
    analyze_gvecs_and_strain(pred_angs_gvec, pred_angs_strain)
    print("Meas Ang ******************")
    analyze_gvecs_and_strain(meas_angs_gvec, meas_angs_strain)
    print("Pred XY ******************")
    analyze_gvecs_and_strain(pred_xy_gvec, pred_xy_strain, fit_strain)
    print("Meas XY ******************")
    analyze_gvecs_and_strain(meas_xy_gvec, meas_xy_strain, fit_strain)
