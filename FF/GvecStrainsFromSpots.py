#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:30:56 2022

@author: djs522
"""

import numpy as np
from hexrd import rotations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(1, '/home/djs522/additional_sw/hedmTools/CHESS_hedmTools/')
#import SingleGrainOrientationDistributions.SGODAnalysis as SGOD
import SingleGrainOrientationDistributions.OrientationTools as OT
import FF.PostProcessStress as PPS

#%%

rod = np.array([[0, 0, 0.4]])
rod1 = np.array([[0, 0, 0.425], [0, 0, 0.45]])

rod = OT.quat2rod(rotations.toFundamentalRegion(OT.rod2quat(rod).T).T)
rod1 = OT.quat2rod(rotations.toFundamentalRegion(OT.rod2quat(rod1).T).T)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(rod[:, 0], rod[:, 1], rod[:, 2], c='r')
ax.scatter(rod1[:, 0], rod1[:, 1], rod1[:, 2], c='r')
ax = OT.PlotFR('cubic', ax)


temp = np.vstack([rod, rod1])

a = OT.calc_closest_orientaiton_configuration_quats(OT.rod2quat(rod).T, OT.rod2quat(rod1).T)
a = OT.calc_closest_orientaiton_configuration_quats(OT.rod2quat(np.atleast_2d(temp[1, :])).T, OT.rod2quat(temp).T)

#print(a)
print(np.degrees(a[1]))

rod1 = OT.quat2rod(a[0].T)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(rod1[:, 0], rod1[:, 1], rod1[:, 2], c='g')
ax = OT.PlotFR('cubic', ax)
plt.show()



#%%
def voigt_stress_t2v(t):
    return np.atleast_2d([t[0,0], t[1,1], t[2,2], t[1,2], t[0,2], t[0,1]])
def voigt_stress_v2t(v):
    return np.array([[v[0], v[5], v[4]],
                    [v[5], v[1], v[3]],
                    [v[4], v[3], v[2]]])
def voigt_strain_t2v(t):
    return np.atleast_2d([t[0,0], t[1,1], t[2,2], 2*t[1,2], 2*t[0,2], 2*t[0,1]])
def voigt_strain_v2t(v):
    return np.array([[v[0], v[5] / 2, v[4] / 2],
                    [v[5] / 2, v[1], v[3] / 2],
                    [v[4] / 2, v[3] / 2, v[2]]])
def voigt_strain_t2v_3d(t):
    return np.atleast_2d([t[:,0,0], t[:,1,1], t[:,2,2], 2*t[:,1,2], 2*t[:,0,2], 2*t[:,0,1]])
def voigt_stress_vt2_3d(v):
    return np.array([[v[:,0], v[:,5], v[:,4]],
                    [v[:,5], v[:,1], v[:,3]],
                    [v[:,4], v[:,3], v[:,2]]]).T
    

def post_process_stress(gr_strain_samp_t, gr_ori_rot_mat, SX_STIFF):
    '''
    Parameters
    ----------
    gr_strain_samp_t : numpy array (n x 3 x 3)
        an array of n grain strain tensors expressed in the sample coord system.
    gr_ori_rot_mat : numpy array (n x 3 x 3)
        an array of n grain orientation rotation matrices.
    SX_STIFF : numpy array (6 x 6)
        stiffness tensor for a crystal in the crystal coord system in Voigt.

    Returns
    -------
    gr_stress_samp_v : numpy array (n x 6)
        an array of n grain stresses expressed in the sample coord.
        
    Notes
    -----
    samp = sample coord system
    crys = crystal coord system
    t = tensor
    v = vector (usually Voigt notation)
    '''
    
    # gather number of grains for processing
    num_grains = gr_strain_samp_t.shape[0]
    
    # initialize return structures
    gr_stress_samp_t = np.zeros([num_grains, 3, 3])
    gr_stress_samp_v = np.zeros([num_grains, 6])
    
    # for each grain in the output
    for i in np.arange(num_grains):
        # grab strain and orientation for grain
        i_strain_samp_t = gr_strain_samp_t[i, :, :]
        i_rot_mat = gr_ori_rot_mat[i, :, :]
        
        # transform strain tensor samp -> crys for grain i
        i_strain_crys_t = np.dot(i_rot_mat.T, np.dot(i_strain_samp_t, i_rot_mat))
        
        # express crystal strain in Voigt (assumes shears need to multiplied by 2 for strain)
        i_strain_crys_v = voigt_strain_t2v(i_strain_crys_t)
                    
        # apply single crystal stiffness tensor for stress in crys coord as vector and tensor
        i_stress_crys_v = np.dot(SX_STIFF, i_strain_crys_v.T).flatten()
        i_stress_crys_t = voigt_stress_v2t(i_stress_crys_v)
        
        # transform stress tensor crys -> samp for grain i
        i_stress_samp_t = np.dot(i_rot_mat, np.dot(i_stress_crys_t, i_rot_mat.T))
        i_stress_samp_v = voigt_stress_t2v(i_stress_samp_t)
        
        # package stress for returning
        gr_stress_samp_v[i] = i_stress_samp_v
        gr_stress_samp_t[i] = i_stress_samp_t
        
    return gr_stress_samp_v, gr_stress_samp_t

def fast_post_process_stress(gr_strain_samp_t, gr_ori_rot_mat, SX_STIFF):
    '''

    Parameters
    ----------
    gr_strain_samp_t : numpy array (n x 3 x 3)
        an array of n grain strain tensors expressed in the sample coord system.
    gr_ori_rot_mat : numpy array (n x 3 x 3)
        an array of n grain orientation rotation matrices.
    SX_STIFF : numpy array (6 x 6)
        stiffness tensor for a crystal in the crystal coord system in Voigt.

    Returns
    -------
    gr_stress_samp_v : numpy array (n x 6)
        an array of n grain stresses expressed in the sample coord.
        
    Notes
    -----
    samp = sample coord system
    crys = crystal coord system
    t = tensor
    v = vector (usually Voigt notation)

    '''
    
    # transform grain strains in sample to crystal coord
    gr_strain_crys_t = np.einsum('tji,tjk->tik', gr_ori_rot_mat, gr_strain_samp_t)
    gr_strain_crys_t = np.einsum('tij,tjk->tik', gr_strain_crys_t, gr_ori_rot_mat)
    
    # express grain strains from tensor to voigt vector for Hooke's Law
    gr_strain_crys_v = voigt_strain_t2v_3d(gr_strain_crys_t)
    
    # perform Hooke's Law
    gr_stress_crys_v = np.dot(SX_STIFF, gr_strain_crys_v)
    
    # express grain stress from voigt vector to tensor
    gr_stress_crys_t = voigt_stress_vt2_3d(gr_stress_crys_v.T)
    
    # transform grain stresses in crystal to sample coord
    gr_stress_samp_t = np.einsum('tij,tjk->tik', gr_ori_rot_mat, gr_stress_crys_t)
    gr_stress_samp_t = np.einsum('tij,tkj->tik', gr_stress_samp_t, gr_ori_rot_mat)
    
    return gr_stress_samp_t
   
    
   
#%%

grain_out = np.loadtxt('/home/djs522/additional_sw/hedmTools/CHESS_hedmTools/FF/analysis/combined_grains_c0_0_ungripped.out')

g, p = PPS.init_correct_strain_for_vertical_beam_variation(grain_out)

#%%
n = 2000

strain_s = np.zeros([n, 3, 3])
strain_c = np.zeros([n, 3, 3])
rot_sc = np.zeros([n, 3, 3])

for i in range(n):
    strain_s[i] = voigt_strain_v2t(grain_out[i, 15:])
    rot_sc[i] = rotations.rotMatOfExpMap(grain_out[i, 3:6])
    
    strain_c[i] = np.dot(rot_sc[i].T, np.dot(strain_s[i], rot_sc[i]))
    
c11 = 259.6e3 #260e3 #MPa
c12 = 179.0e3 #177e3 #MPa
c44 = 109.6e3 #107e3 #MPa
INCONEL_718_SX_STIFF = np.array([[c11, c12, c12,   0,   0,   0], 
                                 [c12, c11, c12,   0,   0,   0], 
                                 [c12, c12, c11,   0,   0,   0],
                                 [  0,   0,   0, c44,   0,   0], 
                                 [  0,   0,   0,   0, c44,   0],
                                 [  0,   0,   0,   0,   0, c44]])

#%%

big_strain = np.repeat(strain_s, 50*6, axis=0)
big_rot = np.repeat(rot_sc, 50*6, axis=0)

print(big_strain.shape)

#%%
import time
t = time.time()
temp = fast_post_process_stress(big_strain, big_rot, INCONEL_718_SX_STIFF)
print(time.time() - t)
t = time.time()
temp2 = post_process_stress(big_strain, big_rot, INCONEL_718_SX_STIFF)
print(time.time() - t)

print(temp[0, :, :])
print(temp2[1][0, :, :])   
print(np.linalg.norm(temp - temp2[1]))
print(np.max(temp - temp2[1]))
print(np.min(temp - temp2[1]))



#%%

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

scan = 37
grain_id = 2
panel_id = 'ff1'

cfg_yaml = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/testing/c0_1_sc%i_testing.yml' %(scan)
spot_file = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/testing/c0_1_sc%i_testing/%s/spots_%05i.out' %(scan, panel_id, grain_id)
grains_file = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/testing/c0_1_sc%i_testing/grains.out' %(scan)

cfg = config.open(cfg_yaml)[0]
instr = cfg.instrument
panel = instr.detector_dict[panel_id]
det = instr._hedm
pd = cfg.material.plane_data


#%%

print(makeEtaFrameRotMat(det._beam_vector, det._eta_vector))

#%%

def calc_gvec_l(angs, beam_energy_kev):
    theta = angs[:, 0] * 0.5
    eta = angs[:, 1]
    const = 4 * np.pi * np.sin(theta) / keVToAngstrom(beam_energy_kev) # 4 pi sin(tth / 2) / lambda
    const = 2 * np.sin(theta) / keVToAngstrom(beam_energy_kev) # 2 sin(tth / 2) / lambda
    gvec_l = np.array([np.cos(theta) * np.cos(eta), np.cos(theta) * np.sin(eta), np.sin(theta)])
    gvec_l = const * gvec_l
    
    return gvec_l.T, np.sin(theta), const

def calc_R_ls(ome, chi):
    cos_o = np.cos(ome)
    sin_o = np.sin(ome)
    cos_c = np.cos(chi)
    sin_c = np.sin(chi)
    return np.array([[cos_o, 0, sin_o], [sin_c * sin_o, cos_c, -sin_c * cos_o], [-cos_c * sin_o, sin_c, cos_c * cos_o]])

def d(tth, beam_energy_kev):
    return keVToAngstrom(beam_energy_kev) / 2.0 * (1 / np.sin(tth / 2.0))

# grain_id = 4
# nvec_mat = []
# strain_mat = []
# diff = []
# for panel_id in instr.detector_dict.keys():
#     panel = instr.detector_dict[panel_id]
#     spot_file = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/testing/c0_1_sc%i_testing/%s/spots_%05i.out' %(scan, panel_id, grain_id)
#     print(panel_id)
#     for i in range(40):
#         # get spot info
#         # [id, pid, h, k, l, sum, max, pred_tth, pred_eta, pred_ome, meas_tth, meas_eta, meas_ome, pred_x, pred_y, meas_x, meas_y]
#         spots = np.loadtxt(spot_file)
#         spots = spots[spots[:, 0] != -999, :]
#         spots = np.atleast_2d(spots[i, :])
#         angs = spots[:, 10:13] # meas
#         #angs = spots[:, 7:10] # pred
#         det_xy = spots[:, 15:] # meas
#         #det_xy = spots[:, 13:15] # pred
        
#         # get grain info
#         grain = np.loadtxt(grains_file)
#         grain = grain[grain[:, 0] == grain_id, :]
#         # get sample to lab and crys to sample transformations
#         R_ls = makeOscillRotMatArray(det.chi, angs[:, 2])[0, :, :]
#         R_sc = rotations.rotMatOfExpMap(grain[:, 3:6].T)
#         # elastic strain in sample, lab, crystal
#         strain_s_t = strainVecToTen(grain[:, 15:].flatten())
#         strain_l_t = np.dot(np.dot(R_ls, strain_s_t), R_ls.T)
#         strain_c_t = np.dot(np.dot(R_sc.T, strain_s_t), R_sc)
#         # stretch in sample, crystal
#         stretch_c_t = np.eye(3) - strain_c_t
#         stretch_s_t = np.dot(R_sc, np.dot(stretch_c_t, R_sc.T))
#         stretch_s_t = vecMVToSymm(grain[:, 9:15].flatten())
#         stretch_c_t = np.dot(np.dot(R_sc.T, stretch_s_t), R_sc)
#         # position in sample
#         pos_s = grain[:, 6:9].T
#         pos_l = np.dot(R_ls, pos_s).T
        
#         # get hkl and recip lattice vectors
#         hkl = spots[:, 2:5]
#         bMat = pd.latVecOps['B']
        
#         # calc gvec_o (undeformed)
#         gvec_c_o = np.dot(bMat, hkl.T).T
#         gvec_s_o = np.dot(R_sc, gvec_c_o.T).T 
#         gvec_l_o = np.dot(R_ls, gvec_s_o.T).T
#         nvec_l_o = (gvec_l_o.T / np.linalg.norm(gvec_l_o, axis=1)).T
        
#         # calc gvec_d (deform) from stretch
#         gvec_c_d = np.dot(stretch_c_t, gvec_c_o.T).T
#         gvec_s_d = np.dot(R_sc, gvec_c_d.T).T
#         gvec_l_d = np.dot(R_ls, gvec_s_d.T).T
#         nvec_l_d = (gvec_l_d.T / np.linalg.norm(gvec_l_d, axis=1)).T
        
#         # calc gvec_a (deformed) from angles
#         gvec_l_a, sin_t, const = calc_gvec_l(np.copy(angs), det._beam_energy)
#         gvec_s_a = np.dot(R_ls.T, gvec_l_a.T).T
#         gvec_c_a = np.dot(R_sc.T, gvec_s_a.T).T
#         nvec_l_a = (gvec_l_a.T / np.linalg.norm(gvec_l_a, axis=1)).T
        
#         # calc gvec_h (deformed) from angles in hexrd code
#         gvec_c_h = anglesToGVec(np.copy(angs), bHat_l=det._beam_vector, eHat_l=det._eta_vector, chi=det.chi, rMat_c=R_sc)
#         gvec_s_h = np.dot(R_sc, gvec_c_h.T).T
#         gvec_l_h = np.dot(R_ls, gvec_s_h.T).T
#         nvec_l_h = (gvec_l_h.T / np.linalg.norm(gvec_l_h, axis=1)).T
        
#         # 
#         [tth_eta, gvec_l_xy] = detectorXYToGvec(det_xy,
#                          rMat_d=panel.rmat, rMat_s=R_ls,
#                          tVec_d=panel._tvec, tVec_s=det.tvec, tVec_c=pos_s,
#                          beamVec=det._beam_vector, etaVec=det.eta_vector)
        
        
#         d_a = d(angs[:, 0], det._beam_energy)
#         d_o = 1.0 / np.linalg.norm(gvec_c_o)
#         d_d = 1.0 / np.linalg.norm(gvec_c_d)
        
#         # print("gvec_c_o: ", gvec_c_o, np.linalg.norm(gvec_c_o))
#         # print("gvec_c_d: ", gvec_c_d, np.linalg.norm(gvec_c_d))
#         # print("gvec_c_a: ", gvec_c_a, np.linalg.norm(gvec_c_a))
#         # print("gvec_c_h: ", gvec_c_h, np.linalg.norm(gvec_c_h))
        
#         # print("nvec_l_o: ", nvec_l_o)
#         # print("nvec_l_d: ", nvec_l_d)
#         # print("nvec_l_a: ", nvec_l_a)
#         # print("nvec_l_h: ", nvec_l_h)
        
#         print("d: ", np.linalg.norm(gvec_l_o) / np.linalg.norm(gvec_l_d) - 1, nvec_l_d @ strain_l_t @ nvec_l_d.T)
#         print("a: ", np.linalg.norm(gvec_l_o) / np.linalg.norm(gvec_l_a) - 1, nvec_l_a @ strain_l_t @ nvec_l_a.T)
#         print("h: ", np.linalg.norm(gvec_l_o) / (2 * np.sin(0.5 * angs[0, 0]) / keVToAngstrom(det._beam_energy)) - 1, nvec_l_h @ strain_l_t @ nvec_l_h.T)
#         print("xy: ", np.linalg.norm(gvec_l_o) / (2 * np.sin(0.5 * tth_eta[0][0]) / keVToAngstrom(det._beam_energy)) - 1, 
#               gvec_l_xy @ strain_l_t @ gvec_l_xy.T)
#         print("d: ", np.linalg.norm(gvec_l_o) / np.linalg.norm(gvec_l_d) - 1 - nvec_l_d @ strain_l_t @ nvec_l_d.T)
#         print("a: ", np.linalg.norm(gvec_l_o) / np.linalg.norm(gvec_l_a) - 1 - nvec_l_a @ strain_l_t @ nvec_l_a.T)
#         print("h: ", np.linalg.norm(gvec_l_o) / (2 * np.sin(0.5 * angs[0, 0]) / keVToAngstrom(det._beam_energy)) - 1 - nvec_l_h @ strain_l_t @ nvec_l_h.T)
#         print("xy: ", np.linalg.norm(gvec_l_o) / (2 * np.sin(0.5 * tth_eta[0][0]) / keVToAngstrom(det._beam_energy)) - 1 - 
#               gvec_l_xy @ strain_l_t @ gvec_l_xy.T)
        
#         diff.append([np.linalg.norm(gvec_l_o) / np.linalg.norm(gvec_l_d) - 1 - nvec_l_d @ strain_l_t @ nvec_l_d.T,
#                      np.linalg.norm(gvec_l_o) / np.linalg.norm(gvec_l_a) - 1 - nvec_l_a @ strain_l_t @ nvec_l_a.T,
#                      np.linalg.norm(gvec_l_o) / (2 * np.sin(0.5 * angs[0, 0]) / keVToAngstrom(det._beam_energy)) - 1 - nvec_l_h @ strain_l_t @ nvec_l_h.T,
#                      np.linalg.norm(gvec_l_o) / (2 * np.sin(0.5 * tth_eta[0][0]) / keVToAngstrom(det._beam_energy)) - 1 - 
#                            gvec_l_xy @ strain_l_t @ gvec_l_xy.T])
        
#         #nvec_mat.append((gvec_l_xy / np.linalg.norm(gvec_l_xy)).flatten())
#         nvec_mat.append((np.dot(R_sc.T, np.dot(R_ls.T, gvec_l_xy.T)) / np.linalg.norm(gvec_l_xy)).flatten())
#         strain_mat.append((np.linalg.norm(gvec_l_o) / (2 * np.sin(0.5 * tth_eta[0][0]) / keVToAngstrom(det._beam_energy)) - 1).flatten())
        
#         #print(tth_eta, angs)
#         #print(gvec_l_xy)
        
#         gHat_c = gvec_c_d.T / np.linalg.norm(gvec_c_d)
#         rMat_c = R_sc
#         rMat_s = R_ls
#         rMat_d=panel.rmat
#         tVec_d=panel._tvec
#         tVec_s=det.tvec
#         tVec_c=pos_s
#         bVec=det._beam_vector
#         etaVec=det.eta_vector
#         calc_xy = gvecToDetectorXY(gHat_c.T,
#                                    rMat_d, rMat_s, rMat_c,
#                                    tVec_d, tVec_s, tVec_c,
#                                    beamVec=bVec)
        
#         print(calc_xy)
#         print(spots[:, 13:])
    
  
# diff = np.array(diff)

# print(np.max(np.abs(diff), axis = 0))
# print(np.mean(diff, axis = 0))

# plt.figure()
# plt.hist(diff[:, 2, 0 ,0], alpha = 0.5)
# plt.hist(diff[:, 3, 0 ,0], alpha = 0.5)
# plt.hist(diff[:, 0, 0 ,0], alpha = 0.5)

 #%%
def strain_rossette(lattice_strain, nvec):
    NMat = np.array([nvec[:, 0]**2,
    nvec[:, 1]**2,
    nvec[:, 2]**2,
    nvec[:, 1] * nvec[:, 2],
    nvec[:, 0] * nvec[:, 2],
    nvec[:, 0] * nvec[:, 1]]).T
    
    return np.linalg.lstsq(NMat, lattice_strain)

# s = np.array(strain_mat)
# n = np.array(nvec_mat)
# #print(s.shape, n.shape)
# print(strain_rossette(s, n)[0].T)
# print(voigt_strain_t2v(strain_c_t))



#%%

def calc_const(tth, beam_energy):
    return (2 * np.sin(0.5 * tth) / keVToAngstrom(beam_energy))


#%%
grains = np.loadtxt(grains_file)
num_grains = grains.shape[0]


nhkl_fams = 4
all_hkls = []
for i_fams in range(nhkl_fams):
    all_hkls.append(pd.hklDataList[i_fams]['symHKLs'])
all_hkls = np.hstack(all_hkls).T

num_spots_per_panel = all_hkls.shape[0]
num_spots = num_spots_per_panel * 2

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
    print(grain_id)
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
        spots_file = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/testing/c0_1_sc%i_testing/%s/spots_%05i.out' %(scan, panel_id, grain_id)
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
                const_pred_angs = calc_const(pred_angs[:, 0], det._beam_energy)
                
                # calc gvech from meas_angs in hexrd code (note: all unit vectors)
                gvech_c_meas_angs = anglesToGVec(meas_angs, bHat_l=det._beam_vector, eHat_l=det._eta_vector, chi=det.chi, rMat_c=R_sc)
                gvech_s_meas_angs = np.dot(R_sc, gvech_c_meas_angs.T).T
                gvech_l_meas_angs = np.dot(R_ls, gvech_s_meas_angs.T).T
                const_meas_angs = calc_const(meas_angs[:, 0], det._beam_energy)
                
                # calc gvech from pred_xy detector spot position in hexrd code (note: all unit vectors)
                [tth_eta, gvech_l_pred_xy] = detectorXYToGvec(pred_xy,
                                 rMat_d=panel.rmat, rMat_s=R_ls,
                                 tVec_d=panel._tvec, tVec_s=det.tvec, tVec_c=pos_s,
                                 beamVec=det._beam_vector, etaVec=det.eta_vector)
                gvech_s_pred_xy = np.dot(R_ls.T, gvech_l_pred_xy.T).T
                gvech_c_pred_xy = np.dot(R_sc.T, gvech_s_pred_xy.T).T
                const_pred_xy = calc_const(tth_eta[0][0], det._beam_energy)
                
                # calc gvech from meas_xy detector spot position in hexrd code (note: all unit vectors)
                [tth_eta, gvech_l_meas_xy] = detectorXYToGvec(meas_xy,
                                 rMat_d=panel.rmat, rMat_s=R_ls,
                                 tVec_d=panel._tvec, tVec_s=det.tvec, tVec_c=pos_s,
                                 beamVec=det._beam_vector, etaVec=det.eta_vector)
                gvech_s_meas_xy = np.dot(R_ls.T, gvech_l_meas_xy.T).T
                gvech_c_meas_xy = np.dot(R_sc.T, gvech_s_meas_xy.T).T
                const_meas_xy = calc_const(tth_eta[0][0], det._beam_energy)
                
                
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
        
    #print(np.sum(pred_angs_gvec[i_grain, :, 0] != bad_const))
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
    
    fit_strain[i_grain, :] = voigt_strain_t2v(strain_c_t)

#%%

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

#print("Pred Ang ******************")
#analyze_gvecs_and_strain(pred_angs_gvec, pred_angs_strain)
#print("Meas Ang ******************")
#analyze_gvecs_and_strain(meas_angs_gvec, meas_angs_strain)
print("Pred XY ******************")
analyze_gvecs_and_strain(pred_xy_gvec, pred_xy_strain, fit_strain)
print("Meas XY ******************")
analyze_gvecs_and_strain(meas_xy_gvec, meas_xy_strain, fit_strain)

#%%
i = 5
plt.figure()
plt.hist((meas_xy_gvec[i, :, 3] - meas_xy_gvec[i, :, 4]))
plt.hist((pred_xy_gvec[i, :, 3] - pred_xy_gvec[i, :, 4]))

