#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:22:28 2022

@author: djs522
"""


# *****************************************************************************
# IMPORTS
# *****************************************************************************
import sys
import os
try:
    import dill as cpl
except(ImportError):
    import pickle as cpl

import numpy as np

import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.cm as cm

from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

from hexrd              import matrixutil as hexrd_mat
import hexrd.fitting.fitpeak as fitpeaks
if sys.version_info[0] < 3:
    # python 2
    from hexrd.xrd          import rotations  as hexrd_rot
    from hexrd.xrd          import symmetry   as hexrd_sym
else:
    # python 3
    from hexrd        import rotations  as hexrd_rot
    from hexrd        import symmetry   as hexrd_sym
    from hexrd        import config

# *****************************************************************************
# CONSTANTS
# *****************************************************************************
pi = np.pi

# single crystal elastic moduli
c11 = 259.6e3 #260e3 #MPa
c12 = 179.0e3 #177e3 #MPa
c44 = 109.6e3 #107e3 #MPa
INCONEL_718_SX_STIFF = np.array([[c11, c12, c12,   0,   0,   0], 
                                 [c12, c11, c12,   0,   0,   0], 
                                 [c12, c12, c11,   0,   0,   0],
                                 [  0,   0,   0, c44,   0,   0], 
                                 [  0,   0,   0,   0, c44,   0],
                                 [  0,   0,   0,   0,   0, c44]])

INCONEL_718_E = 199e9 # found from DIC measurements in elastic regime, averaged over 75 points
INCONEL_718_nu = 0.321 # found from DIC measurements in the elastic regime, averaged over 75 points
INCONEL_718DP_Yield = 940e6 # macroscopic yield around this value
INCONEL_718_SCHMID_TENSOR_LIST = [np.load(os.path.join(os.path.dirname(__file__), 'analysis/inconel_718_schmid_tensors.npy'))]

# *****************************************************************************
# FUNCTION DECLARATION AND IMPLEMENTATION
# *****************************************************************************

def init_sx_stiff_fcc(c11, c12, c44):
    # init_sx_stiff_fcc - Initializes stiffness tensor (in Voigt notiation)
    #   for fcc crystals
    # 
    #   INPUT:
    #   c11 is a float
    #      c11 component of the stiffness tensor
    #   c12 is a float
    #      c12 component of the stiffness tensor
    #   c44 is a float
    #      c44 component of the stiffness tensor
    # 
    #   OUTPUT:
    #   C is a numpy array (6 x 6)
    #      stiffness tensor for fcc crystal
    #      
    #   Notes:
    #      TODO: write a generic ini_sx_stiff for all crystal sym
    #
    
    # initialize stiffness tensor for cubic crystal, C11, C12, C44
    C = np.zeros((6,6))
    C[0,0] = c11; C[1,1] = c11; C[2,2] = c11
    C[0,1] = c12; C[0,2] = c12; C[1,2] = c12
    C[1,0] = c12; C[2,0] = c12; C[2,1] = c12
    C[3,3] = c44; C[4,4] = c44; C[5,5] = c44
    
    return C

def voigt_x(ang, in_deg=True):
    # voigt_x - Voigt notation transformation for a rotation about x-axis by 
    #   angle ang
    # 
    #   INPUT:
    #   ang is a float
    #      angle to rotate around x-axis, can be radians or degrees with in_deg
    #      flat set
    #   in_deg is a bool
    #      True = ang in degrees, False = ang in radians
    # 
    #   OUTPUT:
    #   return is a numpy array (6 x 6)
    #      transformation matrix for rotation about x-axis by ang
    #      
    #   Notes:
    #      None
    #
    if in_deg:
        ang = np.radians(ang)
    c = np.cos(ang)
    s = np.sin(ang)
    c2 = c**2
    s2 = s**2
    return np.array([[  1,    0,   0,      0,   0,   0], 
                     [  0,   c2,  s2,  2*s*c,   0,   0], 
                     [  0,   s2,  c2, -2*s*c,   0,   0], 
                     [  0, -s*c, s*c,  c2-s2,   0,   0], 
                     [  0,    0,   0,      0,   c,  -s], 
                     [  0,    0,   0,      0,   s,   c]])

def voigt_z(ang, in_deg=True):
    # voigt_z - Voigt notation transformation for a rotation about z-axis by 
    #   angle ang
    # 
    #   INPUT:
    #   ang is a float
    #      angle to rotate around z-axis, can be radians or degrees with in_deg
    #      flat set
    #   in_deg is a bool
    #      True = ang in degrees, False = ang in radians
    # 
    #   OUTPUT:
    #   return is a numpy array (6 x 6)
    #      transformation matrix for rotation about z-axis by ang
    #      
    #   Notes:
    #      None
    #
    if in_deg:
        ang = np.radians(ang)
    c = np.cos(ang)
    s = np.sin(ang)
    c2 = c**2
    s2 = s**2
    return np.array([[  c2,  s2,  0,  0,  0,  2*s*c], 
                     [  s2,  c2,  0,  0,  0, -2*s*c], 
                     [   0,   0,  1,  0,  0,      0], 
                     [   0,   0,  0,  c, -s,      0], 
                     [   0,   0,  0,  s,  c,      0],
                     [-s*c, s*c,  0,  0,  0,  c2-s2]])

def gen_sx_stiffness_tensor_in_sample_coord(grain_data, SX_STIFF=INCONEL_718_SX_STIFF):
    # gen_sx_stiffness_tensor_in_sample_coord - generates the Voigt notation
    #   single crystal stiffness tensors transformed to the sample coordinate system
    #   by the orientation of a give grain
    # 
    #   INPUT:
    #   grain_data is a numpy array (n x 21)
    #      typical grains.out format of hexrd in numpy format for n grains
    #   SX_STIFF is a numpy array (6 x 6)
    #      stiffness tensor for a crystal in the crystal coord system
    # 
    #   OUTPUT:
    #   sx_stiff_samp is a numpy array (n x 6 x 6)
    #      stiffness tensors for n crystals in the sample coord system
    #      
    #   Notes:
    #      None
    #
    
    # gather number of grains for processing
    num_grains=grain_data.shape[0]
    
    # initialize return structures
    sx_stiff_samp = np.zeros([num_grains, 6, 6])
    
    # for each grain in the output
    for i in np.arange(num_grains):
        # grab exp map and convert strain vec to true strain vec for grain i
        gr_exp_map = np.atleast_2d(grain_data[i, 3:6]).T
        
        # do voigt notation transformation of voigt stiffness matrix to sample 
        # coordinate systemusing using single grain orientation
        gr_euler_zxz = hexrd_rot.exp_map_2_bunge_euler_zxz(gr_exp_map)
        r_z1 = voigt_z(gr_euler_zxz[0], in_deg=False)
        r_x = voigt_x(gr_euler_zxz[1], in_deg=False)
        r_z2 = voigt_z(gr_euler_zxz[2], in_deg=False)
        gr_rot_mat_voigt = r_z2 @ r_x @ r_z1
        gr_rot_stiff = gr_rot_mat_voigt @ SX_STIFF @ gr_rot_mat_voigt.T
        sx_stiff_samp[i] = gr_rot_stiff
    
    return sx_stiff_samp

def post_process_stress(grain_data, SX_STIFF=INCONEL_718_SX_STIFF, 
                        schmid_T_list=None, stress_macro=None,
                        only_sample_stress=False):
    # func_name - func description
    # 
    #   INPUT:
    #   grain_data is a numpy array (n x 21)
    #      typical grains.out format of hexrd in numpy format for n grains
    #   SX_STIFF is a numpy array (6 x 6)
    #      stiffness tensor for a crystal in the crystal coord system
    #   schmid_T_list is a list of numpy arrays (m x 3 x 3)
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #      None
    #
    
    # gather number of grains for processing
    num_grains=grain_data.shape[0]
    
    # initialize optional Schmid Tensor for RSS
    if schmid_T_list is not None:
        num_slip_systems=schmid_T_list.shape[0]
        RSS=np.zeros([num_grains,num_slip_systems])
    
    # initialize return structures
    stress_mat_samp = np.zeros([num_grains, 6])
    if not only_sample_stress:
        stress_mat_crys = np.zeros([num_grains, 6])
        hydrostatic=np.zeros([num_grains, 1])
        pressure=np.zeros([num_grains, 1])
        von_mises=np.zeros([num_grains, 1])
        stress_coaxiality=np.zeros([num_grains, 1])
    
    # for each grain in the output
    for i in np.arange(num_grains):
        # grab exp map and convert strain vec to true strain vec for grain i
        gr_exp_map = np.atleast_2d(grain_data[i, 3:6]).T
        gr_strain_samp_v = np.atleast_2d(grain_data[i, 15:21]).T
        gr_strain_samp_v[3:6] = 2 * gr_strain_samp_v[3:6] # TODO: check if this is needed
        
        # get the crys -> samp rotation as matrix for grain i
        gr_rot_mat = hexrd_rot.rotMatOfExpMap_opt(gr_exp_map)
        
        # get strain tensor in samp coord for grain i 
        gr_strain_samp_t = hexrd_mat.strainVecToTen(gr_strain_samp_v)
        
        # transform strain tensor samp -> crys for grain i
        gr_strain_crys_v = hexrd_mat.strainTenToVec(np.dot(gr_rot_mat.T, np.dot(gr_strain_samp_t, gr_rot_mat)))
                    
        # apply single crystal stiffness tensor for stress in crys coord as vector and tensor
        gr_stress_crys_v = np.dot(SX_STIFF, gr_strain_crys_v)
        gr_stress_crys_t = hexrd_mat.stressVecToTen(gr_stress_crys_v)
        
        # transform stress tensor crys -> samp for grain i
        gr_stress_samp_t = np.dot(gr_rot_mat, np.dot(gr_stress_crys_t, gr_rot_mat.T))
        gr_stress_samp_v = hexrd_mat.stressTenToVec(gr_stress_samp_t)
        
        if not only_sample_stress:
            # Calculate hydrostatic stress
            hydrostatic_stress = (gr_stress_samp_v[:3].sum()/3)       
            
            # Calculate deviatoric and von Mises stress
            dev_stress_s = gr_stress_samp_t - hydrostatic_stress * np.identity(3)        
            von_mises_stress = np.sqrt((3/2) * (dev_stress_s**2).sum())   
            
            # calculate principal stresses
            prin_vals, prin_dirs = np.linalg.eig(gr_stress_crys_t)
                    
            # Project on to slip systems
            if schmid_T_list is not None:
                for j in np.arange(num_slip_systems):        
                    RSS[i,j]=(gr_stress_crys_t * schmid_T_list[j,:,:]).sum()
            
            # find stress coaxiality
            if stress_macro is not None:
                stress_contraction = np.dot(stress_macro.T, gr_stress_samp_v)
                stress_coaxiality[i] = (np.arccos(stress_contraction / (np.linalg.norm(stress_macro) * np.linalg.norm(gr_stress_samp_v))) * 180 / np.pi) 
            
        # package stress for returning
        stress_mat_samp[i] = gr_stress_samp_v.T
        if not only_sample_stress:
            stress_mat_crys[i] = gr_stress_crys_v.T
            hydrostatic[i, 0] = hydrostatic_stress
            pressure[i, 0] = -hydrostatic_stress
            von_mises[i, 0] = von_mises_stress
    
    stress_data=dict()    
    
    stress_data['stress_S'] = stress_mat_samp
    if not only_sample_stress:
        stress_data['stress_C'] = stress_mat_crys
        stress_data['hydrostatic'] = hydrostatic
        stress_data['pressure'] = pressure
        stress_data['von_mises'] = von_mises
        stress_data['triaxiality'] = hydrostatic / von_mises
        stress_data['coaxiality'] = stress_coaxiality
        stress_data['principal_val'] = prin_vals
        stress_data['principal_dir'] = prin_dirs
    
    if schmid_T_list is not None:
        stress_data['RSS'] = RSS
        
    return stress_data

def post_process_stress_old(grain_data, SX_STIFF=INCONEL_718_SX_STIFF, schmid_T_list=None):
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #      None
    #
    
    num_grains=grain_data.shape[0]

    stress_S=np.zeros([num_grains,6])
    stress_C=np.zeros([num_grains,6])
    hydrostatic=np.zeros([num_grains,1])
    pressure=np.zeros([num_grains,1])
    von_mises=np.zeros([num_grains,1])

    if schmid_T_list is not None:
        num_slip_systems=schmid_T_list.shape[0]
        RSS=np.zeros([num_grains,num_slip_systems])


    for jj in np.arange(num_grains):    
    
        expMap=np.atleast_2d(grain_data[jj,3:6]).T    
        strainTmp=np.atleast_2d(grain_data[jj,15:21]).T
        
        #Turn exponential map into an orientation matrix
        Rsc=hexrd_rot.rotMatOfExpMap(expMap)

        strainTenS = np.zeros((3, 3), dtype='float64')
        strainTenS[0, 0] = strainTmp[0]
        strainTenS[1, 1] = strainTmp[1]
        strainTenS[2, 2] = strainTmp[2]
        strainTenS[1, 2] = strainTmp[3]
        strainTenS[0, 2] = strainTmp[4] 
        strainTenS[0, 1] = strainTmp[5]  
        strainTenS[2, 1] = strainTmp[3] 
        strainTenS[2, 0] = strainTmp[4] 
        strainTenS[1, 0] = strainTmp[5] 

                  
        strainTenC = np.dot(np.dot(Rsc.T,strainTenS),Rsc)
        strainVecC = hexrd_mat.strainTenToVec(strainTenC)
        
        
        #Calculate stress        
        stressVecC = np.dot(SX_STIFF,strainVecC)
        stressTenC = hexrd_mat.stressVecToTen(stressVecC)  
        stressTenS = np.dot(np.dot(Rsc,stressTenC),Rsc.T)
        stressVecS = hexrd_mat.stressTenToVec(stressTenS)       
        
        #Calculate hydrostatic stress
        hydrostaticStress=(stressVecS[:3].sum()/3)       
        
        
        #Calculate Von Mises Stress
        devStressS=stressTenS-hydrostaticStress*np.identity(3)        
        vonMisesStress=np.sqrt((3/2)*(devStressS**2).sum())        
        
        
        #Project on to slip systems
        if schmid_T_list is not None:
            for ii in np.arange(num_slip_systems):        
                RSS[jj,ii]=np.abs((stressTenC*schmid_T_list[ii,:,:]).sum())
            
            
        stress_S[jj,:]=stressVecS.flatten()
        stress_C[jj,:]=stressVecC.flatten()
        
        hydrostatic[jj,0]=hydrostaticStress
        pressure[jj,0]=-hydrostaticStress
        von_mises[jj,0]=vonMisesStress
    
    stress_data=dict()    
    
    stress_data['stress_S']=stress_S
    stress_data['stress_C']=stress_C
    stress_data['hydrostatic']=hydrostatic
    stress_data['pressure']=pressure
    stress_data['von_mises']=von_mises
    
    if schmid_T_list is not None:
        stress_data['RSS']=RSS
        
    return stress_data

def gen_schmid_tensors_from_cfg(cfg_yaml, uvw, hkl):        
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #      None
    #
    
    cfg = config.open(cfg_yaml)[0]
    pd = cfg.material.plane_data
    
    T = gen_schmid_tensors_from_pd(pd, uvw, hkl)

    return T

def gen_schmid_tensors_from_pd(pd, uvw, hkl):        
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #      None
    #
    
    # slip plane directions    
    slipdir  = hexrd_mat.unitVector( np.dot( pd.latVecOps['F'], uvw) ) #  2 -1 -1  0
    slipdir_sym  = hexrd_rot.applySym(slipdir, pd.getQSym(), csFlag=False, cullPM=True, tol=1e-08)
    
    # slip plane plane normals
    n_plane = hexrd_mat.unitVector( np.dot( pd.latVecOps['B'], hkl ) )
    n_plane_sym = hexrd_rot.applySym(n_plane, pd.getQSym(), csFlag=False, cullPM=True, tol=1e-08)

    
    num_slip_plane= n_plane_sym.shape[1]
    
    num_slip_sys=0
    for i in range(num_slip_plane):
        planeID = np.where(abs(np.dot(n_plane_sym[:, i],slipdir_sym)) < 1.e-8)[0]
        num_slip_sys +=planeID.shape[0]
        
    T= np.zeros((num_slip_sys, 3, 3))
    counter=0
        #
    for i in range(num_slip_plane):
        planeID = np.where(abs(np.dot(n_plane_sym[:, i],slipdir_sym)) < 1.e-8)[0]
        for j in np.arange(planeID.shape[0]):    
            print(slipdir_sym[:, planeID[j]], n_plane_sym[:, i])
            T[counter, :, :] = np.dot(slipdir_sym[:, planeID[j]].reshape(3, 1), n_plane_sym[:, i].reshape(1, 3))
            counter+=1
    #Clean some round off errors        
    round_off_err=np.where(abs(T)<1e-8)
    T[round_off_err[0],round_off_err[1],round_off_err[2]]=0.

    return T

def plot_principal_stress_jacks(stress_v):
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #      None
    #
    
    norm = matplotlib.colors.Normalize(vmin=-1500, vmax=1500, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
    
    # transform stress vector to tensor
    stress_t = hexrd_mat.stressVecToTen(stress_v)
    
    # calculate principal stresses
    prin_vals, prin_dirs = np.linalg.eig(stress_t)
    
    # create figure
    main_fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = main_fig.add_subplot(projection='3d')
    
    l = 0.05
    
    for i in range(prin_vals.size):
        ax.quiver(0, 0, 0, prin_dirs[0, i], prin_dirs[1, i], prin_dirs[2, i], 
                       normalize=True, length=l, color=mapper.to_rgba(prin_vals[i]))
        ax.quiver(0, 0, 0, -prin_dirs[0, i], -prin_dirs[1, i], -prin_dirs[2, i], 
                       normalize=True, length=l, color=mapper.to_rgba(prin_vals[i]))
    
    return [main_fig, ax, prin_vals, prin_dirs]    

def obj_func_sx_moduli_with_E(grain_mat_list_dict, C11, C12, C44):
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #      Only compatible with cubic crystals
    #      TODO: make compatible for more crystal symmetry systems
    #
    
    # initialize stiffness tensor for cubic crystal, C11, C12, C44 given in MPa
    C = init_sx_stiff_fcc(C11, C12, C44)
    
    # initialize arrays for macro_stress and elastic modulus for all scans
    grain_mat_list = grain_mat_list_dict['xdata']
    num_load_steps = len(grain_mat_list)
    avg_stress_s_v = np.zeros([num_load_steps, 6])
    avg_elastic_mod = np.zeros(num_load_steps)
    
    # for each grains.out file...
    for i in np.arange(num_load_steps):
        # read grain mat
        grain_mat = grain_mat_list[i]
        grain_mat = np.copy(grain_mat)
                    
        # number of grains in file
        n = grain_mat.shape[0]
        # Weighting - this can be adjusted when we have nearfield data - now all
        # grains are weighted equally. # TODO: Nearfield it will be a volume weight
        # TODO : pass as parameter to this function
        weight = np.ones(n) / n 
        
        # calculate stresses for all grains
        gr_stress_s_v = post_process_stress(grain_mat, SX_STIFF=C, 
                                schmid_T_list=None, stress_macro=None,
                                only_sample_stress=True)['stress_S']
        
        
        # do weighting calculations and reduce to 1 "averaged" grain stress vector
        weight_avg_macro_stress_s_v = np.average(gr_stress_s_v, axis=0, weights=weight)
        
        # gather indices for placing stress in full stress vector
        avg_stress_s_v[i, :] = weight_avg_macro_stress_s_v
        
        # gonvert strain vec to true strain vec for grain i
        gr_strain_s_v = np.atleast_2d(np.copy(grain_mat[:, 15:]))
        gr_strain_s_v[:, 3:] = gr_strain_s_v[:, 3:] * 2 # TODO: check if this is needed
        # do elastic modulus calculations
        avg_elastic_mod[i] = np.average(gr_stress_s_v[:, 1] / gr_strain_s_v[:, 1], weights=weight) / 1e3 # put GPa to help with optimization
    
    avg_stress_s_v = avg_stress_s_v.flatten()
    
    # put elastic modulus at end and return
    avg_stress_s_v = np.hstack([avg_stress_s_v, np.mean(avg_elastic_mod)])
    return avg_stress_s_v

def obj_func_sx_moduli(grain_mat_list_dict, C11, C12, C44):
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #      Only compatible with cubic crystals
    #      TODO: make compatible for more crystal symmetry systems
    #
    
    # initialize stiffness tensor for cubic crystal, C11, C12, C44 given in MPa
    C = init_sx_stiff_fcc(C11, C12, C44)
    
    # initialize arrays for macro_stress and elastic modulus for all scans
    grain_mat_list = grain_mat_list_dict['xdata']
    num_load_steps = len(grain_mat_list)
    avg_stress_s_v = np.zeros([num_load_steps, 6])
    
    # for each grains.out file...
    for i in np.arange(num_load_steps):
        # read grain mat
        grain_mat = grain_mat_list[i]
        grain_mat = np.copy(grain_mat)
                    
        # number of grains in file
        n = grain_mat.shape[0]
        # Weighting - this can be adjusted when we have nearfield data - now all
        # grains are weighted equally. # TODO: Nearfield it will be a volume weight
        # TODO : pass as parameter to this function
        weight = np.ones(n) / n 
         
        # calculate stresses for all grains
        gr_stress_s_v = post_process_stress(grain_mat, SX_STIFF=C, 
                                schmid_T_list=None, stress_macro=None,
                                only_sample_stress=True)['stress_S']
        
        
        # do weighting calculations and reduce to 1 "averaged" grain stress vector
        weight_avg_macro_stress_s_v = np.average(gr_stress_s_v, axis=0, weights=weight)
        
        # gather indices for placing stress in full stress vector
        avg_stress_s_v[i, :] = weight_avg_macro_stress_s_v
    
    avg_stress_s_v = avg_stress_s_v.flatten()
    return avg_stress_s_v

def voigt_reuss_bounds(grain_mat_list, macro_strain_s_v_list, macro_stress_s_v_list, C11=c11, C12=c12, C44=c44, E=INCONEL_718_E):
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #      Only compatible with cubic crystals
    #      TODO: make compatible for more crystal symmetry systems
    #
    
    # initialize stiffness tensor for cubic crystal, C11, C12, C44 given in MPa
    C = init_sx_stiff_fcc(C11, C12, C44)
    
    # initialize arrays for macro_stress and elastic modulus for all scans
    num_load_steps = len(grain_mat_list)
    
    # for each grains.out file...
    for i in np.arange(num_load_steps):
        # read grains.out file and do thresholding
        grain_mat = np.copy(grain_mat_list[i])
        macro_strain_s_v = np.copy(macro_strain_s_v_list[i, :])
        macro_stress_s_v = np.copy(macro_stress_s_v_list[i, :])
        
        if grain_mat.shape[0] < 100:
            print('WARNING: not a sufficient number of grains')
                    
        # number of grains in file
        num_grains = grain_mat.shape[0]
        
        # grab exp map and convert strain vec to true strain vec for grain i
        gr_exp_map_sc = np.atleast_2d(grain_mat[:, 3:6]).T
        gr_strain_s_v = np.atleast_2d(grain_mat[:, 15:]).T
        gr_strain_s_v[3:6] = 2 * gr_strain_s_v[3:6] # TODO: check if this is needed
          
        # get the crys -> samp rotation as matrix for grain i
        gr_rot_mat = hexrd_rot.rotMatOfExpMap_opt(gr_exp_map_sc)
        
        # intialize stress, voigt, and reuss matrices
        voigt_stress_s_v = np.zeros([6, num_grains])
        reuss_strain_s_v = np.zeros([6, num_grains])
        all_gr_stress_s_v  = np.zeros([6, num_grains])
        
        # for each grain in grains.out file...
        for j in range(num_grains):
            # get rotation matrix for crystal instread of indexing each time
            jgr_rot_mat = gr_rot_mat[j, :, :]
            
            # do voigt and reuss calculations
            voigt_strain_c_t = np.matmul(jgr_rot_mat.T, np.matmul(hexrd_mat.strainVecToTen(macro_strain_s_v), jgr_rot_mat))
            voigt_stress_c_v = np.matmul(C, hexrd_mat.stressTenToVec(voigt_strain_c_t))
            voigt_stress_s_t = np.matmul(jgr_rot_mat, np.matmul(hexrd_mat.stressVecToTen(voigt_stress_c_v), jgr_rot_mat.T))
            voigt_stress_s_v[:, j] = hexrd_mat.stressTenToVec(voigt_stress_s_t).T
            
            reuss_stress_c_t = np.matmul(jgr_rot_mat.T, np.matmul(hexrd_mat.stressVecToTen(macro_stress_s_v), jgr_rot_mat))
            reuss_strain_c_v = np.linalg.solve(C, hexrd_mat.stressTenToVec(reuss_stress_c_t))
            reuss_strain_s_t = np.matmul(jgr_rot_mat, np.matmul(hexrd_mat.strainVecToTen(reuss_strain_c_v), jgr_rot_mat.T))
            reuss_strain_s_v[:, j] = hexrd_mat.strainTenToVec(reuss_strain_s_t).T
            
            # do rotation calculations from sample -> crystal for strains
            temp_gr_strain_s_t = hexrd_mat.strainVecToTen(gr_strain_s_v[:, j])
            temp_gr_strain_c_t = np.matmul(jgr_rot_mat.T, np.matmul(temp_gr_strain_s_t, jgr_rot_mat))
            temp_gr_strain_c_v = hexrd_mat.strainTenToVec(temp_gr_strain_c_t)
            
            # do stress and rotation calculations from crystal -> sample for stresses
            temp_gr_stress_c_v = np.matmul(C, temp_gr_strain_c_v)
            temp_gr_stress_c_t = hexrd_mat.stressVecToTen(temp_gr_stress_c_v)
            temp_gr_stress_s_t = np.matmul(jgr_rot_mat, np.matmul(temp_gr_stress_c_t, jgr_rot_mat.T))
            temp_gr_stress_s_v = hexrd_mat.stressTenToVec(temp_gr_stress_s_t)
            
            # record all stresses in indexed matrix
            all_gr_stress_s_v[:, j] = temp_gr_stress_s_v.T         
            
        # do voigt and reuss calculations
        # Voigt provides upper bound to elastic strain energy with constant strain in all grains
        # Reuss provides lower bound to elastic strain energy with constant stress in all grains
        voigt_stress_s_v = np.mean(voigt_stress_s_v, axis=1)
        reuss_strain_s_v = np.mean(reuss_strain_s_v, axis=1)
        # voigt_stress_s_v = np.matmul(C, np.mean(gr_strain_s_v, axis=1))
        
        print('Voigt Stress Bound (Upper): %0.2f MPa' %(voigt_stress_s_v[1]))
        print('Reuss Strain Bound (Lower): %0.2f MPa' %(E/1e6 * reuss_strain_s_v[1]))
        print('Voigt-Reuss-Hill Average: %0.2f MPa' %((voigt_stress_s_v[1] + E/1e6 * reuss_strain_s_v[1]) / 2))
        
        avg_stress_s_v = np.mean(all_gr_stress_s_v, axis=1)
        avg_strain_s_v = np.mean(gr_strain_s_v, axis=1)
        print('Elastic Modulus: %0.2f GPa,    Poissons Ratio: %0.3f' 
              %(avg_stress_s_v[1]/avg_strain_s_v[1]/1e3, 
                -(avg_strain_s_v[0]/avg_strain_s_v[1] + avg_strain_s_v[2]/avg_strain_s_v[1])/2))

def fit_sx_moduli_with_ff_data(grain_mat_list, macro_stress_list, 
                               c11=c11, c12=c12, c44=c44, 
                               comp_thresh=0.8, chi2_thresh=1e-2,
                               fit_E=True, E=INCONEL_718_E):
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #      Only compatible with cubic crystals
    #      TODO: make compatible for more crystal symmetry systems
    #
    
    # initialize starting elastic modulus and single crystal moduli
    macro_elastic_mod = E # MPa
    preC11 = c11 # MPa                       
    preC12 = c12 # MPa
    preC44 = c44 # MPa
    n_s_comp = 6 # number of stress components
    
    # preprocess grain_mat_list with thresholds
    num_load_steps = len(grain_mat_list)
    thresh_grain_mat_list = []
    for i in range(num_load_steps):
        grain_mat = grain_mat_list[i]
        ind_thresh = np.where((grain_mat[:, 1] >= comp_thresh) & (grain_mat[:, 2] <= chi2_thresh))[0]
        grain_mat = grain_mat[ind_thresh, :]
        
        if grain_mat.shape[0] < 100:
            print('WARNING: Load step %i only has %i grains, might fit poorly' %(i, grain_mat.shape[0]))
        thresh_grain_mat_list.append(grain_mat) 
    grain_mat_list = thresh_grain_mat_list
    
    grain_mat_list_dict = {}
    grain_mat_list_dict['xdata'] =  grain_mat_list # weird error in curve_fit if this is left as a numpy array, OK with dictionary

    if fit_E:
        print("********* Fitting with Elastic Modulus *********")
    else:
        print("********* Fitting without Elastic Modulus *********")


    # PRE-OPTIMIZATION CALC *******************************************************
    print('Actual macroscopic stresses')
    for i in range(num_load_steps):
        print('Load step %i sigma_yy: %0.2f MPa' %(i, macro_stress_list[i]))

    print('\nSX_Moduli before optimization')
    print('C_11 = %0.2f GPa \t C_12 = %0.2f GPa \t C_44 = %0.2f GPa ' %(preC11/1e3, preC12/1e3, preC44/1e3))

    # calculate average stress state for the initial moduli
    macro_stress = obj_func_sx_moduli_with_E(grain_mat_list_dict, preC11, preC12, preC44)
    print('\nStresses before optimization')
    for i in range(num_load_steps):
        print('Load step %i sigma_yy: %0.2f MPa' %(i, macro_stress[i*n_s_comp + 1])) # assumes loading in y direction


    # OPTIMIZATION CALC ***********************************************************
    # Input a macro stress in the sigma_yy components and elastic modulus at end, optimize
    if fit_E:
        input_macro_stress = np.zeros(num_load_steps * n_s_comp + 1)
        input_macro_stress[-1] = macro_elastic_mod / 1e3 # put GPa to help with optimization
    else:
        input_macro_stress = np.zeros(num_load_steps * n_s_comp)
    
    # assumes loading in y dir sigma = [0, sigma_yy, 0, 0, 0, 0]; 
    # TODO: fix assumption
    for i in np.arange(num_load_steps):
        input_macro_stress[i*n_s_comp + 1] = macro_stress_list[i]

    if fit_E:
        popt, pcov = curve_fit(obj_func_sx_moduli_with_E, grain_mat_list_dict, input_macro_stress)
    else:
        popt, pcov = curve_fit(obj_func_sx_moduli, grain_mat_list_dict, input_macro_stress)


    # pull single crystal moduli after optimizing, calculate the average stress state
    # the new modulu
    postC11 = popt[0]
    postC12 = popt[1]
    postC44 = popt[2]
    optimal_output_stress = obj_func_sx_moduli(grain_mat_list_dict, postC11, postC12, postC44)


    # POST-OPTIMIZATION CALC ******************************************************
    print('\nSX_Moduli after optimization')
    print('C_11 = %0.2f GPa \t C_12 = %0.2f GPa \t C_44 = %0.2f GPa ' %(postC11/1e3, postC12/1e3, postC44/1e3))

    print('Stresses after optimization')
    # assumes loading in y dir sigma = [0, sigma_yy, 0, 0, 0, 0]; 
    # TODO: fix assumption
    for i in np.arange(num_load_steps):
        print('Load step %i sigma_yy: %0.2f MPa' %(i, optimal_output_stress[i*n_s_comp + 1]))

    print('\nDelta stresses after optimization')
    # assumes loading in y dir sigma = [0, sigma_yy, 0, 0, 0, 0]; 
    # TODO: fix assumption
    for i in np.arange(num_load_steps):
        print('Load step %i delta sigma_yy: %0.2f MPa' %(i, macro_stress_list[i] - optimal_output_stress[i*n_s_comp + 1]))
    
    return [postC11, postC12, postC44]


def extract_strength_for_all_slip_systems(grain_data_list, 
                                          schmid_tensor_list=INCONEL_718_SCHMID_TENSOR_LIST,
                                          SX_STIFF=INCONEL_718_SX_STIFF):
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #
    
    num_load_steps = len(grain_data_list)
    num_ss_fams = len(schmid_tensor_list)
    
    tau_star = [None]*num_ss_fams
    w_tau = [None]*num_ss_fams
    
    # get stress data
    for i, schmid_tensors in enumerate(schmid_tensor_list):
        stress_data = [None]*(num_load_steps)
        for j in np.arange(num_load_steps):
            print('Processing Load Step %i' %(j+1))   
            stress_data[j] = post_process_stress(grain_data_list[j], 
                                                 SX_STIFF=INCONEL_718_SX_STIFF, 
                                                 schmid_T_list=schmid_tensors)
        tau_star[i], w_tau[i]  = extract_strength(stress_data)
    
    return tau_star, w_tau
        

def extract_strength(stress_data):   
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #
    
    num_load_steps = len(stress_data)
    
    tau_star = np.zeros(num_load_steps)
    w_tau = np.zeros(num_load_steps)    
    
    for i in np.arange(num_load_steps):
        max_rss = np.max(stress_data[i]['RSS'], axis=1) # TODO: should this be absolute value of rss?
    
        tau = np.linspace(0., 1.5*np.max(max_rss), num=2000)     

        G = gaussian_kde(max_rss)
        tau_pdf = G.evaluate(tau) 
    
        maxPt = np.argmax(tau_pdf)   
        tmp_pdf = copy.copy(tau_pdf)
        tmp_pdf[:maxPt] = np.max(tau_pdf)
    
        #pfit=fitpeaks.fit_pk_parms_1d([np.max(tau_pdf),tau[maxPt]+1e7,45e6],tau,tmp_pdf,pktype='tanh_stepdown') # assumes stress in Pa
        pfit=fitpeaks.fit_pk_parms_1d([np.max(tau_pdf),tau[maxPt]+1e1,45],tau,tmp_pdf,pktype='tanh_stepdown') # assumes stress in MPa
    
        tau_star[i] = pfit[1]
        w_tau[i] = pfit[2]
        
    return tau_star, w_tau

def plot_strength_curve(tau_star, w_tau, macro_strain=None, plot_color='blue'):  
    # func_name - func description
    # 
    #   INPUT:
    # 
    #   input_name is a input_type
    #      input description
    # 
    # 
    #   OUTPUT:
    # 
    #   output_name is a output_type
    #      output description
    #      
    #   Notes:
    #
    
    if macro_strain is None:
        macro_strain=np.arange(len(tau_star))
    
    tau_star = np.array(tau_star).flatten()
    macro_strain = np.array(macro_strain).flatten()
    strain_fine = np.linspace(macro_strain[0], macro_strain[-1], 100) 
    
    interp_tau_star = interp1d(macro_strain, tau_star)(strain_fine).flatten()
    interp_w_tau = interp1d(macro_strain, w_tau)(strain_fine).flatten()
    
    plt.errorbar(strain_fine,interp_tau_star,yerr=interp_w_tau,color=plot_color, capthick=0)
    plt.plot(macro_strain,tau_star,'s--',markerfacecolor=plot_color,markeredgecolor='k',markeredgewidth=1,color='k')
    plt.plot(strain_fine,interp_tau_star+interp_w_tau,'k--',linewidth=2)
    plt.plot(strain_fine,interp_tau_star-interp_w_tau,'k--',linewidth=2)
    
    plt.grid()

    
def init_correct_strain_for_vertical_beam_variation(grains_out):
    '''
    Purpose: correcting for vertical variation in beam energy based on the 
    initial scan

    Parameters
    ----------
    grains_out : numpy array (n x 21)
        standard data from hexrd grains.out file loaded as numpy array.

    Returns
    -------
    grains_out : numpy array (n x 21)
        standard data from hexrd grains.out file loaded as numpy array, with
        normal strains corrected for variation in vertial beam variation
    p : tuple (2 x 1)
        polynomial fitted paramters for vertial beam variation of strain.

    '''

    y_pos = grains_out[:, 7] # old x
    hydrostatic_strain = np.sum(grains_out[:, 15:18], axis=1) / 3.0 # old y
    p =  np.polyfit(y_pos, hydrostatic_strain, 1) #polynomial fit of gradient p[0] will be the slope p[1] the intercept

    #plot of old values, polyfit
    fig, ax = plt.subplots()
    ax.plot(y_pos, hydrostatic_strain,'gx', label='old')
    ax.plot(np.unique(y_pos), np.poly1d(np.polyfit(y_pos, hydrostatic_strain, 1))(np.unique(y_pos)),'g-',label='polyfit')
    plt.xlabel ('t_vec_c [y]')
    plt.ylabel ('avg_vol_strain')

    #updating hydrostatic_strain to corrected values
    hydrostatic_strain = hydrostatic_strain - p[0] * y_pos - p[1]

    #add corrected values to plot
    ax.plot(y_pos, hydrostatic_strain, 'bo', label='corrected')
    ax.legend(loc='lower right')
    
    # usage of correction as
    grains_out[:, 15:18] = grains_out[:, 15:18] - np.tile((p[0] * grains_out[:, 7] + p[1]), (3,1)).T
    return grains_out, p
    
# *****************************************************************************
# TESTING
# *****************************************************************************

if __name__ == '__main__':
    comp_thresh = 0.8
    chi2_thresh = 1e-2
    
    # data set with two load steps at the following macroscopic stress and strain
    macro_strain = [0.2e-3, 0.35e-3, 0.54e-3, 0.73e-3] # epsilon_yy
    macro_stress = [402, 700, 953, 1020] # MPa, sigma_yy
    
    init_grain_mat_list = [np.loadtxt(os.path.join(os.path.dirname(__file__), 'analysis/combined_grains_c0_1.out')),
                           np.loadtxt(os.path.join(os.path.dirname(__file__), 'analysis/combined_grains_c0_2.out')),
                           np.loadtxt(os.path.join(os.path.dirname(__file__), 'analysis/combined_grains_c0_3.out')),
                           np.loadtxt(os.path.join(os.path.dirname(__file__), 'analysis/combined_grains_c1_1.out'))]
    grain_mat_list = []
    for grain_mat in init_grain_mat_list:
        ind_thresh = np.where((grain_mat[:, 1] >= comp_thresh) & (grain_mat[:, 2] <= chi2_thresh))
        grain_mat_list.append(grain_mat[ind_thresh])
    
    test_schmid_tesnors = False
    if test_schmid_tesnors:
        cfg = os.path.join(os.path.dirname(__file__), 'example_config.yml')
        schmid_tensors = gen_schmid_tensors_from_cfg(cfg, np.array([[1, 1, 0]]).T, np.array([[1, 1, 1]]).T)
    
    test_strength_extract = True
    if test_strength_extract:
        tau_star_list, w_tau_list = extract_strength_for_all_slip_systems(grain_mat_list, 
                                              schmid_tensor_list=INCONEL_718_SCHMID_TENSOR_LIST,
                                              SX_STIFF=INCONEL_718_SX_STIFF)
        plot_strength_curve(tau_star_list, w_tau_list, macro_strain=macro_strain, plot_color='blue')
    
    # test stress calc and stiffness calc
    test_stress_and_stiffness_calc = False
    if test_stress_and_stiffness_calc:
        grain_mat = np.copy(grain_mat_list[0])
        grain_strains = np.copy(grain_mat)[:, 15:]
        grain_strains[:, 3:] = grain_strains[:, 3:] * 2
        sx_stiff_temp = gen_sx_stiffness_tensor_in_sample_coord(np.copy(grain_mat), SX_STIFF=INCONEL_718_SX_STIFF)
        stress_1 = np.einsum('ijk,ik->ij', sx_stiff_temp, grain_strains)
        stress_2 = post_process_stress(np.copy(grain_mat), SX_STIFF=INCONEL_718_SX_STIFF)['stress_S']
        stress_3 = post_process_stress_old(grain_mat, INCONEL_718_SX_STIFF, schmid_T_list=None)['stress_S']
        
        print(np.linalg.norm(stress_1 - stress_2, axis=0).max())
        print(np.linalg.norm(stress_1 - stress_3, axis=0).max())
        print(np.linalg.norm(stress_3 - stress_2, axis=0).max())
    
    
    # test optimization code
    test_sx_moduli_optimization = False
    if test_sx_moduli_optimization:
        grain_mat_list_dict = {}
        grain_mat_list_dict['xdata'] = grain_mat_list
        obj_stress = obj_func_sx_moduli(grain_mat_list_dict, c11, c12, c44)
        
        print(obj_stress)
        
        fit_sx_moduli_with_ff_data(grain_mat_list, macro_stress, 
                                        c11=c11, c12=c12, c44=c44, 
                                        comp_thresh=0.8, chi2_thresh=1e-2,
                                        fit_E=False, E=INCONEL_718_E)
        
        macro_strain_s_v = np.array([[-INCONEL_718_nu * macro_strain[0], macro_strain[0], -INCONEL_718_nu * macro_strain[0], 0, 0, 0],
                                      [-INCONEL_718_nu * macro_strain[1], macro_strain[1], -INCONEL_718_nu * macro_strain[1], 0, 0, 0]])
        macro_stress_s_v = np.array([[0, macro_stress[0], 0, 0, 0, 0],
                                      [0, macro_stress[1], 0, 0, 0, 0]]) # MPa
        voigt_reuss_bounds(grain_mat_list, macro_strain_s_v, macro_stress_s_v, 
                            C11=c11, C12=c12, C44=c44, E=INCONEL_718_E)
    

# functions for UCSB **********************************************************
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
def voigt_strain_vt2_3d(v):
    return np.array([[v[:,0], v[:,5] / 2, v[:,4] / 2],
                    [v[:,5] / 2, v[:,1], v[:,3] / 2],
                    [v[:,4] / 2, v[:,3] / 2, v[:,2] / 2]]).T
def voigt_stress_t2v_3d(t):
    return np.atleast_2d([t[:,0,0], t[:,1,1], t[:,2,2], t[:,1,2], t[:,0,2], t[:,0,1]])

def calc_stress_from_strain_samp(gr_strain_samp_t, gr_ori_rot_mat, SX_STIFF):
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

def calc_stress_from_strain_samp_fast(gr_strain_samp_t, gr_ori_rot_mat, SX_STIFF):
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
