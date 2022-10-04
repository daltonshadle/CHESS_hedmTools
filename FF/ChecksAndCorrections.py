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

# *****************************************************************************
# FUNCTION DECLARATION AND IMPLEMENTATION
# *****************************************************************************
    
def init_correct_chess_vertical_beam_variation(grains_out):
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
    

def init_correct_chess_vertical_beam_variation_extra(grains_out, vert_bnds=[-0.07, 0.07], comp_thresh=0.85, 
                                                     chi2_thresh=1e-2, orig_lattice_params=[3.6], do_plots=True):
    '''
    Purpose: correcting for vertical variation in beam energy based on the 
    initial scan, with extra thresholds and plotting

    Parameters
    ----------
    grains_out : numpy array (n x 21)
        standard data from hexrd grains.out file loaded as numpy array.
    vert_bnds : tuple 2 x 1, optional
        [lower, upper] bounds for vertical thresholding in mm. The default is [-0.07, 0.07].
    comp_thresh : float, optional
        completeness threshold for grains.out to only look at grains ABOVE 
        threshold. The default is 0.85.
    chi2_thresh : float, optional
        chi^2 threshold for grains.out to only look at grains BELOW 
        threshold. The default is 1e-2.
    orig_lattice_params : tuple 1 x 1 or 2 x 1, optional
        original lattice parameters to use for estimating new paramters, if
        cubic, use 1x1 [a0], if hexagonal, use 2x1 [a0, c0], The default is [3.6].
    do_plots : bool, optional
        flag for showing plots. The default is True.

    Returns
    -------
    grains_out : numpy array (n x 21)
        standard data from hexrd grains.out file loaded as numpy array, with
        normal strains corrected for variation in vertial beam variation
    p : tuple (2 x 1)
        polynomial fitted paramters for vertial beam variation of strain.

    '''
    # parse grains_out
    comp_0 = grains_out[:,1]
    chi_0 = grains_out[:,2]

    strain_0 = grains_out[:,15:]
    ori_0 = grains_out[:,3:6]
    pos_0 = grains_out[:,6:9]
    v_strain_0 = np.sum(strain_0[:,0:3],axis=1)/3.
    
    # check if grains are withing completenss and chi^2 levels
    good_grains = np.where((comp_0 >= comp_thresh) & (chi_0 <= chi2_thresh))[0]
    print('Mean Height: %0.3e' %(np.mean(pos_0[good_grains, 1])))
    
    # check if grains are within vertical bounds
    fit_grains = good_grains[np.where((pos_0[good_grains, 1] >= vert_bnds[0]) & (pos_0[good_grains, 1] <= vert_bnds[1]))[0]]
    
    # do a linear fit of vertical position and volumetric strain
    p = np.polyfit(pos_0[fit_grains, 1], v_strain_0[fit_grains], 1)
    new_v_strain_0 = v_strain_0-p[0]*pos_0[:,1]
    
    # check volumetric strain versus vertical position
    if do_plots:
        plt.figure(1)
        plt.plot(pos_0[good_grains, 1], v_strain_0[good_grains],'y')
        plt.plot(pos_0[good_grains,1], new_v_strain_0[good_grains], 'rx')
        
        plt.legend(['raw','energy gradient corrected'])
        
        plt.xlabel('y position')
        plt.ylabel('volumetric strain')
    
    n_grains=len(good_grains)
    print('Energy / Volumetric Strain Slope: %0.3e mm^-1' %(p[0]))
    print('Constant Term: %0.3e (if larger than 1e-4 consider using following plots/output to adjust lattice parameter(s)' %(p[1]))
    
    astrain=np.zeros(n_grains)
    bstrain=np.zeros(n_grains)
    cstrain=np.zeros(n_grains)
    
    # correct energy gradient
    strain_0[:,0]=strain_0[:,0]-(p[0]*pos_0[:,1])
    strain_0[:,1]=strain_0[:,1]-(p[0]*pos_0[:,1])
    strain_0[:,2]=strain_0[:,2]-(p[0]*pos_0[:,1])
    
    if do_plots:
        for ii in np.arange(n_grains):
            ti = good_grains[ii]
            
            # get strain tensor and transform to crystal coord system
            strain_ten_s = hexrd_mat.strainVecToTen(strain_0[ti, :])
            R_sc = hexrd_rot.rotMatOfExpMap(ori_0[ti,:])
            strain_ten_c = np.dot(R_sc.T, np.dot(strain_ten_s, R_sc))
            astrain[ii]=strain_ten_c[0,0]
            bstrain[ii]=strain_ten_c[1,1]
            cstrain[ii]=strain_ten_c[2,2]
        
        
        plt.figure(2)
        plt.plot(astrain,'x')
        plt.title(r'$\Delta a/a (\epsilon_{xx}^C)$')
        plt.xlabel('grain #')
        print('Delta a/a: %0.4e' %(np.mean(astrain)))
                   
                   
        plt.figure(3)
        plt.plot(bstrain,'gx')
        plt.title(r'$\Delta b/b (\epsilon_{yy}^C)$')
        plt.xlabel('grain #')
        print('Delta b/b: %0.4e' %(np.mean(bstrain)))           
        
        plt.figure(4)
        plt.plot(cstrain,'rx')
        plt.title(r'$\Delta c/c (\epsilon_{zz}^C)$')
        plt.xlabel('grain #')
        print('Delta c/c: %0.4e' %(np.mean(cstrain)))
    
    #------------- Correcting for the lattice parameter -----------#
    if len(orig_lattice_params) == 1:
        #FOR CUBIC
        a0 = orig_lattice_params[0] #original lattice parameter
        a = (1.+(np.mean(astrain)+np.mean(bstrain)+np.mean(cstrain))/3.)*a0
        
        print("Original Lattice Paramter a = %0.3e A" %(a0))
        print("New Lattice Paramter a = %0.3e A" %(a))
    elif len(orig_lattice_params) == 2:
        #FOR HEXAGONAL
        a0 = orig_lattice_params[0] #original lattice parameter
        c0 = orig_lattice_params[1] #original lattice parameter
        
        a = (1.+(np.mean(astrain)+np.mean(bstrain))/2.)*a0
        c = (1.+(np.mean(cstrain)))*c0
    
        print("Original Lattice Paramter a = %0.4e A, c = %0.4e A" %(a0, c0))
        print("New Lattice Paramter a = %0.4e A, c = %0.4e A" %(a, c))
    else:
        print("New lattice parameter couldn't be calculated.")
    
    grains_out[:, 15:18] = strain_0[:, 0:3]
    
    return grains_out, p

def bending_load_check(grains_out, do_plots=True):
    '''
    Purpose: Checking if the sample is experiencing a bending moment by 
        monitoring loading direction strain vs x, z position

    Parameters
    ----------
    grains_out : numpy array (n x 21)
        standard data from hexrd grains.out file loaded as numpy array.
    do_plots : bool, optional
        flag for showing plots. The default is True.

    Returns
    -------
    xz_fit : numpy array (4 x 1)
        [x_slope, x_const, z_slope, z_const] from linear fits of loading 
        direction strain and x, z positions.

    '''
    x_pos = grains_out[:, 6] 
    z_pos = grains_out[:, 8] 
    loading_dir_strain = grains_out[:, 16]
    px =  np.polyfit(x_pos, loading_dir_strain, 1) #polynomial fit of gradient p[0] will be the slope p[1] the intercept
    pz =  np.polyfit(z_pos, loading_dir_strain, 1) #polynomial fit of gradient p[0] will be the slope p[1] the intercept

    #plot of old values, polyfit
    if do_plots:
        fig, ax = plt.subplots()
        ax.scatter(x_pos, loading_dir_strain, label='strain-x')
        uni_x_pos = np.unique(x_pos)
        ax.plot(uni_x_pos, px[0] * uni_x_pos + px[1],'g-',label='polyfit')
        plt.xlabel ('t_vec_c [x]')
        plt.ylabel ('loading_dir_strain')
        
        fig, ax = plt.subplots()
        ax.scatter(z_pos, loading_dir_strain, label='strain-z')
        uni_z_pos = np.unique(z_pos)
        ax.plot(uni_z_pos, pz[0] * uni_z_pos + pz[1],'g-',label='polyfit')
        plt.xlabel ('t_vec_c [z]')
        plt.ylabel ('loading_dir_strain')
    
    print('X Position / Strain Slope: %0.3e mm^-1 (if larger than 1e-4, could potentially indicate bending' %(px[0]))
    print('Constant Term X Position: %0.3e' %(px[1]))
    print('Z Position / Strain Slope: %0.3e mm^-1 (if larger than 1e-4, could potentially indicate bending' %(pz[0]))
    print('Constant Term Z Position: %0.3e' %(pz[1]))
    
    xz_fit = np.hstack([px, pz])
    return xz_fit
    
def torsion_load_check(grains_out, do_plots=True):
    '''
    Purpose: Checking if the sample is experiencing a torision by 
        monitoring von mises strain vs radial position

    Parameters
    ----------
    grains_out : numpy array (n x 21)
        standard data from hexrd grains.out file loaded as numpy array.
    do_plots : bool, optional
        flag for showing plots. The default is True.

    Returns
    -------
    pr : tuple (2 x 1)
        polynomial fitted paramters for radial variation of strain.

    '''
    r = np.linalg.norm(grains_out[:, [6, 8]], axis=1)
    s = grains_out[:, 15:]
    von_mises_strain = np.sqrt(((s[:, 0] - s[:, 1])**2 + (s[:, 1] - s[:, 2])**2 + (s[:, 2] - s[:, 0])**2 +
                               3.0 / 2.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)) / 2)
    pr =  np.polyfit(r, von_mises_strain, 1) #polynomial fit of gradient p[0] will be the slope p[1] the intercept

    #plot of old values, polyfit
    if do_plots:
        fig, ax = plt.subplots()
        ax.scatter(r, von_mises_strain, label='strain-x')
        uni_r = np.unique(r)
        ax.plot(uni_r, pr[0] * uni_r + pr[1],'g-',label='polyfit')
        plt.xlabel ('radius')
        plt.ylabel ('von mises strain')
    
    print('Radius Position / Strain Slope: %0.3e mm^-1 (if larger than 1e-4, could potentially indicate torsion' %(pr[0]))
    print('Constant Term Radius: %0.3e' %(pr[1]))
    
    return pr

# *****************************************************************************
# TESTING
# *****************************************************************************

if __name__ == '__main__':
    comp_thresh = 0.95
    chi2_thresh = 5e-3
    
    analysis_path = os.path.join(os.path.dirname(__file__).split('CHESS_hedmTools')[0], 'CHESS_hedmTools/Analysis/')
    
    # data set with two load steps at the following macroscopic stress and strain
    grains_out = np.loadtxt(os.path.join(analysis_path, 'example_grains_out/combined_grains_c0_0_ungripped.out'))
    
    corr_grains_out, v_fit = init_correct_chess_vertical_beam_variation_extra(grains_out, vert_bnds=[-0.07, 0.07], comp_thresh=0.85, 
                                                         chi2_thresh=1e-2, orig_lattice_params=[3.6], do_plots=True)
    
    p_fit = bending_load_check(corr_grains_out[(grains_out[:, 1] >= comp_thresh) & (grains_out[:, 2] <= chi2_thresh), :], do_plots=True)
    
    r_fit = torsion_load_check(corr_grains_out[(grains_out[:, 1] >= comp_thresh) & (grains_out[:, 2] <= chi2_thresh), :], do_plots=True)