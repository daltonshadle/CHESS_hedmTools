#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:35:36 2023

@author: djs522
"""

#%% imports
import os

import numpy as np

from hexrd import imageseries
from hexrd import config
from hexrd import material
from hexrd.fitting import fitpeak

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

#%% helper functions
def make_matl(mat_name, sgnum, lparms, hkl_ssq_max=50):
    '''
    This is a simple function for creating a ceria material quickly without
    a materials config file

    Parameters
    ----------
    mat_name : string
        material name.
    sgnum : float
        space group number.
    lparms : float
        lattice paramters.
    hkl_ssq_max : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    matl : hexrd.materials.Material
        material create bassed on input of function.

    '''
    
    matl = material.Material(mat_name)
    matl.sgnum = sgnum
    matl.latticeParameters = lparms
    matl.hklMax = hkl_ssq_max

    nhkls = len(matl.planeData.exclusions)
    matl.planeData.set_exclusions(np.zeros(nhkls, dtype=bool))
    return matl

def polar_plot_tth_vs_eta(pd, hedm_instr, sim_det_tth_eta, exp_det_tth_eta, 
                          rmin=-0.0025, rmax=0.0025,
                          title='$\\frac{\\theta_{meas} - \\theta_{calc}}{\\theta_{calc}}$ vs $\\eta$',
                          legend=[]):
    '''
    Plot of (tth_{exp} - tth_{sim})  / tth_{sim} vs eta for a given detector

    Parameters
    ----------
    pd : hexrd.crystallography.PlaneData
        plane data object holding ceria.
    hedm_instr : hexrd.Instrument object
        instrument object with panels.
    sim_det_tth_eta : dict
        simulated data save as dictionary with structure:
            dict[det_key][hkl_ring_index][tth_meas_pkfit, eta (from eta_centers)]
    exp_det_tth_eta : dict
        experiment data save as dictionary with structure:
            dict[det_key][hkl_ring_index][tth_meas_pkfit, eta (from eta_centers)]
    rmin : float, optional
        radial plotting minimum. The default is -0.0025.
    rmax : float, optional
        radial plotting maximum. The default is 0.0025.
    title : string, optional
        title for plot. The default is '$\\frac{\\theta_{meas} - \\theta_{calc}}{\\theta_{calc}}$ vs $\\eta$'.
    legend : list, optional
        list of strings for the legend, usually hkl planes. The default is [].

    Returns
    -------
    fig : matplotlib.pyplot.figure
        polar plot figure.
    ax : matplotlib.pyplot.axes
        polar plot axes.

    '''
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.suptitle(title)
    
    # for each ring, collect the data and plot
    for i_ring in np.arange(pd.hkls.shape[1]):
        # initial sim tth, exp tth, and eta lists
        sim = []
        exp = []
        eta = []
        
        # for each detector, add to initalized lists
        for key in hedm_instr.detectors.keys():
            #print(i_ring, key)
            if sim_det_tth_eta[key][i_ring] is not None:
                if len(sim_det_tth_eta[key][i_ring]) > 0:
                    # 0 is tth angle, 1 is eta angle
                    sim.append(sim_det_tth_eta[key][i_ring][:, 0].astype(float))
                    exp.append(exp_det_tth_eta[key][i_ring][:, 0].astype(float))
                    eta.append(sim_det_tth_eta[key][i_ring][:, 1].astype(float))
        
        # transform lists to numpy array
        if len(sim) > 1:
            sim = np.hstack(sim)
            exp = np.hstack(exp)
            eta = np.hstack(eta)
        else:
            sim = np.array(sim)
            exp = np.array(exp)
            eta = np.array(eta)
        
        # create polar plot of (exp ttth - sim tth) / sim tth vs eta
        theta = eta
        r = (exp - sim) / sim
        ax.scatter(theta, r)
        ax.set_rmax(rmax)
        ax.set_rmin(rmin)
        ax.set_rticks([rmin, 0, rmax])  # Less radial ticks
        ax.grid(True)
    fig.legend(legend, loc='lower right')
    plt.show()
    
    return fig, ax

def det_img_plot_vs_data_scatter(pd, hedm_instr, img_dict, 
                         det_x_dict_list, det_y_dict_list, det_data_dict_list,
                         title='Detector Image with Data',
                         vmin=0, vmax=2000, marker_list=list(Line2D.markers.keys()),
                         c_size=10, c_legend='data'):
    '''
    for one figure with multiiple scatters (say exp vs sim position scatter)

    Parameters
    ----------
    pd : hexrd.crystallography.PlaneData
        plane data object holding ceria.
    hedm_instr : hexrd.Instrument object
        instrument object with panels.
    sim_det_tth_eta : dict
        simulated data save as dictionary with structure:
            dict[det_key][hkl_ring_index][tth_meas_pkfit, eta (from eta_centers)]
    img_dict : dict
        detector image data as  dictionary with structure:
            dict[det_key] = image
    det_x_dict_list : dict
        x plotting positions as dictionary with structure:
            dict[det_key][lists of positions] = x_position
    det_y_dict_list : dict
        y plotting positions as dictionary with structure:
           dict[det_key][lists of positions] = y_position
    det_data_dict_list : dict
        data for pllotting at x,y positions as dictionary with structure:
           dict[det_key][lists of data] = data
    title : string, optional
        title of figure. The default is 'Detector Image with Data'.
    vmin : float, optional
        minimum plotting threshold for detector images. The default is 0.
    vmax : float, optional
        maximum plotting threshold for detector images. The default is 2000.
    marker_list : list, optional
        list of markers for each item in data_list. The default is list(Line2D.markers.keys()).
    c_size : float, optional
        size of scatter point in plotting. The default is 10.
    c_legend : list, optional
        list of strings for data plotting in legend. The default is 'data'.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        image plot figure.
    ax : matplotlib.pyplot.axes
        image plot axes.

    '''
    
    # find num detectors
    num_det = len(hedm_instr.detectors.keys())
    key_0 = list(hedm_instr.detectors.keys())[0]
    
    # plotting indices for dexelas
    pl = np.arange(num_det)
    if num_det == 2:
        pl = [1, 0]
        nrows = 1
        ncols = 2
    if num_det == 8:
        pl = [2, 3, 6, 7, 0, 1, 4, 5]
        nrows = 2
        ncols = 4
    
    # create a new figure
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle(title)
    ax = ax.ravel()

    for j, key in enumerate(hedm_instr.detectors.keys()):
        for i in range(len(det_x_dict_list[key_0])):
            ax[pl[j]].imshow(img_dict[key], vmin=vmin, vmax=vmax, cmap='Greys_r')
            
            ax[pl[j]].scatter(det_x_dict_list[key][i], 
                        det_y_dict_list[key][i], 
                        c=det_data_dict_list[key][i], 
                        marker=marker_list[(i % len(marker_list))],
                        s=c_size)
            ax[pl[j]].set_xlabel(key)
    
    fig.legend(c_legend)
    plt.show()
    
    return fig, ax

def det_img_plot_vs_data_colomap(pd, hedm_instr, img_dict, 
                         det_x_dict_list, det_y_dict_list, det_data_dict_list,
                         title='Detector Image with Data',
                         vmin=0, vmax=2000, marker_list=list(Line2D.markers.keys()),
                         c_vmin=0, c_vmax=1, c_size=10, c_label='data'):
    '''
    for multiple figures each with their own colormap

    Parameters
    ----------
    pd : hexrd.crystallography.PlaneData
        plane data object holding ceria.
    hedm_instr : hexrd.Instrument object
        instrument object with panels.
    sim_det_tth_eta : dict
        simulated data save as dictionary with structure:
            dict[det_key][hkl_ring_index][tth_meas_pkfit, eta (from eta_centers)]
    img_dict : dict
        detector image data as  dictionary with structure:
            dict[det_key] = image
    det_x_dict_list : dict
        x plotting positions as dictionary with structure:
            dict[det_key][lists of positions] = x_position
    det_y_dict_list : dict
        y plotting positions as dictionary with structure:
           dict[det_key][lists of positions] = y_position
    det_data_dict_list : dict
        data for pllotting at x,y positions as dictionary with structure:
           dict[det_key][lists of data] = data
    title : string, optional
        title of figure. The default is 'Detector Image with Data'.
    vmin : float, optional
        minimum plotting threshold for detector images. The default is 0.
    vmax : float, optional
        maximum plotting threshold for detector images. The default is 2000.
    marker_list : list, optional
        list of markers for each item in data_list. The default is list(Line2D.markers.keys()).
    c_vmin : float or list, optional
        mimimum threshold for plotting colormap. The default is 0.
    c_vmax : float or list, optional
        maximum threshold for plotting colormap. The default is 1.
    c_size : float, optional
        size of scatter point in plotting. The default is 10.
    c_label : string or list, optional
        DESCRIPTION. The default is 'data'.

    Returns
    -------
    fig_list : list of matplotlib.pyplot.figure
        image plot figure.
    ax_list : list of matplotlib.pyplot.axes
        image plot axes.

    '''
    
    # find num detectors
    num_det = len(hedm_instr.detectors.keys())
    key_0 = list(hedm_instr.detectors.keys())[0]
    
    # plotting indices for dexelas
    pl = np.arange(num_det)
    if num_det == 2:
        pl = [1, 0]
        nrows = 1
        ncols = 2
    if num_det == 8:
        pl = [2, 3, 6, 7, 0, 1, 4, 5]
        nrows = 2
        ncols = 4
    
    # might want to enforce lists have same length
    
    # for each bit of data, create a new figure
    fig_list = []
    ax_list = []
    for i in range(len(det_x_dict_list[key_0])):
        # create a new figure
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        fig.suptitle(title)
        ax = ax.ravel()
        
        for j, key in enumerate(hedm_instr.detectors.keys()):
            ax[pl[j]].imshow(img_dict[key], vmin=vmin, vmax=vmax, cmap='Greys_r')
            
            
            if isinstance(c_vmin, list):
                t_c_vmin = c_vmin[i]
            else:
                t_c_vmin = c_vmin
                
            if isinstance(c_vmax, list):
                t_c_vmax = c_vmax[i]
            else:
                t_c_vmax = c_vmax
            
            if isinstance(c_label, list):
                t_c_label = c_label[i]
            else:
                t_c_label = c_label
                
            ax[pl[j]].scatter(det_x_dict_list[key][i], 
                        det_y_dict_list[key][i], 
                        c=det_data_dict_list[key][i], 
                        marker=marker_list[(i % len(marker_list))],
                        s=c_size,
                        vmin=t_c_vmin, vmax=t_c_vmax)
            ax[pl[j]].set_xlabel(key)
            
        norm = matplotlib.colors.Normalize(vmin=t_c_vmin, vmax=t_c_vmax)
        sm = plt.cm.ScalarMappable(norm=norm)
    
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(sm, cax=cbar_ax, label=t_c_label)
        
        fig_list.append(fig)
        ax_list.append(ax)

    plt.show()
    
    return fig_list, ax_list

def eval_tth_eta(pd, hedm_instr, sim_det_data, exp_det_data):
    '''
    Evaluation of detector calibration (how well measured vs simulated data fit),
    evaluation for each panel calculated with:
        eval_det = mean((tth_{exp} - tth_{sim}) / tth_{sim})  [over all detector]
        eval_total = mean(eval_det) [averaged over all detectors]

    Parameters
    ----------
    pd : hexrd.crystallography.PlaneData
        plane data object holding ceria.
    hedm_instr : hexrd.Instrument object
        instrument object with panels.
    sim_det_tth_eta : dict
        simulated data save as dictionary with structure:
            dict[det_key][hkl_ring_index][tth_meas_pkfit, eta (from eta_centers)]
    exp_det_tth_eta : dict
        experiment data save as dictionary with structure:
            dict[det_key][hkl_ring_index][tth_meas_pkfit, eta (from eta_centers)]

    Returns
    -------
    eval_det_dict : dict
        evaluation of each panel fit with the structure:
            dict[det_key]
    eval_total_instr : float
        evaluation of the total instrument fit.

    '''
    
    # initialize detector evalution dictionary
    eval_det_dict = {}
    eval_all_det = []
    
    # for each detector, add to initalized lists
    for key in hedm_instr.detectors.keys():
        # initial sim tth, exp tth, and eta lists
        sim = []
        exp = []
        eta = []
        
        # for each ring, collect the data and plot
        for i_ring in np.arange(pd.hkls.shape[1]):
            #print(i_ring, key)
            if sim_det_data[key][i_ring] is not None:
                if len(sim_det_data[key][i_ring]) > 0:
                    # 0 is tth angle, 1 is eta angle
                    sim.append(sim_det_data[key][i_ring][:, 0].astype(float))
                    exp.append(exp_det_data[key][i_ring][:, 0].astype(float))
                    eta.append(sim_det_data[key][i_ring][:, 1].astype(float))
        
        # transform lists to numpy array
        if len(sim) > 1:
            sim = np.hstack(sim)
            exp = np.hstack(exp)
            eta = np.hstack(eta)
        else:
            sim = np.array(sim)
            exp = np.array(exp)
            eta = np.array(eta)
        
        # calculate evaluation
        r = (exp - sim) / sim
        
        # this should probably be improved upon for evaluation
        eval_det_dict[key] = np.mean(r)
        eval_all_det.append(r)
    eval_total_instr = np.mean(np.hstack(eval_all_det))
    
    return eval_det_dict, eval_total_instr
    
    
#%% main functions

def evaluate_powder_fit(powder_config_file,
                        eta_tol=1.0,
                        tth_tol=0.2,
                        tth_max=None,
                        eta_centers=np.linspace(-180, 180, num=361),
                        pktype='pvoigt',
                        plane_data_exclusions=[],
                        apply_dis_exp=False,
                        apply_dis_sim=False,
                        plots=True,
                        save_path=None):
    '''
    Function for evaluating the fit or calibration of hexrd instrument file

    Parameters
    ----------
    powder_config_file : string
        path to hexrd config file (not instr config file).
    eta_tol : float, optional
        eta tolerance (degrees) for caking/fitting powder lines. The default is 1.0.
    tth_tol : float, optional
        twotheta tolerance (degrees) for caking/fitting powder lines. The default is 0.2.
    tth_max : float, optional
        twotheta maximum for fitting powder lines. The default is None.
    eta_centers : array, optional
        array for defining eta centers for caking/fitting powder lines. 
        The default is np.linspace(-180, 180, num=361).
    pktype : string, optional
        Peak fitting type (as defined in hexrd) for fitting powder lines. The default is 'pvoigt'.
    plane_data_exclusions : array, optional
        array for indcies of hkl planes to exclude from evaluation. The default is [].
    apply_dis_exp : bool, optional
        apply distortion to experimentally measured data. The default is False.
    apply_dis_sim : bool, optional
        apply distortion to simulated data. The default is False.
    plots : bool, optional
        flag for showing plots during evaluation. The default is True.
    save_path : string, optional
        if not None, path to save plots and data. The default is None.

    Returns
    -------
    eval_total : float
        evaluation of total instrument defined by eval_tth_eta.
    eval_det_dict : dict
        evaluation of each panel defined by eval_tth_eta.
    sim_det_data : dict
        simulated data save as dictionary with structure:
            dict[det_key][hkl_ring_index][tth_meas_pkfit, eta (from eta_centers), x_det, y_det, x_piixel, y_pixel]
    exp_det_data : dict
        experimental data save as dictionary with structure:
            dict[det_key][hkl_ring_index][tth_meas_pkfit, eta (from eta_centers), x_det, y_det, x_piixel, y_pixel]

    '''
    
    # process user input 
    # *************************************************************************
    
    # open hexrd config file
    cfg = config.open(powder_config_file)[0]

    # initialize instruments
    instr = cfg.instrument
    hedm_instr = instr.hedm

    # initialize ceria material 
    matl = make_matl('ceria', 225, [5.41153, ])
    matl.beamEnergy = hedm_instr.beam_energy
    pd = matl.planeData
    if tth_tol is not None:
        pd.tThWidth = np.radians(tth_tol)
    if tth_max is not None:
        pd.exclusions = None
        pd.tThMax = np.radians(tth_max)

    curr_exclusions = pd.exclusions
    for i in plane_data_exclusions:
        if i < curr_exclusions.size:
            curr_exclusions[i] = True
    pd.exclusions = curr_exclusions

    # intialize image series and image dict
    ims_dict = {}
    img_dict = {}
    if cfg.__dict__['_cfg']['image_series']['format'] == 'frame-cache':
        ims_dict = cfg.image_series
        for key in hedm_instr.detectors.keys():
            img_dict[key] = ims_dict[key][0]
        
    elif cfg.__dict__['_cfg']['image_series']['format'] == 'hdf5':
        panel_ops_dict = {'ff1':[('flip', 'v')], 'ff2':[('flip', 'h')]}
        for key in hedm_instr.detectors.keys():
            for i in cfg.__dict__['_cfg']['image_series']['data']:
                if i['panel'] == key:
                    ims = imageseries.open(i['file'], format='hdf5', path='/imageseries')
                    ims_dict[key] = imageseries.process.ProcessedImageSeries(ims, panel_ops_dict[key])
                    img_dict[key] = ims_dict[key][0]


    # extract powder line positions (tth, eta) + (det_x, det_y) from exp. and sim. 
    # *************************************************************************
    print("Extracting Experimental Powder Line Fits %i" %(eta_centers.size))
    exp_line_data = hedm_instr.extract_line_positions(pd, img_dict, 
                                                   eta_centers=eta_centers,
                                                   collapse_eta=True,
                                                   collapse_tth=False,
                                                   eta_tol=eta_tol,
                                                   tth_tol=tth_tol)

    print("Extracting Siimulated Powder Line Fits %i" %(eta_centers.size))
    # pow_angs, pow_xys, tth_ranges = panel.make_powder_rings(
    #     plane_data, merge_hkls=True,
    #     delta_tth=tth_tol, delta_eta=eta_tol,
    #     eta_list=eta_centers, tth_distortion=tth_distr_cls)
    sim_data = hedm_instr.simulate_powder_pattern([matl])
    sim_line_data = hedm_instr.extract_line_positions(pd, sim_data, 
                                                   eta_centers=eta_centers,
                                                   collapse_eta=True,
                                                   collapse_tth=False,
                                                   eta_tol=eta_tol,
                                                   tth_tol=tth_tol)


    # reogranize and fit the data into sim_det_tth_eta and exp_det_tth_eta dicts
    # *************************************************************************
    ''' 
    each dict takes the structure:
    dict[det_key][hkl_ring_index][tth_meas_pkfit, eta (from eta_centers), x_det, y_det, x_piixel, y_pixel]

    Note: hedm_instr.extract_line_positions can extract tth, eta directly with 
    collapse_tth, collapse_eta, but this gives access to all the tth,eta intensity
    data (2D patch data)
    '''

    sim_det_data = {}
    exp_det_data = {}
    for key, panel in hedm_instr.detectors.items():
        print('working on panel %s...' %(key))
        sim_det_data[key] = [None] * pd.hkls.shape[1]
        exp_det_data[key] = [None] * pd.hkls.shape[1]
        
        for i_ring, ringset in enumerate(sim_line_data[key]):
            print('processing i_ring %i' %(i_ring))
            sim = []
            exp = []
            for i_set, temp in enumerate(ringset):
                if len(sim_line_data[key][i_ring][i_set][1]) > 0 and len(exp_line_data[key][i_ring][i_set][1]) > 0:
                    # simulated data
                    sim_angs = sim_line_data[key][i_ring][i_set][0]
                    sim_inten = sim_line_data[key][i_ring][i_set][1]
                    sim_eta_ref = sim_angs[1]
                    
                    sim_tth_centers = np.average(np.vstack([sim_angs[0][:-1], sim_angs[0][1:]]), axis=0)
                    sim_int_centers = np.average(np.vstack([sim_inten[0][:-1], sim_inten[0][1:]]), axis=0)
                    
                    # experimental data
                    exp_angs = exp_line_data[key][i_ring][i_set][0]
                    exp_inten = exp_line_data[key][i_ring][i_set][1]
                    exp_eta_ref = exp_angs[1]
                    exp_tth_centers = np.average(np.vstack([exp_angs[0][:-1], exp_angs[0][1:]]), axis=0)
                    exp_int_centers = np.average(np.vstack([exp_inten[0][:-1], exp_inten[0][1:]]), axis=0)
                    
                    if sim_tth_centers.size == sim_int_centers.size and exp_tth_centers.size == exp_int_centers.size:
                        # peak profile fitting
                        p0 = fitpeak.estimate_pk_parms_1d(sim_tth_centers, sim_int_centers, pktype)
                        p = fitpeak.fit_pk_parms_1d(p0, sim_tth_centers, sim_int_centers, pktype)
                        sim_tth_meas = p[1]
                        #sim_tth_avg = np.average(sim_tth_centers, weights=sim_int_centers)
                        
                        p0 = fitpeak.estimate_pk_parms_1d(exp_tth_centers, exp_int_centers, pktype)
                        p = fitpeak.fit_pk_parms_1d(p0, exp_tth_centers, exp_int_centers, pktype)
                        exp_tth_meas = p[1]
                        #exp_tth_avg = np.average(exp_tth_centers, weights=exp_int_centers)
                        
                        sim_tth_eta = np.vstack([sim_tth_meas, sim_eta_ref]).T
                        exp_tth_eta = np.vstack([exp_tth_meas, exp_eta_ref]).T
                        
                        sim.append(sim_tth_eta)
                        exp.append(exp_tth_eta)
            
            if len(sim) > 0:
                # convert to numpy array
                sim_t_e = np.vstack(sim)
                exp_t_e = np.vstack(exp)
                
                # extracting detector x,y position and pixel indices
                sim_xy_det = hedm_instr.detectors[key].angles_to_cart(sim_t_e,
                                                                   rmat_s=None, tvec_s=None,
                                                                   rmat_c=None, tvec_c=None,
                                                                   apply_distortion=apply_dis_exp)
                sim_xy_det = hedm_instr.detectors[key].clip_to_panel(sim_xy_det, buffer_edges=True)
                sim_pix_ind = hedm_instr.detectors[key].cartToPixel(sim_xy_det[0], pixels=False, apply_distortion=apply_dis_exp)
                
                exp_xy_det = hedm_instr.detectors[key].angles_to_cart(exp_t_e,
                                                                   rmat_s=None, tvec_s=None,
                                                                   rmat_c=None, tvec_c=None,
                                                                   apply_distortion=apply_dis_exp)
                exp_xy_det = hedm_instr.detectors[key].clip_to_panel(exp_xy_det, buffer_edges=True)
                exp_pix_ind = hedm_instr.detectors[key].cartToPixel(exp_xy_det[0], pixels=False, apply_distortion=apply_dis_exp)
                
                sim_det_data[key][i_ring] = np.hstack([sim_t_e, sim_xy_det[0], sim_pix_ind])
                exp_det_data[key][i_ring] = np.hstack([exp_t_e, exp_xy_det[0], exp_pix_ind])
            else:
                sim_det_data[key][i_ring] = None
                exp_det_data[key][i_ring] = None
            
    # plot the initial delta_tth / tth_0 error vs eta
    # *************************************************************************
    if plots:
        fig_polar, ax_polar = polar_plot_tth_vs_eta(pd, hedm_instr, sim_det_data, exp_det_data, 
                                  rmin=-0.0025, rmax=0.0025,
                                  title=cfg.analysis_name + "\n CeO2 $\\frac{\\theta_{exp} - \\theta_{sim}}{\\theta_{sim}}$",
                              legend=[])
        if save_path is not None:
            fig_polar.savefig(os.path.join(save_path, cfg.analysis_name + '_polar_plot_exp_vs_sim.png'))


    # plot det pixel positions of simulated and measured peak fits on ff images
    # *************************************************************************
    if plots:
        # organizing the data for imaage plotting
        det_x_list_dict_s = {}
        det_y_list_dict_s = {}
        det_data_list_dict_s = {}
        det_x_list_dict_c = {}
        det_y_list_dict_c = {}
        det_data_list_dict_c = {}
    
        for j, key in enumerate(hedm_instr.detectors.keys()):
            panel_exp_tth = []
            panel_exp_xy = []
            panel_exp_pix = []
            panel_sim_tth = []
            panel_sim_xy = []
            panel_sim_pix = []
            
            for i_ring in np.arange(pd.hkls.shape[1]):
                if sim_det_data[key][i_ring] is not None:
                    if len(sim_det_data[key][i_ring]) > 0:
                        panel_sim_tth.append(sim_det_data[key][i_ring][:, 0])
                        panel_sim_xy.append(sim_det_data[key][i_ring][:, [2, 3]])
                        panel_sim_pix.append(sim_det_data[key][i_ring][:, [4, 5]])
                        panel_exp_tth.append(exp_det_data[key][i_ring][:, 0])
                        panel_exp_xy.append(exp_det_data[key][i_ring][:, [2, 3]])
                        panel_exp_pix.append(exp_det_data[key][i_ring][:, [4, 5]])
            
            panel_exp_xy = np.vstack(panel_exp_xy)
            panel_sim_xy = np.vstack(panel_sim_xy)
            panel_exp_pix = np.vstack(panel_exp_pix)
            panel_sim_pix = np.vstack(panel_sim_pix)
            panel_exp_tth = np.hstack(panel_exp_tth)
            panel_sim_tth = np.hstack(panel_sim_tth)
            
            # actual data strucutres used for plotting
            det_x_list_dict_s[key] = [panel_sim_pix[:, 1], panel_exp_pix[:, 1]]
            det_y_list_dict_s[key] = [panel_sim_pix[:, 0], panel_exp_pix[:, 0]]
            det_data_list_dict_s[key] = [np.ones(panel_sim_pix[:, 1].size),
                                       np.ones(panel_sim_pix[:, 1].size)]
            det_x_list_dict_c[key] = [panel_sim_pix[:, 1], panel_sim_pix[:, 1]]
            det_y_list_dict_c[key] = [panel_sim_pix[:, 0], panel_sim_pix[:, 0]]
            det_data_list_dict_c[key] = [np.linalg.norm(panel_exp_xy - panel_sim_xy, axis=1),
                                        (panel_exp_tth - panel_sim_tth) / panel_sim_tth]
            
        
        c_legend = ['sim.', 'exp.']
        fig_image, ax_image = det_img_plot_vs_data_scatter(pd, hedm_instr, img_dict, 
                                 det_x_list_dict_s, det_y_list_dict_s, det_data_list_dict_s,
                                 title='Detector Images with Exp. and Sim. Positions',
                                 vmin=0, vmax=4000, marker_list=list(Line2D.markers.keys()),
                                 c_size=10, c_legend=c_legend)
        
        c_vmin = [0, -0.0015]
        c_vmax = [hedm_instr.detectors[list(hedm_instr.detectors.keys())[0]].pixel_size_row*2.0, 0.0015]
        c_label = ['$||x_{exp} - x_{sim} ||$ (mm)', '$\\frac{\\theta_{exp} - \\theta_{sim}}{\\theta_{sim}}$']
        fig_list, ax_list = det_img_plot_vs_data_colomap(pd, hedm_instr, img_dict, 
                                det_x_list_dict_c, det_y_list_dict_c, det_data_list_dict_c,
                                title='Detector Image with Data',
                                vmin=0, vmax=2000, marker_list=['s'],
                                c_vmin=c_vmin, c_vmax=c_vmax, c_size=10, c_label=c_label)
        
        if save_path is not None:
            fig_image.savefig(os.path.join(save_path, cfg.analysis_name + '_detector_image_sim_exp_scatter.png'))
            for i, fig in enumerate(fig_list):
                fig.savefig(os.path.join(save_path, cfg.analysis_name + '_detector_image_sim_exp_colormap_%i.png' %(i)))
    
    if save_path is not None:
        np.savez(os.path.join(save_path, cfg.analysis_name + '_exp_images.npz'), **img_dict)
        np.savez(os.path.join(save_path, cfg.analysis_name + '_sim_images.npz'), **sim_data)
            
    
    # evaluate detector simulation to experiment fit
    eval_det_dict, eval_total = eval_tth_eta(pd, hedm_instr, sim_det_data, exp_det_data)
    
    return eval_total, eval_det_dict, sim_det_data, exp_det_data, sim_line_data, exp_line_data

def index_hkl_ring_pixels(powder_config_file,
                        tth_max=None,
                        plane_data_exclusions=[],
                        strainMag=0.0025,
                        plots=True,
                        save_path=None):
    '''

    Parameters
    ----------
    powder_config_file : string
        path to hexrd config file (not instr config file).
    tth_max : float, optional
        twotheta maximum for fitting powder lines. The default is None.
    plane_data_exclusions : array, optional
       array for indcies of hkl planes to exclude from evaluation. The default is [].
    strainMag : float, optional
        strain resolution in tth ranges . The default is 0.0025.
    plots : bool, optional
        flag for showing plots during evaluation. The default is True.
    save_path : string, optional
        if not None, path to save plots and data. The default is None.

    Returns
    -------
    mask_dict : dict
        dictornary keyed on detector keys of hkl rings index masks. The background 
        has index = -1 (no ring) and the first ring has the index = 0.

    '''
    
    # process user input 
    # *************************************************************************
    
    # open hexrd config file
    cfg = config.open(powder_config_file)[0]

    # initialize instruments
    instr = cfg.instrument
    hedm_instr = instr.hedm

    # initialize ceria material 
    matl = make_matl('ceria', 225, [5.41153, ])
    matl.beamEnergy = hedm_instr.beam_energy
    pd = matl.planeData
    if tth_max is not None:
        pd.exclusions = None
        pd.tThMax = np.radians(tth_max)

    curr_exclusions = pd.exclusions
    for i in plane_data_exclusions:
        if i < curr_exclusions.size:
            curr_exclusions[i] = True
    pd.exclusions = curr_exclusions

    # get instrument tth, eta mask
    mask_dict = {}
    for key in hedm_instr.detectors.keys():
        panel_tth, panel_eta = hedm_instr.detectors[key].pixel_angles()
        mask_dict[key] = np.ones(panel_tth.shape) * -1
        
        tth_ranges = pd.getTThRanges(strainMag=strainMag)
        for i, tth_range in enumerate(tth_ranges):
            tth_range_ind = (panel_tth >= tth_range[0]) & (panel_tth <= tth_range[1])
            mask_dict[key][tth_range_ind] = i
        
        if plots:
            fig_image = plt.figure()
            fig_image.suptitle(key)
            plt.imshow(mask_dict[key])
            plt.show()
            if save_path is not None:
                fig_image.savefig(os.path.join(save_path, cfg.analysis_name + '_%s_hkl_index_mask.png' %(key)))
    
    return mask_dict

#%% testing 
if __name__ == "__main__":
    powder_config_file = '/media/djs522/djs522_nov2020/dexela_distortion/data_for_simon/dexela_61-332kev_ceo2_mruby_instr_config.yml'
    #powder_config_file = '/media/djs522/djs522_nov2020/dexela_distortion/data_for_simon/dexela_61-332kev_ceo2_mruby_final_mpanel_instr_config.yml'
    #powder_config_file = '/media/djs522/djs522_nov2020/dexela_distortion/data_for_simon/dexela_61-332kev_ceo2_mruby_simon_instr_config.yml'
    
    eval_total, eval_det_dict, sim_det_data, exp_det_data, t1, t2 = evaluate_powder_fit(powder_config_file,
                            eta_tol=1.0,
                            tth_tol=0.2,
                            tth_max=14.0,
                            eta_centers=np.linspace(-180, 180, 181), #np.linspace(-180, 180, num=361),
                            pktype='pvoigt',
                            plane_data_exclusions=[],
                            apply_dis_exp=False,
                            apply_dis_sim=False,
                            plots=True,
                            save_path='/media/djs522/djs522_nov2020/dexela_distortion/data_for_simon/')
    hkl_mask_dict = index_hkl_ring_pixels(powder_config_file,
                            tth_max=14.0,
                            plane_data_exclusions=[],
                            strainMag=0.01,
                            plots=True,
                            save_path='/media/djs522/djs522_nov2020/dexela_distortion/data_for_simon/')
