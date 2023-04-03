# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:24:36 2022

@author: Dalton Shadle
"""

#%%
import os

import numpy as np

from hexrd import instrument
from hexrd import imageseries
from hexrd import config
from hexrd import material
from hexrd.fitting import fitpeak

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib


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

def polar_plot_tth_vs_eta(pd, hedm_instr, sim_det_tth_eta, exp_det_tth_eta, 
                          rmin=-0.0025, rmax=0.0025,
                          title='$\\frac{\\theta_{meas} - \\theta_{calc}}{\\theta_{calc}}$ vs $\\eta$',
                          legend=[]):
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
                    c = 0 # c=0 is tth angle
                    sim.append(sim_det_tth_eta[key][i_ring][:, c].astype(float))
                    exp.append(exp_det_tth_eta[key][i_ring][:, c].astype(float))
                    eta.append(sim_det_tth_eta[key][i_ring][:, 2].astype(float))
        
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

def det_img_plot_vs_data(pd, hedm_instr, img_dict, 
                         det_x_dict_list, det_y_dict_list, det_data_dict_list,
                         title='Detector Image with Data',
                         vmin=0, vmax=2000, marker_list=list(Line2D.markers.keys()),
                         c_vmin=0, c_vmax=1, c_size=10):

    # plotting indices for chunked dexelas
    pl = [2, 3, 6, 7, 0, 1, 4, 5]
    if len(hedm_instr.detectors.keys()) > 2:
        fig, ax = plt.subplots(nrows=2, ncols=4)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title)

    j = 0
    for key in hedm_instr.detectors.keys():
        if len(hedm_instr.detectors.keys()) > 2:
            ax[int(np.floor(pl[j] / 4)), int(pl[j] % 4)].imshow(img_dict[key], vmin=vmin, vmax=vmax, cmap='Greys_r')
            
            for i in range(len(det_x_dict_list[key])):
                ax[int(np.floor(pl[j] / 4)), int(pl[j] % 4)].scatter(det_x_dict_list[key][i], 
                                                                     det_y_dict_list[key][i], 
                                                                     c=det_data_dict_list[key][i], 
                                                                     marker=marker_list[(i % len(marker_list))],
                                                                     s=c_size,
                                                                     vmin=c_vmin, vmax=c_vmax)
                
            ax[int(np.floor(pl[j] / 4)), int(pl[j] % 4)].set_xlabel(key)
        else:
            ax[j].imshow(img_dict[key], vmin=vmin, vmax=vmax, cmap='Greys_r')
            
            for i in range(len(det_x_dict_list[key])):
                ax[j].scatter(det_x_dict_list[key][i], 
                            det_y_dict_list[key][i], 
                            c=det_data_dict_list[key][i], 
                            marker=marker_list[(i % len(marker_list))],
                            s=c_size,
                            vmin=c_vmin, vmax=c_vmax)
            
            ax[j].set_xlabel(key)
            
        j += 1

    plt.show()
    
    return fig, ax

#%% user input
all_debug_plot = False
apply_dis_exp = False
apply_dis_sim = False

eta_tol = 1.0
tth_tol = 0.2
tth_max = 8.0
eta_centers = np.linspace(-180, 180, num=720)
pktype = 'pvoigt'
pd_data_exclusions = [0]

base_path = '/media/djs522/djs522_nov2020/dexela_distortion/duncan_april_2018/'
config_name = 'dexela_april2018_41-991kev_ceo2_mruby_instr_config.yml'


base_path = '/media/djs522/djs522_nov2020/dexela_distortion/kirks_dec_2022/'
config_name = '4_dexela_90-524kev_ceo2_instr_mpanel_tilt_and_pos_config.yml'


#%% process user input
# open hexrd config file
cfg = config.open(os.path.join(base_path, config_name))[0]

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
for i in pd_data_exclusions:
    if i < curr_exclusions.size:
        curr_exclusions[0] = True
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


#%% extract powder line positions (tth, eta) + (det_x, det_y) from exp. and sim. 

print("EXP")
exp_line_data = hedm_instr.extract_line_positions(pd, img_dict, 
                                               eta_centers=eta_centers,
                                               collapse_eta=True,
                                               collapse_tth=False,
                                               eta_tol=eta_tol,
                                               tth_tol=tth_tol)

print("SIM")
sim_data = hedm_instr.simulate_powder_pattern([matl])
sim_line_data = hedm_instr.extract_line_positions(pd, sim_data, 
                                               eta_centers=eta_centers,
                                               collapse_eta=True,
                                               collapse_tth=False,
                                               eta_tol=eta_tol,
                                               tth_tol=tth_tol)


#%% reogranize the data into sim_det_tth_eta and exp_det_tth_eta dicts
''' 
each dict takes the structure:
dict[det_key][hkl_ring_index][tth_meas_pkfit, tth_meas_avg, eta]

Note: hedm_instr.extract_line_positions can extract tth, eta directly with 
collapse_tth, collapse_eta, but this gives access to all the tth,eta intensity
data (2D patch data)
'''

sim_det_tth_eta = {}
exp_det_tth_eta = {}
for key, panel in hedm_instr.detectors.items():
    print('working on panel %s...' %(key))
    sim_det_tth_eta[key] = [None] * pd.hkls.shape[1]
    exp_det_tth_eta[key] = [None] * pd.hkls.shape[1]
    
    for i_ring, ringset in enumerate(sim_line_data[key]):
        print('processing i_ring %i' %(i_ring))
        sim = []
        exp = []
        for i_set, temp in enumerate(ringset):
            sim_angs = sim_line_data[key][i_ring][i_set][0]
            sim_inten = sim_line_data[key][i_ring][i_set][1]
            sim_tth_centers = np.average(np.vstack([sim_angs[0][:-1], sim_angs[0][1:]]), axis=0)
            sim_eta_ref = sim_angs[1]
            sim_int_centers = np.average(np.vstack([sim_inten[0][:-1], sim_inten[0][1:]]), axis=0)
            
            exp_angs = exp_line_data[key][i_ring][i_set][0]
            exp_inten = exp_line_data[key][i_ring][i_set][1]
            exp_tth_centers = np.average(np.vstack([exp_angs[0][:-1], exp_angs[0][1:]]), axis=0)
            exp_eta_ref = exp_angs[1]
            exp_int_centers = np.average(np.vstack([exp_inten[0][:-1], exp_inten[0][1:]]), axis=0)
            
            # peak profile fitting
            if sim_tth_centers.size == sim_int_centers.size and exp_tth_centers.size == exp_int_centers.size:
                p0 = fitpeak.estimate_pk_parms_1d(sim_tth_centers, sim_int_centers, pktype)
                p = fitpeak.fit_pk_parms_1d(p0, sim_tth_centers, sim_int_centers, pktype)
                sim_tth_meas = p[1]
                sim_tth_avg = np.average(sim_tth_centers, weights=sim_int_centers)
                
                p0 = fitpeak.estimate_pk_parms_1d(exp_tth_centers, exp_int_centers, pktype)
                p = fitpeak.fit_pk_parms_1d(p0, exp_tth_centers, exp_int_centers, pktype)
                exp_tth_meas = p[1]
                exp_tth_avg = np.average(exp_tth_centers, weights=exp_int_centers)
                
                sim.append([sim_tth_meas, sim_tth_avg, sim_eta_ref])
                exp.append([exp_tth_meas, exp_tth_avg, exp_eta_ref])
        sim_det_tth_eta[key][i_ring] = np.array(sim)
        exp_det_tth_eta[key][i_ring] = np.array(exp)
        
#%% plot the initial delta_tth / tth_0 error vs eta

fig, ax = polar_plot_tth_vs_eta(pd, hedm_instr, sim_det_tth_eta, exp_det_tth_eta, 
                          rmin=-0.0025, rmax=0.0025,
                          title=config_name + "\n Multipanel No Distortion Added     CeO2 $\\frac{\\theta_{meas} - \\theta_{calc}}{\\theta_{calc}}$",
                          legend=np.arange(15))


#%% plot det pixel positions of simulated and measured peak fits on ff images

# organizing the data for plotting
det_x_list_dict = {}
det_y_list_dict = {}
det_data_list_dict = {}
max_data = hedm_instr.detectors[list(hedm_instr.detectors.keys())[0]].pixel_size_row

update_data_max = False

for j, key in enumerate(hedm_instr.detectors.keys()):
    panel_exp_xy = []
    panel_sim_xy = []
    panel_exp_pix = []
    panel_sim_pix = []
    for i_ring in np.arange(pd.hkls.shape[1]):
        if sim_det_tth_eta[key][i_ring] is not None:
            if len(sim_det_tth_eta[key][i_ring]) > 0:
                c = 0
                sim = sim_det_tth_eta[key][i_ring][:, c].astype(float)
                exp = exp_det_tth_eta[key][i_ring][:, c].astype(float)
                eta = sim_det_tth_eta[key][i_ring][:, 2].astype(float)
                        
                exp_tth_eta = np.vstack([exp, eta]).T
                exp_xy_det = hedm_instr.detectors[key].angles_to_cart(exp_tth_eta,
                                                                   rmat_s=None, tvec_s=None,
                                                                   rmat_c=None, tvec_c=None,
                                                                   apply_distortion=apply_dis_exp)
                exp_xy_det = hedm_instr.detectors[key].clip_to_panel(exp_xy_det, buffer_edges=True)
                exp_pix = hedm_instr.detectors[key].cartToPixel(exp_xy_det[0], pixels=False, apply_distortion=apply_dis_exp)
                panel_exp_xy.append(exp_xy_det[0])
                panel_exp_pix.append(exp_pix)
                
                sim_tth_eta = np.vstack([sim, eta]).T
                sim_xy_det = hedm_instr.detectors[key].angles_to_cart(sim_tth_eta,
                                                                   rmat_s=None, tvec_s=None,
                                                                   rmat_c=None, tvec_c=None,
                                                                   apply_distortion=apply_dis_sim)
                sim_xy_det = hedm_instr.detectors[key].clip_to_panel(sim_xy_det, buffer_edges=True)
                sim_pix = hedm_instr.detectors[key].cartToPixel(sim_xy_det[0], pixels=False, apply_distortion=apply_dis_sim)
                panel_sim_xy.append(sim_xy_det[0])
                panel_sim_pix.append(sim_pix)
    
    panel_exp_xy = np.vstack(panel_exp_xy)
    panel_sim_xy = np.vstack(panel_sim_xy)
    panel_exp_pix = np.vstack(panel_exp_pix)
    panel_sim_pix = np.vstack(panel_sim_pix)
    
    # actual data strucutres used for plotting
    det_x_list_dict[key] = [panel_sim_pix[:, 1], panel_exp_pix[:, 1]]
    det_y_list_dict[key] = [panel_sim_pix[:, 0], panel_exp_pix[:, 0]]
    det_data_list_dict[key] = [np.zeros(panel_sim_xy.shape[0]),
                               np.linalg.norm(panel_exp_xy - panel_sim_xy, axis=1)]
    
    if update_data_max:
        max_data = np.max(np.hstack([max_data, np.hstack(det_data_list_dict[key])]))
    
 
fig, ax = det_img_plot_vs_data(pd, hedm_instr, img_dict, 
                         det_x_list_dict, det_y_list_dict, det_data_list_dict,
                         title='Detector Images with Exp. and Sim. Position Difference',
                         vmin=0, vmax=4000, marker_list=list(Line2D.markers.keys()),
                         c_vmin=0, c_vmax=max_data, c_size=10)

fig.legend(['sim', 'exp'])
k = hedm_instr.detectors[list(hedm_instr.detectors.keys())[0]]
norm = matplotlib.colors.Normalize(vmin=0, vmax=max_data)
sm = plt.cm.ScalarMappable(norm=norm)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(sm, cax=cbar_ax, label='mm')

#%% a "poor" attempt at distortion correction using bi-linear fitting of exp to sim data

print_coef = False

# organizing the data for plotting
det_x_list_dict = {}
det_y_list_dict = {}
det_data_list_dict = {}

det_coef_dict = {}
det_coef_inv_dict = {}

for j, key in enumerate(hedm_instr.detectors.keys()):
    panel_exp_xy = []
    panel_sim_xy = []
    panel_exp_pix = []
    panel_sim_pix = []
    
    panel_exp_xy_corr = []
    for i_ring in np.arange(pd.hkls.shape[1]):
        if sim_det_tth_eta[key][i_ring] is not None:
            if len(sim_det_tth_eta[key][i_ring]) > 0:
                c = 0
                sim = sim_det_tth_eta[key][i_ring][:, c].astype(float)
                exp = exp_det_tth_eta[key][i_ring][:, c].astype(float)
                eta = sim_det_tth_eta[key][i_ring][:, 2].astype(float)
                        
                exp_tth_eta = np.vstack([exp, eta]).T
                exp_xy_det = hedm_instr.detectors[key].angles_to_cart(exp_tth_eta,
                                                                   rmat_s=None, tvec_s=None,
                                                                   rmat_c=None, tvec_c=None,
                                                                   apply_distortion=apply_dis_exp)
                exp_xy_det = hedm_instr.detectors[key].clip_to_panel(exp_xy_det, buffer_edges=True)
                exp_pix = hedm_instr.detectors[key].cartToPixel(exp_xy_det[0], pixels=False, apply_distortion=apply_dis_exp)
                panel_exp_xy.append(exp_xy_det[0])
                panel_exp_pix.append(exp_pix)
                
                sim_tth_eta = np.vstack([sim, eta]).T
                sim_xy_det = hedm_instr.detectors[key].angles_to_cart(sim_tth_eta,
                                                                   rmat_s=None, tvec_s=None,
                                                                   rmat_c=None, tvec_c=None,
                                                                   apply_distortion=apply_dis_sim)
                sim_xy_det = hedm_instr.detectors[key].clip_to_panel(sim_xy_det, buffer_edges=True)
                sim_pix = hedm_instr.detectors[key].cartToPixel(sim_xy_det[0], pixels=False, apply_distortion=apply_dis_sim)
                panel_sim_xy.append(sim_xy_det[0])
                panel_sim_pix.append(sim_pix)
    
    panel_exp_xy = np.vstack(panel_exp_xy)
    panel_sim_xy = np.vstack(panel_sim_xy)
    panel_exp_pix = np.vstack(panel_exp_pix)
    panel_sim_pix = np.vstack(panel_sim_pix)
    
    coef = []
    coef_inv = []
    corr_xy = []
    for c in [0, 1]:
        # transform distorted to undistorted (exp to sim)
        det_exp_xy_mat = np.hstack([np.ones([panel_exp_xy.shape[0], 1]),
                                             panel_exp_xy])
        x = np.linalg.lstsq(det_exp_xy_mat, panel_sim_xy[:, c], rcond=None)
        coef.append(x[0])
        
        # transform undistorted to distorted (sim to exp)
        det_sim_xy_mat = np.hstack([np.ones([panel_sim_xy.shape[0], 1]),
                                             panel_sim_xy])
        x_inv = np.linalg.lstsq(det_sim_xy_mat, panel_exp_xy[:, c], rcond=None)
        coef_inv.append(x_inv[0])
        
        corr_xy.append(det_exp_xy_mat @ x[0])
    
    if print_coef:
        distort_string = \
            '    distortion: \n\
          function_name: Dexela_2923 \n\
          parameters: \n\
            - %0.05e \n\
            - %0.05e \n\
            - %0.05e \n\
            - %0.05e \n\
            - %0.05e \n\
            - %0.05e'
        print(key)
        print(distort_string %(coef[0], coef[1], coef[2], coef[3], coef[4], coef[5]))
    
    panel_exp_xy_corr = np.vstack(corr_xy).T
    panel_exp_pix_corr = hedm_instr.detectors[key].cartToPixel(panel_exp_xy_corr, pixels=False, apply_distortion=apply_dis_exp)
    
    det_coef_dict[key] = np.hstack(coef)
    det_coef_inv_dict[key] = np.hstack(coef_inv)
    
    # actual data strucutres used for plotting
    det_x_list_dict[key] = [panel_sim_pix[:, 1], panel_exp_pix_corr[:, 1]]
    det_y_list_dict[key] = [panel_sim_pix[:, 0], panel_exp_pix_corr[:, 0],]
    det_data_list_dict[key] = [np.zeros(panel_sim_xy.shape[0]),
                               np.linalg.norm(panel_exp_xy_corr - panel_sim_xy, axis=1)]
    
 
fig, ax = det_img_plot_vs_data(pd, hedm_instr, img_dict, 
                         det_x_list_dict, det_y_list_dict, det_data_list_dict,
                         title='Detector Images with Exp.-Corr. vs Sim. Position Difference',
                         vmin=0, vmax=4000, marker_list=list(Line2D.markers.keys()),
                         c_vmin=0, c_vmax=max_data, c_size=10)

fig.legend(['sim', 'exp_corr'])
k = hedm_instr.detectors[list(hedm_instr.detectors.keys())[0]]
norm = matplotlib.colors.Normalize(vmin=0, vmax=max_data)
sm = plt.cm.ScalarMappable(norm=norm)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(sm, cax=cbar_ax, label='mm')

#%% plot the corrected delta_tth / tth_0 error vs eta

exp_corr_det_tth_eta = {}

for key, panel in hedm_instr.detectors.items():
    print('working on panel %s...' %(key))
    exp_corr_det_tth_eta[key] = [None] * pd.hkls.shape[1]
    
    for i_ring, ringset in enumerate(sim_line_data[key]):
        print('processing i_ring %i' %(i_ring))
        
        # get exp measured data
        if exp_det_tth_eta[key][i_ring] is not None:
            if len(exp_det_tth_eta[key][i_ring]) > 0:
                c = 0
                exp = exp_det_tth_eta[key][i_ring][:, c].astype(float)
                eta = exp_det_tth_eta[key][i_ring][:, 2].astype(float)
                exp_tth_eta = np.vstack([exp, eta]).T
                panel_exp_xy = hedm_instr.detectors[key].angles_to_cart(exp_tth_eta,
                                                                   rmat_s=None, tvec_s=None,
                                                                   rmat_c=None, tvec_c=None,
                                                                   apply_distortion=apply_dis_exp)
                panel_exp_xy = hedm_instr.detectors[key].clip_to_panel(panel_exp_xy, buffer_edges=True)[0]
                
                # correct exp measured data
                panel_exp_corr_xy = np.zeros(panel_exp_xy.shape)
                for j in range(2):
                    det_exp_xy_mat = np.hstack([np.ones([panel_exp_xy.shape[0], 1]),
                                                         panel_exp_xy])
                    panel_exp_corr_xy[:, j] = det_exp_xy_mat @ det_coef_dict[key][j*3:(j+1)*3]
                    
                panel_exp_corr_angs = hedm_instr.detectors[key].cart_to_angles(panel_exp_corr_xy,
                                                                                rmat_s=None, tvec_s=None,
                                                                                tvec_c=None,
                                                                                apply_distortion=apply_dis_exp)
                
                exp_corr_det_tth_eta[key][i_ring] = np.vstack([panel_exp_corr_angs[0][:, 0], 
                                                               np.zeros(panel_exp_corr_angs[0][:, 0].shape),
                                                               panel_exp_corr_angs[0][:, 1]]).T

fig, ax = polar_plot_tth_vs_eta(pd, hedm_instr, sim_det_tth_eta, exp_corr_det_tth_eta, 
                          rmin=-0.0025, rmax=0.0025,
                          title=config_name + "\n Multipanel Distortion Corrected     CeO2 $\\frac{\\theta_{meas} - \\theta_{calc}}{\\theta_{calc}}$",
                          legend=np.arange(15))



