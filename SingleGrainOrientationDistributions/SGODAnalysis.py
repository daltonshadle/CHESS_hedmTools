#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:13:32 2022

@author: djs522
"""

# *****************************************************************************
# IMPORTS
# *****************************************************************************

import sys

import cc3d

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as plt_colors
#from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import pandas as pd

from scipy.stats import gaussian_kde
#from scipy.optimize import nnls

if sys.version_info[0] < 3:
    # python 2
    from hexrd.xrd          import rotations  as hexrd_rot
    from hexrd.xrd          import symmetry   as hexrd_sym
else:
    # python 3
    from hexrd        import rotations  as hexrd_rot
    from hexrd        import symmetry   as hexrd_sym
    #import scipy.spatial.transform as scipy_trans
    #from sklearn.neighbors import KernelDensity

import CHESS_hedmTools.SingleGrainOrientationDistributions.OrientationTools as OrientationTools
from pyevtk.hl import pointsToVTK

# *****************************************************************************
# FUNCTION DECLARATION AND IMPLEMENTATION
# *****************************************************************************

def plot_grain_dsgod(grain_rod, grain_odf=None, reverse_map=False, 
                     just_faces=False, no_axis=False, cmap=plt.cm.viridis_r,
                     scatter_size=400, fig=None, ori_ax=None):
    '''
    Purpose: plots the discrete single grain orientation distribution (DSGOD) in
      3D Rodigrues Orientation Space with 2D projected distributions on the 
      far faces of the plot area

    Parameters
    ----------
    grain_rod : n x 3 numpy array
        Rodrigues Orientation vectors given as the rows of the array
    grain_odf : n x 1 numpy array
        The weight associated with each orientation, if None it will default
        uniform weights
    reverse_map : bool
        True = reverse the color mapping of the distributions
        False = nothing
    just_faces : bool
        True = plots just the 2D projected distributions on the far faces
        False = plots 3D distribution and 2D projected distributions
    no_axis : bool
        True = removes the axes lines from the plot
        False = leaves the axies lines in the plot
    cmap : matplotlib.pyplot.cm
        matplotlib color map for the distributions
    scatter_size : int
        size of scatter plot points in DSGOD
    fig : matplotlib.pyplot.figure
        matplotlib figure object for the DSGOD plot, if None a figure will be
        generated in the function
    ori_ax : matplotlib.pyplot.ax
        matplotlib ax object for the DSGOD plot, if None an axis will be
        generated in the function

    Returns
    -------
    fig : matplotlib.pyplot.figure
        matplotlib figure object for the DSGOD plot
    ori_ax : matplotlib.pyplot.ax
        matplotlib ax object for the DSGOD plot
        
    Notes
    -----
    TODO: Still some hard-coded variables in this functions, need to replace
    with options parameter
    '''
    
    if grain_odf is None:
        grain_odf = np.ones([grain_rod.shape[0]])
    
    if reverse_map:
        grain_odf = -grain_odf
    
    # main figure
    if fig is None:
        fig = plt.figure(figsize=(12, 12))
    fig.patch.set_alpha(0)
    
    # set up plot area for orientations
    if ori_ax is None:
        ori_ax = Axes3D(fig)
    ori_ax.patch.set_facecolor('white')
    ori_ax.patch.set_alpha(0.0)
    ori_ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ori_ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ori_ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    alpha_max = 1
    alpha_min = 0.6 # 0.6
    my_cmap[:,-1] = np.linspace(alpha_min, alpha_max, cmap.N)
    if reverse_map:
        my_cmap[:,-1] = np.linspace(alpha_max, alpha_min, cmap.N)
    # Create new colormap
    my_cmap = plt_colors.ListedColormap(my_cmap)
    
    # plot rods with odf    
    if not just_faces:
        ori_ax.scatter(grain_rod[:, 0], grain_rod[:, 1], grain_rod[:, 2], 
                       c=grain_odf, cmap=my_cmap, s=scatter_size)
    
    # plot 3 histograims for each face
    for i in range(3):
        x = i % 3
        y = (i + 1) % 3
        z = (i + 2) % 3
        
        # estimate 2D histograim with kernel density
        step = np.abs(grain_rod[:, x].min() - grain_rod[:, x].max()) / 0.8 #0.1
        xmin, ymin = grain_rod[:, x].mean(axis=0)-step, grain_rod[:, y].mean(axis=0)-step
        xmax, ymax = grain_rod[:, x].mean(axis=0)+step, grain_rod[:, y].mean(axis=0)+step
        n = 50j # default is 50
        xi, yi = np.mgrid[xmin:xmax:n, ymin:ymax:n]
        coords = np.vstack([item.ravel() for item in [xi, yi]])
        kde = gaussian_kde(grain_rod[:, [x,y]].T, weights=grain_odf)
        density = kde(coords).reshape(xi.shape)
        density = density / np.sum(density)
        if reverse_map:
            density = -density
        
        edge1_mesh = xi
        edge2_mesh = yi
        
        # set color mapping
        vmin = np.mean(density) / 10
        vmax = np.max(density)
        
        lim_step = 0.02
        c_norm = plt_colors.Normalize(vmin=vmin, vmax=vmax)
        if reverse_map:
            c_norm = plt_colors.Normalize(vmin=-vmax, vmax=-vmin)
        
        scale_map = cm.ScalarMappable(norm=c_norm, cmap=cmap)
        normalized_h_map = scale_map.to_rgba(density)
        normalized_h_map[np.abs(density) < vmin, 3] = 0
        pos_3 = np.min(grain_rod[:, z]) - (lim_step - 1e-3)
        flat_pos_3 = np.full_like(edge1_mesh, pos_3)
        
        # plot surfaces
        if i == 0:
            p = ori_ax.plot_surface(edge1_mesh, edge2_mesh, flat_pos_3, facecolors=normalized_h_map, rstride=1, cstride=1, shade=False)
        elif i == 1:
            p = ori_ax.plot_surface(flat_pos_3, edge1_mesh, edge2_mesh, facecolors=normalized_h_map, rstride=1, cstride=1, shade=False)
        else:
            p = ori_ax.plot_surface(edge2_mesh, flat_pos_3, edge1_mesh, facecolors=normalized_h_map, rstride=1, cstride=1, shade=False)
           
    # label axis and show
    lp = 50
    ori_ax.set_xlabel('$r_{x}$', rotation=0, labelpad=lp)
    ori_ax.set_ylabel('$r_{y}$', rotation=0, labelpad=lp)
    ori_ax.set_zlabel('$r_{z}$', rotation=0, labelpad=lp)
    ori_ax.xaxis.set_rotate_label(True)
    ori_ax.yaxis.set_rotate_label(True)
    ori_ax.zaxis.set_rotate_label(True)
    
    # set limits
    lim_list = [np.min(grain_rod[:, 0]) - lim_step, np.max(grain_rod[:, 0]) + lim_step,
                np.min(grain_rod[:, 1]) - lim_step, np.max(grain_rod[:, 1]) + lim_step,
                np.min(grain_rod[:, 2]) - lim_step, np.max(grain_rod[:, 2]) + lim_step]
    
    ori_ax.set_xlim3d(lim_list[0], lim_list[1])
    ori_ax.set_ylim3d(lim_list[2], lim_list[3])
    ori_ax.set_zlim3d(lim_list[4], lim_list[5])
    
    # rotate graph to isometric
    ori_ax.view_init(25, 45)
    
    # set font size
    plt.rcParams.update({'font.size': 20})
    
    if no_axis:
        ori_ax.grid(True)
        ori_ax.set_xticklabels([])
        ori_ax.set_yticklabels([])
        ori_ax.set_zticklabels([])
        ori_ax.set_xlabel('', rotation=0, labelpad=lp)
        ori_ax.set_ylabel('', rotation=0, labelpad=lp)
        ori_ax.set_zlabel('', rotation=0, labelpad=lp)
    
    return [fig, ori_ax]


def grain_dsgod_to_vtk(vtk_save_dir, grain_rod, grain_odf=None):
    '''
    Purpose: saves the discrete single grain orientation distribution (DSGOD) in
      .vtk file for visualization in Paraview

    Parameters
    ----------
    vtk_save_dir : string
        Path directory to save .vtk file to
    grain_rod : n x 3 numpy array
        Rodrigues Orientation vectors given as the rows of the array
    grain_odf : n x 1 numpy array
        The weight associated with each orientation, if None it will default
        uniform weights

    Returns
    -------
    None
        
    Notes
    -----
    TODO: Still some hard-coded variables in this functions, need to replace
    with options parameter
    '''
    
    if grain_odf is None:
        grain_odf = np.ones([grain_rod.shape[0]])
    
    pointsToVTK(vtk_save_dir, np.array([grain_rod[:, 0]]), np.array([grain_rod[:, 1]]), np.array([grain_rod[:, 2]]), data = {'dsgod': grain_odf})
    
def calc_misorientation_quats(quats, avg_quat=None, disp_stats=False):
    '''
    Purpose: calculates the misorientation quaternions between the average 
        quaternion orientation and a list of quaternion orientations

    Parameters
    ----------
    quats : n x 4 numpy array
        Quaternion orientation vectors given as the rows of the array
    avg_quat : 1 x 4 numpy array
        Quaternion avergae orientation vectors given as the row of the array, 
        if None, the average will be calculated from the given quats
    disp_stats : bool
        True = will display stats as misorientation is calculated

    Returns
    -------
    mis_quats : n x 4 numpy array
        Quaternion misorientation vectors given as the rows of the array
    mis_ang_deg : n x 1 numpy array
        Array of misorientation angles (degrees) for each orientation compared 
        to average orientation
    avg_quat : 1 x 4 numpy array
        Quaternion avergae orientation vectors given as the row of the array
        
    Notes
    -----
    TODO: Still some hard-coded variables in this functions, need to replace
    with options parameter
    '''
    if avg_quat is None:
        avg_quat = hexrd_rot.quatAverageCluster(quats.T, qsym=hexrd_sym.quatOfLaueGroup('Oh')).T
        avg_quat.shape = [1, 4]

    [mis_ang_rad, mis_quats] = hexrd_rot.misorientation(avg_quat.T, quats.T)
    mis_ang_deg = np.rad2deg(mis_ang_rad)
    
    if disp_stats:
        print('Avg misorientation: %0.3f' %(np.mean(mis_ang_deg)))
        print('Max misorientation: %0.3f' %(np.max(mis_ang_deg)))
    
    return [mis_quats.T, mis_ang_deg, avg_quat.T]


def calc_misorient_moments(grain_mis_quat, grain_odf=None, norm_regularizer=0):
    '''
    Purpose: calculates higher order moments of a discrete single grain
       orientation distribution (DSGOD)

    Parameters
    ----------
    grain_mis_quat : n x 4 numpy array
        Quaternion misorientation vectors given as the rows of the array
    grain_odf : n x 1 numpy array
        The weight associated with each orientation, if None it will default
        uniform weights
    norm_regularizer : float
        number to regularize higher order moment calculations (experimental)

    Returns
    -------
    norm_sigma : float
        norm of the sigma (covariance) matrix
    norm_gamma : float
        norm of the gamma (skewness) matrix
    norm_kappa : float
        norm of the kappa (kurtosis) matrix
    sigma : 3 x 3 numpy array
        sigma (covariance) matrix
    gamma : 3 x 3 x 3 numpy array
        gamma (skewness) matrix
    kappa : 3 x 3 x 3 x 3 numpy array
        kappa (kurtosis) matrix
        
    Notes
    -----
    TODO: Still some hard-coded variables in this functions, need to replace
        with options parameter
    Refer to https://link.springer.com/article/10.1007/s40192-019-00153-4 for 
        details on DSGOD higher order analysis, we note that Rodrigues 
        misorientation vectors are used here instead of angle-axis misorientation
        vectors as described in the paper
    '''
    
    # initialize
    norm_sigma = norm_gamma = norm_kappa = sigma = gamma = kappa = 0
    
    # preprocess
    grain_mis_rod = OrientationTools.quat2rod(grain_mis_quat)
    
    if grain_odf is None:
        grain_odf = np.ones([grain_mis_rod.shape[0]])
    
    # get number of orientations
    num_ori = grain_odf.size
    default_num = 37
    
    if num_ori < 2:
        #print('WARNING: Number of orientation (%i) is low, no HOM calculations!' %(num_ori))
        pass
    else:
        # if num_ori < default_num:
        #     print('WARNING: Number of orientation (%i) is low for HOM clculations!' %(num_ori))
        
        # convert misorienation quaternions to angle-axis rep (both methods equal)
        '''
        grain_mis_quat = grain_mis_quat[:, [1,2,3,0]]
        grain_mis_angle_axis = scipy_trans.Rotation(grain_mis_quat).as_rotvec()
        
        #[grain_mis_angle, grain_mis_vec] = hexrd_rot.angleAxisOfRotMat(hexrd_rot.rotMatOfQuat(grain_mis_quat))
        #grain_mis_angle_axis = grain_mis_vec.T * grain_mis_angle[:, np.newaxis]
        '''
        
        # compute covariance
        outer_1 = np.einsum('ij,ik->ijk', grain_mis_rod, grain_mis_rod)
        sigma = (num_ori/(num_ori - 1)
            * np.sum(np.einsum('i,ijk->ijk', grain_odf, outer_1), axis=0))
    
        # compute sigma norm for later
        norm_sigma = np.linalg.norm(sigma)
    
        # compute skewness
        outer_2 = np.einsum('ijk,il->ijkl', outer_1, grain_mis_rod)
        gamma = ((norm_sigma + norm_regularizer)**(-3/2)
            * np.sum(np.einsum('i,ijkl->ijkl', grain_odf, outer_2), axis=0))
    
        # compute gamma norm for later
        norm_gamma = np.linalg.norm(gamma)
    
        # compute kurtosis
        outer_3 = np.einsum('ijkl,im->ijklm', outer_2, grain_mis_rod)
        kappa = ((norm_sigma + norm_regularizer)**(-2)
            * np.sum(np.einsum('i,ijklm->ijklm', grain_odf, outer_3), axis=0))
    
        # compute kappa norm for later
        norm_kappa = np.linalg.norm(kappa)

    return norm_sigma, norm_gamma, norm_kappa, sigma, gamma, kappa
    
    
def animate_dsgods_rod(grain_rod_list, grain_odf_list=None, labels_list=None, 
                 interval=1000, save_gif_dir=None):
    '''
    Purpose: creates an animated gif from a list of discrete single grain orientation
        distributions (DSGODs)

    Parameters
    ----------
    grain_rod_list : list of n x 3 numpy arrays
        list of Rodrigues orientation vectors given as the rows of the array
    grain_odf : list of n x 1 numpy arrays
        list of orientation weights for each DSGOD, must be same length as 
        grain_rod_list, if None will default to uniform distributions
    labels_list : list of strings
        list of labels for each frame in the gif, must be same length as 
        grain_rod_list, if None will default to simple iteration labels
    interval : int
        milliseconds for each frame in the animation
    save_gif_dir : string
        full directory string to save gif to, if None it won't save gif

    Returns
    -------
    None, can save gif
        
    Notes
    -----
    None
    '''
    
    numframes = len(grain_rod_list)
    
    if labels_list is None:
        labels_list = np.arange(numframes)
    
    if grain_odf_list is None:
        grain_odf_list = []
        for i in range(numframes):
            grain_odf_list.append(np.ones([grain_rod_list[i].shape[0]]))
        
    # initialize plot
    fig = plt.figure(figsize=[12, 8])
    ori_ax = Axes3D(fig)
    
    # plot DSGOD
    [fig, ori_ax] = plot_grain_dsgod(grain_rod_list[0], grain_odf=grain_odf_list[0], reverse_map=False, 
                     just_faces=False, no_axis=False, cmap=plt.cm.viridis_r,
                     scatter_size=50, fig=fig, ori_ax =ori_ax)
    fig.suptitle('Time Step: %s' %(labels_list[0]))
    
    # set limits
    init_avg_rod = np.atleast_2d(np.average(grain_rod_list[0], axis=0, weights=grain_odf_list[0]))
    
    # do function animation
    ani = animation.FuncAnimation(fig, update_animate_dsgods_rod, interval=interval,
                                  frames=numframes, fargs=(grain_rod_list, grain_odf_list, 
                                                           labels_list, ori_ax, init_avg_rod, fig))
    #mng = plt.get_current_fig_manager()
    #mng.window.showMaximized()
    
    plt.show()
    if save_gif_dir is not None:
        ani.save(save_gif_dir, writer='imagemagick', dpi=400)

def update_animate_dsgods_rod(i, grain_rod_list, grain_odf_list, labels_list, ori_ax,
                        init_avg_rod, fig):
    '''
    Purpose: companion update function to animate_dsgods_rod

    Parameters
    ----------
    i : int
        frame number given by FuncAnimation
    grain_rod_list : list of n x 3 numpy arrays
        list of Rodrigues orientation vectors given as the rows of the array
    grain_odf : list of n x 1 numpy arrays
        list of orientation weights for each DSGOD, must be same length as 
        grain_rod_list, if None will default to uniform distributions
    labels_list : list of strings
        list of labels for each frame in the gif, must be same length as 
        grain_rod_list, if None will default to simple iteration labels
    ori_ax : matplotlib.pyplot.ax
        matplotlib ax object for the DSGOD plot
    init_avg_rod : 1 x 3 numpy array
        initial average Rodrigues orientation to center all plots on
    fig : matplotliblpyplot.figure
        matplotlib figure object for the DSGOD plot

    Returns
    -------
    None, can save gif
        
    Notes
    -----
    None
    '''
    ori_ax.cla()
      
    [fig, ori_ax] = plot_grain_dsgod(grain_rod_list[i], grain_odf=grain_odf_list[i], reverse_map=False, 
                     just_faces=False, no_axis=False, cmap=plt.cm.viridis_r,
                     scatter_size=50, fig=fig, ori_ax =ori_ax)
    fig.suptitle('Time Step: %s' %(labels_list[i]))

    return ori_ax,


def moments_plotting(col_x, col_y, df, col_k='kind', 
                     color_list=['#D81B60', '#1E88E5', '#FFC107', '#004D40'], 
                     marker_list=['o', '^', 's', 'd'], scatter_alpha=.65, 
                     do_global=False, x_lim=None, y_lim=None, 
                     x_bins=25, y_bins=25, hatch_list=['//', 'xx', '--', '..'],
                     fs=14):
    '''
    Purpose: scatter plot and histogram plots of the moments for a collection 
       of DSGODs

    Parameters
    ----------
    col_x : string
        name of the pandas X column for plotting
    col_y : string
        name of the pandas Y column for plotting
    df : pandas dataframe
        collection of moments data for plotting
    col_k : string 
        name of the pandas grouping kind for plotting
    color_list : list of hex code colors
        list of colors for plotting different moment datasets 
        (default to 4 colorblind friendly colors)
    marker_list : list of markers styles for scatter plots
        list of markers styles for scatter plots for plotting different 
        moment datasets 
    scatter_alpha : float
        transparency of the scatter points
    do_global : boolean
        boolean for plotting global histograms for all datasets
    x_lim : 1 x 2 list or numpy array
        limits for the x-axis
    y_lim : 1 x 2 list or numpy array
        limits for the y-axis
    x_bins : int
        number of histogram bins along the x-axis
    y_bins : int
        number of histogram bins along the y-axis
    hatch_list : list of hatch styles for histograms
        list of hatch styles for histograms for plotting different moment
        datasets
    fs : int
        font size for legend and axes

    Returns
    -------
    fig_ax : seaborn figure axes
        returns the figure for additional adjustments
        
    Notes
    -----
    None
    '''
    
    def colored_scatter(x, y, c=None, marker='.'):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            kwargs['s'] = 125
            kwargs['marker'] = marker
            plt.scatter(*args, **kwargs)

        return scatter
    
    if x_lim is not None:
        x_bins = np.arange(x_lim[0], x_lim[1], (x_lim[1]-x_lim[0])/x_bins)
    if y_lim is not None:
        y_bins = np.arange(y_lim[0], y_lim[1], (y_lim[1]-y_lim[0])/y_bins)
    
    fig_ax = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df,
        xlim=x_lim,
        ylim=y_lim,
        height=5,
    )
    color = None
    legends=[]
    linestyle_list = ['-', (0, (1, 1)), (0, (3, 1, 1, 1)), '..']
    i = 0
    
    col_x_val = []
    col_y_val = []
    
    for name, df_group in df.groupby(col_k, sort=False):
        legends.append(name)
        if color_list is not None:
            color=color_list[i]
        marker=marker_list[i]
        fig_ax.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color,marker),
        )
        
        # gd1 = sns.distplot(
        #     df_group[col_x].values,
        #     ax=g.ax_marg_x,
        #     color=color,
        #     bins=x_bins,
        #     hist=True,
        #     norm_hist=False,
        #     kde=False,
        #     hist_kws={'histtype':'step', 'linewidth':3, 'color':color, 'linestyle':linestyle_list[i]}
        #     #hist_kws={'hatch':hatch_list[i], 'edgecolor':'k', 'stacked':True, 'histtype':'step', 'linewidth':3, 'color':color}
        # )
        # gd2 = sns.distplot(
        #     df_group[col_y].values,
        #     ax=g.ax_marg_y,
        #     color=color,            
        #     vertical=True,
        #     bins=y_bins,
        #     hist=True,
        #     norm_hist=False,
        #     kde=False,
        #     hist_kws={'histtype':'step', 'linewidth':3, 'color':color, 'linestyle':linestyle_list[i]}
        # )
        
        col_x_val.append(df_group[col_x].values)
        col_y_val.append(df_group[col_y].values)
        
        i += 1
    
    gd1 = fig_ax.ax_marg_x.hist(
            col_x_val,
            color=color_list[:i],
            bins=x_bins,
            orientation='vertical',
            rwidth=0.95
        )
    gd2 = fig_ax.ax_marg_y.hist(
            col_y_val,
            color=color_list[:i],   
            bins=y_bins,
            orientation='horizontal',
            rwidth=0.95
        )
        
    fig_ax.ax_marg_y.set_xscale('log')
    fig_ax.ax_marg_x.set_yscale('log')
    fig_ax.ax_marg_y.set_xticks([])
    fig_ax.ax_marg_x.set_yticks([])
        
        
    if do_global:
        # Do also global Hist:
        sns.distplot(
            df[col_x].values,
            ax=fig_ax.ax_marg_x,
            color='grey',
            bins=x_bins
        )
        sns.distplot(
            df[col_y].values.ravel(),
            ax=fig_ax.ax_marg_y,
            color='grey',
            vertical=True,
            bins=y_bins
        )
    
    fig_ax.ax_joint.set_xlabel(col_x, fontsize=fs)
    fig_ax.ax_joint.set_ylabel(col_y, fontsize=fs)
    new_x_ticks = fig_ax.ax_joint.get_xticks()
    x_ticks = []
    for k in range(len(new_x_ticks)):
        x_ticks.append('%0.1f' %(new_x_ticks[k]))
    fig_ax.ax_joint.set_xticklabels(x_ticks, size=fs)
    fig_ax.ax_joint.set_yticklabels(fig_ax.ax_joint.get_yticks(), size=fs)
    plt.legend(legends, fontsize=fs, loc='best')
    
    return fig_ax
    

# *****************************************************************************
# EXPERIEMENTAL FUNCTION DECLARATION AND IMPLEMENTATION
# *****************************************************************************

def process_dsgod_file(dsgod_npz_dir, comp_thresh=0.85, inten_thresh=0, do_avg_ori=True, 
                        do_conn_comp=True, save=False, connectivity_type=18):
    '''
    Purpose: processing raw DSGOD file created from HEDM / VD data

    Parameters
    ----------
    dsgod_npz_dir : string
        path to dsgod .npz file
    comp_thresh : float, optional
        completeness threshold for gathering intensity info to construct dsgod.
        The default is 0.85.
    inten_thresh : float, optional
        intensity threshold to construct dsgod, not recommended. 
        The default is 0.
    do_avg_ori : bool, optional
        flag for using average orientation to identify reference orientaiton 
        cloud if there are multiple clouds in the raw DSGOD file. 
        The default is True.
    do_conn_comp : bool, optional
        flag for using connected components to identify reference orientation 
        cloud if there are multiple clouds in the raw DSGOD file. 
        The default is True.
    save : bool, optional
        flag for saving DSGOD results. The default is False.
    connectivity_type : int, optional
        describes connectivity for connected components, can be 26, 18, and 6. 
        The default is 18.

    Returns
    -------
    [grain_quat, grain_mis_quat, grain_odf]
    grain_quat : numpy array (n x 4)
        list of quaternions in DSGOD
    grain_mis_quat: numpy array (n x 4)
        list of misorientation quaternions (when compared to average 
        orientation) in DSGOD
    grain_odf: numpy array (n x 1)
        list of weights for orientations in DSGOD

    '''
    
    # load grain data
    '''
    np.savez(dsgod_npz_save_dir, dsgod_box_shape=box_shape,
                  dsgod_avg_expmap=cur_exp_maps,
                  dsgod_box_comp=dsgod_box_comp, dsgod_box_quat=dsgod_box_quat,
                  dsgod_box_inten=dsgod_box_inten, dsgod_box_inten_list=dsgod_box_inten_list,
                  dsgod_box_hit_list=dsgod_box_hit_list, dsgod_box_filter_list=dsgod_box_filter_list)
    '''
    
    grain_goe_info = np.load(dsgod_npz_dir)
    grain_goe_box = grain_goe_info['dsgod_box_shape']
    grain_quat = grain_goe_info['dsgod_box_quat'].astype(np.float32)
    grain_inten_arr = grain_goe_info['dsgod_box_inten_list'].astype(np.int32)
    grain_filter_arr = grain_goe_info['dsgod_box_filter_list'].astype(np.int8)
    grain_avg_expmap = grain_goe_info['dsgod_avg_expmap']
    grain_avg_quat = np.atleast_2d(hexrd_rot.quatOfExpMap(grain_avg_expmap.T)).T
    
    # transform orientation to Rodrigues vectors
    grain_quat = np.reshape(grain_quat, [grain_quat.shape[1], grain_quat.shape[2]])
    
    print(grain_inten_arr.shape, grain_filter_arr.shape, grain_quat.shape)
    
    # TIM WORK ****************************************************************
    # reverse sort intensities in high -> low order
    sort_ind = np.argsort(-grain_inten_arr, axis=1)
    sort_grain_inten_arr = np.take_along_axis(grain_inten_arr, sort_ind, axis=1)
    
    # find index of intensity value to use based on completeness thresholding (Tim Long way)
    sum_filter = np.sum(grain_filter_arr, axis=1)
    comp_filter_ind = (comp_thresh * sum_filter).astype(int)
    
    # gather intensity values based on index found above
    grain_inten = sort_grain_inten_arr[np.arange(grain_inten_arr.shape[0]), comp_filter_ind]
    grain_inten[grain_inten < inten_thresh] = 0
    
    if np.any(grain_inten > 0):
        
        if do_conn_comp:
            # CONN COMP WORK ***********************************************************
            
            conn_comp_inten = np.reshape(grain_inten, grain_goe_box)
            conn_comp_map = cc3d.connected_components(conn_comp_inten > inten_thresh, connectivity=connectivity_type)
            
            if do_avg_ori:
                # find nearest non-zero intenisty closest to avg orient as DSGOD group
                nnz_inten_quats = grain_quat[:, grain_inten > inten_thresh]
                
                grain_avg_quat_norms = np.linalg.norm(nnz_inten_quats - grain_avg_quat, axis=0)
                avg_ori_quat = nnz_inten_quats[:, np.where(grain_avg_quat_norms == np.min(grain_avg_quat_norms))[0]]
                avg_ori_ind = np.where((grain_quat == avg_ori_quat).all(axis=0))[0]
                
                # reshape conn comp to find center group number
                group_num = np.reshape(conn_comp_map, [conn_comp_inten.size])[avg_ori_ind]
            else:
                # find max count as GOE group
                # find unique groups and counts
                conn_comp_uni, conn_comp_count = np.unique(conn_comp_map, return_counts=True)
                # remove 0 from list
                conn_comp_count = conn_comp_count[(conn_comp_uni != 0)]
                conn_comp_uni = conn_comp_uni[(conn_comp_uni != 0)]
                # find the group number with max count and filter
                group_num = conn_comp_uni[conn_comp_count == np.max(conn_comp_count)]
            
            # remap values not in group to 0 adn values in group to 1
            conn_comp_map[conn_comp_map != group_num] = 0
            conn_comp_map[conn_comp_map == group_num] = 1
            
            # set all intensities not in group to 0 and reshape
            conn_comp_inten[np.logical_not(conn_comp_map)] = 0
            thresh_grain_inten = np.reshape(conn_comp_inten, [conn_comp_inten.size])
        else:
            thresh_grain_inten = grain_inten
            
        
        # DSGOD WORK **************************************************************
        sum_grain_inten = np.sum(thresh_grain_inten)
        #print(sum_grain_inten)
        
        if sum_grain_inten > 0:
            grain_odf = thresh_grain_inten / sum_grain_inten.astype(float)
            
            print("Number of Diffraction Events: %i" %(grain_inten_arr.shape[1]))
            print('Max inten = %f, Min inten = %f' %(np.max(thresh_grain_inten), np.min(thresh_grain_inten)))
            #print('Max ODF = %f, Min ODF = %f' %(np.max(grain_odf)*100, np.min(grain_odf)*100))
            
            grain_avg_quat = np.atleast_2d(np.average(grain_quat, axis=1, weights=grain_odf)).T
            
            # MISOREINTATION WORK *****************************************************
            [grain_mis_ang_deg, grain_mis_quat] = OrientationTools.calc_misorientation_quat(grain_avg_quat, grain_quat)
            #calc_misorientation(grain_quat[:, grain_odf > 0], avg_quat=grain_avg_quat, disp_stats=False)
        else:
            print('1. Using avg quat')
            grain_quat = grain_avg_quat
            grain_odf = np.ones(grain_quat.shape[1])
            grain_mis_quat = np.zeros(grain_quat.shape)
            grain_mis_quat[0] = 1
            sum_grain_inten = 1
    else:
        print('2. Using avg quat')
        grain_quat = grain_avg_quat
        grain_odf = np.ones(grain_quat.shape[1])
        grain_mis_quat = np.zeros(grain_quat.shape)
        grain_mis_quat[0] = 1
        sum_grain_inten = 0
        
    # RETURN ******************************************************************
    if save:
        dsgod_npz_save_dir = dsgod_npz_dir.split('.npz')[0]
        # np.savez(dsgod_npz_save_dir + '_processed', dsgod_box_shape=grain_goe_box,
        #           dsgod_avg_expmap=grain_avg_expmap, dsgod_avg_quat=grain_avg_quat,
        #           dsgod_box_quat=grain_quat, dsgod_box_mis_quat=grain_mis_quat,
        #           dsgod_box_dsgod=grain_odf, dsgod_comp_thresh=comp_thresh)
        grain_quat = grain_quat[:, grain_odf > 0].T
        grain_mis_quat = grain_mis_quat[:, grain_odf > 0].T
        grain_odf = grain_odf[grain_odf > 0]
        
        comp_str = str(comp_thresh).replace('.', '_')
        np.savez(dsgod_npz_save_dir + '_%s_reduced' %(comp_str), dsgod_box_shape=grain_goe_box,
                  dsgod_avg_expmap=grain_avg_expmap, dsgod_avg_quat=grain_avg_quat,
                  dsgod_box_quat=grain_quat, dsgod_box_mis_quat=grain_mis_quat,
                  dsgod_box_dsgod=grain_odf, dsgod_sum_inten=sum_grain_inten,
                  dsgod_comp_thresh=comp_thresh)
    
    return [grain_quat, grain_mis_quat, grain_odf]

def process_dsgod_file_new(dsgod_npz_dir, comp_thresh=0.85, inten_thresh=0, do_avg_ori=True, 
                        do_conn_comp=True, save=False, connectivity_type=18):
    '''
    Purpose: processing raw DSGOD file created from HEDM / VD data

    Parameters
    ----------
    dsgod_npz_dir : string
        path to dsgod .npz file
    comp_thresh : float, optional
        completeness threshold for gathering intensity info to construct dsgod.
        The default is 0.85.
    inten_thresh : float, optional
        intensity threshold to construct dsgod, not recommended. 
        The default is 0.
    do_avg_ori : bool, optional
        flag for using average orientation to identify reference orientaiton 
        cloud if there are multiple clouds in the raw DSGOD file. 
        The default is True.
    do_conn_comp : bool, optional
        flag for using connected components to identify reference orientation 
        cloud if there are multiple clouds in the raw DSGOD file. 
        The default is True.
    save : bool, optional
        flag for saving DSGOD results. The default is False.
    connectivity_type : int, optional
        describes connectivity for connected components, can be 26, 18, and 6. 
        The default is 18.

    Returns
    -------
    [grain_quat, grain_mis_quat, grain_odf]
    grain_quat : numpy array (n x 4)
        list of quaternions in DSGOD
    grain_mis_quat: numpy array (n x 4)
        list of misorientation quaternions (when compared to average 
        orientation) in DSGOD
    grain_odf: numpy array (n x 1)
        list of weights for orientations in DSGOD

    '''
    
    # load grain data
    '''
    np.savez(dsgod_npz_save_dir, dsgod_box_shape=box_shape,
                  dsgod_avg_expmap=cur_exp_maps,
                  dsgod_box_comp=dsgod_box_comp, dsgod_box_quat=dsgod_box_quat,
                  dsgod_box_inten=dsgod_box_inten, dsgod_box_inten_list=dsgod_box_inten_list,
                  dsgod_box_hit_list=dsgod_box_hit_list, dsgod_box_filter_list=dsgod_box_filter_list)
    '''
    
    grain_goe_info = np.load(dsgod_npz_dir)
    grain_goe_box = grain_goe_info['dsgod_box_shape']
    grain_inten_arr = grain_goe_info['dsgod_box_inten_list'].astype(np.int32)
    grain_filter_arr = grain_goe_info['dsgod_box_filter_list'].astype(np.int8)
    grain_avg_expmap = grain_goe_info['dsgod_avg_expmap']
    grain_avg_quat = np.atleast_2d(hexrd_rot.quatOfExpMap(grain_avg_expmap.T)).T
    misorientation_bnd = grain_goe_info['misorientation_bnd']
    misorientation_spacing = grain_goe_info['misorientation_spacing']
    if 'truncate_comp_thresh' in list(grain_goe_info.keys()):
        truncate_comp_thresh = grain_goe_info['truncate_comp_thresh']
        if truncate_comp_thresh is None:
            truncate_comp_thresh = 0
    else:
        truncate_comp_thresh = 0
    
    # regenerate orientations    
    mis_amt = np.radians(misorientation_bnd)
    mis_spacing = np.radians(misorientation_spacing)
    ori_pts = np.arange(-mis_amt, (mis_amt+(mis_spacing*0.999)), mis_spacing)
    ori_Xs, ori_Ys, ori_Zs = np.meshgrid(ori_pts, ori_pts, ori_pts)
    ori_grid = np.vstack([ori_Xs.flatten(), ori_Ys.flatten(), ori_Zs.flatten()]).T
    grain_exp_maps = ori_grid + grain_avg_expmap
    
    # transform orientation to Rodrigues vectors
    grain_quat = hexrd_rot.quatOfExpMap(grain_exp_maps.T)
    
    # TIM WORK ****************************************************************
    # reverse sort intensities in high -> low order
    sort_ind = np.argsort(-grain_inten_arr, axis=1)
    sort_grain_inten_arr = np.take_along_axis(grain_inten_arr, sort_ind, axis=1)
    
    # find index of intensity value to use based on completeness thresholding (Tim Long way)
    if comp_thresh < truncate_comp_thresh:
        raise ValueError('Completeness threshold cannot be less than truncated completeness of raw data %0.2f' %(truncate_comp_thresh))
    sum_filter = np.sum(grain_filter_arr, axis=1)
    comp_filter_ind = ((comp_thresh - truncate_comp_thresh) * sum_filter / (1 - truncate_comp_thresh)).astype(int)
    
    # gather intensity values based on index found above
    grain_inten = sort_grain_inten_arr[np.arange(grain_inten_arr.shape[0]), comp_filter_ind]
    grain_inten[grain_inten < inten_thresh] = 0
    
    if np.any(grain_inten > 0):
        
        if do_conn_comp:
            # CONN COMP WORK ***********************************************************
            
            conn_comp_inten = np.reshape(grain_inten, grain_goe_box)
            conn_comp_map = cc3d.connected_components(conn_comp_inten > inten_thresh, connectivity=connectivity_type)
            
            if do_avg_ori:
                # find nearest non-zero intenisty closest to avg orient as DSGOD group
                nnz_inten_quats = grain_quat[:, grain_inten > inten_thresh]
                
                grain_avg_quat_norms = np.linalg.norm(nnz_inten_quats - grain_avg_quat, axis=0)
                avg_ori_quat = nnz_inten_quats[:, np.where(grain_avg_quat_norms == np.min(grain_avg_quat_norms))[0]]
                avg_ori_ind = np.where((grain_quat == avg_ori_quat).all(axis=0))[0]
                
                # reshape conn comp to find center group number
                group_num = np.reshape(conn_comp_map, [conn_comp_inten.size])[avg_ori_ind]
            else:
                # find max count as GOE group
                # find unique groups and counts
                conn_comp_uni, conn_comp_count = np.unique(conn_comp_map, return_counts=True)
                # remove 0 from list
                conn_comp_count = conn_comp_count[(conn_comp_uni != 0)]
                conn_comp_uni = conn_comp_uni[(conn_comp_uni != 0)]
                # find the group number with max count and filter
                group_num = conn_comp_uni[conn_comp_count == np.max(conn_comp_count)]
            
            # remap values not in group to 0 adn values in group to 1
            conn_comp_map[conn_comp_map != group_num] = 0
            conn_comp_map[conn_comp_map == group_num] = 1
            
            # set all intensities not in group to 0 and reshape
            conn_comp_inten[np.logical_not(conn_comp_map)] = 0
            thresh_grain_inten = np.reshape(conn_comp_inten, [conn_comp_inten.size])
        else:
            thresh_grain_inten = grain_inten
            
        
        # DSGOD WORK **************************************************************
        sum_grain_inten = np.sum(thresh_grain_inten)
        #print(sum_grain_inten)
        
        if sum_grain_inten > 0:
            grain_odf = thresh_grain_inten / sum_grain_inten.astype(float)
            
            print("Number of Diffraction Events: %i" %(grain_inten_arr.shape[1] + sum_filter.mean()))
            print('Max inten = %f, Min inten = %f' %(np.max(thresh_grain_inten), np.min(thresh_grain_inten)))
            #print('Max ODF = %f, Min ODF = %f' %(np.max(grain_odf)*100, np.min(grain_odf)*100))
            
            grain_avg_quat = np.atleast_2d(np.average(grain_quat, axis=1, weights=grain_odf)).T
            
            # MISOREINTATION WORK *****************************************************
            [grain_mis_ang_deg, grain_mis_quat] = OrientationTools.calc_misorientation_quat(grain_avg_quat, grain_quat)
            #calc_misorientation(grain_quat[:, grain_odf > 0], avg_quat=grain_avg_quat, disp_stats=False)
        else:
            print('1. Using avg quat')
            grain_quat = grain_avg_quat
            grain_odf = np.ones(grain_quat.shape[1])
            grain_mis_quat = np.zeros(grain_quat.shape)
            grain_mis_quat[0] = 1
            sum_grain_inten = 1
    else:
        print('2. Using avg quat')
        grain_quat = grain_avg_quat
        grain_odf = np.ones(grain_quat.shape[1])
        grain_mis_quat = np.zeros(grain_quat.shape)
        grain_mis_quat[0] = 1
        sum_grain_inten = 0
        
    # RETURN ******************************************************************
    if save:
        dsgod_npz_save_dir = dsgod_npz_dir.split('.npz')[0]
        # np.savez(dsgod_npz_save_dir + '_processed', dsgod_box_shape=grain_goe_box,
        #           dsgod_avg_expmap=grain_avg_expmap, dsgod_avg_quat=grain_avg_quat,
        #           dsgod_box_quat=grain_quat, dsgod_box_mis_quat=grain_mis_quat,
        #           dsgod_box_dsgod=grain_odf, dsgod_comp_thresh=comp_thresh)
        grain_quat = grain_quat[:, grain_odf > 0].T
        grain_mis_quat = grain_mis_quat[:, grain_odf > 0].T
        grain_odf = grain_odf[grain_odf > 0]
        
        comp_str = str(comp_thresh).replace('.', '_')
        np.savez(dsgod_npz_save_dir + '_%s_reduced' %(comp_str), dsgod_box_shape=grain_goe_box,
                  dsgod_avg_expmap=grain_avg_expmap, dsgod_avg_quat=grain_avg_quat,
                  dsgod_box_quat=grain_quat, dsgod_box_mis_quat=grain_mis_quat,
                  dsgod_box_dsgod=grain_odf, dsgod_sum_inten=sum_grain_inten,
                  dsgod_comp_thresh=comp_thresh)
    
    return [grain_quat, grain_mis_quat, grain_odf]

# def process_dsgod_file_inv(dsgod_npz_dir, scan=24, compl_thresh=0.85, reg_lambda=None,
#                                do_conn_comp=False, do_avg_ori=False, connectivity_type=18,
#                                do_dist=False, dist_thresh=None):
#     # connectivity_type can be 26, 18, and 6
#     # load odf info ***********************************************************
#     '''
#     np.savez(dsgod_npz_save_dir, dsgod_box_shape=box_shape,
#                  dsgod_avg_expmap=cur_exp_maps,
#                  dsgod_box_comp=dsgod_box_comp, dsgod_box_quat=dsgod_box_quat,
#                  dsgod_box_inten=dsgod_box_inten, dsgod_box_inten_list=dsgod_box_inten_list,
#                  dsgod_box_hit_list=dsgod_box_hit_list, dsgod_box_filter_list=dsgod_box_filter_list)
#     '''
#     grain_dsgod_info = np.load(dsgod_npz_dir)
#     grain_dsgod_box = grain_dsgod_info['dsgod_box_shape']
#     grain_compl = grain_dsgod_info['dsgod_box_comp']
#     grain_quat = grain_dsgod_info['dsgod_box_quat']
#     grain_inten_arr = grain_dsgod_info['dsgod_box_inten_list']
#     grain_eta_arr = grain_dsgod_info['dsgod_box_eta_ind_list']
#     grain_ome_arr = grain_dsgod_info['dsgod_box_ome_ind_list']
#     grain_avg_quat = hexrd_rot.quatOfExpMap(grain_dsgod_info['dsgod_avg_expmap'].T)
    
#     # reshape completeness and orientation
#     grain_compl = grain_compl.T
#     grain_quat = np.reshape(grain_quat, [grain_quat.shape[1], grain_quat.shape[2]]).T
    
#     # threshold by distance
#     if do_dist:
#         # distance from average orientation threshold
#         grain_avg_quat_norms = np.linalg.norm(grain_quat.T - grain_avg_quat, axis=0)
#         dist_thresh_ind = np.where(grain_avg_quat_norms <= dist_thresh)[0]
        
#         grain_quat = grain_quat[dist_thresh_ind, :]
#         grain_inten_arr = grain_inten_arr[dist_thresh_ind, :]
#         grain_eta_arr = grain_eta_arr[dist_thresh_ind, :]
#         grain_ome_arr = grain_ome_arr[dist_thresh_ind, :]
#         grain_compl = grain_compl[dist_thresh_ind]
    
#     # threshold by completeness
#     compl_ind = np.where(grain_compl >= compl_thresh)[0]
#     compl_grain_quat = grain_quat[compl_ind, :]
#     compl_grain_inten_arr = grain_inten_arr[compl_ind, :]
#     compl_grain_eta_arr = grain_eta_arr[compl_ind, :]
#     compl_grain_ome_arr = grain_ome_arr[compl_ind, :]
    
#     # reshape intensity and eta-ome indices
#     inten_flat = compl_grain_inten_arr.flatten()
#     eta_ome_flat = np.vstack([compl_grain_eta_arr.flatten(), compl_grain_ome_arr.flatten()]).T
    
#     # assemble Ax=b ***********************************************************
#     uni_eta_ome, uni_index = np.unique(eta_ome_flat, return_index=True, axis=0)
#     b_inten = inten_flat[uni_index]
    
#     # see if [-1, -1] is the first entry for indices and remove, [-1, -1] means orientation filtered out
#     bad_ind = np.where((uni_eta_ome == [-1,-1]).all(axis=1))[0]
#     uni_eta_ome = np.delete(uni_eta_ome, bad_ind, axis=0)
#     b_inten = np.delete(b_inten, bad_ind, axis=0)
    
#     # gather matrix dimensions 
#     num_ori = compl_grain_inten_arr.shape[0]
#     num_reflections = compl_grain_inten_arr.shape[1]
#     num_inten = b_inten.shape[0]
    
#     # assemble A
#     A_index = np.zeros([num_inten, num_ori])
#     for i in np.arange(num_inten):
#         temp_uni_eta_ome = uni_eta_ome[i, :]
#         temp_grain_ind = np.where((eta_ome_flat == temp_uni_eta_ome).all(axis=1))[0]
#         temp_A_ind = np.floor(temp_grain_ind / num_reflections).astype(int)
#         A_index[i, temp_A_ind] = 1
    
#     # apply Tikhonov regularization, if needed
#     if reg_lambda is not None:
#         A_index = np.vstack([A_index, reg_lambda * np.eye(num_ori)])
#         b_inten = np.hstack([b_inten, np.zeros(num_ori)]).T
    
#     # print information of linear system
#     print("Shape of Mat A: %i x %i (num_pixels x num_orientations)" %(A_index.shape[0], A_index.shape[1]))
#     print("Number of Diffraction Events: %i" %(num_reflections))
    
#     # solve Ax=b, preint residual
#     x_ori_weights, resid = nnls(A_index, b_inten)
#     print("Residual ||Ax - b|| / ||b||= %f" %(resid/np.linalg.norm(b_inten)))
    
#     # process dsgod for connected components (conn comp) **********************
#     if do_conn_comp:
#         # conn comp work set up
#         ori_weight_thresh = 0
        
#         # assemble conn ccomp map of ori weightssss
#         conn_comp_ori_weights = np.zeros(grain_dsgod_box[0] * grain_dsgod_box[1] * grain_dsgod_box[2])
#         conn_comp_ori_weights[compl_ind] = x_ori_weights
#         conn_comp_ori_weights.shape = grain_dsgod_box
#         conn_comp_map = cc3d.connected_components(conn_comp_ori_weights > ori_weight_thresh, connectivity=connectivity_type)
        
#         if do_avg_ori:
#             # find avg orient as DSGOD group
#             grain_avg_quat_norms = np.linalg.norm(grain_quat.T - grain_avg_quat, axis=0)
#             avg_ori_ind = np.where(grain_avg_quat_norms == np.min(grain_avg_quat_norms))
            
#             # reshape conn comp to find center group number
#             group_num = np.reshape(conn_comp_map, [conn_comp_ori_weights.size])[avg_ori_ind]
#         else:
#             # find max count as DSGOD group
#             # find unique groups and counts
#             conn_comp_uni, conn_comp_count = np.unique(conn_comp_map, return_counts=True)
            
#             # remove 0 from list
#             conn_comp_count = conn_comp_count[(conn_comp_uni != 0)]
#             conn_comp_uni = conn_comp_uni[(conn_comp_uni != 0)]
            
#             # find the group number with max count and filter
#             group_num = conn_comp_uni[conn_comp_count == np.max(conn_comp_count)]
        
#         # remap values not in group to 0 adn values in group to 1
#         conn_comp_map[conn_comp_map != group_num] = 0
#         conn_comp_map[conn_comp_map == group_num] = 1
        
#         # set all intensities not in group to 0 and reshape
#         conn_comp_ori_weights[np.logical_not(conn_comp_map)] = 0
#         thresh_ori_weights = np.reshape(conn_comp_ori_weights, [conn_comp_ori_weights.size])
        
#         # threshold non-zero weights
#         grain_ori_weights_ind = np.where(thresh_ori_weights > 0)[0]
#         grain_compl = grain_compl[grain_ori_weights_ind]
#         grain_quat = grain_quat[grain_ori_weights_ind, :]
#         grain_ori_weights = thresh_ori_weights[grain_ori_weights_ind]
    
#     else:
#         # return orientation weights as is, no connected components work
#         grain_ori_weights = x_ori_weights
#         grain_compl = grain_compl[compl_ind]
#         grain_quat = compl_grain_quat
    
#     return [grain_quat, grain_compl, grain_ori_weights]


