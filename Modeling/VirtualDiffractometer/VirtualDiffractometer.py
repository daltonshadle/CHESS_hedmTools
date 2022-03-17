# ============================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC. 
# Produced at the Lawrence Livermore National Laboratory. 
# Written by Joel Bernier <bernier2@llnl.gov> and others. 
# LLNL-CODE-529294. 
# All rights reserved.
# 
# This file is part of HEXRD. For details on downloading the source,
# see the file COPYING.
# 
# Please also see the file LICENSE.
# 
# This program is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free Software
# Foundation) version 2.1 dated February 1999.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the 
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# ============================================================

#%% Imports
###############################################################################

import sys

import os

import matplotlib.pyplot as plt

import numpy as np

import yaml

from hexrd import config
from hexrd import instrument
from hexrd import imageseries
from hexrd import material
from hexrd.gridutil import cellIndices

import scipy as sp
from scipy.signal import fftconvolve

from joblib import Parallel, delayed

#%% Old VD Functions
###############################################################################

# multiprocessing of frames
def old_process_frame_mp(pixd, filter_pad_transform, nrows, ncols, nframes, cts_per_event=1e3, min_intensity=0.01, ncpus=1):
    #frame_cache_data=[sp.sparse.coo_matrix([nrows,ncols],dtype='uint16')]*nframes
    
    image_stack = Parallel(n_jobs=ncpus, verbose=2)(delayed(old_process_frame)(idx, nrows, ncols, pixd, cts_per_event, filter_pad_transform, min_intensity) for idx in range(nframes))
    image_stack = np.array(image_stack)
    
    return image_stack

def old_process_frame(i, nrows, ncols, pixd, cts_per_event, filterPadTransform, min_intensity):    
    this_frame = np.zeros((nrows, ncols), dtype=float)
    
    all_ij = pixd[:4, pixd[2, :] == i].astype(int)
    
    # THIS MAY NEED TO BE FIXED IF ALL "THESE_INTENSITIES" AREN'T 1
    #uni_ij, count_ij = np.unique(these_ij, axis=1, return_counts=True)
    this_frame[all_ij[0, :], all_ij[1, :]] += cts_per_event * all_ij[3, :]
    
    this_frame_transform=np.fft.fft2(this_frame)
    this_frame_convolved=np.real(np.fft.ifft2(this_frame_transform*filterPadTransform))  
    tmp=np.where(this_frame_convolved<min_intensity)
    this_frame_convolved[tmp]=0.
    
    debug = False
    if debug:
        print('Frame %i Count Max: %f' %(i, np.max(this_frame_convolved)))
        sys.stdout.flush()
    return sp.sparse.coo_matrix(this_frame_convolved,dtype='uint16')


#%% hexrd helper Functions
###############################################################################
# plane data
def load_pdata_hexrd3(h5, key):
    temp_mat = material.Material(material_file=h5, name=key)
    return temp_mat.planeData

# images
def load_images_hexrd3(yml):
    return imageseries.open(yml, format="image-files")

# instrument
def load_instrument_hexrd3(yml):
    with open(yml, 'r') as f:
        icfg = yaml.safe_load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


#%% Detector Intercept Functions
###############################################################################

# multiprocessing of pixd
def calc_pixel_intercepts_mp(grain_mat, plane_data, instr, eta_range, ome_period_rad, ome_edges, row_edges, col_edges, chunk_size, ncpus=1):
    num_events = grain_mat.shape[0]
    total_tasks = int(num_events/chunk_size) + 1
    chunks = np.arange(0, total_tasks + 1) * chunk_size
    chunks[-1] = num_events
    
    print('Total Tasks: %i' %(total_tasks))
    pixd_list = Parallel(n_jobs=ncpus, verbose=7)(delayed(calc_pixel_intercepts)(idx, grain_mat[chunks[idx]:chunks[idx+1], :], 
                         plane_data, instr, eta_range, ome_period_rad, 
                         ome_edges, row_edges, col_edges, total_tasks) for idx in range(total_tasks))
    
    num_items = 0
    for pixd_arr in pixd_list:
        num_items += pixd_arr.shape[1]
    
    print('Total Events: %i' %num_items)
    # pixd = [i_row_idx, j_col_idx, frame_num_idx]
    pixd = np.empty([3, num_items])
    idx = 0
    for pixd_arr in pixd_list:
        next_idx = idx+pixd_arr.shape[1]
        pixd[:, idx:next_idx] = pixd_arr
        idx = next_idx
    
    return pixd

def calc_pixel_intercepts(i, grain_mat, plane_data, instr, eta_range, ome_period_rad, ome_edges, row_edges, col_edges, total_tasks):    
    det_keys = instr.detectors.keys()
    
    simg = instr.simulate_rotation_series(plane_data, 
                                          grain_mat[:, 3:15],
                                          eta_ranges=eta_range,
                                          ome_ranges=[(-np.pi, np.pi), ],
                                          ome_period=ome_period_rad,
                                          wavelength=None)
    
    #valid_ids = np.empty([0, 1])
    #valid_hkl = np.empty([0, 3])
    valid_ang = np.empty([0, 3])
    valid_xy = np.empty([0, 2])
    #ang_ps = np.empty([0, 2])
    for det_key in det_keys:
        [det_valid_ids, det_valid_hkl, det_valid_ang, det_valid_xy, det_ang_ps] = simg[det_key]
        
        #valid_ids = np.append(valid_ids, np.atleast_2d(np.array(det_valid_ids).flatten(order='C')).T, axis=0)
        #valid_hkl = np.append(valid_hkl, np.vstack(det_valid_hkl), axis=0)
        valid_ang = np.append(valid_ang, np.vstack(det_valid_ang), axis=0)
        valid_xy = np.append(valid_xy, np.vstack(det_valid_xy), axis=0)
        #ang_ps = np.append(ang_ps, np.vstack(det_ang_ps), axis=0)
    
    
    frame_indices = cellIndices(ome_edges, np.degrees(valid_ang[:, 2]))
    i_row = cellIndices(row_edges, valid_xy[:, 1])
    j_col = cellIndices(col_edges, valid_xy[:, 0])
    
    return np.vstack([i_row, j_col, frame_indices])

# multiprocessing of frames
def reduce_intercept_frames_mp(pixel_intercepts, nframes, nrows, ncols, ncpus=1):   
    reduce_list = Parallel(n_jobs=ncpus, verbose=2)(delayed(reduce_intercept_frames)
                                                    (idx, pixel_intercepts, nrows, ncols) for idx in range(nframes))
    
    pixel_intercepts = []
    frame_cache_list = []
    
    for i in range(len(reduce_list)):
        pixel_intercepts.append(reduce_list[i][0])
        frame_cache_list.append(reduce_list[i][1])
        
    return np.hstack(pixel_intercepts), frame_cache_list

def reduce_intercept_frames(i, pixel_intercepts, nrows, ncols):    
    idx = np.where(pixel_intercepts[2, :] == i)[0]
            
    i_uni_pdi, i_uni_pdi_count = np.unique(pixel_intercepts[:3, idx].astype(int), axis=1, return_counts=True)
    i_uni_pdi = i_uni_pdi.astype(int)
    
    i_pixel_intercepts = np.vstack([i_uni_pdi, i_uni_pdi_count])
    
    i_frame_cache = sp.sparse.csr_matrix((i_uni_pdi_count, (i_uni_pdi[0, :], i_uni_pdi[1, :])), 
                                   shape=(nrows, ncols), 
                                   dtype='uint16')
    
    return [i_pixel_intercepts, i_frame_cache]

def calc_diffraction_detector_intercepts_and_frame(grain_mat, cfg_file, 
                                                   output_dir, ome_period=[-180.0, 180.0], 
                                                   ncpus=1, chunk_size=None,
                                                   save_pixel_intercepts=False, 
                                                   save_frame_cache_ims=False):
    
    #% Process user input
    ###############################################################################
    
    # load config file
    cfg = config.open(cfg_file)[0] # NOTE: always a list of cfg objects
    
    # instrument info
    instr = cfg.instrument.hedm
    #instr = load_instrument_hexrd3(cfg.instrument)
    det_keys = list(instr.detectors.keys())
    if len(det_keys) > 1:
        print("Error: Only built for one detector VD simulations. Multi-detector coming soon!")
        return
    for det_key in det_keys:
        row_edges = instr.detectors[det_key].row_edge_vec
        col_edges = instr.detectors[det_key].col_edge_vec
        nrows = instr.detectors[det_key].rows
        ncols = instr.detectors[det_key].cols
    
    # process omega range for detector frames
    ome_period_rad = (np.deg2rad(ome_period[0]), np.deg2rad(ome_period[1]))
    delta_ome = cfg.find_orientations.omega.tolerance
    nframes = int(abs(ome_period[1]-ome_period[0])/float(delta_ome))
    ome_edges = np.arange(nframes + 1)*delta_ome + ome_period[0]
    
    # process eta ranges
    eta_range = np.deg2rad(cfg.find_orientations.eta.range)
    
    # process material and plane data
    plane_data = load_pdata_hexrd3(cfg.material.definitions, cfg.material.active)
    plane_data.tThMax = np.radians(cfg.fit_grains.tth_max)
    #plane_data.set_exclusions(np.zeros(len(plane_data.exclusions), dtype=bool))
    
    # output frame cache location
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    pixel_intercepts_output_name = '%s_%s_%s' %(cfg.analysis_name, det_key.lower(), 'pixel_intercepts')
    pixel_intercepts_output_full_dir = os.path.join(output_dir, pixel_intercepts_output_name)
    frame_cache_output_name = '%s_%s_%s' %(cfg.analysis_name, det_key.lower(), 'pixel_intercepts_cache')
    frame_cache_output_full_dir = os.path.join(output_dir, frame_cache_output_name)
    
    if chunk_size == None:
        chunk_size = int(np.ceil(grain_mat.shape[0] / ncpus))
    
    #% Do diffraction-detector event calcuations
    ###############################################################################
    if ncpus > 1:
        # multiprocessing        
        pixel_intercepts = calc_pixel_intercepts_mp(grain_mat, 
                                                    plane_data, instr, 
                                                    eta_range, ome_period_rad, 
                                                    ome_edges, row_edges, col_edges, 
                                                    chunk_size=chunk_size, ncpus=ncpus)
    
    else:
        pixel_intercepts = []
        
        print('Do diffraction-detector event calcuation')
        total_tasks = np.ceil(grain_mat.shape[0]/chunk_size)
        for ii in np.arange(total_tasks):
            print(ii)
            start_i = int(ii * chunk_size)
            end_i = int((ii+1) * chunk_size)
            if end_i > grain_mat.shape[0]:
                end_i = grain_mat.shape[0]
            
            chunk_pixel_intercepts = calc_pixel_intercepts(ii, grain_mat[start_i:end_i, :], 
                                                           plane_data, instr, 
                                                           eta_range, ome_period_rad, 
                                                           ome_edges, row_edges, col_edges, 
                                                           total_tasks)
            
            pixel_intercepts.append(chunk_pixel_intercepts)
           
        # assemble pixel detector intercepts
        pixel_intercepts = np.hstack(pixel_intercepts)
    
    old = False
    if old:
        print("Reducing pixel intercepts")
        uni_pdi, uni_pdi_count = np.unique(pixel_intercepts[:3, :].astype(int), axis=1, return_counts=True)
        uni_pdi = uni_pdi.astype(int)
        #frame_cache = np.zeros([nframes, nrows, ncols], dtype='uint16')
        #frame_cache[uni_pdi[2, :], uni_pdi[0, :], uni_pdi[1, :]] = uni_pdi_count
        
        # pixel_intercepts = [i_row_idx, j_col_idx, frame_num_idx, number_of_intercepts]
        pixel_intercepts = np.vstack([uni_pdi, uni_pdi_count])
        
        if save_pixel_intercepts:
            print("Saving pixel intercepts")
            np.savez(pixel_intercepts_output_full_dir + '.npz', pixel_intercepts=pixel_intercepts)
        
        print("Assembling frame cache")
        frame_cache_list = []
        for i in range(nframes):
            idx = np.where(uni_pdi[2, :] == i)[0]
            frame_cache_list.append(sp.sparse.csr_matrix((uni_pdi_count[idx], (uni_pdi[0, idx], uni_pdi[1, idx])), 
                                           shape=(nrows, ncols), 
                                           dtype='uint16'))
    else:        
        print("Reducing pixel intercepts and assembling frame cache")
        
        pixel_intercepts, frame_cache_list = reduce_intercept_frames_mp(pixel_intercepts, nframes, nrows, ncols, ncpus=ncpus)
        
        '''
        frame_cache_list = []
        new_pixel_intercepts = np.empty([4, 0])
        for i in range(nframes):
            print(i)
            idx = np.where(pixel_intercepts[2, :] == i)[0]
            
            i_uni_pdi, i_uni_pdi_count = np.unique(pixel_intercepts[:3, idx].astype(int), axis=1, return_counts=True)
            i_uni_pdi = i_uni_pdi.astype(int)
            
            new_pixel_intercepts = np.hstack([new_pixel_intercepts, 
                                              np.vstack([i_uni_pdi, i_uni_pdi_count])])
            
            frame_cache_list.append(sp.sparse.csr_matrix((i_uni_pdi_count, (i_uni_pdi[0, :], i_uni_pdi[1, :])), 
                                           shape=(nrows, ncols), 
                                           dtype='uint16'))
        pixel_intercepts = new_pixel_intercepts
        '''
    
    print("Creating frame cache image series")
    frame_omega_steps = np.vstack([ome_edges[:nframes], ome_edges[1:nframes+1]]).T
    intercept_ims = imageseries.open(frame_cache_output_name, 'frame-cache', data=frame_cache_list, style='csr_mat_list', 
                                meta={'panel_id': det_keys[0], 'omega': frame_omega_steps})
    if save_pixel_intercepts:
            print("Saving pixel intercepts")
            np.savez(pixel_intercepts_output_full_dir + '.npz', pixel_intercepts=pixel_intercepts)
    
    if save_frame_cache_ims:
        print("Saving frame cache")
        ims_writer = imageseries.save.WriteFrameCache(intercept_ims, frame_cache_output_name + '.npz', 
                                                      style='npz', threshold=1, 
                                                      cache_file=frame_cache_output_full_dir + '.npz')
        ims_writer._write_frames()
    
    return pixel_intercepts, intercept_ims


#%% Filter Functions
###############################################################################
# Filters For Point Spread
def make_gaussian_filter(size,fwhm):
    sigma=fwhm/(2.*np.sqrt(2.*np.log(2.)))   
    gaussFilter=np.zeros(size)
    cenRow=size[0]/2.
    cenCol=size[1]/2.
    
    pixRowCens=np.arange(size[0])+0.5
    pixColCens=np.arange(size[1])+0.5
    
    y=cenRow-pixRowCens
    x=pixColCens-cenCol
    
    xv, yv = np.meshgrid(x, y, sparse=False)
    
    r=np.sqrt(xv**2.+yv**2.)
    gaussFilter=np.exp(-r**2./(2*sigma**2))
    gaussFilter=gaussFilter/gaussFilter.sum()
    
    return gaussFilter

def make_lorentzian_filter(size,fwhm):
    
    gamma=fwhm/2.  
    
    lorentzianFilter=np.zeros(size)
    cenRow=size[0]/2.
    cenCol=size[1]/2.
    
    pixRowCens=np.arange(size[0])+0.5
    pixColCens=np.arange(size[1])+0.5
    
    y=cenRow-pixRowCens
    x=pixColCens-cenCol
    
    xv, yv = np.meshgrid(x, y, sparse=False)
    
    r=np.sqrt(xv**2.+yv**2.)
    lorentzianFilter=gamma**2 / ((r)**2 + gamma**2)
    lorentzianFilter=lorentzianFilter/lorentzianFilter.sum()
    
    return lorentzianFilter

# multiprocessing of frames
def apply_filter_to_frame_mp(frame_cache, im_filter, chunk_size, cts_per_event=1e3, min_intensity=0.01, ncpus=1):   
    nframes = len(frame_cache)
    image_stack = Parallel(n_jobs=ncpus, verbose=2)(delayed(apply_filter_to_frame)
                                                    (idx, frame_cache[idx], im_filter, cts_per_event, min_intensity) for idx in range(nframes))
    
    return image_stack

def apply_filter_to_frame(i, frame, im_filter, cts_per_event, min_intensity):    
    this_frame_convolved = fftconvolve(frame*cts_per_event, im_filter, mode='same')
    this_frame_convolved[this_frame_convolved<min_intensity]=0.
    
    debug = False
    if debug:
        print('Frame %i Count Max: %f' %(i, np.max(this_frame_convolved)))
        sys.stdout.flush()
    return sp.sparse.csr_matrix(this_frame_convolved, dtype='uint16')

def apply_ps_filter_to_frame_cache(cfg_file, in_frame_cache_ims, output_dir,
                                   det_psf_fwhm=2.0, cts_per_event=1e4, gauss_or_lorentz=False, 
                                   min_intensity=0.01, max_intensity=65000, ncpus=1,
                                   save_frame_cache_ims=True, chunk_size=None):
    
    #% Process user input
    ###############################################################################
    
    # load config file
    cfg = config.open(cfg_file)[0] # NOTE: always a list of cfg objects
    
    # instrument info
    instr = cfg.instrument.hedm
    det_keys = list(instr.detectors.keys())
    if len(det_keys) > 1:
        print("Error: Only built for one detector VD simulations. Multi-detector coming soon!")
        return
    pixel_pitch = instr.detectors[det_keys[0]].pixel_size_row
    
    # output frame cache location
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # take care of filter string
    if gauss_or_lorentz:
        filter_str = 'gauss_cache'
    else:
        filter_str = 'lorentz_cache'
    
    # output frame cache location
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    frame_cache_output_name = '%s_%s_%s' %(cfg.analysis_name, det_keys[0].lower(), filter_str)
    frame_cache_output_full_dir = os.path.join(output_dir, frame_cache_output_name)
    
    nframes = len(in_frame_cache_ims)
    
    if chunk_size == None:
        chunk_size = int(np.ceil(nframes / ncpus))
    
    #% Do filter point spread and initial frame cache assembly
    ###############################################################################
    
    # process filter
    filter_size = np.round(det_psf_fwhm * (1.5 / float(pixel_pitch)))
    if filter_size % 2 == 0:
        filter_size+=1
    if gauss_or_lorentz:
        psf_filter=make_gaussian_filter([int(filter_size),int(filter_size)],det_psf_fwhm)
    else:
        psf_filter=make_lorentzian_filter([int(filter_size),int(filter_size)],det_psf_fwhm)
    
    # make pad four fast fourier tranform
    #filterPad=np.zeros(frame_cache_ims.shape, dtype=float)
    #filterPad[:psf_filter.shape[0],:psf_filter.shape[1]]=psf_filter
    #filterPadTransform=np.fft.fft2(filterPad)
    
    # Build images and apply point spread    
    if ncpus > 1:
        out_frame_cache_list = apply_filter_to_frame_mp(in_frame_cache_ims, psf_filter, 
                                 cts_per_event=cts_per_event, 
                                 min_intensity=min_intensity, 
                                 ncpus=ncpus, chunk_size=chunk_size)
    else:
        out_frame_cache_list = []
        for i in np.arange(nframes):
            print("processing frame %d of %d" % (i+1, nframes))  
            frame = apply_filter_to_frame(i, in_frame_cache_ims[i], psf_filter, 
                                 cts_per_event=cts_per_event, 
                                 min_intensity=min_intensity)
            out_frame_cache_list.append(frame)
    
    #% Save Frame Cache Data in HEXRD Format
    ###############################################################################
    
    out_frame_cache_ims = imageseries.open(frame_cache_output_name, 'frame-cache', 
                                           data=out_frame_cache_list, style='csr_mat_list', 
                                           meta=in_frame_cache_ims.metadata)
    
    if save_frame_cache_ims:
        ims_writer = imageseries.save.WriteFrameCache(out_frame_cache_ims,
                                                      frame_cache_output_name + '.npz', 
                                                      style='npz', threshold=min_intensity, 
                                                      cache_file=frame_cache_output_full_dir + '.npz')
        ims_writer._write_frames()
    
    return out_frame_cache_ims

#%% Testing
###############################################################################
if __name__ == '__main__':
    # path variables
    base_dir = '/home/djs522/additional_sw/hedmTools/CHESS_hedmTools/Modeling/VirtualDiffractometer/'
    cfg_file = os.path.join(base_dir, 'VirtDiffConfig.yml')
    output_dir = os.path.join(base_dir, 'vd_output/')
    
    # virt diff variables
    ome_period = [-180, 180] # since variable is deprecated in hexrd.config, define here
    fwhm = 1.2
    cts_per_event = 100.0
    min_inten = 1.0
    max_inten = 65000
    ncpus = 2
    
    vox_size = 0.015
    x = np.arange(-0.2, 0.2, step=vox_size)
    y = np.arange(-0.2, 0.2, step=vox_size)
    z = np.arange(-0.2, 0.2, step=vox_size)
    
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    pos = np.vstack([xv.flatten(), yv.flatten(), zv.flatten()]).T
    
    num_vox = xv.size
    split_idx = int(num_vox / 2)
    print("Number of voxels in example microstructure: %i" %(num_vox))
    grain_mat = np.zeros([num_vox, 21])
    grain_mat[:split_idx, 0] = 0 
    grain_mat[:split_idx, 3:6] = np.array([4.2537e-01,   2.0453e-01 ,  5.8538e-01])
    grain_mat[:split_idx, 6:9] = pos[:split_idx, :]
    
    grain_mat[split_idx:, 0] = 1
    grain_mat[split_idx:, 3:6] = np.array([3.3462e-01,   1.2188e-01 ,  6.0102e-01])
    grain_mat[split_idx:, 6:9] = pos[split_idx:, :]
    
    grain_mat[:, 9:12] = 1
    
    
    [pixel_intercepts, intercept_frame_cache] = calc_diffraction_detector_intercepts_and_frame(grain_mat, cfg_file, 
                                                   output_dir, ome_period=[-180.0, 180.0], 
                                                   ncpus=ncpus, save_pixel_intercepts=False, 
                                                   chunk_size=None)
    
    filter_frame_cache = apply_ps_filter_to_frame_cache(cfg_file, intercept_frame_cache, output_dir,
                                   det_psf_fwhm=fwhm, cts_per_event=cts_per_event, gauss_or_lorentz=False, 
                                   min_intensity=0.01, max_intensity=65000, ncpus=ncpus,
                                   save_frame_cache_ims=False)

    fig = plt.figure()
    plt.imshow(intercept_frame_cache[509])
    
    fig = plt.figure()
    plt.imshow(filter_frame_cache[509])
    
    plt.show()