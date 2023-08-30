#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 16:22:28 2023

@author: djs522
"""

# *****************************************************************************
# IMPORTS
# *****************************************************************************

import sys
import os
import argparse

import numpy as np

import timeit

import multiprocessing
from joblib import Parallel, delayed

from hexrd import instrument
from hexrd import indexer
from hexrd import config
from hexrd.xrdutil import EtaOmeMaps
from hexrd import rotations
from hexrd import constants
from hexrd.transforms import xfcapi

from hexrd.constants import USE_NUMBA
if USE_NUMBA:
    import numba

import logging
logger = logging.getLogger()
logger.setLevel('INFO')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel('INFO')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# *****************************************************************************
# PARAMTERS
# *****************************************************************************
omega_period_DFLT = np.radians(np.r_[-180., 180.])
paramMP = None
nCPUs_DFLT = multiprocessing.cpu_count()

# *****************************************************************************
# HELPER FUNCTIONS
# *****************************************************************************


# *****************************************************************************
# CLASS DEFINITIONS
# *****************************************************************************

# =============================================================================
# Methods
# =============================================================================
def process_grain(j, grain_mat, start_grain_ind, dsgod_npz_save_fname, num_grains_curr_iter, num_ori_per_grain,
                  tot_inten_list, tot_filter_list, truncate_comp_thresh, box_shape, misorientation_bnd, misorientation_spacing):
    # location and name of npz file output    
    curr_grain_id = grain_mat[start_grain_ind+j, 0]
    dsgod_npz_save_string = dsgod_npz_save_fname %(curr_grain_id)
    logger.info("Processing grain %i / %i -" %(j+1, num_grains_curr_iter))
    
    # set up orientations to search for building DSGOD clouds
    curr_s_ind = (j) * num_ori_per_grain
    curr_e_ind = (j+1) * num_ori_per_grain
    curr_exp_map = grain_mat[start_grain_ind+j, 3:6]
    #print(curr_s_ind, curr_e_ind, i, j, tot_inten_list.shape, curr_grain_id)
    curr_inten = tot_inten_list[curr_s_ind:curr_e_ind, :]
    curr_filter = tot_filter_list[curr_s_ind:curr_e_ind, :]
    
    # process truncating threshold
    if truncate_comp_thresh is not None:
        # reverse sort intensities in high -> low order
        sort_ind = np.argsort(-curr_inten, axis=1)
        sort_dsgod_box_inten_list = np.take_along_axis(curr_inten, sort_ind, axis=1)
        sort_dsgod_box_filter_list = np.take_along_axis(curr_filter, sort_ind, axis=1)
        
        # find index of intensity value to use based on completeness thresholding (Tim Long way)
        sum_filter = np.sum(sort_dsgod_box_filter_list, axis=1)
        comp_filter_ind = (truncate_comp_thresh * sum_filter).astype(int)
        
        # create new dsgod_box_inten and dsgod_box_filter
        min_size = sort_dsgod_box_inten_list.shape[1] - comp_filter_ind.min()
        max_size = sort_dsgod_box_inten_list.shape[1] - comp_filter_ind.max()
        end_comp_filter_ind = (sort_dsgod_box_inten_list.shape[1] - comp_filter_ind).astype(int)
        inten_list = np.zeros([sort_dsgod_box_inten_list.shape[0], min_size])
        filter_list = np.zeros([sort_dsgod_box_inten_list.shape[0], min_size])
        
        # be nice to vectorize this indexing and assignment somehow
        for j in range(inten_list.shape[0]):
            inten_list[j, :end_comp_filter_ind[j]] = sort_dsgod_box_inten_list[j, comp_filter_ind[j]:]
            filter_list[j, :end_comp_filter_ind[j]] = sort_dsgod_box_filter_list[j, comp_filter_ind[j]:]
        
    # re-type to save on space
    dsgod_box_inten_list = inten_list.astype(np.int32)
    dsgod_box_filter_list = filter_list.astype(bool)
    
    np.savez(dsgod_npz_save_string, 
             dsgod_box_shape=box_shape,
             dsgod_avg_expmap=curr_exp_map,
             dsgod_box_inten_list=dsgod_box_inten_list,
             dsgod_box_filter_list=dsgod_box_filter_list,
             misorientation_bnd=misorientation_bnd,
             misorientation_spacing=misorientation_spacing,
             truncate_comp_thresh=truncate_comp_thresh
             )

#%%
def paintGrid_dsgod(quats, etaOmeMaps,
              threshold=None, bMat=None,
              omegaRange=None, etaRange=None,
              omeTol=constants.d2r, etaTol=constants.d2r,
              omePeriod=omega_period_DFLT,
              doMultiProc=False,
              nCPUs=None, debug=False):
    """
    COPY OF hexrd.indexer.paintGrid function and helpter functions
    
    Spherical map-based indexing algorithm, i.e. paintGrid.

    Given a list of trial orientations `quats` and an eta-omega intensity map
    object `etaOmeMaps`, this method executes a test to produce a completeness
    ratio for each orientation across the spherical inensity maps.

    Parameters
    ----------
    quats : (4, N) ndarray
        hstacked array of trial orientations in the form of unit quaternions.
    etaOmeMaps : object
        an spherical map object of type `hexrd.instrument.GenerateEtaOmeMaps`.
    threshold : float, optional
        threshold value on the etaOmeMaps.
    bMat : (3, 3) ndarray, optional
        the COB matrix from the reciprocal lattice to the reference crystal
        frame.  In not provided, the B in the planeData class in the etaOmeMaps
        is used.
    omegaRange : array_like, optional
        list of valid omega ranges in radians,
        e.g. np.radians([(-60, 60), (120, 240)])
    etaRange : array_like, optional
        list of valid eta ranges in radians,
        e.g. np.radians([(-85, 85), (95, 265)])
    omeTol : float, optional
        the tolerance to use in the omega dimension in radians.  Default is
        1 degree (0.017453292519943295)
    etaTol : float, optional
        the tolerance to use in the eta dimension in radians.  Default is
        1 degree (0.017453292519943295)
    omePeriod : (2, ) array_like, optional
        the period to use for omega angles in radians,
        e.g. np.radians([-180, 180])
    doMultiProc : bool, optional
        flag for enabling multiprocessing
    nCPUs : int, optional
        number of processes to use in case doMultiProc = True
    debug : bool, optional
        debugging mode flag
    dsgod : bool, optional
        flag for constructing dsgod. Default is False

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    retval : (N, ) list
        completeness score list for `quats`.


    Notes
    -----
    Notes about the implementation algorithm (if needed).

    This can have multiple paragraphs.

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a Greek symbol like :math:`\omega` inline.

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print([x + 3 for x in a])
    [4, 5, 6]
    >>> print("a\nb")
    a
    b
    """
    quats = np.atleast_2d(quats)
    if quats.size == 4:
        quats = quats.reshape(4, 1)

    planeData = etaOmeMaps.planeData

    hklIDs = np.r_[etaOmeMaps.iHKLList]
    hklList = np.atleast_2d(planeData.hkls[:, hklIDs].T).tolist()
    nHKLS = len(hklIDs)

    numEtas = len(etaOmeMaps.etaEdges) - 1
    numOmes = len(etaOmeMaps.omeEdges) - 1
    
    if threshold == 'None' or threshold == 'none':
    	threshold = None
    if threshold is None:
        threshold = np.zeros(nHKLS)
        for i in range(nHKLS):
            threshold[i] = np.mean(
                np.r_[
                    np.mean(etaOmeMaps.dataStore[i]),
                    np.median(etaOmeMaps.dataStore[i])
                    ]
                )
    elif threshold is not None and not hasattr(threshold, '__len__'):
        threshold = threshold * np.ones(nHKLS)
    elif hasattr(threshold, '__len__'):
        if len(threshold) != nHKLS:
            raise RuntimeError("threshold list is wrong length!")
        else:
            print("INFO: using list of threshold values")
            threshold = np.array(threshold)
    else:
        raise RuntimeError(
            "unknown threshold option. should be a list of numbers or None"
            )
    if bMat is None:
        bMat = planeData.latVecOps['B']

    # ???
    # not positive why these are needed
    etaIndices = np.arange(numEtas)
    omeIndices = np.arange(numOmes)

    omeMin = None
    omeMax = None
    if omegaRange is None:  # FIXME
        omeMin = [np.min(etaOmeMaps.omeEdges), ]
        omeMax = [np.max(etaOmeMaps.omeEdges), ]
    else:
        omeMin = [omegaRange[i][0] for i in range(len(omegaRange))]
        omeMax = [omegaRange[i][1] for i in range(len(omegaRange))]
    if omeMin is None:
        omeMin = [-np.pi, ]
        omeMax = [np.pi, ]
    omeMin = np.asarray(omeMin)
    omeMax = np.asarray(omeMax)

    etaMin = None
    etaMax = None
    if etaRange is not None:
        etaMin = [etaRange[i][0] for i in range(len(etaRange))]
        etaMax = [etaRange[i][1] for i in range(len(etaRange))]
    if etaMin is None:
        etaMin = [-np.pi, ]
        etaMax = [np.pi, ]
    etaMin = np.asarray(etaMin)
    etaMax = np.asarray(etaMax)

    multiProcMode = nCPUs_DFLT > 1 and doMultiProc

    if multiProcMode:
        nCPUs = nCPUs or nCPUs_DFLT
        chunksize = max(quats.shape[1] // nCPUs, 10) # !!! CHECK THIS AND WHY 10
        logger.info(
            "using multiprocessing with %d processes and a chunk size of %d",
            nCPUs, chunksize
            )
    else:
        logger.info("running in serial mode")
        nCPUs = 1

    # Get the symHKLs for the selected hklIDs
    symHKLs = planeData.getSymHKLs()
    symHKLs = [symHKLs[id] for id in hklIDs]
    # Restructure symHKLs into a flat NumPy HKL array with
    # each HKL stored contiguously (C-order instead of F-order)
    # symHKLs_ix provides the start/end index for each subarray
    # of symHKLs.
    symHKLs_ix = np.add.accumulate([0] + [s.shape[1] for s in symHKLs])
    symHKLs = np.vstack([s.T for s in symHKLs])

    # Pack together the common parameters for processing
    params = {
        'symHKLs': symHKLs,
        'symHKLs_ix': symHKLs_ix,
        'wavelength': planeData.wavelength,
        'hklList': hklList,
        'omeMin': omeMin,
        'omeMax': omeMax,
        'omeTol': omeTol,
        'omeIndices': omeIndices,
        'omePeriod': omePeriod,
        'omeEdges': etaOmeMaps.omeEdges,
        'etaMin': etaMin,
        'etaMax': etaMax,
        'etaTol': etaTol,
        'etaIndices': etaIndices,
        'etaEdges': etaOmeMaps.etaEdges,
        'etaOmeMaps': np.stack(etaOmeMaps.dataStore),
        'bMat': bMat,
        'threshold': np.asarray(threshold)
        }

    # do the mapping
    start = timeit.default_timer()
    retval = None
    if multiProcMode:
        # multiple process version
        pool = multiprocessing.Pool(nCPUs, paintgrid_init_dsgod, (params, ))
        retval = pool.map(paintGridThis_dsgod, quats.T, chunksize=chunksize)
        pool.close()
    else:
        # single process version.
        global paramMP
        paintgrid_init_dsgod(params)    # sets paramMP
        retval = list(map(paintGridThis_dsgod, quats.T))
        paramMP = None    # clear paramMP
    elapsed = (timeit.default_timer() - start)
    logger.info("paintGrid took %.3f seconds", elapsed)

    return retval

def paintgrid_init_dsgod(params):
    """
    Initialize global variables for paintGrid.

    Parameters
    ----------
    params : dict
        multiprocessing parameter dictionary.

    Returns
    -------
    None.
    """
    global paramMP
    paramMP = params

    # create valid_eta_spans, valid_ome_spans from etaMin/Max and omeMin/Max
    # this allows using faster checks in the code.
    # TODO: build valid_eta_spans and valid_ome_spans directly in paintGrid
    #       instead of building etaMin/etaMax and omeMin/omeMax. It may also
    #       be worth handling range overlap and maybe "optimize" ranges if
    #       there happens to be contiguous spans.
    paramMP['valid_eta_spans'] = indexer._normalize_ranges(paramMP['etaMin'],
                                                   paramMP['etaMax'],
                                                   -np.pi)

    paramMP['valid_ome_spans'] = indexer._normalize_ranges(paramMP['omeMin'],
                                                   paramMP['omeMax'],
                                                   min(paramMP['omePeriod']))
    return


def _check_dilated_dsgod(eta, ome, dpix_eta, dpix_ome, etaOmeMap, threshold): 
    
    i_max, j_max = etaOmeMap.shape
    ome_start, ome_stop = (
        max(ome - dpix_ome, 0),
        min(ome + dpix_ome + 1, i_max)
    )
    eta_start, eta_stop = (
        max(eta - dpix_eta, 0),
        min(eta + dpix_eta + 1, j_max)
    )
    
    
    ome_range = range(ome_start, ome_stop)
    eta_range = range(eta_start, eta_stop)
    
    max_inten = 0
    max_eta_ind = -1
    max_ome_ind = -1
    
    dist_thresh = 1.5
    
    for i in ome_range:
        for j in eta_range:
            dist = abs(i-ome) + abs(j-eta)
            
            if dist <= dist_thresh:
                if etaOmeMap[i, j] > max_inten:
                    max_inten = etaOmeMap[i, j]
                    max_eta_ind = j
                    max_ome_ind = i
                if np.isnan(etaOmeMap[i, j]):
                    #print('nan')
                    return -1, -1, -1
    
    # return [inten, eta, ome]
        #   if inten > 0:  orientation is on map and is hit
        #   if inten = 0:  orientation is on map and is not hit
        #   if inten = -1: orientation is not on map and is not hit
    return max_inten, max_eta_ind, max_ome_ind

if USE_NUMBA:
    def paintGridThis_dsgod(quat):
        """Single instance paintGrid call.

        Note that this version does not use omeMin/omeMax to specify the valid
        angles. It uses "valid_eta_spans" and "valid_ome_spans". These are
        precomputed and make for a faster check of ranges than
        "validateAngleRanges"
        """
        symHKLs = paramMP['symHKLs']  # the HKLs
        symHKLs_ix = paramMP['symHKLs_ix']  # index partitioning of symHKLs
        bMat = paramMP['bMat']
        wavelength = paramMP['wavelength']
        omeEdges = paramMP['omeEdges']
        omeTol = paramMP['omeTol']
        omePeriod = paramMP['omePeriod']
        valid_eta_spans = paramMP['valid_eta_spans']
        valid_ome_spans = paramMP['valid_ome_spans']
        omeIndices = paramMP['omeIndices']
        etaEdges = paramMP['etaEdges']
        etaTol = paramMP['etaTol']
        etaIndices = paramMP['etaIndices']
        etaOmeMaps = paramMP['etaOmeMaps']
        threshold = paramMP['threshold']

        # dpix_ome and dpix_eta are the number of pixels for the tolerance in
        # ome/eta. Maybe we should compute this per run instead of per
        # quaternion
        del_ome = abs(omeEdges[1] - omeEdges[0])
        del_eta = abs(etaEdges[1] - etaEdges[0])
        dpix_ome = int(round(omeTol / del_ome))
        dpix_eta = int(round(etaTol / del_eta))

        # FIXME
        debug = False
        if debug:
            print(
                "using ome, eta dilitations of (%d, %d) pixels"
                % (dpix_ome, dpix_eta)
            )

        # get the equivalent rotation of the quaternion in matrix form (as
        # expected by oscillAnglesOfHKLs

        rMat = xfcapi.makeRotMatOfQuat(quat)

        # Compute the oscillation angles of all the symHKLs at once
        oangs_pair = xfcapi.oscillAnglesOfHKLs(symHKLs, 0., rMat, bMat,
                                               wavelength)
        
        return _filter_and_count_hits_dsgod(oangs_pair[0], oangs_pair[1], symHKLs_ix,
                                      etaEdges, valid_eta_spans,
                                      valid_ome_spans, omeEdges, omePeriod,
                                      etaOmeMaps, etaIndices, omeIndices,
                                      dpix_eta, dpix_ome, threshold)
    
    @numba.njit(nogil=True, cache=True)
    def _angle_is_hit_dsgod(ang, eta_offset, ome_offset, hkl, valid_eta_spans,
                      valid_ome_spans, etaEdges, omeEdges, etaOmeMaps,
                      etaIndices, omeIndices, dpix_eta, dpix_ome, threshold):
        
        tth, eta, ome = ang
        
        if np.isnan(tth):
            #print('nan_tth')
            return -1, -1, -1

        eta = indexer._map_angle(eta, eta_offset)
        if indexer._find_in_range(eta, valid_eta_spans) & 1 == 0:
            # index is even: out of valid eta spans
            #print('nan_eta')
            return -1, -1, -1

        ome = indexer._map_angle(ome, ome_offset)
        if indexer._find_in_range(ome, valid_ome_spans) & 1 == 0:
            # index is even: out of valid ome spans
            #print('nan_ome')
            return -1, -1, -1

        # discretize the angles
        eta_idx = indexer._find_in_range(eta, etaEdges) - 1
        if eta_idx < 0:
            # out of range
            #print('idx_eta')
            return -1, -1, -1

        ome_idx = indexer._find_in_range(ome, omeEdges) - 1
        if ome_idx < 0:
            # out of range
            #print('idx_ome',omeEdges[0]*180/np.pi,omeEdges[-1]*180/np.pi, ome*180/np.pi)
            return -1, -1, -1
        
        
        
        # pixel indices for eta, omega
        pix_ind_eta = etaIndices[eta_idx]
        pix_ind_ome = omeIndices[ome_idx]
        eta_ome_inten, pix_ind_eta, pix_ind_ome = _check_dilated_dsgod(pix_ind_eta, pix_ind_ome, dpix_eta, dpix_ome, etaOmeMaps[hkl], threshold[hkl])
        
        
        # return [inten, eta, ome]
        #   if inten > 0:  orientation is on map and is hit
        #   if inten = 0:  orientation is on map and is not hit
        #   if inten = -1: orientation is not on map and is not hit
        return eta_ome_inten, pix_ind_eta, pix_ind_ome
    
    @numba.njit(nogil=True, cache=True)
    def _filter_and_count_hits_dsgod(angs_0, angs_1, symHKLs_ix, etaEdges,
                               valid_eta_spans, valid_ome_spans, omeEdges,
                               omePeriod, etaOmeMaps, etaIndices, omeIndices,
                               dpix_eta, dpix_ome, threshold):
        """assumes:
        we want etas in -pi -> pi range
        we want omes in ome_offset -> ome_offset + 2*pi range

        Instead of creating an array with the angles of angs_0 and angs_1
        interleaved, in this numba version calls for both arrays are performed
        getting the angles from angs_0 and angs_1. this is done in this way to
        reuse hkl computation. This may not be that important, though.

        """
        eta_offset = -np.pi
        ome_offset = np.min(omePeriod)
        hits = 0
        total = 0
        curr_hkl_idx = 0
        end_curr = symHKLs_ix[1]
        count = len(angs_0)
        
        # the total summed intensity of one orientation at each of its measured 
        # diffraction events
        inten_list = []
        hit_list = []
        filter_list = []
        eta_ind_list = []
        ome_ind_list = []

        for i in range(count):
            if i >= end_curr:
                curr_hkl_idx += 1
                end_curr = symHKLs_ix[curr_hkl_idx+1]

            # first solution            
            f_inten, f_eta_ind, f_ome_ind = _angle_is_hit_dsgod(
                      angs_0[i], eta_offset, ome_offset, curr_hkl_idx,
                      valid_eta_spans, valid_ome_spans, etaEdges, omeEdges, 
                      etaOmeMaps, etaIndices, omeIndices, dpix_eta, dpix_ome, threshold)
            
            hit = 0
            not_filter = 0
            if f_inten > 0:
                hit = 1
            if f_inten > -1:
                not_filter = 1
            
            hits += hit
            total += not_filter
            
            inten_list.append(f_inten)
            hit_list.append(hit)
            filter_list.append(not_filter)
            eta_ind_list.append(f_eta_ind)
            ome_ind_list.append(f_ome_ind)

            # second solution
            f_inten, f_eta_ind, f_ome_ind = _angle_is_hit_dsgod(
                      angs_1[i], eta_offset, ome_offset, curr_hkl_idx,
                      valid_eta_spans, valid_ome_spans, etaEdges, omeEdges, 
                      etaOmeMaps, etaIndices, omeIndices, dpix_eta, dpix_ome, threshold)
            
            hit = 0
            not_filter = 0
            if f_inten > 0:
                hit = 1
            if f_inten > -1:
                not_filter = 1
            
            hits += hit
            total += not_filter
            
            inten_list.append(f_inten)
            hit_list.append(hit)
            filter_list.append(not_filter)
            eta_ind_list.append(f_eta_ind)
            ome_ind_list.append(f_ome_ind)
        
        if total != 0:
            comp = float(hits)/float(total)
        else:
            comp = 0
        return (comp, inten_list, hit_list, filter_list, eta_ind_list, ome_ind_list)

    # use a jitted version of _check_dilated
    _check_dilated_dsgod = numba.njit(nogil=True, cache=True)(_check_dilated_dsgod)
else:
    raise  ImportError("Only Numba implementations supported.")


# *****************************************************************************
# MAIN CALL
# *****************************************************************************
if __name__ == '__main__':

    # Run preprocessor
    parser = argparse.ArgumentParser(description="Generate Eta-Omega Map for DSGODs")
    
    parser.add_argument('cfg_yml_file',
                       metavar='cfg',
                       type=str,
                       help='Path to configuration yaml file for experiment')    
    parser.add_argument('--output_dir', metavar='output_dir', nargs='?', default=None,
                        help="Path to output directory for eta-omega maps", type=str)
    parser.add_argument('--mis_bound', metavar='mis_bound', nargs='?', default=3.0,
                        help="Bound (in degrees) for the maximum amount of misorientation", type=float)
    parser.add_argument('--mis_spacing', metavar='mis_spacing', nargs='?', default=0.25,
                        help="Spacing steps (in degrees) for grid of misorientation", type=float)
    parser.add_argument('--start_omega', metavar='start_omega', nargs='?', default=None,
                        help="Starting omega rotation angle (in degrees)", type=float)
    parser.add_argument('--end_omega', metavar='end_omega', nargs='?', default=None,
                        help="Ending omega rotation angle (in degrees)", type=float)
    parser.add_argument('--select_grain_ids', metavar='select_ids', nargs='?', default=None,
                        help="Path to .npy or .txt file with array of grain ids to construct", type=str)
    parser.add_argument('--truncate_comp_thresh', metavar='truncate_comp_thresh', nargs='?', default=None,
                        help="Completeness threshold for truncating saved DSGOD data", type=float)
    parser.add_argument('--paint_grid_chunk_size', metavar='paint_grid_chunk_size', nargs='?', default=int(2e6),
                        help="Chunk size for the amount of orientations ingested into paint grid", type=float)
    parser.add_argument('--rebuild_eta_ome_maps', metavar='rebuild_eta_ome_maps', nargs='?', default=True,
                        help="Flag for rebuilding eta omega maps", type=bool)

    args = parser.parse_args()
    cfg_file = args.cfg_yml_file
    output_dir = args.output_dir
    misorientation_bnd = args.mis_bound
    misorientation_spacing = args.mis_spacing
    starting_omega = args.start_omega
    ending_omega = args.end_omega
    select_grain_ids = args.select_grain_ids
    truncate_comp_thresh = args.truncate_comp_thresh
    max_ori_per_paint_grid_iter = args.paint_grid_chunk_size
    rebuild_eta_ome_maps = args.rebuild_eta_ome_maps

    # LOAD YML FILE
    logger.info("Loading config file and preprocessing...")
    cfg = config.open(cfg_file)[0]
    
    # location and name of npz file output, make output directory if doesn't exist
    if output_dir is None:
        output_dir = cfg.working_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    main_dsgod_save_dir = os.path.join(output_dir, 'dsgod/')
    if not os.path.exists(main_dsgod_save_dir):
        os.mkdir(main_dsgod_save_dir)
        
    eta_ome_npz_save_dir = os.path.join(main_dsgod_save_dir,  'eta_ome_maps/')
    eta_ome_npz_save_fname = os.path.join(eta_ome_npz_save_dir, '%s_eta_ome_maps.npz' %(cfg.analysis_id))
    if not os.path.exists(eta_ome_npz_save_dir):
        os.mkdir(eta_ome_npz_save_dir)  
    
    dsgod_npz_save_dir = os.path.join(main_dsgod_save_dir,  'dsgod_data_%s/' %(cfg.analysis_id))
    dsgod_npz_save_fname = os.path.join(dsgod_npz_save_dir, 'grain_%d_dsgod_data.npz')
    if not os.path.exists(dsgod_npz_save_dir):
        os.mkdir(dsgod_npz_save_dir)
    
    # initialize grain mat from grains.out
    grain_mat = np.loadtxt(cfg.fit_grains.estimate)
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
            max_tth = np.degrees(float(max_tth))
    else:
        max_tth = None
    
    # load plane data
    plane_data = cfg.material.plane_data
    plane_data.tThMax = max_tth
    
    # load instrument
    instr = cfg.instrument.hedm
    det_keys = instr.detectors.keys()
    
    # threshold on frame cache building eta-omega maps
    build_map_threshold = cfg.find_orientations.orientation_maps.threshold
    
    # threshold on eta-ome maps for buidling dsgods
    on_map_threshold = cfg.find_orientations.threshold
    
    # eta tolerance for eta-ome maps
    eta_tol = cfg.find_orientations.eta.tolerance
    
    # grab image series
    if cfg.__dict__['_cfg']['image_series']['format'] == 'frame-cache':
        ims_dict = cfg.image_series
    else:
        raise ValueError("Only frame-caches supported! (%s not supported)" %(cfg.__dict__['_cfg']['image_series']['format']))
    
    # set omega period
    omegas = ims_dict[next(iter(ims_dict))].omega
    if starting_omega is None:
        starting_omega = omegas[0, 0]
    if ending_omega is None:
        ending_omega = omegas[-1, 1]
    omega_period = [starting_omega, ending_omega]
    
    # set up multiprocessing details
    grain_mat_ids = grain_mat[:, 0]
    num_grains = grain_mat_ids.size
    ncpus = num_grains if num_grains < cfg.multiprocessing else cfg.multiprocessing
    
    # start building DSGODs
    logger.info('Number of grains %i...' %(num_grains))
    
    # check if eta omega map exists, if not generate it, if so load it
    if not os.path.isfile(eta_ome_npz_save_fname) or rebuild_eta_ome_maps:
        # generate eta omega map for file
        logger.info('Building eta_ome maps...')
        
        eta_ome = instrument.GenerateEtaOmeMaps(image_series_dict=cfg.image_series, 
        instrument=instr, 
        plane_data=plane_data, 
        threshold=build_map_threshold, 
        ome_period=omega_period, #cfg.find_orientations.omega.period is depricated
        active_hkls=active_hkls, 
        eta_step=eta_tol)
        
        # TODO: put code here for scaling eta ome map by structure factor in plane data
        
        eta_ome.save(filename=eta_ome_npz_save_fname)
        
    else:
        # load eta-ome maps for this scan for all grains
        eta_ome = EtaOmeMaps(eta_ome_npz_save_fname)
        
    # start building DSGODs
    logger.info('Building DSGODs from eta_ome maps...')
    
    all_mis_quats = []
    num_ori_per_grain = 1
    for igrain, cur_grain_id in enumerate(grain_mat_ids):
        
        # TODO: Fix this active hkl indexing
        #eta_ome_hkl, eta_ome_hkl_ind, no_need = np.intersect1d(eta_ome.iHKLList, eta_ome_active_hkls, return_indices=True)
        #eta_ome.iHKLList = eta_ome.iHKLList[eta_ome_hkl_ind]
        #eta_ome.dataStore = eta_ome.dataStore[eta_ome_hkl_ind, :, :]
        
        # set up orientations to search for building DSGOD clouds
        cur_exp_maps = grain_mat[igrain, 3:6]
            
        mis_amt = np.radians(misorientation_bnd)
        mis_spacing = np.radians(misorientation_spacing)
        mis_ori_pts = np.arange(-mis_amt, (mis_amt+(mis_spacing*0.999)), mis_spacing)
        mis_ori_Xs, mis_ori_Ys, mis_ori_Zs = np.meshgrid(mis_ori_pts, mis_ori_pts, mis_ori_pts)
        ori_grid = np.vstack([mis_ori_Xs.flatten(), mis_ori_Ys.flatten(), mis_ori_Zs.flatten()]).T
        box_shape = mis_ori_Xs.shape
        
        grain_mis_exp_maps = ori_grid + cur_exp_maps
        num_oris = grain_mis_exp_maps.shape[0]
        num_ori_per_grain = max(num_ori_per_grain, num_oris)
        grain_mis_quats = rotations.quatOfExpMap(grain_mis_exp_maps.T)
        
        all_mis_quats.append(grain_mis_quats)
    
    all_mis_quats = np.hstack(all_mis_quats)
    logger.info("will test %d quaternions using %d processes" % (all_mis_quats.shape[1], ncpus))
    
    total_num_oris = all_mis_quats.shape[1]
    num_paint_grid_iter = int(np.ceil(total_num_oris / max_ori_per_paint_grid_iter))
    num_grains_per_iter = int(np.ceil(num_grains / num_paint_grid_iter))
    
    grain_counter = 0
    for i in range(num_paint_grid_iter):
        start_ind = i * num_grains_per_iter * num_ori_per_grain
        end_ind = min((i+1) * num_grains_per_iter * num_ori_per_grain, total_num_oris)
        tot_inten_list = []
        tot_filter_list = []
    
        # =============================================================================
        # % ORIENTATION SCORING
        # =============================================================================
        logger.info("using map search with paintGrid iteration %i / %i on %d processes - %s" %(i+1, num_paint_grid_iter, ncpus, cfg.analysis_id))
        start = timeit.default_timer()
        
        # return format: (comp, inten_list, hit_list, filter_list, eta_ind_list, ome_ind_list)
        retval = paintGrid_dsgod(
            all_mis_quats[:, start_ind:end_ind],
            eta_ome,
            etaRange=np.radians(cfg.find_orientations.eta.range),
            omeTol=np.radians(cfg.find_orientations.omega.tolerance),
            etaTol=np.radians(cfg.find_orientations.eta.tolerance),
            omePeriod=np.radians(omega_period), #cfg.find_orientations.omega.period) is depricated
            threshold=on_map_threshold,
            doMultiProc=ncpus > 1,
            nCPUs=ncpus
           )
        
        # process return value
        for j in range(len(retval)):
            tot_inten_list.append(retval[j][1])
            tot_filter_list.append(retval[j][3])
        tot_inten_list = np.array(tot_inten_list)
        tot_filter_list = np.array(tot_filter_list)
        
        del retval
        
        # process paaint grid results for each grain
        num_grains_curr_iter = int(int(end_ind - start_ind) / num_ori_per_grain)
        start_grain_ind = i * num_grains_per_iter
        end_grain_ind = start_grain_ind + num_grains_curr_iter
        
        # process_grain(j, grain_mat, start_grain_ind, dsgod_npz_save_fname, num_grains_curr_iter, num_ori_per_grain,
        #                   tot_inten_list, tot_filter_list, truncate_comp_thresh, box_shape, misorientation_bnd, misorientation_spacing)
        
        Parallel(n_jobs=48)(delayed(process_grain)(j, grain_mat, start_grain_ind, dsgod_npz_save_fname, num_grains_curr_iter, num_ori_per_grain,
                          tot_inten_list, tot_filter_list, truncate_comp_thresh, box_shape, misorientation_bnd, misorientation_spacing) for j in range(num_grains_curr_iter))
        
        logger.info("search map paint grid dsgod took %f seconds..." % (timeit.default_timer() - start))
    
