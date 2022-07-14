import os
import glob
import argparse

try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml

import numpy as np

from multiprocessing import Pool
from functools import partial

from hexrd import instrument
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd import constants as ct
from hexrd.transforms.xfcapi import mapAngle
from hexrd.valunits import valWUnit
from hexrd import config
from hexrd import material


# plane data
def load_pdata(cpkl, key):
    mat = material.load_materials_hdf5(cpkl)
    return mat[key].planeData

# images
def load_images(yml):
    return imageseries.open(yml, format="frame-cache", style="npz")

# instrument
def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)

class GenerateEtaOmeMaps(object):
    """
    eta-ome map class derived from new image_series and YAML config

    ...for now...

    must provide:

    self.dataStore
    self.planeData
    self.iHKLList
    self.etaEdges # IN RADIANS
    self.omeEdges # IN RADIANS
    self.etas     # IN RADIANS
    self.omegas   # IN RADIANS

    """

    def __init__(self, grain_params, image_series_dict, instrument, plane_data,
                 active_hkls=None, eta_step=0.25, threshold=None,
                 ome_period=(0, 360), 
                 eta_ome_npz_save_dir='', analysis_id='default_analysis'):
        """
        image_series must be OmegaImageSeries class
        instrument_params must be a dict (loaded from yaml spec)
        active_hkls must be a list (required for now)
        """
        grain_id = grain_params[0]
        print('Starting Grain %i' %(grain_id))
        class_analysis_id = analysis_id + '-grain-%d' % grain_id
        
        self._planeData = plane_data

        if active_hkls is None:
            n_rings = len(plane_data.getTTh())
            self._iHKLList = range(n_rings)
        else:
            self._iHKLList = active_hkls
            n_rings = len(active_hkls)

        # ???: need to pass a threshold?
        eta_mapping, etas = instrument.extract_polar_maps(
            plane_data, image_series_dict, 
            active_hkls=active_hkls, threshold=threshold,
            tth_tol=None, eta_tol=eta_step, grain_params=grain_params)
        
        # grab a det key
        # WARNING: this process assumes that the imageseries for all panels
        # have the same length and omegas
        # det_key = eta_mapping.keys()[0]
        det_key = list(eta_mapping.keys())[0]
        data_store = []
        for i_ring in range(n_rings):
            full_map = np.zeros_like(eta_mapping[det_key][i_ring])
            nan_mask_full = np.zeros(
                (len(eta_mapping), full_map.shape[0], full_map.shape[1])
            )
            i_p = 0
            
            for det_key, eta_map in eta_mapping.items():
                nan_mask = ~np.isnan(eta_map[i_ring])
                nan_mask_full[i_p] = nan_mask
                full_map[nan_mask] += eta_map[i_ring][nan_mask]
                i_p += 1
            re_nan_these = np.sum(nan_mask_full, axis=0) == 0
            full_map[re_nan_these] = np.nan
            
            data_store.append(full_map)
        self._dataStore = data_store

        # handle omegas
        omegas_array = image_series_dict[det_key].metadata['omega']
        self._omegas = mapAngle(
            np.radians(np.average(omegas_array, axis=1)),
            np.radians(ome_period)
        )
        self._omeEdges = mapAngle(
            np.radians(np.r_[omegas_array[:, 0], omegas_array[-1, 1]]),
            np.radians(ome_period)
        )
        #self._omeEdges = np.sort(self._omeEdges)

        # !!! must avoid the case where omeEdges[0] = omeEdges[-1] for the
        # indexer to work properly
        if abs(self._omeEdges[0] - self._omeEdges[-1]) <= ct.sqrt_epsf:
            # !!! SIGNED delta ome
            del_ome = np.radians(omegas_array[0, 1] - omegas_array[0, 0])
            self._omeEdges[-1] = self._omeEdges[-2] + del_ome

        # handle etas
        # WARNING: unlinke the omegas in imageseries metadata,
        # these are in RADIANS and represent bin centers
        self._etas = etas
        self._etaEdges = np.r_[
            etas - 0.5*np.radians(eta_step),
            etas[-1] + 0.5*np.radians(eta_step)]
        #self._etaEdges = np.sort(self._etaEdges)
        

        self.save(eta_ome_npz_save_dir + class_analysis_id + "_maps.npz")

    @property
    def dataStore(self):
        return self._dataStore

    @property
    def planeData(self):
        return self._planeData

    @property
    def iHKLList(self):
        return np.atleast_1d(self._iHKLList).flatten()

    @property
    def etaEdges(self):
        return self._etaEdges

    @property
    def omeEdges(self):
        return self._omeEdges

    @property
    def etas(self):
        return self._etas

    @property
    def omegas(self):
        return self._omegas

    def save(self, filename):
        """
        self.dataStore
        self.planeData
        self.iHKLList
        self.etaEdges
        self.omeEdges
        self.etas
        self.omegas
        """
        #pd_params = list(self.planeData.getParams())
        #pd_params[0] = pd_params[0][0]
        #args = pd_params[:4]
        args = np.array(self.planeData.getParams())[:4]
        args[2] = valWUnit('wavelength', 'length', args[2], 'angstrom')
        hkls = self.planeData.hkls
        save_dict = {'dataStore': self.dataStore,
                     'etas': self.etas,
                     'etaEdges': self.etaEdges,
                     'iHKLList': self.iHKLList,
                     'omegas': self.omegas,
                     'omeEdges': self.omeEdges,
                     'planeData_args': args,
                     'planeData_hkls': hkls}
        np.savez_compressed(filename, **save_dict)
        return
    pass  # end of class: GenerateEtaOmeMaps


if __name__ == '__main__':

    # Run preprocessor
    parser = argparse.ArgumentParser(description="Generate Eta-Omega Maps for Grains")
    
    parser.add_argument('cfg_yml_file',
                       metavar='cfg',
                       type=str,
                       help='Path to configuration yaml file for experiment')    
    parser.add_argument('sample_name', metavar='samp_name',
                        help="Sample name", type=str)
    parser.add_argument('scan_num', metavar='scan_num',
                        help="Scan number for grains", type=int)
    parser.add_argument('grains_out', metavar='grains_out_dir',
                        help="Path to grains.out directory for scan", type=str)
    parser.add_argument('frame_cache_dir', metavar='frame_cache_dir',
                        help="Path to data directory of frame caches", type=str)
    parser.add_argument('output_dir', metavar='output_dir',
                        help="Path to output directory for eta-omega maps", type=str)
    parser.add_argument('--start_omega', metavar='start_omega', nargs='?', default=0.0,
                        help="Starting omega rotation angle (in degrees)", type=float)
    parser.add_argument('--end_omega', metavar='end_omega', nargs='?', default=360.0,
                        help="Ending omega rotation angle (in degrees)", type=float)
    parser.add_argument('--select_grain_ids', metavar='select_ids', nargs='?', default=None,
                        help="Path to .npy file with array of grain ids to construct", type=str)

    args = parser.parse_args()
    cfg_file = args.cfg_yml_file
    scan_num = args.scan_num
    samp_name = args.sample_name
    scan_grains_out_dir = args.grains_out
    data_dir = args.frame_cache_dir
    output_dir = args.output_dir
    starting_omega = args.start_omega
    ending_omega = args.end_omega
    select_grain_ids = args.select_grain_ids

# location and name of npz file output, make output directory if doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
dsgod_save_folder = os.path.join(output_dir, 'dsgod/')
if not os.path.exists(dsgod_save_folder):
    os.mkdir(dsgod_save_folder)
eta_ome_npz_save_dir = os.path.join(dsgod_save_folder,  'eta-ome_%s_%i/' %(samp_name, scan_num))
if not os.path.exists(eta_ome_npz_save_dir):
    os.mkdir(eta_ome_npz_save_dir)

# initialize grain mat from grains.out
grain_mat = np.loadtxt(scan_grains_out_dir)
if select_grain_ids is not None:
    good_ids = np.load(select_grain_ids).astype(int)
    grain_mat = np.atleast_2d(grain_mat[good_ids, :])

# intialize frame cache stem
fc_stem = "%s_%s_%%s*.npz" % (samp_name, scan_num)

# LOAD YML FILE
cfg = config.open(cfg_file)[0]

# number of processes
ncpus = cfg.multiprocessing

# get analysis name for outputting
analysis_id = cfg.analysis_id

# set omega period
omega_period = [starting_omega, ending_omega]

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
plane_data = load_pdata(cfg.material.definitions, cfg.material.active)
plane_data.tThMax = max_tth

# load instrument
instr = cfg.instrument.hedm
det_keys = instr.detectors.keys()

# threshold on frame cache building eta-omega maps
build_map_threshold = cfg.find_orientations.orientation_maps.threshold

# threshold on eta-ome maps for buidling dsgods
on_map_threshold = cfg.find_orientations.threshold

eta_tol = cfg.find_orientations.eta.tolerance

print("Loading image series")
# grab image series
imsd = dict.fromkeys(det_keys)
for det_key in det_keys:
    fc_file = sorted(
        glob.glob(
            os.path.join(
                data_dir,
                fc_stem % det_key.lower()
            )
        )
    )
    if len(fc_file) != 1:
        raise(RuntimeError, 'cache file not found, or multiple found')
    else:
        ims = load_images(fc_file[0])
        imsd[det_key] = OmegaImageSeries(ims)

# =============================================================================
# % SET UP AND PROCESSING OF ETA OME MAP - MULTI-PROCESSING
#      The code below generates all the eta-ome maps for all grains in the scan
# =============================================================================

grain_mat_ids = grain_mat[:, 0]
num_grains = grain_mat_ids.size
ncpus = num_grains if num_grains < ncpus else ncpus

print('building eta_ome maps for %i grains using %i processes...' %(num_grains, ncpus))

# initialize multiprocessing pool
pool = Pool(processes=ncpus)

# active hkls hardcoded to [0,1,2,3,4]
# NOTE: changed to be passed as optional parameter, defaults to the hardcoded value [0,1,2,3,4]
eta_ome_partial = partial(GenerateEtaOmeMaps, 
                          image_series_dict=imsd, 
                          instrument=instr, 
                          plane_data=plane_data, 
                          threshold=build_map_threshold, 
                          ome_period=omega_period, #cfg.find_orientations.omega.period is depricated
                          active_hkls=active_hkls, 
                          eta_step=eta_tol,
                          analysis_id=analysis_id, 
                          eta_ome_npz_save_dir=eta_ome_npz_save_dir)

eta_ome = pool.map(eta_ome_partial, list(grain_mat))
pool.close()
