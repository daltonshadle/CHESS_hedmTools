import os
import argparse
import timeit

try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml

import numpy as np

from hexrd import instrument
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd.xrdutil import EtaOmeMaps
from hexrd import config
from hexrd import rotations
from hexrd import indexer
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


if __name__ == '__main__':

    # Run preprocessor
    parser = argparse.ArgumentParser(description="Construct DSGODs for Grains")
    
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
    parser.add_argument('output_dir', metavar='output_dir',
                        help="Path to output directory for eta-omega maps", type=str)
    parser.add_argument('--mis_bound', metavar='mis_bound', nargs='?', default=4.0,
                        help="Bound (in degrees) for the maximum amount of misorientation", type=float)
    parser.add_argument('--mis_spacing', metavar='mis_spacing', nargs='?', default=0.25,
                        help="Spacing steps (in degrees) for grid of misorientation", type=float)
    parser.add_argument('--start_omega', metavar='start_omega', nargs='?', default=0.0,
                        help="Starting omega rotation angle (in degrees)", type=float)
    parser.add_argument('--end_omega', metavar='end_omega', nargs='?', default=360.0,
                        help="Ending omega rotation angle (in degrees)", type=float)
    parser.add_argument('--select_grain_ids', metavar='select_ids', nargs='?', default=None,
                        help="Path to .npy file with array of grain ids to construct", type=str)
    parser.add_argument('--eta_omega_map_path', metavar='eta_omega_map_path', nargs='?', default=None,
                        help="Path to eta_omega_map file for constructing DSGOD. Use GEN to generate eta_omega_map.", type=str)

    args = parser.parse_args()
    cfg_file = args.cfg_yml_file
    scan_num = args.scan_num
    samp_name = args.sample_name
    scan_grains_out_dir = args.grains_out
    output_dir = args.output_dir
    misorientation_bnd = args.mis_bound
    misorientation_spacing = args.mis_spacing
    starting_omega = args.start_omega
    ending_omega = args.end_omega
    select_grain_ids = args.select_grain_ids
    eta_omega_map_path = args.eta_omega_map_path

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
    if ".txt" in select_grain_ids:
        good_ids  = np.loadtxt(select_grain_ids).astype(int)
    else:
        good_ids = np.load(select_grain_ids).astype(int)
    grain_mat = grain_mat[good_ids, :]

# LOAD YML FILE
cfg = config.open(cfg_file)[0]

# number of processes
ncpus = cfg.multiprocessing

# get analysis name for outputting
analysis_id = cfg.analysis_id

# get active hkls for generating eta-omega maps
active_hkls = cfg.find_orientations.orientation_maps.active_hkls
eta_ome_active_hkls = active_hkls
if active_hkls == 'all':
    active_hkls = None

max_tth = cfg.fit_grains.tth_max
if max_tth:
    if type(cfg.fit_grains.tth_max) != bool:
        max_tth = np.degrees(float(max_tth))
else:
    max_tth = None

# set omega period
omega_period = [starting_omega, ending_omega]

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
if (on_map_threshold == 'none') or (on_map_threshold == 'None'):
    on_map_threshold = None

# =============================================================================
# % STARTING DSGOD BUILDS
# =============================================================================

grain_mat_ids = grain_mat[:, 0]
grain_mat_ids_list = list(np.array(grain_mat_ids).astype('int').T)
num_grains = grain_mat_ids.size

if eta_omega_map_path in "GEN":
    # generate eta omega map for file
    print("Generating Eta-Omega Map for Scan")
    eta_tol = cfg.find_orientations.eta.tolerance
    
    eta_ome = instrument.GenerateEtaOmeMaps(image_series_dict=cfg.image_series, 
    instrument=instr, 
    plane_data=plane_data, 
    threshold=build_map_threshold, 
    ome_period=omega_period, #cfg.find_orientations.omega.period is depricated
    active_hkls=active_hkls, 
    eta_step=eta_tol)
    
    eta_ome.save(filename=eta_ome_npz_save_dir)
    
elif eta_omega_map_path is not None:
    eta_ome_maps = eta_omega_map_path
    # load eta-ome maps for this scan for all grains
    if os.path.exists(eta_ome_maps):
        eta_ome = EtaOmeMaps(eta_ome_maps)
    else:
        raise ValueError('ETA-OME MAP NOT FOUND!')
    

print('building DSGODs from eta_ome maps...')
for igrain, grain in enumerate(grain_mat_ids_list):
    print("Processing grain %i / %i" %(igrain, num_grains))
    # location and name of npz file output    
    dsgod_npz_save_string = 'grain_%d' % grain + '_dsgod_data.npz'
    dsgod_npz_save_dir = dsgod_save_folder + 'dsgod_%s_%i/' %(samp_name, scan_num) #for npz file
    
    # make output directory if doesn't exist
    if not os.path.exists(dsgod_npz_save_dir):
        os.mkdir(dsgod_npz_save_dir)
    
    # location eta-ome maps and name of analysis_id
    if eta_omega_map_path is None:
        eta_ome_maps = eta_ome_npz_save_dir + analysis_id + '-grain-%d' % grain
        eta_ome_maps = eta_ome_maps + "_maps.npz"
        
        # load eta-ome maps for this grain and scan
        if os.path.exists(eta_ome_maps):
            eta_ome = EtaOmeMaps(eta_ome_maps)
        else:
            raise ValueError('ETA-OME MAP NOT FOUND FOR GRAIN %i!' %(grain))
    
    # TODO: Fix this active hkl indexing
    #eta_ome_hkl, eta_ome_hkl_ind, no_need = np.intersect1d(eta_ome.iHKLList, eta_ome_active_hkls, return_indices=True)
    #eta_ome.iHKLList = eta_ome.iHKLList[eta_ome_hkl_ind]
    #eta_ome.dataStore = eta_ome.dataStore[eta_ome_hkl_ind, :, :]
    
    # set up orientations to search for building DSGOD clouds
    cur_exp_maps = np.zeros([1,3])
    cur_grain_id = np.zeros([1,1])    
    cur_exp_maps[0,:] = grain_mat[igrain, 3:6]
    cur_grain_id[0,:] = grain_mat[igrain, 0]
        
    mis_amt=misorientation_bnd*np.pi/180
    spacing=misorientation_spacing*np.pi/180
    
    ori_pts = np.arange(-mis_amt, (mis_amt+(spacing*0.999)), spacing)
    num_ori_grid_pts=ori_pts.shape[0]**3
    num_oris = cur_exp_maps.shape[0]
    
    Xs0, Ys0, Zs0 = np.meshgrid(ori_pts, ori_pts, ori_pts)
    grid0 = np.vstack([Xs0.flatten(), Ys0.flatten(), Zs0.flatten()]).T
    
    exp_maps_expanded=np.zeros([num_ori_grid_pts*num_oris,3])
    
    for ii in np.arange(num_oris):
        pts_to_use=np.arange(num_ori_grid_pts) + ii*num_ori_grid_pts
        exp_maps_expanded[pts_to_use,:] =grid0 + np.r_[cur_exp_maps[ii,:]]
    
    exp_maps=exp_maps_expanded
    box_shape = Xs0.shape
    
    pts_to_use = np.arange(exp_maps.shape[0]) 
    rMat_c = rotations.quatOfExpMap(exp_maps.T)
    
    qfib=rMat_c
    print("INFO: will test %d quaternions using %d processes"
          % (qfib.shape[1], ncpus))
    
    # =============================================================================
    # % ORIENTATION SCORING
    # =============================================================================
    print(cfg.find_orientations.omega.period)
    print("INFO:\tusing map search with paintGrid on %d processes"
          % ncpus)
    start = timeit.default_timer()
    
    # return format: (comp, inten_list, hit_list, filter_list, eta_ind_list, ome_ind_list)
    retval = indexer.paintGrid(
        qfib,
        eta_ome,
        etaRange=np.radians(cfg.find_orientations.eta.range),
        omeTol=np.radians(cfg.find_orientations.omega.tolerance),
        etaTol=np.radians(cfg.find_orientations.eta.tolerance),
        omePeriod=np.radians(omega_period), #cfg.find_orientations.omega.period) is depricated
        threshold=on_map_threshold,
        doMultiProc=ncpus > 1,
        nCPUs=ncpus,
        dsgod=True
       )
    
    # process return value
    completeness = []
    inten_list = []
    filter_list = []
    for i in range(len(retval)):
        completeness.append(retval[i][0])
        inten_list.append(retval[i][1])
        filter_list.append(retval[i][3])
    
    completeness = np.array(completeness)
    inten_list = np.array(inten_list)
    filter_list = np.array(filter_list)
    
    # initialize capture things to save for DSGOD
    dsgod_box_quat = np.zeros([1,4, len(pts_to_use)])
    dsgod_box_comp = np.zeros([1,len(pts_to_use)])
    dsgod_box_inten_list = np.zeros([1, len(pts_to_use), inten_list.shape[1]])
    dsgod_box_filter_list = np.zeros([1,len(pts_to_use), filter_list.shape[1]])
    
    # capture things to save for DSGOD
    dsgod_box_quat[0,:,:] = qfib[:,:]
    dsgod_box_comp[0,:] = completeness[:]
    dsgod_box_inten_list = inten_list[:,:]
    dsgod_box_filter_list = filter_list[:,:]
    
    # re-type to save on space
    dsgod_box_quat = dsgod_box_quat.astype(np.float32)
    dsgod_box_comp = dsgod_box_comp.astype(np.float16)
    dsgod_box_inten_list = dsgod_box_inten_list.astype(np.int32)
    dsgod_box_filter_list = dsgod_box_filter_list.astype(np.int8)
    
    np.savez(dsgod_npz_save_dir + dsgod_npz_save_string, 
             dsgod_box_shape=box_shape,
             dsgod_avg_expmap=cur_exp_maps,
             dsgod_box_comp=dsgod_box_comp, 
             dsgod_box_quat=dsgod_box_quat,
             dsgod_box_inten_list=dsgod_box_inten_list,
             dsgod_box_filter_list=dsgod_box_filter_list)
    
    print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
