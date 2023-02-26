#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:32:06 2018

@author: djs522
"""


# *****************************************************************************
#%% IMPORTS
# *****************************************************************************
import sys
import os
import stat
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if sys.version_info[0] < 3:
    from hexrd.xrd import rotations as rot
    from hexrd.grainmap import nfutil
    from hexrd.grainmap import vtkutil
    from hexrd.xrd import symmetry as sym
else:
    from hexrd import rotations as rot
import scipy.io as sio    
import cc3d

sys.path.insert(1, '/home/djs522/additional_sw/hedmTools/')
from CHESS_hedmTools.SingleGrainOrientationDistributions import OrientationTools as OT

import logging
logger = logging.getLogger()
logger.setLevel('INFO')
handler = logging.StreamHandler(sys.stdout)
handler.setLevel('INFO')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# *****************************************************************************
#%% CONSTANTS
# *****************************************************************************
GRAIN_MAP = 'grain_map'
CONF_MAP = 'confidence_map'
X_COORD = 'Xs'
Y_COORD = 'Ys'
Z_COORD = 'Zs'
ORI_LIST = 'ori_list'

# *****************************************************************************
#%% FUNCTIONS
# *****************************************************************************
def calc_coord(gm_shape):
    # Create meshed grid for data points [0,1] (centers of voxels)
    n_x = gm_shape[1]
    n_y = gm_shape[0]
    n_z = gm_shape[2]
    
    x = np.linspace(1/(2*n_x), 1-1/(2*n_x), num=n_x, endpoint=True)
    y = np.linspace(1/(2*n_y), 1-1/(2*n_y), num=n_y, endpoint=True)
    z = np.linspace(1/(2*n_z), 1-1/(2*n_z), num=n_z, endpoint=True)
    
    [X,Y,Z] = np.meshgrid(x,y,z,indexing='ij')
    
    return [X, Y, Z]

def volume_fraction_conf(grain_map, conf_map):
    old_ids = np.unique(grain_map)
    vol_frac_conf = np.zeros(old_ids.shape)
    
    for i in range(old_ids.size):
        ind = (grain_map == old_ids[i])
        vol_frac_conf[i] = np.sum(conf_map[ind]) / float(np.sum(ind))
        
    return vol_frac_conf

def reorder_ids(gm, voxel_threshold=0, do_cc=False, conf_map=None, connectivity=18):
    '''
    

    Parameters
    ----------
    gm : numpy array (i x j x k)
        nf grain id map array.
    voxel_threshold : int, optional
        threshold for minimum voxels to consider a grain, otherwise ignore. The default is 0.
    do_cc : bool, optional
        True = do connected components segmentation of grain ids for reassignment. The default is False.
    conf_map : numpy array (i x j x k), optional
        nf confidence threshold map (conf_map > conf_thresh). The default is None.
    connectivity : int, optional
        connectivity  for cconnected componenets algorithm (6, 18, 26). The default is 18.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    ret_gm : numpy array (i x j x k)
        nf grain id map array with reassigned sequential ids
    new_ids : numpy array (n)
        array of n new grain ids.
    new_old_ids : numpy array (n)
        array of old grain ids that align and pair with the new grain ids
    new_grain_size_distrib : numpy array (n)
        array of number of voxels in grain for new ids (grain size).
    old_grain_size_distrib : numpy array (o)
        array of number of voxels in grain for old ids (grain size).

    '''
    
    #initialize variables
    old_ids = np.unique(gm)
    ret_gm = np.zeros(gm.shape)
    new_old_ids = []
    old_grain_size_distrib = []
    new_grain_size_distrib = []
    
    # replace grain ids under conf_thresh map with -1
    if conf_map is not None:
        gm[~conf_map] = -1
    
    # initialize starting grain id
    curr_gid = 1
    if do_cc:
        # do connected components of grain ids in grain map, segementing all grains
        # connectivity = 18 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
        labels_out = cc3d.connected_components(gm.astype(int), connectivity=connectivity)
        labels_out += 1 # increment all label ids by 1 (now starting at 1)
        labels_out[gm == -1] = 0 # all voxels under conf_thresh map get 0
        N = np.max(labels_out)
        
        new_old_ids = []
        uni_label, label_count = np.unique(labels_out, return_counts=True)
        if 0 in uni_label:
            uni_label = uni_label[1:] # remove the 0 (under conf_thresh)
            label_count = label_count[1:] # remove the 0 (under conf_thresh)
        
        logger.info("%i individual connected components" %(N))
        logger.info("%i individual connected components above voxel threshold" %(np.sum(label_count >= voxel_threshold)))
        
        # for each label...
        for i, label in enumerate(uni_label):
            if label <= 0:
                # ignore 0 label (under conf_thresh)
                continue
            else:
                # add to old grain size distribution
                old_grain_size_distrib.append(label_count[i])
                
                # if this connected component label is above the voxel threshold
                if label_count[i] >= voxel_threshold:
                    # find the indices of the connected component
                    ind = (labels_out == label)
                    
                    # reassign grain id in return grain map
                    ret_gm[ind] = curr_gid
                    
                    # add to new grain sie distribution
                    new_grain_size_distrib.append(np.sum(ind))
                    
                    # check old grain map to ensure only 1 grain id exists in connected region
                    # might not be necessary but a nice check for now
                    temp_old_id = np.unique(gm[ind])
                    if temp_old_id.size == 1:
                        new_old_ids.append(temp_old_id)
                    else:
                        raise Exception(temp_old_id)
                    
                    # inrecement current grain id
                    curr_gid = curr_gid+1
        
        
    else:
        # for each old grain id in the old grain map...
        for i in range(old_ids.size):
            # find the indices of the grain
            ind = (gm == old_ids[i])
            
            # add to old grain size distribution
            old_grain_size_distrib.append(np.sum(ind))
            
            # if this grain is above the voxel threshold
            if np.sum(ind) >= voxel_threshold:
                # reassign grain id in return grain map
                ret_gm[ind] = curr_gid
                
                # add to new grain sie distribution
                new_old_ids.append(old_ids[i])
                new_grain_size_distrib.append(np.sum(ind))
                
                # inrecement current grain id
                curr_gid = curr_gid+1
            else:
                ret_gm[ind] = 0 # cell_id = 0 means ignore/fill-in in Neper
                
    # initialize new ids starting at grain id 1, all zeros in return grain map are ignore/fill-in for Neper
    new_ids = np.arange(curr_gid-1) + 1
    
    new_ids = new_ids.astype(int)    
    new_old_ids = np.array(new_old_ids).astype(int)
    new_grain_size_distrib = np.array(new_grain_size_distrib).astype(int)
    old_grain_size_distrib = np.array(old_grain_size_distrib).astype(int)
    return ret_gm, new_ids, new_old_ids, new_grain_size_distrib, old_grain_size_distrib

# *****************************************************************************
#%% MAIN CALL
# *****************************************************************************
script_cmd_gen = 'python /home/djs522/additional_sw/hedmTools/CHESS_hedmTools/Modeling/Neper/nf_grain_map_to_neper.py {map.npz} \
    --output_dir {output_dir} --output_stem {output_stem} --ori_sym {cubic} --ori_conv {passive} --comp_thresh {0.75} \
    --voxel_size {0.005} --voxel_thresh {0} --do_cc3d {True} --cc3d_connectivity {26} --do_reorder_ids {True} --tess_max_iter {1000} --debug {True} \
'
script_cmd = 'python /home/djs522/additional_sw/hedmTools/CHESS_hedmTools/Modeling/Neper/nf_grain_map_to_neper.py map.npz \
    --output_dir output_dir --output_stem output_stem --ori_sym cubic --ori_conv passive --comp_thresh 0.75 \
    --voxel_size 0.005 --voxel_thresh 0 --do_cc3d True --cc3d_connectivity 26 --do_reorder_ids True --tess_max_iter 1000 --debug True \
'

if __name__ == '__main__':

    # Run preprocessor
    parser = argparse.ArgumentParser(description="Generate Neper Files and Script for Constructing Neper Tesselation\n  %s" %(script_cmd_gen))
    
    parser.add_argument('grain_map_path',
                       metavar='grain_map_path',
                       type=str,
                       help='Path to nf grain map .npz for experiment')    
    parser.add_argument('--output_dir', metavar='output_dir', nargs='?', default=os.getcwd(),
                        help="Path to output directory for Neper files", type=str)
    parser.add_argument('--output_stem', metavar='output_stem', nargs='?', default='default_ff_neper',
                        help="Stem name for output of Neper files", type=str)
    parser.add_argument('--orientation_sym', metavar='ori_sym', nargs='?', default='cubic',
                        help="Orientation symmetry to use", type=str)
    parser.add_argument('--orientation_conv', metavar='ori_conv', nargs='?', default='passive',
                        help="Orientation convention to use (active or passive)", type=str)
    parser.add_argument('--completeness_thresh', metavar='comp_thresh', nargs='?', default=0.75,
                        help="Completeness threshold to use for nf confidence map data", type=float)
    parser.add_argument('--voxel_size', metavar='voxel_size', nargs='?', default=0.005,
                        help="Voxel size of tesr file (in mm)", type=float)
    parser.add_argument('--voxel_thresh', metavar='voxel_thresh', nargs='?', default=0,
                        help="Voxel threshold for minimum number of voxels per grain", type=int)
    parser.add_argument('--do_cc3d', metavar='do_cc3d', nargs='?', default=True,
                        help="bool flag True = do connected components analysis to segment grains that have the same grain ID but are not connected, False = ignore", type=bool)
    parser.add_argument('--cc3d_connectivity', metavar='cc3d_connectivity', nargs='?', default=26,
                        help="Connectivity type for cc3d algorithm (6, 18, 26)", type=int)
    parser.add_argument('--do_reorder_ids', metavar='do_reorder_ids', nargs='?', default=True,
                        help="bool flag True = reorder grain ids in grain map to sequentially range from 1-n needed for Mech-Suite, False = ignore", type=bool)
    parser.add_argument('--tess_max_iter', metavar='tess_max_iter', nargs='?', default=1000,
                        help="maximum number of iterations for Neper tesselation", type=int)
    parser.add_argument('--debug', metavar='debug', nargs='?', default=True,
                        help="Boolean for creating debug plots and statements", type=bool)
    

    args = parser.parse_args()
    grain_map_path = args.grain_map_path
    output_dir = args.output_dir
    output_stem = args.output_stem
    ori_sym = args.orientation_sym
    ori_conv = args.orientation_conv
    comp_thresh = args.completeness_thresh
    voxel_size = args.voxel_size
    voxel_thresh = args.voxel_thresh
    do_cc3d = args.do_cc3d
    cc3d_connectivity = args.cc3d_connectivity
    do_reorder_ids = args.do_reorder_ids
    tess_max_iter = args.tess_max_iter
    debug = args.debug
    

# *****************************************************************************
#%% PROCESS USER INPUT
# *****************************************************************************
if not os.path.isfile(grain_map_path):
    raise ValueError("%s is not a valid nf grain map file" %(grain_map_path))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

ori_sym = ori_sym.lower()
if ori_sym not in ['cubic', 'hexagonal']:
    raise Warning("%s symmetry not supported by MechSuite" %(ori_sym))

ori_conv = ori_conv.lower()
if ori_conv not in ['active', 'passive']:
    raise ValueError("%s convention not supported by Neper, only [active, passive]" %(ori_conv))

if comp_thresh < 0 or comp_thresh > 1:
    raise ValueError("%0.3f completeness threshold must be between [0, 1]" %(comp_thresh))
    
if voxel_size <= 0:
    raise ValueError("%0.3f voxel size must be greater than 0" %(voxel_size))

if cc3d_connectivity not in [6, 18, 26]:
    raise ValueError("cc3d_connectivity must be either 6 (face), 18 (edge), 26 (vertice), not %i" %(cc3d_connectivity))


# *****************************************************************************
#%% LOAD DATA
# *****************************************************************************

if do_cc3d:
    updated_ext = '_conf%i_voxel%i_cc3d%i' %(int(comp_thresh*100), voxel_thresh, cc3d_connectivity)
else:
    updated_ext = '_conf%i_voxel%i' %(int(comp_thresh*100), voxel_thresh)
updated_stem = output_stem + updated_ext

data = np.load(grain_map_path)  

grain_map = data[GRAIN_MAP]
conf_map = data[CONF_MAP]
Xs = data[X_COORD]
Ys = data[Y_COORD]
Zs = data[Z_COORD]
old_exp_maps = data[ORI_LIST]
old_quat = rot.quatOfExpMap(old_exp_maps.T)
old_rod = OT.quat2rod(old_quat.T)

if np.any(grain_map < 0):
    raise ValueError('Grain map contains grain ids less than zero!')
#vol_frac_conf_orig = volume_fraction_conf(grain_map, conf_map) 

# *****************************************************************************
#%% REORDER IDS AND ORIENTATIONS
# *****************************************************************************

if do_reorder_ids:
    logger.info("Reordering IDs...")
    grain_map_ori = np.copy(grain_map)
    grain_map_id, new_ids, new_to_old_ids, new_gsd, old_gsd = reorder_ids(grain_map, voxel_threshold=voxel_thresh, 
                                          do_cc=do_cc3d, conf_map=(conf_map >= comp_thresh), connectivity=cc3d_connectivity)

    #vol_frac_conf_new = volume_fraction_conf(grain_map, conf_map) 
    
    new_exp_maps = old_exp_maps[new_to_old_ids.flatten(), :]
    new_quat = rot.quatOfExpMap(new_exp_maps.T)
    new_rod = OT.quat2rod(new_quat.T)

# *****************************************************************************
#%% CHECK SHAPE OF GRAIN MAP
# *****************************************************************************

# check shape of grain map, get list of grain id with removed 0
gm_list = np.unique(grain_map_id)
gm_list = np.trim_zeros(gm_list)
gm_shape = grain_map_id.shape
IS_CUBE = (gm_shape[0] == gm_shape[1] == gm_shape[2])

# *****************************************************************************
#%% CREATE ASSEMBLED DATA -- LIST OF [VOXEL COORDINATES (X,Y,Z),GRAIN ID]
# *****************************************************************************
o = 'F' # C=row-major order, F=col-major order, Neper says col-major
coordinate_list = np.vstack((Xs.ravel(order=o), Ys.ravel(order=o), Zs.ravel(order=o)))
assembled_data = np.hstack((coordinate_list.T, np.atleast_2d(grain_map_id.ravel(order=o)).T))

# coordinate_list = np.vstack((Xs.ravel(), Ys.ravel(), Zs.ravel()))
# assembled_data = np.hstack((coordinate_list.T, np.atleast_2d(grain_map_id.ravel()).T))

# *****************************************************************************
#%% PREPARING STRINGS
# *****************************************************************************
logger.info("Preparing strings...")
np.set_printoptions(threshold=np.inf)
l1  = '***tesr'
l2  = ' **format'
l3  = '   2.0 ascii'
l4  = ' **general'
l5  = '   3'
l6  = '   ' + str(gm_shape[0]) + ' ' + str(gm_shape[1])  + ' ' + str(gm_shape[2]) 
l7  = '   ' + str(voxel_size) + ' ' + str(voxel_size) + ' ' + str(voxel_size)
l8  = ' **cell';
l9  = '   ' + str(len(gm_list))
l10 = '  *id';
l11 = '   ' + str(gm_list.astype('int').T)[1:-1] # [1:-1] to remove brackets in string
l11_25 = '  *ori \n   rodrigues:%s' %(ori_conv) 
l11_5 = '  *crysym \n   %s' %(ori_sym)
l12 = ' **data'
l14 = '***end'


# *****************************************************************************
#%% WRITING TESR
# *****************************************************************************
logger.info("Writing tesr...")
output = open('%s.tesr' %(os.path.join(output_dir, updated_stem)), 'w');
output.write('%s\n' % l1)
output.write('%s\n' % l2)
output.write('%s\n' % l3)
output.write('%s\n' % l4)
output.write('%s\n' % l5)
output.write('%s\n' % l6)
output.write('%s\n' % l7)
output.write('%s\n' % l8)
output.write('%s\n' % l9)
output.write('%s\n' % l10)
output.write('%s\n' % l11)
output.write('%s\n' % l11_25)
np.savetxt(output, new_rod, fmt='%0.4f')
output.write('%s\n' % l11_5)
output.write('%s\n' % l12)
output.write('   ')
np.savetxt(output, np.atleast_2d(assembled_data[:, 3]).T,fmt='%d')
output.write('%s\n' % l14)

output.close()


# *****************************************************************************
#%% WRITING OTHER OUTPUT
# *****************************************************************************

logger.info("Writing to .npz...")
rod_save_path = os.path.join(output_dir, updated_stem + '_rod.txt')
if do_reorder_ids:
    np.savez(os.path.join(output_dir, updated_stem + '.npz'), GRAIN_MAP=grain_map, CONFIDENCE_MAP=conf_map, X_COORD=Xs, Y_COORD=Ys, Z_COORD=Zs, 
             OLD_IDS=new_to_old_ids, NEW_IDS=new_ids, OLD_EXP_MAPS=old_exp_maps, NEW_EXP_MAPS=new_exp_maps)
    np.savetxt(rod_save_path, new_rod)
else:
    np.savez(os.path.join(output_dir, updated_stem + '.npz'), GRAIN_MAP=grain_map, CONFIDENCE_MAP=conf_map, X_COORD=Xs, Y_COORD=Ys, Z_COORD=Zs, 
             OLD_EXP_MAPS=old_exp_maps)
    np.savetxt(rod_save_path, old_rod)
    

logger.info("Writing bash neper script...")
run_neper_path = os.path.join(output_dir, 'run_neper_nf_%s.sh' %(output_stem))
with open(run_neper_path, 'w') as f:
    f.write('#!/bin/bash\n\n')
    f.write('MY_PRE="%s"\n' %(updated_stem))
    f.write('MICRO_FN_OUT="${MY_PRE}"\n')
    f.write('MICRO_PNG_FN_OUT="${MY_PRE}_png"\n\n')
    
    f.write('# Visualize tesr\n')
    f.write('neper -V "${MICRO_FN_OUT}.tesr" -datacellcol id -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_tesr"\n')
    f.write('neper -V "${MICRO_FN_OUT}.tesr" -datacellcol ori -datacellcolscheme "ipf(y)" -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_ipfy_tesr"\n')
    
    f.write('# Create tessellation\n')
    f.write('neper -T -n from_morpho -domain "cube(%0.4f,%0.4f,%0.4f)" -morpho "tesr:file("${MICRO_FN_OUT}.tesr")" -morphooptiobjective "tesr:pts(region=surf, res=7)" -morphooptistop itermax=%i -transform "grow" -ori "file(%s [,des=rodrigues:%s])" -reg 1 -o "${MICRO_FN_OUT}"\n' %(gm_shape[0]*voxel_size, gm_shape[1]*voxel_size, gm_shape[2]*voxel_size, tess_max_iter, rod_save_path, ori_conv))
    
    f.write('# Visualize tessellation\n')
    f.write('neper -V "${MICRO_FN_OUT}.tess" -datacellcol id -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_tess"\n')
    f.write('neper -V "${MICRO_FN_OUT}.tess" -datacellcol ori -datacellcolscheme "ipf(y)" -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_ipfy_tess"\n')
    
    f.write('# Mesh tessellation\n')
    f.write('neper -M "${MICRO_FN_OUT}.tess" -part 32:8 -for msh,fepx:legacy -rcl 1.0 -pl 1.5 -order 2 -faset z0,z1,x0,x1,y0,y1 -o "${MICRO_FN_OUT}"\n')
    
    f.write('# Visualize mesh\n')
    f.write('neper -V "${MICRO_FN_OUT}.tess,${MICRO_FN_OUT}.msh" -dataelsetco id -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_mesh"\n')
    f.write('neper -V "${MICRO_FN_OUT}.tess,${MICRO_FN_OUT}.msh" -dataelsetcol ori -dataelsetcolscheme "ipf(y)" -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_ipfy_mesh"\n')
    
    f.write('# Mesh stats\n')
    f.write('neper -M -loadmesh "${MICRO_FN_OUT}.msh" -statmesh nodenb,eltnb -o ${MICRO_FN_OUT}_mesh_stats\n')
    
    f.write('echo "Done!" \nexit 0')

st = os.stat(run_neper_path)
os.chmod(run_neper_path, st.st_mode | stat.S_IEXEC)
    
logger.info("Wrote script for running Neper to: %s" %(run_neper_path))


