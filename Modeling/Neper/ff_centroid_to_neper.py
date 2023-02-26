#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:32:06 2018

@author: djs522
"""

script_cmd_gen = 'python /home/djs522/additional_sw/hedmTools/CHESS_hedmTools/Modeling/Neper/ff_centroid_to_neper.py {grains.out} \
    --output_dir {output_dir} --output_stem {output_stem} --ori_sym {cubic} --ori_conv {passive} --comp_thresh {0.75} --chi2_thresh {0.1} \
    --voxel_size {0.005} --x_lower {-0.5} --x_upper {0.5} --y_lower {-0.5} --y_upper {0.5} --z_lower {-0.5} --z_upper {0.5} --plot True \
'

script_cmd = 'python /home/djs522/additional_sw/hedmTools/CHESS_hedmTools/Modeling/Neper/ff_centroid_to_neper.py grains.out \
    --output_dir output_dir --output_stem output_stem --ori_sym cubic --ori_conv passive --comp_thresh 0.75 --chi2_thresh 0.1 \
    --voxel_size 0.005 --x_lower -0.5 --x_upper 0.5 --y_lower -0.5 --y_upper 0.5 --z_lower -0.5 --z_upper 0.5 --plot True \
'

# *****************************************************************************
#%% IMPORTS
# *****************************************************************************

import sys
import os
import stat
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from hexrd import rotations

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
#%% MAIN CALL
# *****************************************************************************
if __name__ == '__main__':

    # Run preprocessor
    parser = argparse.ArgumentParser(description="Generate Neper Files and Script for Constructing Neper Tesselation\n  %s" %(script_cmd_gen))
    
    parser.add_argument('grains_out_path',
                       metavar='grains_out_path',
                       type=str,
                       help='Path to grains.out for experiment')    
    parser.add_argument('--output_dir', metavar='output_dir', nargs='?', default=os.getcwd(),
                        help="Path to output directory for Neper files", type=str)
    parser.add_argument('--output_stem', metavar='output_stem', nargs='?', default='default_ff_neper',
                        help="Stem name for output of Neper files", type=str)
    parser.add_argument('--orientation_sym', metavar='ori_sym', nargs='?', default='cubic',
                        help="Orientation symmetry to use", type=str)
    parser.add_argument('--orientation_conv', metavar='ori_conv', nargs='?', default='passive',
                        help="Orientation convention to use (active or passive)", type=str)
    parser.add_argument('--completeness_thresh', metavar='comp_thresh', nargs='?', default=0.75,
                        help="Completeness threshold to use for ff-HEDM grains.out data", type=float)
    parser.add_argument('--chi2_thresh', metavar='chi2_thresh', nargs='?', default=1e-1,
                        help="chi^2 threshold to use for ff-HEDM grains.out data", type=float)
    parser.add_argument('--voxel_size', metavar='voxel_size', nargs='?', default=0.005,
                        help="Voxel size of tesr file (in mm)", type=float)
    parser.add_argument('--x_lower', metavar='x_lower', nargs='?', default=-0.5,
                        help="Lower bound of x-pos (in mm) of COM position in ff-HEDM grains.out data", type=float)
    parser.add_argument('--x_upper', metavar='x_upper', nargs='?', default=0.5,
                        help="Upper bound of x-pos (in mm) of COM position in ff-HEDM grains.out data", type=float)
    parser.add_argument('--y_lower', metavar='y_lower', nargs='?', default=-0.5,
                        help="Lower bound of y-pos (in mm) of COM position in ff-HEDM grains.out data", type=float)
    parser.add_argument('--y_upper', metavar='y_upper', nargs='?', default=0.5,
                        help="Upper bound of y-pos (in mm) of COM position in ff-HEDM grains.out data", type=float)
    parser.add_argument('--z_lower', metavar='z_lower', nargs='?', default=-0.5,
                        help="Lower bound of z-pos (in mm) of COM position in ff-HEDM grains.out data", type=float)
    parser.add_argument('--z_upper', metavar='z_upper', nargs='?', default=0.5,
                        help="Upper bound of z-pos (in mm) of COM position in ff-HEDM grains.out data", type=float)
    parser.add_argument('--plotting', metavar='plot', nargs='?', default=True,
                        help="Boolean for creating debug plots", type=bool)
    

    args = parser.parse_args()
    grains_out_path = args.grains_out_path
    output_dir = args.output_dir
    output_stem = args.output_stem
    ori_sym = args.orientation_sym
    ori_conv = args.orientation_conv
    comp_thresh = args.completeness_thresh
    chi2_thresh = args.chi2_thresh
    voxel_size = args.voxel_size
    x_lower = args.x_lower
    x_upper = args.x_upper
    y_lower = args.y_lower
    y_upper = args.y_upper
    z_lower = args.z_lower
    z_upper = args.z_upper
    plot = args.plotting

# *****************************************************************************
#%% PROCESS USER INPUT
# *****************************************************************************
if not os.path.isfile(grains_out_path):
    raise ValueError("%s is not a valid grains.out file" %(grains_out_path))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
ori_save_path = os.path.join(output_dir, output_stem + '_rod.txt')
pos_save_path = os.path.join(output_dir, output_stem + '_pos.npos')
grainids_save_path = os.path.join(output_dir, output_stem + '_grainids.txt')

ori_sym = ori_sym.lower()
if ori_sym not in ['cubic', 'hexagonal']:
    raise Warning("%s symmetry not supported by MechSuite" %(ori_sym))

ori_conv = ori_conv.lower()
if ori_conv not in ['active', 'passive']:
    raise ValueError("%s convention not supported by Neper, only [active, passive]" %(ori_conv))

if comp_thresh < 0 or comp_thresh > 1:
    raise ValueError("%0.3f completeness threshold must be between [0, 1]" %(comp_thresh))

if chi2_thresh < 0 or comp_thresh > 1:
    raise Warning("%0.3f chi^2 threshold might be off" %(chi2_thresh))

if x_lower > x_upper:
    raise ValueError("%0.4f lower > %0.4f upper x bound" %(x_lower, x_upper))

if y_lower > y_upper:
    raise ValueError("%0.4f lower > %0.4f upper y bound" %(y_lower, y_upper))

if z_lower > z_upper:
    raise ValueError("%0.4f lower > %0.4f upper z bound" %(z_lower, z_upper))

scan_bounds = [[x_lower, x_upper], [y_lower, y_upper], [z_lower, z_upper]]
scan_dimen = [np.round(np.abs(scan_bounds[0][0] - scan_bounds[0][1]), decimals=4),
              np.round(np.abs(scan_bounds[1][0] - scan_bounds[1][1]), decimals=4),
              np.round(np.abs(scan_bounds[2][0] - scan_bounds[2][1]), decimals=4)]

# *****************************************************************************
#%% PROCESS GRAINS.OUT DATA
# *****************************************************************************

# load grain mat
grain_mat = np.loadtxt(grains_out_path)
logger.info("%i grains in original grains.out" %(grain_mat.shape[0]))

# threshold completeness and chi^2 from fitting of grain mat
good_grain_mat = grain_mat[((grain_mat[:, 1] >= comp_thresh) & (grain_mat[:, 2] <= chi2_thresh)), :]
logger.info("%i grains in comp_thresh and chi2_thresh reduced grains.out" %(good_grain_mat.shape[0]))

# threshold position of grains in grain mat
xyz = good_grain_mat[:, 6:9]
good_pos_ind = ( (xyz[:, 0] > scan_bounds[0][0]) & (xyz[:, 0] < scan_bounds[0][1]) 
               & (xyz[:, 1] > scan_bounds[1][0]) & (xyz[:, 1] < scan_bounds[1][1]) 
               & (xyz[:, 2] > scan_bounds[2][0]) & (xyz[:, 2] < scan_bounds[2][1]))
good_pos_grain_mat = good_grain_mat[good_pos_ind, :]
logger.info("%i grains in position bound reduced grains.out" %(good_pos_grain_mat.shape[0]))


# PLOT CENTER OF MASS POSITIONS OF RAW DATA AND THRESHOLD GRAINS
if plot:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(good_grain_mat[:, 6], good_grain_mat[:, 7], good_grain_mat[:, 8])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(good_pos_grain_mat[:, 6], good_pos_grain_mat[:, 7], good_pos_grain_mat[:, 8])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.show()

# *****************************************************************************
#%% SAVE NEPER FILES AND WRITE NEPER COMMAND SCRIPT
# *****************************************************************************

# save positions file
np.savetxt(pos_save_path, good_pos_grain_mat[:, 6:9])

# save orienations file
exp_maps = good_pos_grain_mat[:, 3:6]
quat = rotations.quatOfExpMap(exp_maps.T).T
rod = OT.quat2rod(quat)
np.savetxt(ori_save_path, rod)

# save grain ids file
np.savetxt(grainids_save_path, good_pos_grain_mat[:, 0])


neper_cmd = 'neper -T -n %i -dim 3 -domain "cube(%.3f,%.3f,%.3f)" -loadpoint "msfile(%s)" \
-ori "file(%s [,des=rodrigues:%s])" -oridescriptor rodrigues:%s -oricrysym %s \
-reg 1 -mloop 10 -morpho voronoi -morphooptistop itermax=5000 -format tess,tesr \
-tesrsize "%i:%i:%i" -tesrformat "ascii" -o $MICRO_FN_OUT' \
      %(good_pos_grain_mat.shape[0],
        scan_dimen[0], 
        scan_dimen[1], 
        scan_dimen[2],
        pos_save_path,
        ori_save_path,
        ori_conv,
        ori_conv,
        ori_sym,
        scan_dimen[0] / voxel_size, 
        scan_dimen[1] / voxel_size, 
        scan_dimen[2] / voxel_size)

logger.info("Neper cmd: \n %s" %(neper_cmd))

run_neper_path = os.path.join(output_dir, 'run_neper_ff_%s.sh' %(output_stem))
with open(run_neper_path, 'w') as f:
    f.write('#!/bin/bash\n\n')
    f.write('MY_PRE="%s"\n' %(output_stem))
    f.write('MICRO_FN_OUT="${MY_PRE}"\n')
    f.write('MICRO_PNG_FN_OUT="${MY_PRE}_png"\n\n')
    
    f.write('# Create tessellation and tesr\n')
    f.write('%s\n' %(neper_cmd))
    
    f.write('# Visualize tessellation\n')
    f.write('neper -V "${MICRO_FN_OUT}.tess" -datacellcol id -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_tess"\n')
    f.write('neper -V "${MICRO_FN_OUT}.tess" -datacellcol ori -datacellcolscheme "ipf(y)" -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_ipfy_tess"\n')
    
    f.write('# Visualize tesr\n')
    f.write('neper -V "${MICRO_FN_OUT}.tesr" -datacellcol id -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_tesr"\n')
    f.write('neper -V "${MICRO_FN_OUT}.tesr" -datacellcol ori -datacellcolscheme "ipf(y)" -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_ipfy_tesr"\n')
    
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
