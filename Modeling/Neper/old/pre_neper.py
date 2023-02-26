# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import print_function

IMPORT_MSP_DIR = '/home/millerlab/djs522/chess_bt_2019-12/dp718-2_hexrd/FIT_HEDM/'
IMPORT_HEXRD_SCRIPT_DIR = '/home/millerlab/djs522/chess_bt_2019-12/dp718-2_hexrd/Scripts/'

import sys
sys.path.insert(0, IMPORT_MSP_DIR)
sys.path.insert(0, IMPORT_HEXRD_SCRIPT_DIR)


import numpy as np

import hexrd.matrixutil as hexrd_mat
import hexrd.xrd.rotations as hexrd_rot

import post_process_stress as pp_stress


# =============================================================================
# CONSTANTS
# =============================================================================
FILENAME = '/home/millerlab/djs522/chess_bt_2019-12/neper_dp718-2/det6_load0MPa_grains.out'
SAVE_NAME = 'det6_load0MPa'
X_LIM = [-0.5, 0.5]
Y_LIM = [-0.15, 0.15]
Z_LIM = [-0.5, 0.5]

# =============================================================================
# FUNCTION DECLARATION AND IMPLEMENTATION
# =============================================================================

# transform euler angles to kocks angles
def euler2kocks(euler):
    # euler angles (rad) to kocks angles (rad)
    kocks = np.vstack([np.mod(np.pi/2 - euler[:, 2], np.pi * 2), np.mod(euler[:, 1], np.pi * 2), np.mod(euler[:, 0] - np.pi/2, np.pi * 2)]).T
    return kocks

# create .kocks file
def create_kocks_file(exp_maps, save_name):
    num_grains = exp_maps.shape[0]
    
    # do rotation transformations to go from exp_maps to kocks
    rotmat = hexrd_rot.rotMatOfExpMap_opt(exp_maps.T)
    eulers = np.zeros([num_grains, 3])
    for i in range(num_grains):
        eulers[i] = hexrd_rot.angles_from_rmat_xyz(rotmat[i])
    kocks_rad = euler2kocks(eulers)
    
    # write output file
    fo = open(save_name + '.kocks', "w")
    fo.writelines('grain-orientations\n')
    fo.writelines(str(num_grains) + '\n')
    
    kocks_deg = np.rad2deg(kocks_rad)
    grain_ids = np.arange(num_grains) + 1
    
    write_orient = np.hstack([kocks_deg, grain_ids[:, np.newaxis]])
    np.savetxt(fo, write_orient, fmt='%.8f \t%.8f \t%.8f \t%i')
    
    fo.writelines('EOF')
    fo.close()

# create .npos file
def create_npos_file(centroids, save_name):
    num_grains = centroids.shape[0]
    
    # write output file
    fo = open(save_name + '.npos', "w")
    np.savetxt(fo, centroids, fmt='%.8f \t%.8f \t%.8f')
    fo.close()
    

# read grains.out and do thresholding
grain_mat = np.loadtxt(FILENAME)
comp_thresh = 0.95
chi2_thresh = 1.5e-2
grain_mat = grain_mat[np.where((grain_mat[:, 1] >= comp_thresh) & (grain_mat[:, 2] <= chi2_thresh))]
centroids = grain_mat[:, 6:9]
orient = grain_mat[:, 3:6]

# theshold centroids
bulk_ind = np.where((centroids[:, 0] >= X_LIM[0]) & (centroids[:, 0] <= X_LIM[1]) &
                    (centroids[:, 1] >= Y_LIM[0]) & (centroids[:, 1] <= Y_LIM[1]) &
                    (centroids[:, 2] >= Z_LIM[0]) & (centroids[:, 2] <= Z_LIM[1]))
centroids = centroids[bulk_ind]
orient = orient[bulk_ind]

# gather number of grains, centroids, and orientations
num_grains = centroids.shape[0]
print('Number of grains: %i' %(num_grains))
grain_ids = np.arange(num_grains) + 1


# create .kocks file
create_kocks_file(orient, SAVE_NAME)

# create .npos file (neper centroid position file)
create_npos_file(centroids, SAVE_NAME)
    






















