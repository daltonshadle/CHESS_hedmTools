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
import os

import hexrd.matrixutil as hexrd_mat
import hexrd.xrd.rotations as hexrd_rot

import post_process_stress as pp_stress


# =============================================================================
# CONSTANTS
# =============================================================================
SAVE_NAME = 'det6_load0MPa'

# =============================================================================
# FUNCTION DECLARATION AND IMPLEMENTATION
# =============================================================================

# create .grain file
def create_grain_file(opt_file, save_name):
    # open opt file, read lines, and remove first line
    with open(opt_file) as f_opt:
        lines_opt = f_opt.readlines()
    
    lines_opt.pop(0)
    
    # write output file
    f_gr = open(save_name + '.grain', "w")
    f_gr.writelines(lines_opt)
    f_gr.close()

# create .mesh file
def create_mesh_file(mesh_file, params_file, save_name):
    # open parm file, read lines, and get node, element numbers
    with open(params_file) as f_param:
        lines_param = f_param.readlines()    
    
    first_line = lines_param[0].split(' ')
    num_ele = first_line[0]
    num_nodes = first_line[1]
    num_npe = first_line[2]
    
    with open(mesh_file) as f_mesh:
        lines_mesh = f_mesh.readlines()
    first_line = lines_mesh[0].split(' ')
    num_npe = len(first_line) - 3
    
    # remove line 1.0 1.0 1.0
    lines_mesh.pop(int(num_ele))
    
    # insert header
    lines_mesh.insert(0, '%i %i %i\n' %(int(num_ele), int(num_nodes), int(num_npe)))
    
    # write output file
    f_mesh_out = open(save_name + '.mesh', "w")
    f_mesh_out.writelines(lines_mesh)
    f_mesh_out.close()
    


root_dir = '/home/millerlab/djs522/chess_bt_2019-12/neper_dp718-2'
base_str = 'det6_load0MPa_mesh'
opt_file = os.path.join(root_dir, base_str + '.opt')
parm_file = os.path.join(root_dir, base_str + '.parms')
mesh_file = os.path.join(root_dir, base_str + '.mesh')

# create .kocks file
create_grain_file(opt_file, SAVE_NAME)

# create .npos file (neper centroid position file)
create_mesh_file(mesh_file, parm_file, SAVE_NAME)
    
print("Done!")





















