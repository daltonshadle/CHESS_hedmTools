#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:58:21 2021

@author: djs522
"""

#%% IMPORTS
import numpy as np
import os
import sys
if sys.version_info[0] < 3:
    from hexrd.xrd import rotations as hexrd_rot
    import hexrd.xrd.symmetry as hexrd_sym
else:
    from hexrd import rotations as hexrd_rot
    from hexrd import symmetry as hexrd_sym
import scipy.io as sio  

#%% CONSTANTS

# nf-HEDM GRAIN MAP KEYS
GRAIN_MAP = 'grain_map'
X_COORD = 'Xs'
Y_COORD = 'Ys'
Z_COORD = 'Zs'
ORI_LIST = 'ori_list'
OLD_IDS = 'old_ids'
NEW_IDS = 'new_ids'

#%% FUNCTIONS
def parse_tesr(filename):
    general_key = '**general' # 0: dimen, 1: domain, 2: voxel size
    cell_key = '**cell' # 0: num grains
    id_key = '*id' # lines of ids until next *
    voxel_key = '**data' # lines of voxel ides until next *    
    
    dimen = []
    domain = []
    voxel_size = []
    num_grains = []
    ids = ''
    voxels = ''
    
    with open(filename) as myfile:
        all_lines = myfile.readlines()
        
        print("Lines Read")
        for i, line in enumerate(all_lines):
            if general_key in line:
                dimen = list(map(int, all_lines[i+1].strip().split()))
                domain = list(map(int, all_lines[i+2].strip().split()))
                voxel_size = list(map(float, all_lines[i+3].strip().split()))
            if cell_key in line:
                num_grains = list(map(int, all_lines[i+1].strip().split()))
            if id_key in line:
                j = i+1
                while ('*' not in all_lines[j]):
                    ids = ids +  ' ' + all_lines[j].strip()
                    j += 1
                ids = list(map(int, ids.split()))
            if voxel_key in line:
                j = i+1
                while ('*' not in all_lines[j]):
                    voxels = voxels +  ' ' + all_lines[j].strip()
                    j += 1
                voxels = list(map(int, voxels.split()))
                break
    return [dimen, domain, voxel_size, num_grains, ids, voxels]

def combine_tesr(list_of_tesr, stack_dir=2, stack_order=None):
    # list_of_tesr: ID-ordered list containing tesr parsed object for parse_tesr()
    # stack_dir: direction to stack tesrs, 0-2 for xyz
    
    n_tesr = len(list_of_tesr)
    
    domain_list = []
    vox_size_list = []
    n_grains_list = []
    voxel_maps_list = []
    
    for i in range(n_tesr):
        domain_list.append(list_of_tesr[i][1])
        vox_size_list.append(list_of_tesr[i][2])
        n_grains_list.append(list_of_tesr[i][3])
        
        copy_domain = np.flip(np.copy(list_of_tesr[i][1]))
        voxel_maps_list.append(np.array(list_of_tesr[i][5]).reshape(copy_domain).T)
        print(voxel_maps_list[i].shape)
        print(domain_list[i])
        print(vox_size_list[i])
    
    domain_list = np.array(domain_list)
    vox_size_list = np.array(vox_size_list)
    n_grains_list = np.array(n_grains_list)
    
    total_grains = np.sum(n_grains_list)
    total_grain_ids = np.arange(total_grains) + 1
    
    # do checks
    vox_check = np.all(vox_size_list == vox_size_list[0, :], axis=0)
    if vox_check.all():
        print('All .tesr files have the same voxel size')
    else:
        raise ValueError('.tesr files do not share the same voxel size')
    
    if stack_dir == 0:
        check_ind = [1, 2]
    elif stack_dir == 1:
        check_ind = [0, 2]
    else:
        check_ind = [0, 1]
        
    
    domain_check = np.all(domain_list[:, check_ind] == domain_list[0, check_ind], axis=0)
    if domain_check.all():
        print('All .tesr files have the same domain orthogonal to stack direction')
    else:
        print(domain_list)
        raise ValueError('.tesr files do not share the same domain orthogonal to stack direction')
    
    # do id reassignment mapping
    start = 0
    end = 0
    total_voxel_map_list = []
    
    print(voxel_maps_list[0].shape)
    print(np.zeros(domain_list[0]).shape)
    
    for i in range(n_tesr):
        end += n_grains_list[i]
        end = int(end)
        ids_remap = np.vstack([list_of_tesr[i][4], total_grain_ids[start:end]]).T
        start += n_grains_list[i]
        start = int(start)
        
        print(ids_remap.shape)
        
        temp_voxel_map = np.zeros(domain_list[i])
        for j in range(ids_remap.shape[0]):
            #print((voxel_maps_list[i] == ids_remap[j, 0]).shape)
            temp_voxel_map[voxel_maps_list[i] == ids_remap[j, 0]] = ids_remap[j, 1]
        total_voxel_map_list.append(temp_voxel_map)
    
    # stack voxel map
    if stack_order is None:
        stack_order = range(len(total_voxel_map_list))
    total_vox_map = np.concatenate([total_voxel_map_list[i] for i in stack_order], axis=stack_dir) 
    
    return total_vox_map

def calc_coord(gm_shape):
    # Create meshed grid for data points [0,1] (centers of voxels)
    n_x = gm_shape[0]
    n_y = gm_shape[1]
    n_z = gm_shape[2]
    
    x = np.linspace(1/(2*n_x), 1-1/(2*n_x), num=n_x, endpoint=True)
    y = np.linspace(1/(2*n_y), 1-1/(2*n_y), num=n_y, endpoint=True)
    z = np.linspace(1/(2*n_z), 1-1/(2*n_z), num=n_z, endpoint=True)
    
    [X,Y,Z] = np.meshgrid(x,y,z)
    
    return [X, Y, Z]

def reorder_ids(gm):
    old_ids = np.unique(gm)
    
    ret_gm = np.copy(gm)
    
    for i in range(old_ids.size):
        ret_gm[gm == old_ids[i]] = i + 1
    
    new_ids = np.arange(old_ids.size) + 1
    
    return ret_gm, new_ids.astype(int), old_ids.astype(int)

def grainmap2tesr(filename, savename, voxel_spacing=0.005, HAVE_COORD=True, HAVE_ORI=True, SAVE_NPZ=True):
    # NOTES: 
    #  - Make sure data is modified to be ORTHOGONAL TO THE ARRAY from sample coord sys
    #  - The example_HEDM_map file is saved in the order ['x_coord', 'z_coord', 'grain_id_map', 'y_coord']
    #  - Any file saved with this program will have the order ['grain_id_map', 'x_coord', 'y_coord', 'z_coord']
    #  - HAVE_COORD = True if coordinates are attached, False if coordinates need to be calculated
    #  - EXAMPLE = True if using example_HEDM_map
    #  - SAVE_NPZ = True if user wants to save .npz of data (for instance, if coordinates are calculated)
    
    # LOAD DATA
    data=np.load(open(filename,'rb'))  

    if HAVE_COORD:
        grain_map = data[GRAIN_MAP]
        Xs = data[X_COORD]
        Ys = data[Y_COORD]
        Zs = data[Z_COORD] 
    else:
        grain_map = data[GRAIN_MAP]
        grain_map, new_ids, old_ids = reorder_ids(grain_map)
        if HAVE_ORI:
            new_exp_maps = data[ORI_LIST][old_ids, :]
        [Xs, Ys, Zs] = calc_coord(grain_map.shape)

    # check shape of grain map, get list of grain id with removed 0
    gm_list = np.unique(grain_map)
    gm_list = np.trim_zeros(gm_list)
    gm_shape = grain_map.shape
    print('Grain map shape: %i, %i, %i' %(gm_shape[0], gm_shape[1], gm_shape[2]))
    IS_CUBE = (gm_shape[0] == gm_shape[1] == gm_shape[2])

    #CREATE ASSEMBLED DATA -- LIST OF [VOXEL COORDINATES (X,Y,Z),GRAIN ID]
    coordinate_list=np.vstack((Xs.ravel(),Ys.ravel(),Zs.ravel()))
    assembled_data=np.hstack((coordinate_list.T,np.atleast_2d(grain_map.ravel()).T))
    
    # PREPARE STRINGS FOR TESR
    print('Preparing strings...')
    np.set_printoptions(threshold=np.inf)
    l1  = '***tesr'
    l2  = ' **format'
    l3  = '   2.0 ascii'
    l4  = ' **general'
    l5  = '   3'
    # l6  = '   ' + str(grain_map.shape[1]) + ' ' + str(grain_map.shape[0])  + ' ' + str(grain_map.shape[2]) 
    l6  = '   ' + str(gm_shape[2]) + ' ' + str(gm_shape[1])  + ' ' + str(gm_shape[0]) 
    l7  = '   ' + str(voxel_spacing) + ' ' + str(voxel_spacing) + ' ' + str(voxel_spacing)
    l8  = ' **cell';
    l9  = '   ' + str(len(gm_list))
    l10 = '  *id';
    # l11 = '   ' + str(np.arange(1,np.max(grain_map)+1).astype('int').T)[1:-1]
    l11 = '   ' + str(gm_list.astype('int').T)[1:-1]
    l12 = ' **data'
    #l13 = '   ' + str(assembled_data[:,3].astype('int'))[1:-1]
    l14 = '***end'


    # WRITE TESR
    print('Writing to tesr...')
    output = open('%s.tesr'%(savename),'w');
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
    output.write('%s\n' % l12)
    output.write('   ')
    np.savetxt(output,np.atleast_2d(assembled_data[:,3]).T,fmt='%d')
    #output.write('%s\n' % l13)
    output.write('%s\n' % l14)
    
    output.close()
    
    if not IS_CUBE:
        print('NOTE: THIS GRAIN MAP DOES NOT HAVE A CUBE SHAPE')
        print('NOTE: DOMAIN FOR THIS MAP IS NOW')
        print('   FROM: -domain "cube(0.5,0.5,0.5)"')
        print('   TO:   -domain "cube(%.3f,%.3f,%.3f)"' %(gm_shape[2]*voxel_spacing, gm_shape[1]*voxel_spacing, gm_shape[0]*voxel_spacing))
    
    if SAVE_NPZ:
        print('Writing to .npz...')
        if HAVE_ORI:
            np.savez(savename+'.npz', grain_map=grain_map, Xs=Xs, Ys=Ys, Zs=Zs, ori_list=new_exp_maps, old_ids=old_ids, new_ids=new_ids)
        else:
            np.savez(savename+'.npz', grain_map=grain_map, Xs=Xs, Ys=Ys, Zs=Zs, old_ids=old_ids, new_ids=new_ids)
    
    print('Done!')

def grainsout2npos(path, filename, savename, COMP_THRESH=0.75, CHI2_THRESH=1e-2, VOXEL_SIZE=0.005, 
                   SCAN_BOUNDS=[[-0.5, 0.5], [-0.125, 0.125], [-0.5, 0.5]], SAVE_ORI=True):

    # load grain mat
    grain_mat = np.loadtxt(os.path.join(path, filename))
    
    # threshold completeness and chi^2 from fitting of grain mat
    good_grain_mat = grain_mat[((grain_mat[:, 1] >= COMP_THRESH) & (grain_mat[:, 2] <= CHI2_THRESH)), :]
    
    # threshold position of grains in grain mat
    SCAN_DIMEN = [np.round(np.abs(SCAN_BOUNDS[0][0] - SCAN_BOUNDS[0][1]), decimals=4),
                  np.round(np.abs(SCAN_BOUNDS[1][0] - SCAN_BOUNDS[1][1]), decimals=4),
                  np.round(np.abs(SCAN_BOUNDS[2][0] - SCAN_BOUNDS[2][1]), decimals=4)]
    
    xyz = good_grain_mat[:, 6:9]
    good_pos_ind = ( (xyz[:, 0] > SCAN_BOUNDS[0][0]) & (xyz[:, 0] < SCAN_BOUNDS[0][1]) 
                   & (xyz[:, 1] > SCAN_BOUNDS[1][0]) & (xyz[:, 1] < SCAN_BOUNDS[1][1]) 
                   & (xyz[:, 2] > SCAN_BOUNDS[2][0]) & (xyz[:, 2] < SCAN_BOUNDS[2][1]))
    good_pos_grain_mat = good_grain_mat[good_pos_ind, :]

    np.savetxt(os.path.join(path, savename + '.npos'), good_pos_grain_mat[:, [6,8,7]])
    
    if SAVE_ORI:
        np.save(os.path.join(path, savename + '_exp_maps.npy'), good_pos_grain_mat[:, 3:6])

    print('NOTE: UPDATE THESE COMMANDS IN THE NEPER TESSLATION SCRIPT')
    print('   NUMBER:   -n "%i"' %(good_pos_grain_mat.shape[0]))
    print('   DOMAIN:   -domain "cube(%.3f,%.3f,%.3f)"' %(SCAN_DIMEN[0], SCAN_DIMEN[2], SCAN_DIMEN[1]))
    print('   POS_LOAD:   -loadpoint "file("%s"):dim"' %(os.path.join(path, savename + '.npos')))
    print('   TESR_SIZE:   -tesrsize "%i:%i:%i"' %(SCAN_DIMEN[0] / VOXEL_SIZE, SCAN_DIMEN[2] / VOXEL_SIZE, SCAN_DIMEN[1] / VOXEL_SIZE))
    
    print('neper -T -n %i -dim 3 -domain "cube(%.3f,%.3f,%.3f)" -loadpoint "file("%s"):dim" \
    -reg 1 -mloop 10 -morpho voronoi -morphooptistop itermax=5000 -format tess,tesr \
    -tesrsize "%i:%i:%i" -tesrformat "ascii" -o $MICRO_FN_OUT' 
          %(good_pos_grain_mat.shape[0],
            SCAN_DIMEN[0], 
            SCAN_DIMEN[2], 
            SCAN_DIMEN[1],
            os.path.join(path, savename + '.npos'),
            SCAN_DIMEN[0] / VOXEL_SIZE, 
            SCAN_DIMEN[2] / VOXEL_SIZE, 
            SCAN_DIMEN[1] / VOXEL_SIZE))

# transform euler angles to kocks angles
def euler2kocks(euler):
    # euler angles (rad) to kocks angles (rad)
    kocks = np.vstack([np.mod(np.pi/2 - euler[:, 2], np.pi * 2), np.mod(euler[:, 1], np.pi * 2), np.mod(euler[:, 0] - np.pi/2, np.pi * 2)]).T
    return kocks

# create .kocks file
def create_kocks_file(path, out_f_name_kocks, exp_maps, grain_ids):
    num_grains = exp_maps.shape[0]
    
    # do rotation transformations to go from exp_maps to kocks
    rotmat = hexrd_rot.rotMatOfExpMap_opt(exp_maps.T)
    eulers = np.zeros([num_grains, 3])
    for i in range(num_grains):
        eulers[i] = hexrd_rot.angles_from_rmat_xyz(rotmat[i])
    kocks_rad = euler2kocks(eulers)
    
    # write output file
    out_f_kocks = open(os.path.join(path, out_f_name_kocks), "w")
    out_f_kocks.writelines('grain-orientations\n')
    out_f_kocks.writelines(str(num_grains) + '\n')
    
    kocks_deg = np.rad2deg(kocks_rad)
    
    write_orient = np.hstack([kocks_deg, grain_ids[:, np.newaxis]])
    np.savetxt(out_f_kocks, write_orient, fmt='%.8f \t%.8f \t%.8f \t%i')
    
    out_f_kocks.writelines('EOF')
    out_f_kocks.close()

# create .grain file
def create_grain_file(path, f_name_opt, out_f_name_grain):
    # open opt file, read lines, and remove first line
    with open(os.path.join(path, f_name_opt)) as f_opt:
        lines_opt = f_opt.readlines()
    
    # remove first and last line
    lines_opt.pop(0)
    lines_opt.pop(-1)
    
    # write grain file
    out_f_grain = open(os.path.join(path, out_f_name_grain), "w")
    out_f_grain.writelines(lines_opt)
    out_f_grain.close()

# create .mesh file
def create_mesh_file(path, f_name_params, f_name_mesh, out_f_name_mesh):
    # read params file
    f_params = open(os.path.join(path, f_name_params), "r")
    f_params_lines = f_params.readlines()
    f_params.close()
    
    # read mesh file
    f_mesh = open(os.path.join(path, f_name_mesh), "r")
    f_mesh_lines = f_mesh.readlines()
    f_mesh.close()
    
    f_mesh_lines.remove('  1.0   1.0   1.0\n')
    
    # write mesh file
    out_f_mesh = open(os.path.join(path, out_f_name_mesh), "w")
    out_f_mesh.write(f_params_lines[0])
    out_f_mesh.writelines(f_mesh_lines)
    out_f_mesh.close()

# create .matl file
def create_matl_file(path, out_f_name_matl, comp, c_11, c_12, c_44, crss_1, crss_2):
    # write matl file
    out_f_matl = open(os.path.join(path, out_f_name_matl), "w")
    out_f_matl.write('%i \n' %(comp))
    out_f_matl.write('%0.3e \t %0.3e \t %0.3e \n' %(c_11, c_12, c_44))
    out_f_matl.write('%0.3e \t %0.3e \n' %(crss_1, crss_2))
    out_f_matl.close()





