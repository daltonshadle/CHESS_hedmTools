#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:58:21 2021

@author: djs522
"""

#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


IN_PATH = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/neper/'
FILENAME_ff_top = 'ss718_ff_sc28.tesr'
FILENAME_nf_mid = 'ss718_total_tesr.tesr'
FILENAME_ff_bot = 'ss718_ff_sc32.tesr'


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

# [dimen, domain, voxel_size, num_grains, ids, voxels]
print('Reading top')
top_tesr_list = parse_tesr(IN_PATH + FILENAME_ff_top)
print('Reading mid')
mid_tesr_list = parse_tesr(IN_PATH + FILENAME_nf_mid)
print('Reading bot')
bot_tesr_list = parse_tesr(IN_PATH + FILENAME_ff_bot)
print('Done')


#%%

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

total_vox_map = combine_tesr([mid_tesr_list, top_tesr_list, bot_tesr_list], stack_dir=2, 
                             stack_order=[1, 0, 2])

#%%

NEPER_PATH = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/neper/'
top_exp_map = np.load(NEPER_PATH + 'ss718_sc28_neper_ori.npy')
mid_exp_map = np.load(NEPER_PATH + 'ss718_total_tesr.npz')['ORI_LIST']
bot_exp_map = np.load(NEPER_PATH + 'ss718_sc32_neper_ori.npy')

exp_map_list = np.vstack([mid_exp_map, top_exp_map, bot_exp_map])


#%%

print(total_vox_map.shape)

#temp_vox_map_djs = total_vox_map #np.array(mid_tesr_list[5]).reshape([96, 198, 198]).T #mid_tesr_list[1])
img = total_vox_map[:, :, 80]

fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.voxels(temp_vox_map_djs, edgecolor="k")
plt.imshow(img)
plt.show()



#%%
np.savez(NEPER_PATH + 'total_nf_with_ff_buffer_map.npz', grain_map=total_vox_map, ori_list=exp_map_list)


#%%

tot_grain_mat = np.zeros([mid_exp_map.shape[0], 21])
tot_grain_mat[:, 0] = np.arange(tot_grain_mat.shape[0])
tot_grain_mat[:, 1] = 1
tot_grain_mat[:, 9:12] = 1
tot_grain_mat[:, 3:6] = mid_exp_map

head = ['grain ID', 'completeness',  'chi^2', 'exp_map_c[0]', 'exp_map_c[1]', 'exp_map_c[2]', 't_vec_c[0]', 't_vec_c[1]',
        't_vec_c[2]', 'inv(V_s)[0,0]', 'inv(V_s)[1,1]', 'inv(V_s)[2,2]', 'inv(V_s)[1,2]*sqrt(2)', 'inv(V_s)[0,2]*sqrt(2)',
        'inv(V_s)[0,2]*sqrt(2)', 'ln(V_s)[0,0]', 'ln(V_s)[1,1]', 'ln(V_s)[2,2]', 'ln(V_s)[1,2]', 'ln(V_s)[0,2]', 
        'ln(V_s)[0,1]']
head = 'ID completeness   chi^2         exp_map_c[0]             exp_map_c[1]             exp_map_c[2]             t_vec_c[0]               t_vec_c[1]               t_vec_c[2]               inv(V_s)[0,0]            inv(V_s)[1,1]            inv(V_s)[2,2]            inv(V_s)[1,2]*sqrt(2)    inv(V_s)[0,2]*sqrt(2)    inv(V_s)[0,2]*sqrt(2)    ln(V_s)[0,0]             ln(V_s)[1,1]             ln(V_s)[2,2]             ln(V_s)[1,2]             ln(V_s)[0,2]             ln(V_s)[0,1]           '
fmt = ['%i', '%.6f', '%.6e'] + ['%.15e'] * 18
np.savetxt('/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf_analysis/ss718_actual_nf_grains.out', tot_grain_mat, fmt=fmt, header=head, delimiter='    ')



