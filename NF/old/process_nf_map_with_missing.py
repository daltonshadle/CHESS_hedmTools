#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:48:25 2021

@author: djs522
"""

#%% Necessary Dependencies

from __future__ import print_function

import sys

import time

import numpy as np

import matplotlib.pyplot as plt

import multiprocessing as mp

import os

from scipy import ndimage
from skimage import measure

from hexrd.grainmap import nfutil
from hexrd.grainmap import tomoutil
from hexrd.grainmap import vtkutil
from hexrd.xrd import rotations  as hexrd_rot
import hexrd.xrd.symmetry as hexrd_sym
from hexrd.xrd import transforms_CAPI as xfcapi

# *****************************************************************************
# %% CONSTANTS
# *****************************************************************************
pi = np.pi


#==============================================================================
# %% NF ADDITIONAL FUNCTIONS
#==============================================================================
def bunge2quat(bunge, units='radians'):
    '''
    % bunge2quat - Bunge Euler angles to quaternion conversion
    %   
    %   USAGE:
    %
    %   quat = bunge2quat(bunge, units='radians')
    %
    %   INPUT:
    %
    %   bunge is n x 3, 
    %        n Bunge Euler angle vectors for conversion
    %   units is string,
    %        gives units in 'radians' or 'degrees'
    %
    %   OUTPUT:
    %
    %   quat is n x 4, 
    %        returns an array of n quaternions
    %
    %   NOTES:  
    %
    %   *  None
    %
    '''
    
    if (units == 'degrees'):
        indeg = True
    elif (units == 'radians'):
        indeg = False
    else:
        raise ValueError('angle units need to be specified:  ''degrees'' or ''radians''')
    
    if (indeg):
        bunge = np.deg2rad(bunge)
        
    # phi1 = bunge[:, 0]
    # theta = bunge[:, 1]
    # phi2 = bunge[:, 2]
    
    sigma = 0.5 * (bunge[:, 0] + bunge[:, 2])
    delta = 0.5 * (bunge[:, 0] - bunge[:, 2])
    c = np.cos(bunge[:, 1] / 2)
    s = np.sin(bunge[:, 1] / 2)
    
    quat = np.vstack([c * np.cos(sigma), -s * np.cos(delta), -s * np.sin(delta), -c * np.sin(sigma)]).T
    return quat

def quat2rod(quat):
    """
    % quat2rod - Rodrigues parameterization from quaternion.
    %
    %   USAGE:
    %
    %   rod = quat2rod(quat)
    %
    %   INPUT:
    %
    %   quat is n x 3,
    %        an array whose columns are quaternion paramters;
    %        it is assumed that there are no binary rotations
    %        (through 180 degrees) represented in the input list
    %
    %   OUTPUT:
    %
    %  rod is n x 3,
    %      an array whose columns form the Rodrigues parameterization
    %      of the same rotations as quat
    %
    """
    return np.true_divide(quat[:, 1:4], np.tile(quat[:, 0], (3, 1)).T)

def quat2exp_map(quat):
    '''
    % quat2exp_map - quaternion to exponential map conversion
    %   
    %   USAGE:
    %
    %   exp_map = quat2exp_map(quat)
    %
    %   INPUT:
    %
    %   quat is n x 4, 
    %        n quaternions for conversion
    %
    %   OUTPUT:
    %
    %   exp_map is n x 3, 
    %        returns an array of n exponential maps
    %
    %   NOTES:  
    %
    %   *  None
    %
    '''
    
    phi = 2 * np.arccos(quat[:, 0])
    norm = xfcapi.unitRowVector(quat[:, 1:])
    
    exp_map = norm * phi[:, None]
    
    return exp_map

def discretizeFundamentalRegion(phi1_bnd=[0, 180], phi1_step=1,
                                theta_bnd=[0, 90], theta_step=1,
                                phi2_bnd=[0, 180], phi2_step=1,
                                crys_sym='cubic', ret_type='quat'):
    '''
    % discretizeFundamentalRegion - Bunge Euler angles to dicretize fundamental region
    %   
    %   USAGE:
    %
    %   orientation_mat = discretizeFundamentalRegion(
    %                           phi1_bnd=[0, 90], phi1_step=1,
    %                           theta_bnd=[0, 90], theta_step=1,
    %                           phi2_bnd=[0, 90], phi2_step=1,
    %                           crys_sym='cubic', ret_type='quat'
    %
    %   INPUT:
    %
    %   phi1_bnd is 1 x 2, 
    %        1st Bunge angle bounds (in degrees)
    %   phi1_step is float,
    %        1st Bunge angle step size (in degrees)
    %   theta_bnd is 1 x 2, 
    %        2nd Bunge angle bounds (in degrees)
    %   theta_step is float,
    %        2nd Bunge angle step size (in degrees)
    %   phi2_bnd is 1 x 2, 
    %        3rd Bunge angle bounds (in degrees)
    %   phi2_step is float,
    %        3rd Bunge angle step size (in degrees)
    %   crys_sym is string, 
    %        determines the crystal symmetry type (e.g. cubic, hex)
    %   ret_type is string,
    %        determines the return type of the orientation (e.g. quat, rod)
    %
    %   OUTPUT:
    %
    %   discrete_FR_ret is m x n, 
    %        the array of orientations given by m (number of elements in unit orientation
    %        representation) and n (number of orientations)
    %
    %   NOTES:  
    %
    %   *  None
    %
    '''
    
    
    if crys_sym is 'cubic':
        phi1_mat, theta_mat, phi2_mat = np.meshgrid(np.arange(phi1_bnd[0], phi1_bnd[1] + phi1_step, phi1_step),
                                                    np.arange(theta_bnd[0], theta_bnd[1] + theta_step, theta_step),
                                                    np.arange(phi2_bnd[0], phi2_bnd[1] + phi2_step, phi2_step))
        discrete_FR_bunge = np.vstack([phi1_mat.flatten(), theta_mat.flatten(), phi2_mat.flatten()]).T
        
        discrete_FR_quat = bunge2quat(discrete_FR_bunge, units='degrees')
        discrete_FR_quat = hexrd_sym.toFundamentalRegion(discrete_FR_quat.T, crysSym='Oh').T
        discrete_FR_ret = discrete_FR_quat
        
        if ret_type is 'rod':
            discrete_FR_ret = quat2rod(discrete_FR_quat)
        
        discrete_FR_ret  = np.round(discrete_FR_ret, decimals=8)
        discrete_FR_ret = np.unique(discrete_FR_ret, axis=0)
        return discrete_FR_ret
    else:
        print('Crystal symmetry type is not supported at this time.')



def new_experiment_ori(quat, experiment):
    '''
    Input:
        quat (n x 4) - quaternion orientation representation to overwrite with
        experiment (object) - experiment object to overwrite
    Output:
        experiment (object) - new experiment object with new orientations
    '''
    
    experiment.n_grains = quat.shape[0]
    experiment.rMat_c = hexrd_rot.rotMatOfQuat(quat.T)
    experiment.exp_maps = quat2exp_map(quat)
    
    return experiment



#==============================================================================
# %% NF-IMAGE CLEAN IMAGE FUNCTION
#==============================================================================
import skimage.io as img
import skimage.filters as filters
import scipy.ndimage.morphology as morphology
from skimage import feature
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank
from joblib import Parallel, delayed

def gen_nf_cleaned_image_stack(data_folder,img_nums,dark,ome_dilation_iter,threshold,nrows,ncols,stem='nf_',num_digits=5,ext='.tif',non_peak_thresh=1.7,gaussian_sigma=3):
    num_imgs = img_nums.shape[0]
    image_stack=np.zeros([num_imgs,nrows,ncols],dtype=bool)

    print('Loading and Cleaning Images...')
    for ii in np.arange(num_imgs):
        sys.stdout.write('\rImage #: %i/%i' %(ii+1, num_imgs))
        sys.stdout.flush()
        tmp_img=img.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
        #image procesing
        
        if threshold == -1:
            im = tmp_img
        else:
            tmp_img[tmp_img < threshold] = 0
            im = filters.gaussian(tmp_img, sigma=gaussian_sigma) #gaussian filter, may need to change higher for deformed peaks (3-6)
            im = morphology.grey_closing(im,size=(3,3)) #grey filter, smooths mins
        where_remove = np.where(im<non_peak_thresh) #identify non-peak region, will need to be lower for deformed peaks (0.5-1.5)
        im[where_remove] = 0 #set non-peak region to 0
        binary_img = morphology.binary_fill_holes(im) #make image binary and  fill holes
        image_stack[ii,:,:]=binary_img #put into image stack
        
    #%A final dilation that includes omega
    print('Final Dilation Including Omega....')
    image_stack=morphology.binary_dilation(image_stack,iterations=ome_dilation_iter)
    
    return image_stack

def gen_nf_cleaned_image_stack_mp(data_folder,img_nums,dark,ome_dilation_iter,threshold,nrows,ncols,stem='nf_',num_digits=5,ext='.tif',non_peak_thresh=1.7,gaussian_sigma=3,ncpus=1):
    num_imgs = img_nums.shape[0]
    image_stack = np.zeros([num_imgs,nrows,ncols],dtype=float)
    ret_image_stack = np.zeros([num_imgs,nrows,ncols],dtype=bool)
    
    if num_imgs < 50:
        ncpus=1
    print('Loading Images...')
    for ii in np.arange(num_imgs):
        sys.stdout.write('\rImage #: %i/%i' %(ii+1, num_imgs))
        sys.stdout.flush()
        image = img.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
        image[image <= threshold] = 0
        image_stack[ii,:,:] = image
    print('\n')
    print('Cleaning Images...')
    results = Parallel(n_jobs=ncpus, verbose=2)(delayed(gen_nf_cleaned_image)(image_stack[idx, :, :], non_peak_thresh, gaussian_sigma) for idx in range(num_imgs))
    image_stack = np.array(results)
    
    #%A final dilation that includes omega
    print('Final Dilation Including Omega....')
    ret_image_stack=morphology.binary_dilation(image_stack,iterations=ome_dilation_iter)
    
    iso_struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for i in range(ret_image_stack.shape[0]):
        conv_mask = ndimage.convolve(ret_image_stack[i, :, :].astype(int), iso_struct) > 3
        ret_image_stack[i, :, :] = ret_image_stack[i, :, :].astype(int) * conv_mask.astype(int)
    
    return ret_image_stack

def gen_nf_cleaned_image(image, non_peak_thresh=1.7, gaussian_sigma=3):    
    #image procesing
    #    im = filters.gaussian(image, sigma=gaussian_sigma) #gaussian filter, may need to change higher for deformed peaks (3-6)
    #    im = morphology.grey_closing(im,size=(5,5)) #grey filter, smooths mins
    #    where_remove = np.where(im<non_peak_thresh) #identify non-peak region, will need to be lower for deformed peaks (0.5-1.5)
    #    im[where_remove] = 0 #set non-peak region to 0
    #    binary_img = morphology.binary_fill_holes(im) #make image binary and  fill holes
    
    #    im = np.copy(image)
    #    im = morphology.grey_erosion(im,size=(2,2))
    #    im = morphology.grey_dilation(im,size=(2,2))
    #    
    #    gauss = 4.0
    #    can_thresh = 0.05
    #    im = feature.canny(im, sigma=gauss, low_threshold=can_thresh, high_threshold=can_thresh)
    #    
    #    structure = np.ones([3,3])
    #    im = morphology.binary_closing(im, iterations=2, structure=structure) 
    #    im = morphology.binary_dilation(im, iterations=1, structure=structure) 
    #    
    #    binary_img = morphology.binary_fill_holes(im) 
    
    binary_img = (image > 0.0)

    iso_struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    conv_mask = ndimage.convolve(binary_img.astype(int), iso_struct) > 2
    binary_img = binary_img.astype(int) * conv_mask.astype(int)
    
    iso_struct = np.ones([4,4])
    conv_mask = ndimage.convolve(binary_img.astype(int), iso_struct) > 3
    binary_img = binary_img.astype(int) * conv_mask.astype(int)
    
    binary_img = morphology.binary_closing(binary_img, iterations=1) 
    binary_img = morphology.binary_fill_holes(binary_img) 
    
    float_img = np.copy(image)
    float_img = float_img / float(float_img.max())
    float_img[np.logical_not(binary_img)] = 0
    
    selem = disk(20)
    float_img = rank.autolevel(float_img, selem=selem)
    float_img = float_img / float(255.0)
    
    blur1 = filters.gaussian(float_img, sigma=4.0) #4.0
    num1 = float_img - blur1
    
    blur2 = filters.gaussian(num1*num1, sigma=20.0) #20.0
    den2 = np.power(blur2, 0.5)
    
    float_img = num1 / den2
    
    binary_img = (float_img > 0.0)

    iso_struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    conv_mask = ndimage.convolve(binary_img.astype(int), iso_struct) > 2
    binary_img = binary_img.astype(int) * binary_img.astype(int)
    
    binary_img = morphology.binary_closing(binary_img, iterations=1) 
    binary_img = morphology.binary_fill_holes(binary_img) 
    
    return binary_img
    
def gen_nf_cleaned_image_stack_mp_inten(data_folder,img_nums,dark,threshold,nrows,ncols,stem='nf_',num_digits=5,ext='.tif',ncpus=1):
    num_imgs = img_nums.shape[0]
    image_stack = np.zeros([num_imgs,nrows,ncols],dtype=float)
    ret_image_stack = np.zeros([num_imgs,nrows,ncols],dtype=np.uint8)
    
    if num_imgs < 50:
        ncpus=1
    print('Loading Images...')
    for ii in np.arange(num_imgs):
        sys.stdout.write('\rImage #: %i/%i' %(ii+1, num_imgs))
        sys.stdout.flush()
        image = img.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
        image_stack[ii,:,:] = image
    print('\n')
    print('Cleaning Images...')
    results = Parallel(n_jobs=ncpus, verbose=2)(delayed(gen_nf_cleaned_image_inten)(image_stack[idx, :, :], threshold) for idx in range(num_imgs))
    image_stack = np.array(results)
    
    #%A final dilation that includes omega
    print('Final Dilation Including Omega....')
    ret_image_stack = morphology.grey_dilation(image_stack, footprint=np.ones([3,1,1]))
    
    return ret_image_stack.astype(np.uint8)

def gen_nf_cleaned_image_inten(image, threshold=3):    
    #image procesing
    
    # copy and account for beam stop
    float_img = np.copy(image)
    float_img[842:1242, :] = 0
    
    # create a binary mask
    binary_img = (float_img > threshold)

    iso_struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    conv_mask = ndimage.convolve(binary_img.astype(int), iso_struct) > 2
    binary_img = binary_img.astype(int) * conv_mask.astype(int)
    
    iso_struct = np.ones([4,4])
    conv_mask = ndimage.convolve(binary_img.astype(int), iso_struct) > 3
    binary_img = binary_img.astype(int) * conv_mask.astype(int)
    
    binary_img = morphology.binary_closing(binary_img, iterations=1) 
    binary_img = morphology.binary_fill_holes(binary_img) 
    
    # apply binary mask
    float_img[np.logical_not(binary_img)] = 0
    
    # normalize
    float_img = float_img / float(float_img.max())
    
    # apply autolevel
    selem = disk(30)
    float_img = rank.autolevel(float_img, selem=selem)
    
    # re-normalize
    uint8_img = (float_img / float(float_img.max()) * 255.0).astype(int)
    
    return uint8_img.astype(np.uint8)

#==============================================================================
# %% FILES TO LOAD -CAN BE EDITED
#==============================================================================
#These files are attached, retiga.yml is a detector configuration file
#The near field detector was already calibrated

#A materials file, is a cPickle file which contains material information like lattice
#parameters necessary for the reconstruction

det_file='/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf_analysis/retiga.yml'
mat_file='/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf_analysis/materials_dp718_36000.cpl'

print(det_file)
print(mat_file)
#==============================================================================
# %% OUTPUT INFO -CAN BE EDITED
#==============================================================================

# scan number to be processed
# nf: scan 8 = layer 1, scan 9 = layer 2, scan 10 = layer 3
# ff: scan 18 = layer 1, scan 19 = layer 2, scan 20 = layer 3
my_scan_num = 18

# set up output directory and filename
output_dir = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf_analysis/output/new_dp718_nf_missing/'
output_stem = 'dp718_scan%s_out' %(my_scan_num)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# location of grains.out folder
grain_out_file='/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_0_gripped/c0_0_gripped_sc%i_nf/grains.out' %(my_scan_num)

if my_scan_num == 18:
    # location of nf data images
    data_folder='/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf/8/nf/'
    img_start=10337#for 0.25 degree/steps and 5 s exposure, end up with 0 junk frames up front, this is the 6th
    
elif my_scan_num == 19:
    # location of nf data images
    data_folder='/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf/9/nf/'
    img_start=11784#for 0.25 degree/steps and 5 s exposure, end up with 0 junk frames up front, this is the 6th
    
elif my_scan_num == 20:
    # location of nf data images
    data_folder='/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf/10/nf/'
    img_start=13231#for 0.25 degree/steps and 5 s exposure, end up with 0 junk frames up front, this is the 6th


# TOMO IS ONLY [0, 180] in omega
# location of tomo bright field images
tbf_data_folder='/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf/12/nf/'
tbf_img_start=14703 #for this rate, this is the 6th file in the folder
tbf_num_imgs=10

#Locations of tomography images
tomo_data_folder='/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf/14/nf/'
tomo_img_start=14963 #for this rate, this is the 6th file in the folder
tomo_num_imgs=720

num_imgs=1441
img_nums=np.arange(img_start,img_start+num_imgs,1)
    

print('grains.out: %s' %(grain_out_file))
print('nf data: %s' %(data_folder))
print('tomo bf: %s' %(tbf_data_folder))
print('tomo data: %s' %(tomo_data_folder))

#==============================================================================
# %% USER OPTIONS -CAN BE EDITED
#==============================================================================

x_ray_energy=61.332 #keV

#name of the material for the reconstruction
mat_name='dp718-2'

#reconstruction with misorientation included, for many grains, this will quickly
#make the reconstruction size unmanagable
misorientation_bnd=0.0 #degrees 
misorientation_spacing=1.0 #degrees

beam_stop_width=0.6#mm, assumed to be in the center of the detector


ome_range_deg=[(0.,360.)] #degrees 


max_tth=12. #degrees, if a negative number is input, all peaks that will hit the detector are calculated

#image processing
do_nf_image_clean = True
#DO NOT TOUCH
num_for_dark=250#num images to use for median data
threshold=2.0
num_erosions=3 #num iterations of images erosion, don't mess with unless you know what you're doing
num_dilations=2 #num iterations of images erosion, don't mess with unless you know what you're doing
ome_dilation_iter=1. #num iterations of 3d image stack dilations, don't mess with unless you know what you're doing

chunk_size=500#chunksize for multiprocessing, don't mess with unless you know what you're doing

#thresholds for grains in reconstructions
comp_thresh=0.3 #only use orientations from grains with completnesses ABOVE this threshold
chi2_thresh=1.0 #only use orientations from grains BELOW this chi^2


#tomography options
layer_row=1024 # row of layer to use to find the cross sectional specimen shape
recon_thresh=0.0005#usually varies between 0.0001 and 0.0005
#Don't change these unless you know what you are doing, this will close small holes
#and remove noise
noise_obj_size=500
min_hole_size=500


cross_sectional_dim=1.3 #cross sectional to reconstruct (should be at least 20%-30% over sample width)
#voxel spacing for the near field reconstruction
voxel_spacing=0.005#in mm
#vertical (y) reconstruction voxel bounds in mm
v_bnds=[-0.15,0.1]
#v_bnds=[0,0]


ncpus=66

# debugging parameters
do_param_opt = False
show_images = True
save_confidence_map = False
save_grain_map = True
save2vtk = True
do_brute_force_ori = False

#==============================================================================
# %% LOAD GRAIN AND EXPERIMENT DATA
#==============================================================================
status = 'EXP GENERATION'
print('STARTED %s' %(status))
experiment, nf_to_ff_id_map  = nfutil.gen_trial_exp_data(grain_out_file,det_file,mat_file, 
                                                         x_ray_energy, mat_name, max_tth, comp_thresh, 
                                                         chi2_thresh, misorientation_bnd,
                                                         misorientation_spacing,ome_range_deg, 
                                                         num_imgs, beam_stop_width)

print('FINISHED %s' %(status))
#==============================================================================
# %% TOMO PROCESSING - GENERATE BRIGHT FIELD
#==============================================================================
status = 'GENERATING BRIGHT-FIELD'
print('STARTED %s' %(status))
tbf=tomoutil.gen_bright_field(tbf_data_folder,tbf_img_start,tbf_num_imgs,experiment.nrows,experiment.ncols,num_digits=6)
print('FINISHED %s' %(status))

#==============================================================================
# %% TOMO PROCESSING - BUILD RADIOGRAPHS
#==============================================================================
status = 'GENERATING ATTENUATION'
print('STARTED %s' %(status))

tomo_stack_file = 'tomo_rad_stack.npy'
load_tomo_stack = False

if load_tomo_stack:
    rad_stack = np.load(tomo_stack_file)
else:
    rad_stack=tomoutil.gen_attenuation_rads(tomo_data_folder,tbf,tomo_img_start,tomo_num_imgs,experiment.nrows,experiment.ncols,num_digits=6)

inf_pts=np.isinf(rad_stack)
rad_stack[inf_pts]=0.

save_rad_stack = True
if save_rad_stack:
    np.save(tomo_stack_file, rad_stack)
print('FINISHED %s' %(status)) 

#==============================================================================
# %% TOMO PROCESSING - INVERT SINOGRAM
#==============================================================================
djs_start = -0.0
ome_range_deg=[(djs_start, djs_start + 180.)]
    
status = 'GENERATING TOMO RECONSTRUCTION'
print('STARTED %s' %(status))
reconstruction_fbp=tomoutil.tomo_reconstruct_layer(rad_stack,cross_sectional_dim,layer_row=layer_row,\
                                                   start_tomo_ang=ome_range_deg[0][0],end_tomo_ang=ome_range_deg[0][1],\
                                                   tomo_num_imgs=tomo_num_imgs, center=experiment.detector_params[3])
print('FINISHED %s' %(status)) 

##==============================================================================
# %% TOMO PROCESSING - VIEW RAW FILTERED BACK PROJECTION
##==============================================================================
#
if show_images:
    #plt.close('all')
    fig = plt.figure()
    plt.imshow(reconstruction_fbp,vmin=0.6e-3,vmax=2e-3)
    plt.show()
#Use this image to view the raw reconstruction, estimate threshold levels. and
#figure out if the rotation axis position needs to be corrected

#==============================================================================
# %% TOMO PROCESSING - CLEAN TOMO RECONSTRUCTION
#==============================================================================
status = 'THRESHOLD AND CLEAN TOMO'
print('STARTED %s' %(status))
binary_recon=tomoutil.threshold_and_clean_tomo_layer(reconstruction_fbp,recon_thresh, noise_obj_size,min_hole_size)
print('FINISHED %s' %(status)) 
#==============================================================================
# %%  TOMO PROCESSING - RESAMPLE TOMO RECONSTRUCTION
#==============================================================================
status = 'CROP TOMO'
print('STARTED %s' %(status))

tomo_mask_file = 'tomo_mask.npy'

load_tomo_mask = True
if load_tomo_mask:
    tomo_mask = np.load(tomo_mask_file)
else:
    tomo_mask=tomoutil.crop_and_rebin_tomo_layer(binary_recon,recon_thresh,voxel_spacing,experiment.pixel_size[0],cross_sectional_dim)

save_tomo_mask = False
if save_tomo_mask:
    np.save(tomo_mask_file, tomo_mask)


print('FINISHED %s' %(status))
#==============================================================================
# %%  TOMO PROCESSING - VIEW TOMO_MASK FOR SAMPLE BOUNDS
#==============================================================================
if show_images:
    plt.figure(1)
    plt.imshow(tomo_mask,interpolation='none')

#==============================================================================
# %%  TOMO PROCESSING - CONSTRUCT DATA GRID
#==============================================================================
status = 'GENERATING GRID COORD FOR NF'
print('STARTED %s' %(status))
use_tomo_test_coords = False # both do the basically the same thing

if use_tomo_test_coords:
    test_crds, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid_tomo(tomo_mask.shape[1], tomo_mask.shape[0], v_bnds, voxel_spacing)
else:
    if v_bnds[0]==v_bnds[1]:
        Xs,Ys,Zs=np.meshgrid(np.arange(0.,cross_sectional_dim+voxel_spacing,voxel_spacing),v_bnds[0],np.arange(0.,cross_sectional_dim+voxel_spacing,voxel_spacing))        
    else:
        Xs,Ys,Zs=np.meshgrid(np.arange(0.,cross_sectional_dim+voxel_spacing,voxel_spacing),np.arange(v_bnds[0]+voxel_spacing/2.,v_bnds[1],voxel_spacing),np.arange(0.,cross_sectional_dim+voxel_spacing,voxel_spacing))
        #note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))
    
    Zs=(Zs-cross_sectional_dim/2.)
    Xs=(Xs-cross_sectional_dim/2.)
    
    test_crds = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    n_crds = len(test_crds)
print('FINISHED %s' %(status))
#==============================================================================
# %% NEAR FIELD - MAKE MEDIAN DARK
#==============================================================================
status = 'GENERATING NF DARK IMAGE'
print('STARTED %s' %(status))

nf_dark_file = 'nf_dark_ffsc%i.npy' %my_scan_num

load_dark = True
if load_dark:
    dark = np.load(nf_dark_file)
else:
    dark = nfutil.gen_nf_dark(data_folder,img_nums,num_for_dark,experiment.nrows,experiment.ncols,num_digits=6)

save_dark = False
if save_dark:
    np.save(nf_dark_file, dark)
print('FINISHED %s' %(status))

fig = plt.figure()
plt.imshow(dark, vmax=20)
plt.show()

#==============================================================================
# %% NEAR FIELD - LOAD IMAGE DATA AND PROCESS
#==============================================================================
status = 'GENERATING NF IMAGE STACK'
print('STARTED %s' %(status))

non_peak_thresh = 3.5
threshold = 3.0
gauss_sigma = 1.4
num_erosions=3 #num iterations of images erosion, don't mess with unless you know what you're doing
num_dilations=3 #num iterations of images erosion, don't mess with unless you know what you're doing
ome_dilation_iter=1 #num iterations of 3d image stack dilations, don't mess with unless you know what you're doing

nf_image_stack_file = 'nf_image_stack_ffsc%i_nonpeakthresh%0.2f_thresh%0.2f.npy' %(my_scan_num, non_peak_thresh, threshold)
nf_image_stack_file = 'nf_image_stack_ffsc%i_inten.npy' %(my_scan_num)

load_image_stack = False
if load_image_stack:
    image_stack = np.load(nf_image_stack_file)
else:
    if do_nf_image_clean:
        start = time.time()
        # multi-processing clean nf images
        #image_stack = gen_nf_cleaned_image_stack_mp(data_folder,img_nums,dark,ome_dilation_iter,threshold,experiment.nrows,experiment.ncols,num_digits=6,non_peak_thresh=non_peak_thresh,gaussian_sigma=gauss_sigma,ncpus=ncpus)
        # serial processing clean nf images
        #image_stack = gen_nf_cleaned_image_stack(data_folder,img_nums[[0, 100, 200]],dark,ome_dilation_iter,threshold,experiment.nrows,experiment.ncols,stem='nf_',num_digits=6,non_peak_thresh=non_peak_thresh,gaussian_sigma=gauss_sigma)
        
        image_stack = gen_nf_cleaned_image_stack_mp_inten(data_folder,img_nums,dark,threshold,experiment.nrows,experiment.ncols,num_digits=6,ncpus=ncpus)
        
        end = time.time()
        print(end - start)
    else:
        image_stack = nfutil.gen_nf_image_stack(data_folder,img_nums,dark,num_erosions,num_dilations,ome_dilation_iter,threshold,experiment.nrows,experiment.ncols,num_digits=6)

save_image_stack = True
if save_image_stack:
    np.save(nf_image_stack_file, image_stack)

print('\n')
print('FINISHED %s' %(status))

#%%

im_num = 180

stem='nf_'
num_digits=6
ext='.tif'
tmp_img=img.imread(data_folder+'%s'%(stem)+str(img_nums[im_num]).zfill(num_digits)+ext)


non_peak_thresh = 1.1
threshold = 1.0
gauss_sigma = 2.0

num_erosions=3 #num iterations of images erosion, don't mess with unless you know what you're doing
num_dilations=3 #num iterations of images erosion, don't mess with unless you know what you're doing
ome_dilation_iter=1 #num iterations of 3d image stack dilations, don't mess with unless you know what you're doing
my_img1 = gen_nf_cleaned_image_stack(data_folder,img_nums[[im_num, im_num+1]],dark,ome_dilation_iter,threshold,experiment.nrows,experiment.ncols,stem='nf_',num_digits=6,non_peak_thresh=non_peak_thresh,gaussian_sigma=gauss_sigma) 

plt.figure()
plt.imshow(my_img1[0, :, :])


non_peak_thresh = 3.5
threshold = 1.0
gauss_sigma = 0.75

num_erosions=3 #num iterations of images erosion, don't mess with unless you know what you're doing
num_dilations=3 #num iterations of images erosion, don't mess with unless you know what you're doing
ome_dilation_iter=1 #num iterations of 3d image stack dilations, don't mess with unless you know what you're doing
my_img2 = gen_nf_cleaned_image_stack(data_folder,img_nums[[im_num, im_num+1]],dark,ome_dilation_iter,threshold,experiment.nrows,experiment.ncols,stem='nf_',num_digits=6,non_peak_thresh=non_peak_thresh,gaussian_sigma=gauss_sigma) 

plt.figure()
plt.imshow(my_img2[0, :, :])

plt.figure()
plt.imshow(my_img2[0, :, :].astype(int) - my_img1[0, :, :].astype(int) )

#plt.figure()
#plt.imshow(tmp_img-dark, vmin=0, vmax=5)

plt.show()

#==============================================================================
# %% VIEW IMAGES FOR DEBUGGING TO LOOK AT IMAGE PROCESSING PARAMETERS
#==============================================================================
if show_images:
    img_to_view=[2, 200, 400, 899, 941] # check a lot of these
    #img_to_view=[0, 1, 2] # check a lot of these
    for temp_img in img_to_view:
        plt.figure()
        plt.imshow(image_stack[temp_img,:,:],interpolation='none')
        plt.draw()
    plt.show()

#==============================================================================
# %% INSTANTIATE CONTROLLER - RUN BLOCK NO EDITING
#==============================================================================
status = 'SETTING UP MULTI-PROCESSING'
print('STARTED %s' %(status))
progress_handler = nfutil.progressbar_progress_observer()
save_handler = nfutil.forgetful_result_handler()
    
controller = nfutil.ProcessController(save_handler, progress_handler,
                               ncpus=ncpus, chunk_size=chunk_size)

multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'
print('FINISHED %s' %(status))

#==============================================================================
# %% TEST ORIENTATIONS - RUN BLOCK NO EDITING
#==============================================================================
status = 'GENEARTING NF CONFIDENCE MAP'
print('STARTED %s' %(status))
overall_start_time = time.ctime(time.time())
print("Local time:", overall_start_time)	

# process nf images and orientations for this round
raw_confidence=nfutil.test_orientations(image_stack, experiment, test_crds,
   controller,multiprocessing_start_method)

local_time = time.ctime(time.time())
print("Local time:", local_time)
print('FINISHED %s' %(status))

#%%

#temp_raw_conf = np.copy(raw_confidence)
#new_shape = np.hstack([raw_confidence.shape[0], Xs.shape])
#temp_raw_conf = np.reshape(temp_raw_conf, new_shape)

top_n_confidence, top_n_index = nfutil.save_raw_confidence_n(raw_confidence, output_dir, output_stem, n=2)
del top_n_confidence
del top_n_index

#==============================================================================
# %% POST PROCESS W WHEN TOMOGRAPHY HAS BEEN USED
#==============================================================================
status = 'PROCESSING NF CONFIDENCE MAP'
print('STARTED %s' %(status))

use_tomo_mask = False

if use_tomo_mask:
    grain_map, confidence_map = nfutil.process_raw_confidence(raw_confidence,Xs.shape,tomo_mask=tomo_mask,id_remap=nf_to_ff_id_map)
else:
    grain_map, confidence_map = nfutil.process_raw_confidence(raw_confidence,Xs.shape,id_remap=nf_to_ff_id_map)

print('FINISHED %s' %(status)) 

#==============================================================================
# %% SAVE RAW CONFIDENCE FILES 
#============================================================================

#This will be a very big file, don't save it if you don't need it
if save_confidence_map:
    status = 'SAVING CONFIDENCE MAP'
    print('STARTED %s' %(status))
    nfutil.save_raw_confidence(output_dir,output_stem,raw_confidence)
    print('FINISHED %s' %(status)) 


#==============================================================================
# %% SAVE PROCESSED GRAIN MAP DATA
#==============================================================================
if save_grain_map:
    status = 'SAVING NF MAP'
    print('STARTED %s' %(status))
    nfutil.save_nf_data(output_dir,output_stem,grain_map,confidence_map,Xs,Ys,Zs,experiment.exp_maps)
    print('FINISHED %s' %(status)) 

#==============================================================================
# %% SAVE DATA AS VTK
#==============================================================================
if save2vtk:
    status = 'SAVING NF MAP TO VTK'
    print('STARTED %s' %(status))
    vtkutil.output_grain_map_vtk(output_dir,[output_stem],output_stem,0.1)
    print('FINISHED %s' %(status))

#==============================================================================
# %% DETECTOR CALIBRATION
#==============================================================================
if do_param_opt:
    parm_to_opt=3 #0 is distance, 1 is rotation axis position, 2 is xtilt, 3 is ytilt, 4 ztilt
    #degree values are going to be in radians, positions in mm
    parm_vector=np.linspace(-0.01, 0.01, 10, endpoint=True)
    #parm_vector=np.arange(-9.2,-9.45,-0.05)
    print(parm_vector)
    test=nfutil.scan_detector_parm(image_stack, experiment,test_crds,controller,parm_to_opt,parm_vector,[grain_map.shape[1],grain_map.shape[2]])
      
    my_int_thresh = 1.0
    for i in range(parm_vector.size):
        param_im = np.squeeze(test[i,:,:])
        plt.figure()
        plt.imshow(param_im,vmin=0.5,vmax=my_int_thresh)
        plt.title(parm_vector[i])
        gy, gx = np.gradient(param_im)
        sharpness = np.average(np.sqrt(gx**2 + gy**2))
        print('parm vector: %f' %(parm_vector[i]))
        print('Image sharpness: %f' %(sharpness))
        print('Percent above threshold: %f %%' %(100 * np.sum(param_im>my_int_thresh)/float(np.size(param_im))))
        print('sum: %f' %(np.sum(np.sum(param_im>0.5))))

#==============================================================================
# %% PLOTTING SINGLE LAYERS FOR DEBUGGING
#==============================================================================
if show_images:
    layer_no = 25
    plt.figure(2)
    #nfutil.plot_ori_map(grain_map, confidence_map, experiment.exp_maps, layer_no)
    
    m = 2
    temp_tomo_mask = ndimage.binary_erosion(tomo_mask, structure=np.ones((m,m)))
    plt.imshow(temp_tomo_mask,interpolation='none')
    
    map_angle = -2.25 #-2.25
    
    im_layer = confidence_map[layer_no]
    im_layer = temp_tomo_mask
    im_layer = ndimage.rotate(im_layer, angle=map_angle, reshape=False, order=0)
    #temp_tomo_mask = ndimage.rotate(temp_tomo_mask, angle=map_angle, reshape=False, order=0)
    
    plt.figure(3)
    plt.imshow(im_layer)
    
    voxel_spacing = 0.005
    space = 0
    samp_pix = int(1 / voxel_spacing)
    z_start = 28 - space
    x_start = 33 - space
    z_end = z_start + samp_pix + space*2
    x_end = x_start + samp_pix + space*2
    
    print(x_start, x_end, z_start, z_end)
    
    temp_mask = np.zeros(im_layer.shape)
    temp_mask[x_start:x_end, z_start:z_end] = 1
    
    plt.figure(4)
    plt.imshow(temp_mask - im_layer)
    
    plt.show()
    
#    my_int_thresh = 0.5
#    gy, gx = np.gradient(im_layer)
#    sharpness = np.average(np.sqrt(gx**2 + gy**2))
#    print('Image sharpness: %f' %(sharpness))
#    print('Percent above threshold: %f %%' %(100 * np.sum(im_layer>my_int_thresh)/float(np.size(im_layer))))


#%%

overwrite_now = True
if overwrite_now:
    old_grain_map = np.copy(grain_map)
    old_conf_map = np.copy(confidence_map)
    old_Xs = np.copy(Xs)
    old_Ys = np.copy(Ys)
    old_Zs = np.copy(Zs)

#%%

grain_map = np.copy(old_grain_map)
confidence_map = np.copy(old_conf_map)
Xs = np.copy(old_Xs)
Ys = np.copy(old_Ys)
Zs = np.copy(old_Zs)


#%%
do_map_fix = True

top_ind = 10
bot_ind = top_ind + 35 # 0.17 / 0.005

if do_map_fix:
    grain_map = ndimage.rotate(grain_map, angle=map_angle, axes=[1,2], reshape=False, order=0) 
    confidence_map = ndimage.rotate(confidence_map, angle=map_angle, axes=[1,2], reshape=False, order=0)
    Xs = ndimage.rotate(Xs, angle=map_angle, axes=[1,2], reshape=False, order=0) 
    Ys = ndimage.rotate(Ys, angle=map_angle, axes=[1,2], reshape=False, order=0) 
    Zs = ndimage.rotate(Zs, angle=map_angle, axes=[1,2], reshape=False, order=0) 
    
    
    grain_map = grain_map[top_ind:bot_ind, x_start:x_end, z_start:z_end]
    confidence_map = confidence_map[top_ind:bot_ind, x_start:x_end, z_start:z_end]
    Xs = Xs[top_ind:bot_ind, x_start:x_end, z_start:z_end]
    Ys = Ys[top_ind:bot_ind, x_start:x_end, z_start:z_end]
    Zs = Zs[top_ind:bot_ind, x_start:x_end, z_start:z_end]

print(grain_map.shape)
red_output_stem = output_stem + '_reduced'
nfutil.save_nf_data(output_dir,red_output_stem,grain_map,confidence_map,Xs,Ys,Zs,experiment.exp_maps)
vtkutil.output_grain_map_vtk(output_dir,[red_output_stem],red_output_stem,0.1)

#==========================================================================================================================
#%% MISSING GRAINS ROUTINEs
#==========================================================================================================================
load_old = True
if load_old:
    temp_grain_data = np.load(os.path.join(output_dir, 'dp718_scan%i_out_reduced_grain_map_data.npz' %my_scan_num))
    grain_map = temp_grain_data['grain_map']
    confidence_map = temp_grain_data['confidence_map']
    Xs = temp_grain_data['Xs']
    Ys = temp_grain_data['Ys']
    Zs = temp_grain_data['Zs']
    exp_maps = temp_grain_data['ori_list']


#==============================================================================
#%% IDENTIFY LOW CONFIDENCE REGION
#==============================================================================

conf_threshold_high = 0.5 #please note, I had very poor quality data in this example so you will likely want a much higher number than this. 
conf_threshold_low = 0.0

low_conf = np.logical_and(confidence_map < conf_threshold_high, confidence_map >= conf_threshold_low)

#==============================================================================
#%% CHECK THAT YOUR THRESHOLD IS IDENTIFYING AREAS YOU WANT
#==============================================================================
layer_no = 25

plt.figure('area check')
plt.imshow(confidence_map[layer_no,:,:])
plt.hold('on')
plt.imshow(low_conf[layer_no,:,:], alpha = 0.2)

#==============================================================================
#%% ADDITIONAL CHECK - LOOK AT HOW THE AREAS WILL SEGMENT AND MAKE SURE IT DOESNT RECOGNIZE AS ONE CONTIGUOUS BLOB
#==============================================================================

all_labels = measure.label(low_conf[0:32,:,:])
blob_labels = measure.label(low_conf[0:32,:,:], background = 0)

plt.figure('labels',figsize=(9, 3.5))
plt.subplot(131)
plt.imshow(low_conf[layer_no,:,:], cmap='gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(all_labels[layer_no,:,:], cmap='nipy_spectral')
plt.axis('off')
plt.subplot(133)
plt.imshow(blob_labels[layer_no,:,:], cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()

#==============================================================================
#%% CREATE CENTROID MAP OF LOW CONFIDENCE REGION
#==============================================================================

blob_labels_2 = ndimage.label(low_conf[:,:,:])[0]
centroids_2 = ndimage.measurements.center_of_mass(low_conf[:,:,:], blob_labels_2, np.unique(blob_labels_2))

centroid_point_map = np.zeros(np.shape(confidence_map))
centroid_new = np.empty([0,3])

min_blob_size = 10
miss_centroid_count_list = []
for i in range(1,len(centroids_2)):
    where=len(np.where(blob_labels_2==i)[0])
    if where >= min_blob_size:
        print(i, where)
        miss_centroid_count_list.append(where)
        centroid_new = np.append(centroid_new,np.reshape(np.array(centroids_2[i]),[1,3]),axis=0)
        centroid_point_map[np.rint(centroids_2[i][0]).astype('int'),np.rint(centroids_2[i][1]).astype('int'), np.rint(centroids_2[i][2]).astype('int')] = 10
print(len(miss_centroid_count_list))

#==============================================================================
#%% CAN CHECK THE CENTROIDS ARE IN LOCATIONS EXPECTED. 
#==============================================================================
        
layer_to_check = np.arange(20) + 0
#layer_to_check = np.arange(15) + 20
#layer_to_check = np.arange(10) + 40

for layer_no in layer_to_check:
    plt.figure()
    #plt.imshow(confidence_map[layer_no,:,:])
    plt.hold('on')
    plt.imshow(low_conf[layer_no,:,:], alpha = 0.5)
    plt.imshow(centroid_point_map[layer_no,:,:], alpha = 0.5)

plt.show()

#==============================================================================
#%% SAVE FILE OF CENTROIDS
#==============================================================================

vox_ind_centroid_save_file = os.path.join(output_dir, 'centroid_diff_sc%i_miss_grain.npy' %(my_scan_num))
np.save(vox_ind_centroid_save_file, centroid_new)

#==============================================================================
#%% CONVERT CENTROIDS FROM VOXEL INDICES TO REAL SPACE POSITIONS
#==============================================================================

centroid_new = np.load(vox_ind_centroid_save_file).astype(int)

test_crds_1 = np.zeros([centroid_new.shape[0],3])
for ii in np.arange(centroid_new.shape[0]):
    test_crds_1[ii,0]=Xs[centroid_new[ii,0],centroid_new[ii,1],centroid_new[ii,2]]
    test_crds_1[ii,1]=Ys[centroid_new[ii,0],centroid_new[ii,1],centroid_new[ii,2]]
    test_crds_1[ii,2]=Zs[centroid_new[ii,0],centroid_new[ii,1],centroid_new[ii,2]]

real_pos_centroid_save_file = os.path.join(output_dir, 'missing_coords_vol_sc%i.npy' %(my_scan_num))
np.save(real_pos_centroid_save_file, test_crds_1)

#==============================================================================
#%% RERUN NF-HEDM PROCESSING ON MISSING CENTROIDS TO FIND MISSING GRAINS
#==============================================================================

status = 'EXP GENERATION'
print('STARTED %s' %(status))
experiment_miss, nf_to_ff_id_map_miss  = nfutil.gen_trial_exp_data(grain_out_file,det_file,mat_file, 
                                                         x_ray_energy, mat_name, max_tth, comp_thresh, 
                                                         chi2_thresh, misorientation_bnd,
                                                         misorientation_spacing,ome_range_deg, 
                                                         num_imgs, beam_stop_width)
print('FINISHED %s' %(status))

#==============================================================================
#%% RERUN NF-HEDM PROCESSING ON MISSING CENTROIDS TO FIND MISSING GRAINS
#==============================================================================

deg_step = 1.0
missing_grain_coordinates = real_pos_centroid_save_file #missing coordinate list identified from the find missing centroids script.
quaternion_test_list = 'quat_list_%0.2f_deg_discrete_FR.npy' %deg_step #I recommend a fine discritization over the fundamental region of orientation space for your material.
new_quat_save_output = 'quats_to_add_vol_sc%i.npy'%(my_scan_num)


load_quat_list = True
if load_quat_list:
    quat_FR = np.load(os.path.join(output_dir, quaternion_test_list))
else:
    quat_FR = discretizeFundamentalRegion(phi1_bnd=[0, 180], phi1_step=deg_step,
                                theta_bnd=[0, 90], theta_step=deg_step,
                                phi2_bnd=[0, 180], phi2_step=deg_step).T
    np.save(os.path.join(output_dir, quaternion_test_list), quat_FR)

miss_test_crds_load = np.load(real_pos_centroid_save_file)
miss_test_crds = miss_test_crds_load[:, :]
n_crds = miss_test_crds.shape[0]

#==============================================================================
#%% TRANSFROM QUAT LIST TO EXP_MAPS AND ADD TO EXPERIMENT OBJECT
#==============================================================================

n_grains = quat_FR.shape[1]
rMat_c = hexrd_rot.rotMatOfQuat(quat_FR)
exp_maps = np.zeros([quat_FR.shape[1],3])
for i in range(0, quat_FR.shape[1]):
    phi = 2*np.arccos(quat_FR[0,i])
    n = xfcapi.unitRowVector(quat_FR[1:,i])
    exp_maps[i,:] = phi*n

experiment_miss.n_grains = n_grains
experiment_miss.rMat_c = rMat_c
experiment_miss.exp_maps = exp_maps

print(n_grains)
print(n_crds)
#==============================================================================
# %% INSTANTIATE CONTROLLER - RUN BLOCK NO EDITING
#==============================================================================

progress_handler = nfutil.progressbar_progress_observer()
save_handler=nfutil.forgetful_result_handler()

miss_chunk_size = max(int(miss_test_crds.shape[0] / ncpus), 1)
print(miss_chunk_size)

controller = nfutil.ProcessController(save_handler, progress_handler,
                               ncpus=ncpus, chunk_size=miss_chunk_size)

multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

#==============================================================================
# %% TEST ORIENTATIONS - RUN BLOCK NO EDITING
#==============================================================================

missing_raw_confidence = nfutil.test_orientations(image_stack, experiment_miss, miss_test_crds,
                  controller, multiprocessing_start_method)


#==============================================================================
# %% FIND BEST QUAT FOR EACH TEST COORDINATE
#==============================================================================

best_quaternion = np.zeros([miss_test_crds.shape[0],4])
for i in range(0,missing_raw_confidence.shape[1]):
    where = np.where(missing_raw_confidence[:,i] == np.max(missing_raw_confidence[:,i]))
    best_quaternion[i,:] = quat_FR[:,where[0][0]]
    print(miss_centroid_count_list[i], np.max(missing_raw_confidence[:,i]))

#==============================================================================
# %% SAVE ORIENTATIONS
#==============================================================================

np.save(os.path.join(output_dir, new_quat_save_output), best_quaternion)

#==============================================================================
# %% RERUN EXPERIMENT
#==============================================================================

l_experiment, l_nf_to_ff_id_map  = nfutil.gen_trial_exp_data(grain_out_file,det_file,mat_file, 
                                                         x_ray_energy, mat_name, max_tth, comp_thresh, 
                                                         chi2_thresh, misorientation_bnd,
                                                         misorientation_spacing,ome_range_deg, 
                                                         num_imgs, beam_stop_width)

#%%

add_missing_quat = True
if add_missing_quat:
    miss_quats = np.load(os.path.join(output_dir, new_quat_save_output))
    
    l_experiment.n_grains = l_experiment.n_grains + miss_quats.shape[0]
    l_experiment.rMat_c = np.concatenate((l_experiment.rMat_c, hexrd_rot.rotMatOfQuat(miss_quats.T)), axis=0)
    l_experiment.exp_maps = np.concatenate((l_experiment.exp_maps, quat2exp_map(miss_quats)), axis=0)
    
    l_nf_to_ff_id_map = np.arange(l_experiment.n_grains)

low_conf_Xs = Xs[low_conf]
low_conf_Ys = Ys[low_conf]
low_conf_Zs = Zs[low_conf]

low_conf_test_crds = np.vstack([low_conf_Xs.flatten(), low_conf_Ys.flatten(), low_conf_Zs.flatten()]).T
n_low_conf_crds = len(low_conf_test_crds)

#%%
l_progress_handler = nfutil.progressbar_progress_observer()
l_save_handler = nfutil.forgetful_result_handler()

l_controller = nfutil.ProcessController(l_save_handler, l_progress_handler,
                               ncpus=ncpus, chunk_size=chunk_size)

l_multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

# process nf images and orientations for this round
l_raw_confidence = nfutil.test_orientations(image_stack, l_experiment, low_conf_test_crds,
   l_controller,l_multiprocessing_start_method)


#%%

l_top_n_confidence, l_top_n_index = nfutil.save_raw_confidence_n(l_raw_confidence, None, None, n=1)

grain_map[low_conf] = l_top_n_index.flatten()
confidence_map[low_conf] = l_top_n_confidence.flatten()

red_output_stem = output_stem + '_reduced_mis'
nfutil.save_nf_data(output_dir,red_output_stem,grain_map,confidence_map,Xs,Ys,Zs,l_experiment.exp_maps)
vtkutil.output_grain_map_vtk(output_dir,[red_output_stem],red_output_stem,0.1)














