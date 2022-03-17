#%% Necessary Dependencies

from __future__ import print_function

import sys

import time

import numpy as np

import matplotlib.pyplot as plt

import multiprocessing as mp

import os

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
        im = filters.gaussian(tmp_img, sigma=gaussian_sigma) #gaussian filter, may need to change higher for deformed peaks (3-6)
        im = morphology.grey_closing(im,size=(5,5)) #grey filter, smooths mins
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
        image_stack[ii,:,:] = img.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
    print('\n')
    print('Cleaning Images...')
    results = Parallel(n_jobs=ncpus, verbose=2)(delayed(gen_nf_cleaned_image)(image_stack[idx, :, :], threshold, non_peak_thresh, gaussian_sigma) for idx in range(num_imgs))
    image_stack = np.array(results)
    
    #%A final dilation that includes omega
    print('Final Dilation Including Omega....')
    ret_image_stack=morphology.binary_dilation(image_stack,iterations=ome_dilation_iter)
    
    return ret_image_stack

def gen_nf_cleaned_image(image, threshold, non_peak_thresh=1.7, gaussian_sigma=3):    
    #image procesing
    im = filters.gaussian(image, sigma=gaussian_sigma) #gaussian filter, may need to change higher for deformed peaks (3-6)
    im = morphology.grey_closing(im,size=(5,5)) #grey filter, smooths mins
    where_remove = np.where(im<non_peak_thresh) #identify non-peak region, will need to be lower for deformed peaks (0.5-1.5)
    im[where_remove] = 0 #set non-peak region to 0
    binary_img = morphology.binary_fill_holes(im) #make image binary and  fill holes
    return binary_img
    

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
my_scan_num = 29

# set up output directory and filename
output_dir = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf_analysis/output/'
output_stem = 'new_ss718_scan%s_out' %(my_scan_num)

# location of grains.out folder
grain_out_file='/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/c0_0_gripped/c0_0_gripped_sc%i_nf/grains.out' %(my_scan_num)

if my_scan_num == 29:
    # location of nf data images
    data_folder='/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf/25/nf/'
    img_start=1815#for 0.25 degree/steps and 5 s exposure, end up with 0 junk frames up front, this is the 6th
    
elif my_scan_num == 30:
    # location of nf data images
    data_folder='/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf/26/nf/'
    img_start=3262#for 0.25 degree/steps and 5 s exposure, end up with 0 junk frames up front, this is the 6th
    
elif my_scan_num == 31:
    # location of nf data images
    data_folder='/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf/27/nf/'
    img_start=4709#for 0.25 degree/steps and 5 s exposure, end up with 0 junk frames up front, this is the 6th


# TOMO IS ONLY [0, 180] in omega
# location of tomo bright field images
tbf_data_folder='/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf/11/nf/'
tbf_img_start=31 #for this rate, this is the 6th file in the folder
tbf_num_imgs=10

#Locations of tomography images
tomo_data_folder='/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/nf/12/nf/'
tomo_img_start=57 #for this rate, this is the 6th file in the folder
tomo_num_imgs=720

num_imgs=1441
img_nums=np.arange(img_start,img_start+num_imgs,1)
    

print('grains.out: %s' %(grain_out_file))
print('nf data: %s' %(data_folder))
print('tomo bf: %s' %(tbf_data_folder))
print('tomo data: %s' %(tomo_data_folder))
#==============================================================================
# %% NEAR FIELD DATA FILES -CAN BE EDITED
#==============================================================================

#These are the near field data files used for the reconstruction, a grains.out file
#from the far field analaysis is used as orientation guess for the grid that will 
#be used for the near field reconstruction
#grain_out_file='/home/millerlab/djs522/chess_bt_2019-12/dp718-2_hexrd/det8_scans/dp718-2_sc25_det0/grains.out'

#%%
#Locations of near field images
#data_folder='/home/millerlab/djs522/chess_bt_2019-12/nf_dp718-2/19/nf/'
#img_start=14726#for 0.25 degree/steps and 5 s exposure, end up with 0 junk frames up front, this is the 6th
#num_imgs=1441
#img_nums=np.arange(img_start,img_start+num_imgs,1)

#==============================================================================
# %% TOMOGRAPHY DATA FILES -CAN BE EDITED
#==============================================================================

#Locations of tomography bright field images
#tbf_data_folder='/home/millerlab/djs522/chess_bt_2019-12/nf_dp718-2/10/nf/'
#tbf_img_start=13092 #for this rate, this is the 6th file in the folder
#tbf_num_imgs=10

#Locations of tomography images
#tomo_data_folder='/home/millerlab/djs522/chess_bt_2019-12/nf_dp718-2/11/nf/'
#tomo_img_start=13118 #for this rate, this is the 6th file in the folder
#tomo_num_imgs=720

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


cross_sectional_dim=1.05 #cross sectional to reconstruct (should be at least 20%-30% over sample width)
#voxel spacing for the near field reconstruction
voxel_spacing=0.005#in mm
#vertical (y) reconstruction voxel bounds in mm
v_bnds=[-0.12, 0.09]
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
load_tomo_stack = True

if load_tomo_stack:
    rad_stack = np.load(tomo_stack_file)
else:
    rad_stack=tomoutil.gen_attenuation_rads(tomo_data_folder,tbf,tomo_img_start,tomo_num_imgs,experiment.nrows,experiment.ncols,num_digits=6)
    
    inf_pts=np.isinf(rad_stack)
    rad_stack[inf_pts]=0.

save_rad_stack = False
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
    l = 25
    w = 20
    plt.imshow(tomo_mask[l:(l+212), w:(w+212)],interpolation='none')

tomo_mask = tomo_mask[l:(l+212), w:(w+212)]

#==============================================================================
# %%  TOMO PROCESSING - CONSTRUCT DATA GRID
#==============================================================================
status = 'GENERATING GRID COORD FOR NF'
print('STARTED %s' %(status))
use_tomo_test_coords = True # both do the basically the same thing

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
#==============================================================================
# %% NEAR FIELD - LOAD IMAGE DATA AND PROCESS
#==============================================================================
status = 'GENERATING NF IMAGE STACK'
print('STARTED %s' %(status))

nf_image_stack_file = 'nf_image_stack_ffsc%i.npy' %my_scan_num

load_image_stack = True
if load_image_stack:
    image_stack = np.load(nf_image_stack_file)
else:
    if do_nf_image_clean:
        threshold = 3.0
        non_peak_thresh = 1.7
        gauss_sigma = 3
        start = time.time()
        # multi-processing clean nf images
        image_stack = gen_nf_cleaned_image_stack_mp(data_folder,img_nums,dark,ome_dilation_iter,threshold,experiment.nrows,experiment.ncols,num_digits=6,non_peak_thresh=non_peak_thresh,gaussian_sigma=gauss_sigma,ncpus=ncpus)
        # serial processing clean nf images
        #image_stack = gen_nf_cleaned_image_stack(data_folder,img_nums,dark,ome_dilation_iter,threshold,experiment.nrows,experiment.ncols,stem='nf_',num_digits=6,non_peak_thresh=non_peak_thresh,gaussian_sigma=gauss_sigma)
        end = time.time()
        print(end - start)
    else:
        image_stack = nfutil.gen_nf_image_stack(data_folder,img_nums,dark,num_erosions,num_dilations,ome_dilation_iter,threshold,experiment.nrows,experiment.ncols,num_digits=6)

save_image_stack = False
if save_image_stack:
    np.save(nf_image_stack_file, image_stack)

print('\n')
print('FINISHED %s' %(status))
#==============================================================================
# %% VIEW IMAGES FOR DEBUGGING TO LOOK AT IMAGE PROCESSING PARAMETERS
#==============================================================================
if show_images:
    img_to_view=[100, 300, 500, 700, 900, 1100, 1300] # check a lot of these
    for temp_img in img_to_view:
        plt.figure()
        plt.imshow(image_stack[temp_img,:,:],interpolation='none')
        plt.draw()

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

if do_brute_force_ori:
    top_n = 10 # number of orientations to save in confidence matirx
    step_range = 30
    phi1_range = np.arange(0, 180 + step_range, step_range) # in degrees
    phi2_range = np.arange(0, 180 + step_range, step_range) # in degrees
    theta_range = np.arange(0, 90 + step_range, step_range) # in degrees
    step_size = 2 # in degrees
    
    # number of ranges
    num_phi1 = phi1_range.shape[0] - 1
    num_phi2 = phi2_range.shape[0] - 1
    num_theta = theta_range.shape[0] - 1
    
    for i in range(num_phi1):
        for j in range(num_phi2):
            for k in range(num_theta):
                
                
                # print necessary round info
                temp_start_time = time.ctime(time.time())
                print("Round start time:", temp_start_time)
                print('i = %i, j = %i, k = %i' %(i,j,k))
                print('Starting round %i/%i' %((k+1) + j*num_theta + i*num_theta*num_phi2, num_phi1*num_phi2*num_theta))
                
                # set up bounds for this round
                phi1_bnd = [phi1_range[i], phi1_range[i+1]]
                phi2_bnd = [phi2_range[j], phi2_range[j+1]]
                theta_bnd = [theta_range[k], theta_range[k+1]]
                
                print(phi1_bnd)
                print(phi2_bnd)
                print(theta_bnd)
                
                # discretize fundamental region 
                fr_quats = discretizeFundamentalRegion(phi1_bnd=phi1_bnd, phi1_step=step_size,
                                                        theta_bnd=theta_bnd, theta_step=step_size,
                                                        phi2_bnd=phi2_bnd, phi2_step=step_size,
                                                        crys_sym='cubic', ret_type='quat')
                experiment = new_experiment_ori(fr_quats, experiment)
                nf_to_ff_id_map = np.arange(experiment.n_grains)
                print('BRUTE FORCE METHOD: Testing %i Orientations' %(experiment.n_grains))
                
                # process nf images and orientations for this round
                raw_confidence=nfutil.test_orientations(image_stack, experiment, test_crds,
                   controller,multiprocessing_start_method)
                
                print('Done with raw confidence')
                # pull top_n highest confidence orienations from confidence matrix
                raw_confidenceT = raw_confidence.T
                sorted_row_ind = np.argsort(raw_confidenceT, axis=1)[:, raw_confidenceT.shape[1]-top_n::]
                col_ind = np.arange(raw_confidenceT.shape[0])[:, None]
                
                top_raw_conf = raw_confidenceT[col_ind, sorted_row_ind]
                print('Done with sorting raw confidence')
                
                # save info
                round_save_stem = '/home/millerlab/djs522/chess_bt_2019-12/dp718-2_hexrd/FIT_HEDM/scan25_nf_output/nf_vol2_sc25_phi1_%i_%i_phi2_%i_%i_theta_%i_%i' %(phi1_bnd[0], phi1_bnd[1], phi2_bnd[0], phi2_bnd[1], theta_bnd[0], theta_bnd[1])
                
                np.savez(round_save_stem, top_raw_conf=top_raw_conf, sorted_row_ind=sorted_row_ind)
                
                print('Done with saving top confidence')
                
                del raw_confidence
                del raw_confidenceT
                del sorted_row_ind
                del col_ind
                del top_raw_conf
                
                print('Done with deleting raw confidence')
                
                temp_end_time = time.ctime(time.time())
                print("Round end time:", temp_end_time)

else:
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

top_n_confidence, top_n_index = nfutil.save_raw_confidence_n(raw_confidence, output_dir, output_stem, n=3)


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

#%%
scan_num = 20

scan_data = np.load('/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/nf_analysis/output/dp718_scan%i_out_grain_map_data.npz' %(scan_num))

confidence_map = scan_data['confidence_map']
grain_map = scan_data['grain_map']

Ys = scan_data['Ys'] - 0.25

out_bounds=np.where(tomo_mask==0)  
confidence_map[:,out_bounds[0],out_bounds[1]] =-0.001   
grain_map[:,out_bounds[0],out_bounds[1]] =-1  

output_stem = 'dp718_scan%i_out_tomo_mask1' %(scan_num)

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
# %% PLOTTING SINGLE LAYERS FOR DEBUGGING
#==============================================================================
if show_images:
    import scipy.ndimage as im_func
    
    layer_no = 25
    plt.figure(2)
    #nfutil.plot_ori_map(grain_map, confidence_map, experiment.exp_maps, layer_no)
    
    m = 2
    temp_tomo_mask = im_func.binary_erosion(tomo_mask, structure=np.ones((m,m)))
    plt.imshow(temp_tomo_mask,interpolation='none')
    
    map_angle = -2.25 #-2.25
    
    im_layer = confidence_map[layer_no]
    im_layer = temp_tomo_mask
    im_layer = im_func.rotate(im_layer, angle=map_angle, reshape=False, order=0)
    #temp_tomo_mask = im_func.rotate(temp_tomo_mask, angle=map_angle, reshape=False, order=0)
    
    plt.figure(3)
    plt.imshow(im_layer)
    
    space = 0
    samp_pix = int(1 / voxel_spacing)
    x_start = 28 - space
    y_start = 33 - space
    x_end = x_start + samp_pix + space*2
    y_end = y_start + samp_pix + space*2
    
    print(x_start, x_end, y_start, y_end)
    
    temp_mask = np.zeros(im_layer.shape)
    temp_mask[y_start:y_end, x_start:x_end] = 1
    
    plt.figure(4)
    plt.imshow(temp_mask - im_layer)
    
    plt.show()
    
    my_int_thresh = 0.5
    gy, gx = np.gradient(im_layer)
    sharpness = np.average(np.sqrt(gx**2 + gy**2))
    print('Image sharpness: %f' %(sharpness))
    print('Percent above threshold: %f %%' %(100 * np.sum(im_layer>my_int_thresh)/float(np.size(im_layer))))


do_rotation_fix = True
if do_rotation_fix:
    rot_grain_map = im_func.rotate(im_layer, angle=map_angle, reshape=False, order=0)

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
    
#%%    
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


# %%
do_map_fix = False
if do_map_fix:
    map_angle = -4.25
    x_start = 45
    y_start = 42
    samp_pix = int(1 / voxel_spacing) - 6 # remove 3 pixels on either side
    num_layers = grain_map.shape[0]
    
    clean_grain_map = np.zeros([grain_map.shape[0], samp_pix, samp_pix])
    clean_confidence = np.zeros([grain_map.shape[0], samp_pix, samp_pix])
    clean_xs = np.zeros([grain_map.shape[0], samp_pix, samp_pix])
    clean_ys = np.zeros([grain_map.shape[0], samp_pix, samp_pix])
    clean_zs = np.zeros([grain_map.shape[0], samp_pix, samp_pix])
    
    for i in np.arange(num_layers):
        clean_grain_map[i, :, :] = im_func.rotate(grain_map[i, :, :], angle=map_angle, reshape=True)[x_start:x_start+samp_pix, y_start:y_start+samp_pix]
        clean_confidence[i, :, :] = im_func.rotate(confidence_map[i, :, :], angle=map_angle, reshape=True)[x_start:x_start+samp_pix, y_start:y_start+samp_pix]
        clean_xs[i, :, :] = im_func.rotate(Xs[i, :, :], angle=map_angle, reshape=True)[x_start:x_start+samp_pix, y_start:y_start+samp_pix]
        clean_ys[i, :, :] = im_func.rotate(Ys[i, :, :], angle=map_angle, reshape=True)[x_start:x_start+samp_pix, y_start:y_start+samp_pix]
        clean_zs[i, :, :] = im_func.rotate(Zs[i, :, :], angle=map_angle, reshape=True)[x_start:x_start+samp_pix, y_start:y_start+samp_pix]
    
    y_rm = 1
    new_clean_grain_map = clean_grain_map[y_rm:num_layers-y_rm, :, :]
    new_clean_confidence = clean_confidence[y_rm:num_layers-y_rm, :, :]
    clean_xs = clean_xs[y_rm:num_layers-y_rm, :, :]
    clean_ys = clean_ys[y_rm:num_layers-y_rm, :, :]
    clean_zs = clean_zs[y_rm:num_layers-y_rm, :, :]
    temp_output_stem = output_stem + '_clean'
    nfutil.save_nf_data(output_dir,temp_output_stem,new_clean_grain_map,new_clean_confidence,clean_xs,clean_ys,clean_zs,experiment.exp_maps)
    vtkutil.output_grain_map_vtk(output_dir,[temp_output_stem],temp_output_stem,0.1)
    
#%% Stitch ff-HEDM data from scans
from hexrd.xrd          import rotations  as rot
from hexrd.xrd          import symmetry   as sym

# location of grains.out folder
grain_out_sc18 = np.loadtxt('/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_0_gripped/c0_0_gripped_sc%i_nf/grains.out' %(18)) #top
grain_out_sc19 = np.loadtxt('/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_0_gripped/c0_0_gripped_sc%i_nf/grains.out' %(19)) #middle
grain_out_sc20 = np.loadtxt('/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_0_gripped/c0_0_gripped_sc%i_nf/grains.out' %(20)) #bottom

# make vertical axis adjustements
temp_layer_size = 0.15 #mm
grain_out_sc18[:, 7] += temp_layer_size
grain_out_sc20[:, 7] -= temp_layer_size

# make grain_id adjustment
grain_out_sc18[:, 0] += 2000
grain_out_sc19[:, 0] += 4000
grain_out_sc20[:, 0] += 6000

# large stack grain mat
grain_out_total = np.vstack([grain_out_sc18, grain_out_sc19, grain_out_sc20])
#grain_out_total = grain_out_total[:5, :]

old_grain_ids = np.copy(grain_out_total[:, 0])
new_grain_ids = np.copy(grain_out_total[:, 0])
tot_grains = grain_out_total.shape[0]

# set thresholds
dist_thresh = 0.25 # mm
misori_thresh = 5 # degrees
qsyms = 'Oh'

# create a graph of duplicate grains
grain_dup_graph = np.zeros((tot_grains,tot_grains))
grain_dup_list = []

# create duplicate graph array
for i in np.arange(tot_grains):
    temp_grain = np.atleast_2d(grain_out_total[i, :])
    other_grains = grain_out_total[i:, :]
    
    dist_comp = np.linalg.norm(temp_grain[:, 6:9] - other_grains[:, 6:9], axis=1)
    
    close_dist_comp_grains = other_grains[dist_comp <= dist_thresh]
    
    if close_dist_comp_grains.shape[0] > 0:
        q1 = sym.toFundamentalRegion(np.atleast_2d(rot.quatOfExpMap(temp_grain[:,3:6].T)), crysSym=qsyms)
        q2 = sym.toFundamentalRegion(np.atleast_2d(rot.quatOfExpMap(close_dist_comp_grains[:,3:6].T)), crysSym=qsyms)
        
        misorientation = np.zeros(q2.shape[1])
        for j in range(q2.shape[1]):
            misorientation[j] = rot.misorientation(q1, np.atleast_2d(q2[:, j]).T)[0] * 180./np.pi
    
        close_ori_comp_grains = close_dist_comp_grains[misorientation <= misori_thresh, :]
        
        if close_ori_comp_grains.shape[0] > 0:
            dup_grains_ind = np.where(np.in1d(grain_out_total[:, 0], close_ori_comp_grains[:, 0]))[0]
            grain_dup_graph[i, dup_grains_ind] = 1
            grain_dup_graph[dup_grains_ind, i] = 1

# process dup graph array
new_grain_out_total = np.copy(grain_out_total)
row_sum_dup_graph = np.sum(grain_dup_graph, axis=0)
idx_dup_graph = np.where(row_sum_dup_graph > 1)[0]

print("Number of Duplicates: %i" %idx_dup_graph.size)

for i in idx_dup_graph:
    #print(i)
    temp_row_dup_graph = grain_dup_graph[:, i]
    #print(temp_row_dup_graph)
    temp_row_idx = np.where(temp_row_dup_graph == 1)[0]
    #print(temp_row_idx)
    
    if temp_row_idx.size > 0:
        temp_row_idx_minus_grain = np.setdiff1d(temp_row_idx, i)
        #print(temp_row_idx_minus_grain)
        
        temp_dup_grains = grain_out_total[temp_row_idx, :]
        #print(temp_dup_grains.shape)
        
        new_grain_out_total[i, 1:] = np.mean(temp_dup_grains[:, 1:])
        
        grain_dup_graph[:, temp_row_idx_minus_grain] = 0
        grain_dup_graph[temp_row_idx_minus_grain, :] = 0
        
        new_grain_ids[temp_row_idx_minus_grain] = grain_out_total[i, 0]

uni_new_grain_ids = np.unique(new_grain_ids)
new_grain_out_total = new_grain_out_total[np.in1d(new_grain_out_total[:, 0], uni_new_grain_ids), :]

for i, uni_id in enumerate(uni_new_grain_ids):
    new_grain_ids[new_grain_ids == uni_id] = i
 
grain_id_old_new_map = np.vstack([old_grain_ids, new_grain_ids]).T
new_grain_out_total[:, 0] = np.arange(uni_new_grain_ids.size)
np.save('dp718_sc18_19_20_ff_stitch.npy', new_grain_out_total)

sc18_old_new = grain_id_old_new_map[0:grain_out_sc18.shape[0], :]
sc19_old_new = grain_id_old_new_map[grain_out_sc18.shape[0]:grain_out_sc18.shape[0]+grain_out_sc19.shape[0], :]
sc20_old_new = grain_id_old_new_map[grain_out_sc18.shape[0]+grain_out_sc19.shape[0]:, :]
sc18_old_new[:, 0] -= 2000
sc19_old_new[:, 0] -= 4000
sc20_old_new[:, 0] -= 6000
np.save('dp718_sc18_old_to_new_ids.npy', sc18_old_new)
np.save('dp718_sc19_old_to_new_ids.npy', sc19_old_new)
np.save('dp718_sc20_old_to_new_ids.npy', sc20_old_new)

print(sc18_old_new.shape, grain_out_sc18.shape[0])
print(sc19_old_new.shape, grain_out_sc19.shape[0])
print(sc20_old_new.shape, grain_out_sc20.shape[0])



#%% stitch ff to nf

temp_scan_num = 20
temp_offset = -0.15

new_grain_out_total = np.load('dp718_sc18_19_20_ff_stitch.npy')
scan_new_grains = np.load('dp718_sc%i_old_to_new_ids.npy' %temp_scan_num).astype(int)   # [old, new]

# grain_map=grain_map,confidence_map=confidence_map,Xs=Xs,Ys=Ys,Zs=Zs,ori_list=ori_list,id_remap=id_remap
scan_nf = np.load('dp718_scan%i_out_grain_map_data.npz' %temp_scan_num)
scan_grain_map = scan_nf['grain_map']

new_scan_grain_map = np.zeros(scan_grain_map.shape)
for i in range(scan_new_grains.shape[0]):
    new_scan_grain_map[scan_grain_map == scan_new_grains[i, 0]] = int(scan_new_grains[i, 1])
    if i % 50 == 0:
        print("%i / %i" %(i, scan_new_grains.shape[0]))
    
scan_expmaps = new_grain_out_total[scan_new_grains[:, 1].astype(int), 3:6]

new_out_stem = 'dp718_sc%i_new_ids' %(temp_scan_num)
if save_grain_map:
    status = 'SAVING NF MAP'
    print('STARTED %s' %(status))
    nfutil.save_nf_data(output_dir,new_out_stem,new_scan_grain_map,
                        scan_nf['confidence_map'],scan_nf['Xs'],scan_nf['Ys']+temp_offset,scan_nf['Zs'],scan_expmaps)
    print('FINISHED %s' %(status)) 

if save2vtk:
    status = 'SAVING NF MAP TO VTK'
    print('STARTED %s' %(status))
    vtkutil.output_grain_map_vtk(output_dir,[new_out_stem],new_out_stem,0.1)
    print('FINISHED %s' %(status))


#%% stitch all nf layers

temp_scans = [18, 19, 20]
temp_offsets = [[0.225, 0.075], [0.075, -0.075], [-0.075, -0.225]]

full_X_dim = (int(cross_sectional_dim / voxel_spacing)
full_Z_dim = (int(cross_sectional_dim / voxel_spacing)
full_Y_dim = (int(0.45 / voxel_spacing)

full_grain_map = np.zeros(full_Y_dim, full_X_dim, full_Z_dim))
Xs,Ys,Zs=np.meshgrid(np.arange(full_X_dim) * voxel_spacing, np.arange(full_Y_dim) * voxel_spacing, np.arange(full_Z_dim) * voxel_spacing)   
Zs=(Zs-cross_sectional_dim/2.)
Xs=(Xs-cross_sectional_dim/2.)

for i, scan in enumerate(temp_scans):
    scan_nf = np.load('dp718_sc%i_new_ids_grain_map_data.npz' %scan)
    scan_offset = temp_offsets[i]
    
    







