#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:59:42 2023

@author: djs522
"""

#==============================================================================
# %% IMPORTS
#==============================================================================

import sys

import numpy as np

import scipy.ndimage as img

import skimage.io as imgio
import skimage.filters as filters

from joblib import Parallel, delayed

#==============================================================================
# %% NF-IMAGE CLEAN IMAGE FUNCTION (CURRENT)
#==============================================================================

def gen_nf_cleaned_image_stack(data_folder,img_nums,dark,nrows,ncols, \
                               process_type='gaussian',process_args=[4.5,5], \
                               threshold=1.5,ome_dilation_iter=1,stem='nf_', \
                               num_digits=5,ext='.tif'):
    
    image_stack=np.zeros([img_nums.shape[0],nrows,ncols],dtype=bool)

    print('Loading and Cleaning Images...')
    
    
    if process_type=='gaussian':
        sigma=process_args[0]
        size=process_args[1].astype(int) #needs to be int
    
        for ii in np.arange(img_nums.shape[0]):
            print('Image #: ' + str(ii))
            tmp_img=imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
            #image procesing
            tmp_img = filters.gaussian(tmp_img, sigma=sigma)

            tmp_img = img.morphology.grey_closing(tmp_img,size=(size,size))

            binary_img = img.morphology.binary_fill_holes(tmp_img>threshold)
            image_stack[ii,:,:]=binary_img
    
    else:    

        num_erosions=process_args[0]
        num_dilations=process_args[1]


        for ii in np.arange(img_nums.shape[0]):
            print('Image #: ' + str(ii))
            tmp_img=imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
            #image procesing
            image_stack[ii,:,:]=img.morphology.binary_erosion(tmp_img>threshold,iterations=num_erosions)
            image_stack[ii,:,:]=img.morphology.binary_dilation(image_stack[ii,:,:],iterations=num_dilations)
    
    
    #%A final dilation that includes omega
    print('Final Dilation Including Omega....')
    image_stack=img.morphology.binary_dilation(image_stack,iterations=ome_dilation_iter)
    
    
    return image_stack

#==============================================================================
# %% NF-IMAGE CLEAN IMAGE FUNCTION (MULTIPROCESSING)
#==============================================================================

def gen_nf_cleaned_image_stack_mp(data_folder,img_nums,dark,nrows,ncols, \
                               process_type='gaussian',process_args=[4.5,5], \
                               threshold=1.5,ome_dilation_iter=1,stem='nf_', \
                               num_digits=5,ext='.tif',ncpus=1):
    num_imgs = img_nums.shape[0]
    image_stack = np.zeros([num_imgs.shape[0],nrows,ncols],dtype=bool)

    print('Loading and Cleaning Images...')
    if process_type=='gaussian':
        # gaussian filter
        if ncpus > 1:
            # mutli-processing
            for ii in np.arange(num_imgs):
                sys.stdout.write('\rImage #: %i/%i' %(ii+1, num_imgs))
                sys.stdout.flush()
                image_stack[ii,:,:] = imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
            print('\n')
            results = Parallel(n_jobs=ncpus, verbose=2)(delayed(gen_nf_cleaned_image_gauss)(image_stack[idx, :, :], process_args, threshold) for idx in range(num_imgs))
            image_stack = np.array(results)
        else:
            # serial processing
            for ii in np.arange(img_nums.shape[0]):
                sys.stdout.write('\rImage #: %i/%i' %(ii+1, num_imgs))
                sys.stdout.flush()
                tmp_img=imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
                #image procesing
                image_stack[ii,:,:] = gen_nf_cleaned_image_gauss(tmp_img, process_args, threshold)
            print('\n')
    
    else: 
        # binary closing filter
        if ncpus > 1:
            # mutli-processing
            for ii in np.arange(num_imgs):
                sys.stdout.write('\rImage #: %i/%i' %(ii+1, num_imgs))
                sys.stdout.flush()
                image_stack[ii,:,:] = imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
            print('\n')
            results = Parallel(n_jobs=ncpus, verbose=2)(delayed(gen_nf_cleaned_image_closing)(image_stack[idx, :, :], process_args, threshold) for idx in range(num_imgs))
            image_stack = np.array(results)
        else:
            # serial processing
            for ii in np.arange(img_nums.shape[0]):
                sys.stdout.write('\rImage #: %i/%i' %(ii+1, num_imgs))
                sys.stdout.flush()
                tmp_img=imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
                #image procesing
                image_stack[ii,:,:] = gen_nf_cleaned_image_closing(tmp_img, process_args, threshold)
            print('\n')
    
    
    #%A final dilation that includes omega
    print('Final Dilation Including Omega....')
    image_stack = img.morphology.binary_dilation(image_stack, iterations=ome_dilation_iter)
    
    return image_stack

def gen_nf_cleaned_image_gauss(image, process_args, threshold):    
    # guass filter nf image procesing
    sigma = process_args[0]
    size = process_args[1].astype(int) # needs to be int
    
    tmp_img = filters.gaussian(image, sigma=sigma)
    tmp_img = img.morphology.grey_closing(tmp_img, size=(size,size))
    binary_img = img.morphology.binary_fill_holes(tmp_img > threshold)
    
    return binary_img

def gen_nf_cleaned_image_closing(image, process_args, threshold):    
    # binary closing nf image procesing
    num_erosions = process_args[0]
    num_dilations = process_args[1]
    
    tmp_img = img.morphology.binary_erosion(image > threshold, iterations=num_erosions)
    binary_img = img.morphology.binary_dilation(tmp_img, iterations=num_dilations)
    
    return binary_img