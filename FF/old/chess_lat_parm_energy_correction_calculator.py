#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:49:39 2017

@author: dcp99
"""

import sys

import numpy as np

import matplotlib.pyplot as plt

from hexrd import rotations  as rot

#%% Input and Adjustable Parameters

grain_out_file = sys.argv[0] + 'analysis/combined_grains_c0_0_ungripped.out'

#grain metrics to use
comp_thresh=0.9 #completeness
chi2_thresh=0.005 #chi squared

#bnds of grains to use to for analysis, probably want to ignore grains at edges of beam, adjust to ~+/- half the beam size
# for this example, beam height was 0.170 mm
top_bnd=0.07#mm
bot_bnd=-0.07#mm

#%%%

plt.close('all')


tmp_data_0=np.loadtxt(grain_out_file)

comp_0=tmp_data_0[:,1]
chi_0=tmp_data_0[:,2]

strain_0=tmp_data_0[:,15:]
ori_0=tmp_data_0[:,3:6]
pos_0=tmp_data_0[:,6:9]
v_strain_0=np.sum(strain_0[:,0:3],axis=1)/3.

#check if grains are withing completenss and chi^2 levels
good_grains=np.where(np.logical_and(comp_0>comp_thresh,chi_0<chi2_thresh))[0]

plt.figure(1)
plt.plot(pos_0[good_grains,1],v_strain_0[good_grains],'x')


print('Mean Height: ' + str(np.mean(pos_0[good_grains,1])))

good_grains_a=np.where(np.abs(pos_0[good_grains,1])<top_bnd)[0]
good_grains_b=np.where(np.abs(pos_0[good_grains,1])>bot_bnd)[0]
good_grains_2=np.intersect1d(good_grains_a,good_grains_b)

p=np.polyfit(pos_0[good_grains[good_grains_2],1],v_strain_0[good_grains[good_grains_2]],1)

v_strain_0=v_strain_0-p[0]*pos_0[:,1]

plt.plot(pos_0[good_grains,1],v_strain_0[good_grains],'rx')

n_grains=len(good_grains)


plt.legend(['raw','energy gradient corrected'])

plt.xlabel('y position')
plt.ylabel('volumetric strain')

print('Energy / Volumetric Strain Slope: ' + str(p[0])+ ' mm^-1')
print('Constant Term: ' + str(p[1]) + ' (if larger than 1e-4 consider using following plots/output to adjust lattice parameter(s)')


astrain=np.zeros(n_grains)
bstrain=np.zeros(n_grains)
cstrain=np.zeros(n_grains)

#correct energy gradient
strain_0[:,0]=strain_0[:,0]-(p[0]*pos_0[:,1])
strain_0[:,1]=strain_0[:,1]-(p[0]*pos_0[:,1])
strain_0[:,2]=strain_0[:,2]-(p[0]*pos_0[:,1])


for ii in np.arange(n_grains):
    ti=good_grains[ii]
    strain_ten=np.array([[strain_0[ti,0],strain_0[ti,5],strain_0[ti,4]],[strain_0[ti,5],strain_0[ti,1],strain_0[ti,3]],[strain_0[ti,4],strain_0[ti,3],strain_0[ti,2]]])
    R=rot.rotMatOfExpMap(ori_0[ti,:])
    
    strain_ten_c=np.dot(R.T,np.dot(strain_ten,R))
    #print(strain_ten_c)
    astrain[ii]=strain_ten_c[0,0]
    #astrain2[ii]=np.mean(strain_0[ti,0:3])
    bstrain[ii]=strain_ten_c[1,1]
    cstrain[ii]=strain_ten_c[2,2]

plt.figure(2)
plt.plot(astrain,'x')
plt.title(r'$\Delta a/a (\epsilon_{xx}^C)$')
plt.xlabel('grain #')
print('Delta a/a: ' + str(np.mean(astrain)))
           
           
plt.figure(3)
plt.plot(bstrain,'gx')
plt.title(r'$\Delta b/b (\epsilon_{yy}^C)$')
plt.xlabel('grain #')
print('Delta b/b: ' + str(np.mean(bstrain)))           

plt.figure(4)
plt.plot(cstrain,'rx')
plt.title(r'$\Delta c/c (\epsilon_{zz}^C)$')
plt.xlabel('grain #')
print('Delta c/c: ' + str(np.mean(cstrain)))


