import matplotlib.pyplot as plt
import numpy as np

import hexrd.xrd.rotations as rot

import scipy.ndimage as img
from skimage.transform import iradon, radon, rescale

import hexrd.matrixutil as mutil

from scipy.stats import gaussian_kde
from PIL import Image
import cStringIO

from hexrd              import matrixutil as mutil
from hexrd          import rotations  as rot
from hexrd          import symmetry   as sym

import copy
from cycler import cycler

import cPickle as cpl

#%%
#------------Import data into new array-------------#
#please note, forloops and labelling for importing data will be based on however you saved your analysis folders for grains.out scans
#please change accordingly

grain_out_root = '/nfs/chess/aux/user/ken38/Ti7_project/ff_data_all/new-build/ti7-05-new/'
test_name = 'ti7-05'
initial_scan = 11
init_grain_out_file= grain_out_root + 'new-' + test_name + '-scan-%d/grains.out' % (initial_scan)

tmp_data_0=np.loadtxt(init_grain_out_file)

id_0=tmp_data_0[:,0]
comp_0=tmp_data_0[:,1]
chi_0=tmp_data_0[:,2]
strain_0=tmp_data_0[:,15:]
ori_0=tmp_data_0[:,3:6]
pos_0=tmp_data_0[:,6:9]
v_strain_0=np.sum(strain_0[:,0:3],axis=1)/3.

#%%
#----------Good grain parameters ------------#
#this block defines what should be considered a "good" grain

good_grains=np.where(np.logical_and(np.logical_and(comp_0>0.8,chi_0<0.009),np.abs(v_strain_0)<0.001))[0]

#%%
#---------- Correcting for vertical variation in beam energy ----------
#this block corrects the vertical variation in beam energy based on the initial scan

x = pos_0[good_grains,1]
y = v_strain_0[good_grains]
p =  np.polyfit(x, y, 1) #polynomial fit of gradient p[0] will be the slope p[1] the intercept

#plot of old values, polyfit
fig, ax = plt.subplots()
ax.plot(x,y,'gx', label='old')
ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),'g-',label='polyfit')
plt.xlabel ('t_vec_c [y]')
plt.ylabel ('avg_vol_strain')

#updating v_strain_0 to corrected values
v_strain_0 = v_strain_0-p[0]*pos_0[:,1]-p[1]

#add corrected values to plot
ax.plot(pos_0[good_grains,1],v_strain_0[good_grains],'bo', label='corrected')
legend = ax.legend(loc='lower right')
#%%
n_grains=len(good_grains)

astrain=np.zeros(n_grains)
astrain2=np.zeros(n_grains)
cstrain=np.zeros(n_grains)

strain_0[:,0]=strain_0[:,0]-(p[0]*pos_0[:,1]+p[1])
strain_0[:,1]=strain_0[:,1]-(p[0]*pos_0[:,1]+p[1])
strain_0[:,2]=strain_0[:,2]-(p[0]*pos_0[:,1]+p[1])

for ii in np.arange(n_grains):
    ti=good_grains[ii]
    strain_ten=np.array([[strain_0[ti,0],strain_0[ti,5],strain_0[ti,4]],[strain_0[ti,5],strain_0[ti,1],strain_0[ti,3]],[strain_0[ti,4],strain_0[ti,3],strain_0[ti,2]]])
    R=rot.rotMatOfExpMap(ori_0[ti,:])

    strain_ten_c=np.dot(R.T,np.dot(strain_ten,R))
    #print(strain_ten_c)
    astrain[ii]=strain_ten_c[0,0]
    astrain2[ii]=strain_ten_c[1,1]
    cstrain[ii]=strain_ten_c[2,2]

plt.figure(2)
plt.plot(astrain,'x')
plt.plot(astrain2,'gx')
plt.plot(cstrain,'rx')


#==============================================================================
# %% ELASTIC MODULI (Ti-7Al) #user should edit for their particular material
#==============================================================================
#Values in GPa

C11=176.1 #Venkataraman DOI: 10.1007/s11661-017-4024-y (2017)
C33=190.5
C44=50.8
C66=44.6

C12=86.9
C13=68.3

B=110

c_mat_C=np.array([[C11,C12,C13,0.,0.,0.],
                  [C12,C11,C13,0.,0.,0.],
                  [C13,C13,C33,0.,0.,0.],
                  [0.,0.,0.,C44,0.,0.],
                  [0.,0.,0.,0.,C44,0.],
                  [0.,0.,0.,0.,0.,C66]])*1e9

#==============================================================================
# %% Calculating Stresses
#==============================================================================
#this block is used to calculate stress tensors and particular stresses
#you will want to edit this for your own scripts

start_scan = 11
end_scan = 66
loadsteps = np.abs(start_scan - end_scan)+1
bad_scan = 54 #needs to have value, if you have no missing scans just put in a number larger than end scan

great_grains = good_grains
num_grains=len(great_grains)

count = 0
cmap=plt.cm.CMRmap
c = cycler('color',cmap(np.linspace(num_grains,0,1)))
plt.rcParams["axes.prop_cycle"] = c

mycmap = cmap(np.linspace(1,0,num_grains))

great_grains_stresses = np.zeros([loadsteps,num_grains,6])
great_grains_chi = np.zeros([loadsteps,num_grains])
great_grains_comp = np.zeros([loadsteps,num_grains])

#plt.close("all")
for i in range(start_scan,end_scan+1,1):
    if i != bad_scan:
        grains_out_file= grain_out_root + 'new-' + test_name + '-scan-%d/grains.out' % (i)
        tmp_data_x=np.loadtxt(grains_out_file)
        strain_x=tmp_data_x[:,15:]
        pos_x=tmp_data_x[:,6:9]
        strain_x[:,0]=strain_x[:,0]-(p[0]*pos_x[:,1]+p[1])
        strain_x[:,1]=strain_x[:,1]-(p[0]*pos_x[:,1]+p[1])
        strain_x[:,2]=strain_x[:,2]-(p[0]*pos_x[:,1]+p[1])
        exmap_x=tmp_data_x[:,3:6]
        strainTmp=np.atleast_2d(strain_x[great_grains]).T
        expMap=np.atleast_2d(exmap_x[great_grains]).T

        stress_S=np.zeros([num_grains,6])
        stress_C=np.zeros([num_grains,6])
        stress_prin=np.zeros([num_grains,3])
        hydrostatic=np.zeros([num_grains,1])
        pressure=np.zeros([num_grains,1])
        max_shear=np.zeros([num_grains,1])
        von_mises=np.zeros([num_grains,1])
           #Turn exponential map into an orientation matrix
        chi_x_t= tmp_data_x[:,2]
        chi_x=chi_x_t[great_grains]

        comp_x_t=tmp_data_x[:,1]
        comp_x=comp_x_t[great_grains]

        id_x_t=tmp_data_x[:,0]
        id_x=id_x_t[great_grains]
        count = 0
       # plt.figure(ii)
        for ii in range (0,num_grains-1,1):
            count+=1
            #if chi_x[great_grains[ii]]<0.002 and comp_x[great_grains[ii]]>0.99:
            great_grains_chi[i-start_scan,ii]=chi_x[ii]
            great_grains_comp[i-start_scan,ii]=comp_x[ii]
         #       plt.figure((great_grains[ii]))
            Rsc=rot.rotMatOfExpMap(expMap[:,ii])

            strainTenS = np.zeros((3, 3), dtype='float64')
            strainTenS[0, 0] = strainTmp[0,ii]
            strainTenS[1, 1] = strainTmp[1,ii]
            strainTenS[2, 2] = strainTmp[2,ii]
            strainTenS[1, 2] = strainTmp[3,ii]
            strainTenS[0, 2] = strainTmp[4,ii]
            strainTenS[0, 1] = strainTmp[5,ii]
            strainTenS[2, 1] = strainTmp[3,ii]
            strainTenS[2, 0] = strainTmp[4,ii]
            strainTenS[1, 0] = strainTmp[5,ii]

            strainTenC=np.dot(np.dot(Rsc.T,strainTenS),Rsc)
            strainVecC = mutil.strainTenToVec(strainTenC)

            v_strain_gg=np.trace(strainTenS)/3

        #Calculate stress
            stressVecC=np.dot(c_mat_C,strainVecC)
            stressTenC = mutil.stressVecToTen(stressVecC)
            stressTenS = np.dot(np.dot(Rsc,stressTenC),Rsc.T)
            stressVecS = mutil.stressTenToVec(stressTenS)

        #Calculate hydrostatic stress
            hydrostaticStress=(stressVecS[:3].sum()/3)
            w,v = np.linalg.eig(stressTenC)
            maxShearStress=(np.max(w)-np.min(w))/2.

        #Calculate Von Mises Stress
            devStressS=stressTenS-hydrostaticStress*np.identity(3)
            vonMisesStress=np.sqrt((3/2)*(devStressS**2).sum())

            stress_data=dict()

            stress_data['stress_S']=stress_S
            stress_data['stress_C']=stress_C
            stress_data['hydrostatic']=hydrostaticStress
            stress_data['max_shear']=max_shear
            stress_data['pressure']=pressure
            stress_data['von_mises']=von_mises
            stress_data['principal']=stress_prin

#
            great_grains_stresses[i-start_scan,ii,:] = stressVecS.T
#
# Save the image in memory in PNG format

#%%LOAD DIC DATA --- this is dic stress-strain data saved in npy format

dic_stress_strain = np.load('dic_stress_strain.npy')

#%%Pull dic strains correlating to scans in ff

strain_corr = dic_stress_strain[6:62,0]

#%% plotting parameters
SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%% coloring based on stress to make linear distribution for one strain
count = 0
cmap=plt.cm.inferno
good_chi = np.array(np.where((great_grains_chi[:,:]<0.0045).all(axis=0))).T
#good_chi = np.array(np.unique(np.where(np.logical_and(np.logical_and((great_grains_chi[:,:]<0.003).all(axis=0),(great_grains_comp[:21,:]>0.8).all(axis=0)), (great_grains_chi[50:56,:]<0.8).all(axis=0)))))
good_grains = id_x[good_chi]

c = cycler('color',cmap(np.linspace(len(good_chi),0,1)))
plt.rcParams["axes.prop_cycle"] = c
mycmap = cmap(np.linspace(1,0,len(good_chi)))

y_stresd= np.reshape(great_grains_stresses[:,good_chi,1],[great_grains_stresses.shape[0],len(good_chi)]).T
y_stress_sort = y_stresd[np.argsort(y_stresd[:,15])].T
#%% plotting stress strain data
scan = np.linspace(start_scan, end_scan, loadsteps)
plt.figure(20)
count=0
for i in range (0,len(good_chi)):
     #plt.plot(strain_corr[:28],great_grains_stresses[:,good_chi[i],1]/10**6,'x',color=mycmap[count],alpha=0.25)
#    count += 1
     #value = (great_grains_stresses[8,good_chi[i],1]-np.min(great_grains_stresses[8,good_chi,1]))/(np.max(great_grains_stresses[8,good_chi,1])-np.min(great_grains_stresses[8,good_chi,1]))
         #a = np.asscalar(value_r[i])
    plt.plot(strain_corr[:bad_scan-start_scan],y_stress_sort[:bad_scan-start_scan,i]/10**6,linestyle='-',color=mycmap[count],alpha=0.25)
    plt.plot(strain_corr[(bad_scan-start_scan)+1:loadsteps],y_stress_sort[(bad_scan-start_scan)+1:,i]/10**6,linestyle='-',color=mycmap[count],alpha=0.25)
    #plt.plot(scan[:bad_scan-start_scan],y_stress_sort[:bad_scan-start_scan,i]/10**6,linestyle='-',color=mycmap[count],alpha=0.25)
    #plt.plot(scan[bad_scan-start_scan+1:loadsteps-1],y_stress_sort[(bad_scan-start_scan)+1:loadsteps-1,i]/10**6,linestyle='-',color=mycmap[count],alpha=0.25)
    count += 1

#%%------------PLOT MACROSCOPIC STRESS STRAIN CURVE FROM DIC----------------#
plt.plot(strain_corr[:bad_scan-start_scan],y_stress_sort[:bad_scan-start_scan,3]/10**6,linestyle='-',color='black',alpha=0.25)
plt.plot(strain_corr[(bad_scan-start_scan)+1:loadsteps],y_stress_sort[(bad_scan-start_scan)+1:,3]/10**6,linestyle='-',color='black',alpha=0.25)
plt.ylim(0,900)
#plt.xlim(start_scan,end_scan)
#plt.xlim(0,np.max(strain_corr))
plt.xlabel('Macroscopic Strain')
plt.ylabel('Grain Stress_yy (MPa)')
