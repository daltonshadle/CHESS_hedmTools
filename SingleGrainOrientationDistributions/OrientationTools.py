#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:13:32 2022

@author: djs522
"""

# *****************************************************************************
# IMPORTS
# *****************************************************************************
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if sys.version_info[0] < 3:
    # python 2
    import hexrd.xrd.symmetry as hexrd_sym
    from hexrd.xrd import transforms_CAPI as xfcapi
    from hexrd.xrd import rotations  as hexrd_rot
else:
    # python 3
    from hexrd import symmetry as hexrd_sym
    from hexrd import rotations as hexrd_rot
    from hexrd.transforms import xfcapi
    from hexrd.xrdutil import EtaOmeMaps

# *****************************************************************************
# CONSTANTS
# *****************************************************************************
pi = np.pi

# *****************************************************************************
# FUNCTION DECLARATION AND IMPLEMENTATION
# *****************************************************************************
def mat_row_normalize(mat):
    """
    Purpose: Normalizes the rows in each matrix
    Input:   mat (m x n matrix) - 2D matrix to normalize rows
    Output:  norm_mat (m x n matrix) - mat with normalized row values
    Notes:   None
    """
    return np.divide(mat, np.linalg.norm(mat, axis=1)[:, None])

def PlotFR(sym, ax, elev=30, azim=45):
    if sym == 'cubic':
        [verts, plot_order, _, _]= CubPolytope()

        for i in range(0, plot_order.shape[0]):
            ax.plot(verts[plot_order[i,:],0], verts[plot_order[i,:],1],
            verts[plot_order[i,:],2], color='black')
        rod_lim = np.sqrt(2) - 1
        ax.set_xlim(-rod_lim, rod_lim)
        ax.set_ylim(-rod_lim, rod_lim)
        ax.set_zlim(-rod_lim, rod_lim)

        ax.set_xlabel('$R_x$')
        ax.set_ylabel('$R_y$')
        ax.set_zlabel('$R_z$')
        
        ax.view_init(elev=elev, azim=azim)
        
        return ax
    else:
        print('Symmetry type not supported')

def CubPolytope():
    b1 = np.tan(np.pi/8)

    x = b1
    z = b1**2

    # vertices of fundamental region
    verts = np.array([[ x, x, z], [ x, z, x], [ z, x, x],
                      [-x, x, z], [-x, z, x], [-z, x, x],
                      [-x,-x, z], [-x,-z, x], [-z,-x, x],
                      [ x,-x, z], [ x,-z, x], [ z,-x, x],
                      [ x, x,-z], [ x, z,-x], [ z, x,-x],
                      [-x, x,-z], [-x, z,-x], [-z, x,-x],
                      [-x,-x,-z], [-x,-z,-x], [-z,-x,-x],
                      [ x,-x,-z], [ x,-z,-x], [ z,-x,-x]])

    # faces of fundamental region with orders of vertices
    f100 = np.array([[ 1,     2,    11,    10,    22,    23,    14,    13],
                     [ 4,     5,     8,     7,    19,    20,    17,    16],
                     [ 1,     3,     6,     4,    16,    18,    15,    13],
                     [ 7,     9,    12,    10,    22,    24,    21,    19],
                     [ 2,     3,     6,     5,     8,     9,    12,    11],
                     [14,    15,    18,    17,    20,    21,    24,    23]])

    f111 = np.array([[ 1,  2,  3], [ 4,  5,  6], [ 7,  8,  9], [10, 11, 12],
                     [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]])

    # fix off by one indexing with MATLAB to Python
    f100 = f100 - 1
    f111 = f111 - 1

    connect = np.array([ 12,    0,     0,     0,     0,     0,     0,     0])
    plot_order = np.vstack([f100, connect])

    return [verts, plot_order, f100, f111]

def kocks2bunge(kocks, units='radians'):
    '''
    % BungeOfKocks - Bunge angles from Kocks angles.
    %   
    %   USAGE:
    %
    %   bunge = BungeOfKocks(kocks, units)
    %
    %   INPUT:
    %
    %   kocks is 3 x n,
    %         the Kocks angles for the same orientations
    %   units is a string,
    %         either 'degrees' or 'radians'
    %
    %   OUTPUT:
    %
    %   bunge is 3 x n,
    %         the Bunge angles for n orientations 
    %
    %   NOTES:
    %
    %   *  The angle units apply to both input and output.
    %
    '''
    if (units == 'degrees'):
        indeg = True
    elif (units == 'radians'):
        indeg = False
    else:
        raise ValueError('angle units need to be specified:  ''degrees'' or ''radians''')
    
    if (indeg):
        pi_over_2 = 90
    else:
        pi_over_2 = pi/2
    
    bunge = kocks
    
    bunge[0, :] = kocks[0, :] + pi_over_2
    bunge[2, :] = pi_over_2 - kocks[2, :]
    
    return bunge

def bunge2kocks(bunge, units='radians'):
    '''
    % KocksOfBunge - Kocks angles from Bunge angles.
    %   
    %   USAGE:
    %
    %   bunge = KocksOfBunge(bunge, units)
    %
    %   INPUT:
    %
    %   bunge is 3 x n,
    %         the Bunge angles for the same orientations
    %   units is a string,
    %         either 'degrees' or 'radians'
    %
    %   OUTPUT:
    %
    %   kocks is 3 x n,
    %         the K angles for n orientations 
    %
    %   NOTES:
    %
    %   *  The angle units apply to both input and output.
    %
    '''
    if (units == 'degrees'):
        indeg = True
    elif (units == 'radians'):
        indeg = False
    else:
        raise ValueError('angle units need to be specified:  ''degrees'' or ''radians''')
    
    if (indeg):
        pi_over_2 = 90
    else:
        pi_over_2 = pi/2
    
    kocks = bunge
    
    kocks[0, :] = bunge[0, :] - pi_over_2
    kocks[2, :] = pi_over_2 + bunge[2, :]
    return kocks

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

def rod2quat(rod):
    """
    % QuatOfRod - Quaternion from Rodrigues vectors.
    %
    %   USAGE:
    %
    %   quat = QuatOfRod(rod)
    %
    %   INPUT:
    %
    %   rod  is n x 3,
    %        an array whose columns are Rodrigues parameters
    %
    %   OUTPUT:
    %
    %   quat is n x 4,
    %        an array whose columns are the corresponding unit
    %        quaternion parameters; the first component of
    %        `quat' is nonnegative
    %
    """
    cphiby2 = np.cos(np.arctan(np.sqrt((rod*rod).sum(axis=1))))
    cphiby2 = cphiby2.reshape(cphiby2.shape[0], 1)
    return np.hstack([cphiby2, np.multiply(np.tile(cphiby2, (1, 3)), rod) ])

def rod2bunge(rod):
    """
    % QuatOfRod - Quaternion from Rodrigues vectors.
    %
    %   USAGE:
    %
    %   quat = QuatOfRod(rod)
    %
    %   INPUT:
    %
    %   rod  is n x 3,
    %        an array whose columns are Rodrigues parameters
    %
    %   OUTPUT:
    %
    %   quat is n x 4,
    %        an array whose columns are the corresponding unit
    %        quaternion parameters; the first component of
    %        `quat' is nonnegative
    %
    """
    
    # precalculate terms
    r_sum = np.arctan(rod[:, 2])
    r_dif = np.arctan(rod[:, 1] / rod[:, 0])
    
    # assign bunge array values
    bunge = np.zeros(rod.shape)
    bunge[:, 0] = r_sum + r_dif
    bunge[:, 1] = 2.0 * np.arctan(rod[:, 1] * np.cos(r_sum) / np.sin(r_dif))
    bunge[:, 2] = r_sum - r_dif
    
    return bunge
    
def CubSymmetries():
    '''
    % CubSymmetries - Return quaternions for cubic symmetry group.
    %
    %   USAGE:
    %
    %   csym = CubSymmetries
    %
    %   INPUT:  none
    %
    %   OUTPUT:
    %
    %   csym is 4 x 24, 
    %        quaternions for the cubic symmetry group
    %
    '''
    AngleAxis = np.array([  [0.0,      1,   1,   1], # identity
                            [pi*0.5,   1,   0,   0], # fourfold about x1
                            [pi,       1,   0,   0],
                            [pi*1.5,   1,   0,   0],
                            [pi*0.5,   0,   1,   0], # fourfold about x2
                            [pi,       0,   1,   0],
                            [pi*1.5,   0,   1,   0],
                            [pi*0.5,   0,   0,   1], # fourfold about x3
                            [pi,       0,   0,   1],
                            [pi*1.5,   0,   0,   1],
                            [pi*2/3,   1,   1,   1], # threefold about 111
                            [pi*4/3,   1,   1,   1],
                            [pi*2/3,  -1,   1,   1], # threefold about 111
                            [pi*4/3,  -1,   1,   1],
                            [pi*2/3,   1,  -1,   1], # threefold about 111
                            [pi*4/3,   1,  -1,   1],
                            [pi*2/3,  -1,  -1,   1], # threefold about 111
                            [pi*4/3,  -1,  -1,   1],
                            [pi,       1,   1,   0], # twofold about 110
                            [pi,      -1,   1,   0],
                            [pi,       1,   0,   1],
                            [pi,       1,   0,  -1],
                            [pi,       0,   1,   1],
                            [pi,       0,   1,  -1]]).T
                 
    Angle = AngleAxis[0,:]
    Axis  = AngleAxis[1:4,:]
    # Axis does not need to be normalized; it is done
    # in call to QuatOfAngleAxis.
    return QuatOfAngleAxis(Angle, Axis);

def ElementTypeStruct(typename):
    '''
    % ElementTypeStruct - Element type structure
    %
    %   USAGE:
    %
    %   etype = ElementTypeStruct(typename)
    %   
    %   INPUT:
    %
    %   typename is a string
    %            Choices are:
    %            (0-D)   'point'
    %            (1-D)   'lines:2', 'lines:3'
    %            (2-D)   'triangles:3', 'triangles:6', 'quads:4'
    %            (3-D)   'tets:4', 'tets:10', 'tets:10:fepx', 'bricks:8'
    %
    %   OUTPUT:
    %
    %   NOTES:
    %
    %   * Standard ordering of all elements is dictionary(z, y, x)  [z is sorted first]
    %     i.e. x varies fastest, then y, then z 
    %
    %   * Need to do 'quads:8', 'quads:9', 'bricks:20'
    %
    '''
    
    tname = str.lower(typename)
    
    # 0-D Elements
    if tname == 'point':
        dimension      = 0;
        is_rectangular = 1;
        is_simplicial  = 1;
        surftype = '';
        elsurfs  = [];
    # 1-D Elements
    elif tname == 'lines:2':
        dimension      = 1;
        is_rectangular = 1;
        is_simplicial  = 1;
        surftype = 'point';
        elsurfs  = np.array([1, 2]);
    elif tname == 'lines:3':
        dimension      = 1;
        is_rectangular = 1;
        is_simplicial  = 1;
        surftype = 'point';
        elsurfs  = np.array([1, 3]);
    # 2-D Elements
    elif tname == 'triangles:3':
        dimension      = 2;
        is_rectangular = 0;
        is_simplicial  = 1;
        surftype = 'lines:2';
        elsurfs  = np.array([[1, 2], [3, 1], [2, 3]]);
    elif tname == 'triangles:6':
        dimension      = 2;
        is_rectangular = 0;
        is_simplicial  = 1;
        surftype = 'lines:3';
        elsurfs  = np.array([[1, 2, 3], [6, 4, 1], [3, 5, 6]]);
    elif tname == 'quads:4':
        dimension      = 2;
        is_rectangular = 1;
        is_simplicial  = 0;
        surftype = 'lines:2';
        elsurfs  = np.array([[1, 2], [3, 1], [2, 4], [4, 3]]);
    # 3-D Elements
    elif tname == 'tets:4':
        dimension      = 3;
        is_rectangular = 0;
        is_simplicial  = 1;
        surftype = 'triangles:3';
        elsurfs  = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 2], [2, 4, 3]]);
    elif tname == 'tets:10:fepx':
        dimension      = 3;
        is_rectangular = 0;
        is_simplicial  = 1;
        surftype = 'triangles:6';
        elsurfs  = np.array([[1, 2, 3,  4,  5, 6],
    	                     [1, 6, 5,  9, 10, 7],
    	                     [1, 7, 10, 8,  3, 2],
    	                     [3, 8, 10, 9,  5, 4]]);
    elif tname == 'tets:10':
        dimension      = 3;
        is_rectangular = 0;
        is_simplicial  = 1;
        surftype = 'triangles:6';
        elsurfs  = np.array([[1,  7, 10, 8,  3, 2],
    	                     [1,  2,  3, 5,  6, 4],
    	                     [1,  4,  6, 9, 10, 7],
    	                     [10, 9,  6, 5,  3, 8]]);
    elif tname == 'bricks:8':
        dimension      = 3;
        is_rectangular = 1;
        is_simplicial  = 0;
        surftype = 'quads:4';
        elsurfs  = np.array([[1, 2, 3, 4],
    	                     [2, 1, 6, 5],
    	                     [1, 3, 5, 7],
    	                     [4, 2, 8, 6],
                             [3, 4, 7, 8],
                             [7, 8, 5, 6]]);
    else:
        raise ValueError('no such element type:  %s' %(typename))
    
    etype_dict = {'name': tname,
                  'surfs': elsurfs,
                  'surftype': surftype,
                  'dimension': dimension,
                  'is_rectangular': is_rectangular,
                  'is_simplicical': is_simplicial}
    return etype_dict

def HexSymmetries():
    '''
    % HexSymmetries - Quaternions for hexagonal symmetry group.
    %
    %   USAGE:
    %
    %   hsym = HexSymmetries
    %
    %   INPUT:  none
    %
    %   OUTPUT:
    %
    %   hsym is 4 x 12,
    %        it is the hexagonal symmetry group represented
    %        as quaternions
    %   
    '''
    p3  = pi/3;
    p6  = pi/6;
    ci  = np.cos(p6*np.arange(6));
    si  = np.sin(p6*np.arange(6));
    z6  = np.zeros([1, 6]);
    w6  = np.ones([1, 6]);
    pi6 = pi*np.ones([1, 6]);
    
    sixfold = np.vstack([p3*np.arange(6), z6, z6, w6])
    twofold = np.vstack([pi6, ci, si, z6])
    
    AngleAxis = np.hstack([sixfold, twofold]);
    
    Angle = AngleAxis[0, :];
    Axis  = AngleAxis[1:4, :];
    
    #  Axis does not need to be normalized in call to QuatOfAngleAxis.
    return QuatOfAngleAxis(Angle, Axis);

def QuatOfAngleAxis(angle, raxis):
    '''
    function quat = QuatOfAngleAxis(angle, raxis)
    % QuatOfAngleAxis - Quaternion of angle/axis pair.
    %
    %   USAGE:
    %
    %   quat = QuatOfAngleAxis(angle, rotaxis)
    %
    %   INPUT:
    %
    %   angle is an n-vector, 
    %         the list of rotation angles
    %   raxis is 3 x n, 
    %         the list of rotation axes, which need not
    %         be normalized (e.g. [1 1 1]'), but must be nonzero
    %
    %   OUTPUT:
    %
    %   quat is 4 x n, 
    %        the quaternion representations of the given
    %        rotations.  The first component of quat is nonnegative.
    %   
    '''
    
    '''
    %
    rescale = sphiby2 ./sqrt(dot(raxis, raxis, 1));
    %
    quat = [cphiby2; repmat(rescale, [3 1]) .* raxis ] ;
    '''
    
    halfangle = 0.5*angle.T;
    cphiby2   = np.cos(halfangle);
    sphiby2   = np.sin(halfangle);
    
    rescale = np.divide(sphiby2, np.linalg.norm(raxis, axis=0).T)
    
    quat = np.hstack([cphiby2[:, np.newaxis], np.multiply(np.tile(rescale, (3, 1)), raxis).T]).T
    
    q1negative = (quat[0, :] < 0)
    quat[:, q1negative] = -1*quat[:, q1negative]
    
    return quat

def rmat2quat(rmat):
    '''
    function quat = QuatOfRMat(rmat)
    % QuatOfRMat - Quaternion from rotation matrix
    %   
    %   USAGE:
    %
    %   quat = QuatOfRMat(rmat)
    %
    %   INPUT:
    %
    %   rmat is 3 x 3 x n,
    %        an array of rotation matrices
    %
    %   OUTPUT:
    %
    %   quat is 4 x n,
    %        the quaternion representation of `rmat'
    
    % 
    %  Find angle of rotation.
    %
    '''
    
    if len(rmat.shape) < 3:
        rmat = rmat.reshape(3,3,1)
    
    rmat_trace = rmat[0, 0, :] + rmat[1, 1, :] + rmat[2, 2, :]
    
    ca = 0.5 * (rmat_trace - 1)
    ca[ca > 1] = 1
    ca[ca < -1] = -1
    angle = np.squeeze(np.arccos(ca)).T
    
    #  Three cases for the angle:   
    #  *   near zero -- matrix is effectively the identity
    #  *   near pi   -- binary rotation; need to find axis
    #  *   neither   -- general case; can use skew part
    
    tol = 1.0e-4;
    anear0 = (angle < tol);
    angle[anear0] = 0;
    
    
    raxis = np.vstack([[(rmat[2, 1, :] - rmat[1, 2, :]).T],
                       [(rmat[0, 2, :] - rmat[2, 0, :]).T], 
                       [(rmat[1, 0, :] - rmat[0, 1, :]).T]])
    
    raxis = np.squeeze(raxis)
    raxis[:, anear0] = 1
    
    special = angle > (pi - tol)
    num_spec = np.count_nonzero(special)
    
    if num_spec > 0:
        rmat_spec = rmat[:, :, special]
        tmp = rmat_spec + np.tile(np.eye(3), [num_spec, 1, 1]).T
        tmpr = np.reshape(tmp, [3, 3*num_spec])
        tmpnrm = np.reshape(np.sum(tmpr.T * tmpr, axis=0), [3, num_spec])
        ind = np.argmax(tmpnrm)
        ind = ind + (np.arange(num_spec) * 3)
        saxis = np.reshape(tmpr[:, ind], [3, num_spec])
        raxis[:, special] = saxis

    quat = QuatOfAngleAxis(angle, raxis)
    return quat

def bunge2rmat(bunge, units='radians'):
    '''
    function rmat = RMatOfBunge(bunge, units)
    % RMatOfBunge - Rotation matrix from Bunge angles.
    %
    %   USAGE:
    %
    %   rmat = RMatOfBunge(bunge, units)
    %
    %   INPUT:
    %
    %   bunge is 3 x n,
    %         the array of Bunge parameters
    %   units is a string,
    %         either 'degrees' or 'radians'
    %       
    %   OUTPUT:
    %
    %   rmat is 3 x 3 x n,
    %        the corresponding rotation matrices
    %   
    '''
    if (units == 'degrees'):
        indeg = True
        bunge = bunge*(pi/180)
    elif (units == 'radians'):
        indeg = False
    else:
        raise ValueError('angle units need to be specified:  ''degrees'' or ''radians''')
    
    n    = bunge.shape[1]
    cbun = np.cos(bunge)
    sbun = np.sin(bunge)
    
    rmat = np.array([
                     [np.multiply(cbun[0, :], cbun[2, :]) - np.multiply(np.multiply(sbun[0, :], cbun[1, :]), sbun[2, :])],
                     [np.multiply(sbun[0, :], cbun[2, :]) + np.multiply(np.multiply(cbun[0, :], cbun[1, :]), sbun[2, :])],
                     [np.multiply(sbun[1, :], sbun[2, :])],
                     [np.multiply(-cbun[0, :], sbun[2, :]) - np.multiply(np.multiply(sbun[0, :], cbun[1, :]), cbun[2, :])],
                     [np.multiply(-sbun[0, :], sbun[2, :]) + np.multiply(np.multiply(cbun[0, :], cbun[1, :]), cbun[2, :])],
                     [np.multiply(sbun[1, :], cbun[2, :])],
                     [np.multiply(sbun[0, :], sbun[1, :])],
                     [np.multiply(-cbun[0, :], sbun[1, :])],
                     [cbun[1, :]]
                    ])
    rmat = np.reshape(rmat, [3, 3, n]);
    return rmat

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
    
    
    if crys_sym == 'cubic':
        phi1_mat, theta_mat, phi2_mat = np.meshgrid(np.arange(phi1_bnd[0], phi1_bnd[1] + phi1_step, phi1_step),
                                                    np.arange(theta_bnd[0], theta_bnd[1] + theta_step, theta_step),
                                                    np.arange(phi2_bnd[0], phi2_bnd[1] + phi2_step, phi2_step))
        discrete_FR_bunge = np.vstack([phi1_mat.flatten(), theta_mat.flatten(), phi2_mat.flatten()]).T
        
        discrete_FR_quat = bunge2quat(discrete_FR_bunge, units='degrees')
        discrete_FR_quat = hexrd_sym.toFundamentalRegion(discrete_FR_quat.T, crysSym='Oh').T
        discrete_FR_ret = discrete_FR_quat
        
        if ret_type == 'rod':
            discrete_FR_ret = quat2rod(discrete_FR_quat)
        
        discrete_FR_ret  = np.round(discrete_FR_ret, decimals=8)
        discrete_FR_ret = np.unique(discrete_FR_ret, axis=0)
        return discrete_FR_ret
    else:
        print('Crystal symmetry type is not supported at this time.')

def exp_map2rod(exp_map):
    return quat2rod(hexrd_rot.quatOfExpMap(exp_map).T)

def rod2exp_map(rod):
    return quat2exp_map(rod2quat(rod))

def calc_misorientation_quat(quat1, quat2_mat):
    '''
    Purpose: calculates misorientation between rod1 and rod2_mat

    Parameters
    ----------
    quat1 : TYPE
        DESCRIPTION.
    quat2_mat : TYPE
        DESCRIPTION.

    Returns
    -------
    mis_ang_deg : TYPE
        DESCRIPTION.
    mis_rod : TYPE
        DESCRIPTION.

    '''
    
    # calculate misorientation angles and vectors
    [mis_ang_rad, mis_quat] = hexrd_rot.misorientation(quat1, quat2_mat)
    mis_ang_deg = 180/np.pi * mis_ang_rad
    
    return mis_ang_deg, mis_quat

def calc_misorientation_rod(rod1, rod2_mat):
    '''
    Purpose: calculates misorientation between rod1 and rod2_mat

    Parameters
    ----------
    rod1 : TYPE
        DESCRIPTION.
    rod2_mat : TYPE
        DESCRIPTION.

    Returns
    -------
    mis_ang_deg : TYPE
        DESCRIPTION.
    mis_rod : TYPE
        DESCRIPTION.

    '''
    
    # convert rods to quats
    quat1 = rod2quat(rod1).T
    quat2_mat = rod2quat(rod2_mat).T
    
    # calculate misorientation angles and vectors
    [mis_ang_rad, mis_quat] = hexrd_rot.misorientation(quat1, quat2_mat)
    mis_ang_deg = 180/np.pi * mis_ang_rad
    mis_rod = quat2rod(mis_quat.T)
    
    return mis_ang_deg, mis_rod

def calc_misorientation_expmap(exp1, exp2_mat):
    '''
    Purpose: calculates misorientation between rod1 and rod2_mat

    Parameters
    ----------
    rod1 : TYPE
        DESCRIPTION.
    rod2_mat : TYPE
        DESCRIPTION.

    Returns
    -------
    mis_ang_deg : TYPE
        DESCRIPTION.
    mis_rod : TYPE
        DESCRIPTION.

    '''
    
    # convert rods to quats
    quat1 = hexrd_rot.quatOfExpMap(np.atleast_2d(exp1).T)
    quat2_mat = hexrd_rot.quatOfExpMap(exp2_mat.T)
    
    # calculate misorientation angles and vectors
    [mis_ang_rad, mis_quat] = hexrd_rot.misorientation(quat1, quat2_mat)
    mis_ang_deg = 180/np.pi * mis_ang_rad
    mis_expmap = quat2exp_map(mis_quat)
    
    return mis_ang_deg, mis_expmap

def calc_closest_orientaiton_configuration_quats(ref_quat, quats, sym='cubic'):
    '''
    Purpose: Find the closest quaternion configuration (inside or outside the 
             fundamental region) of quats to the ref_quat, this can be used in
             an effort to "rejoin" orientations separated once mapped to FR

    Parameters
    ----------
    ref_quat : numpy array (4 x 1)
        reference quaternion for all the other quats to find the closest
        configuration to it
    quats : numpy array (4 x n)
        an array of n quaternions to find closest configurations of
    sym : string, optional
        choose symmetry type. The default (and only option) is 'cubic'.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    q2_remap : numpy array (4 x n)
        remapped configuation of quats that is closest to ref_quat
    angle : numpy array (n)
        list of all misorientation angles of q2_remap quats and ref_quat in 
        radians

    '''
    
    q1 = ref_quat
    q2 = quats
    
    # *************************************************************************
    # the following code is augmented from hexrd.rotations.misorientation
    if not isinstance(q1, np.ndarray) or not isinstance(q2, np.ndarray):
        raise RuntimeError("quaternion args are not of type `numpy ndarray'")

    if q1.ndim != 2 or q2.ndim != 2:
        raise RuntimeError(
            "quaternion args are the wrong shape; must be 2-d (columns)"
        )

    if q1.shape[1] != 1:
        raise RuntimeError(
            "first argument should be a single quaternion, got shape %s"
            % (q1.shape,)
        )
    
    if sym == 'cubic':
        # in original code for hrexrd.rotations, sym is a tuple consisting of 
        # (crystal_symmetry, *sample_symmetry)
        sym = [CubSymmetries()]
        if len(sym) == 1:
            if not isinstance(sym[0], np.ndarray):
                raise RuntimeError("symmetry argument is not an numpy array")
            else:
                # add triclinic sample symmetry (identity)
                sym += (np.c_[1., 0, 0, 0].T,)
        elif len(sym) == 2:
            if not isinstance(sym[0], np.ndarray) \
              or not isinstance(sym[1], np.ndarray):
                raise RuntimeError(
                    "symmetry arguments are not an numpy arrays"
                )
        elif len(sym) > 2:
            raise RuntimeError(
                "symmetry argument has %d entries; should be 1 or 2"
                % (len(sym))
            )
        

    # set some lengths
    n = q2.shape[1]             # length of misorientation list
    m = sym[0].shape[1]         # crystal (right)
    p = sym[1].shape[1]         # sample  (left)

    # tile q1 inverse
    q1i = hexrd_rot.quatProductMatrix(hexrd_rot.invertQuat(q1), mult='right').squeeze()

    # convert symmetries to (4, 4) qprod matrices
    rsym = hexrd_rot.quatProductMatrix(sym[0], mult='right')
    lsym = hexrd_rot.quatProductMatrix(sym[1], mult='left')

    # Do R * Gc, store as
    # [q2[:, 0] * Gc[:, 0:m], ..., q2[:, n-1] * Gc[:, 0:m]]
    q2 = np.dot(rsym, q2).transpose(2, 0, 1).reshape(m*n, 4).T

    # Do Gs * (R * Gc), store as
    # [Gs[:, 0:p]*q[:,   0]*Gc[:, 0], ..., Gs[:, 0:p]*q[:,   0]*Gc[:, m-1], ...
    #  Gs[:, 0:p]*q[:, n-1]*Gc[:, 0], ..., Gs[:, 0:p]*q[:, n-1]*Gc[:, m-1]]
    q2 = np.dot(lsym, q2).transpose(2, 0, 1).reshape(p*m*n, 4).T

    # Calculate the class misorientations for full symmetrically equivalent
    # classes for q1 and q2.  Note the use of the fact that the application
    # of the symmetry groups is an isometry.
    eqvMis = hexrd_rot.fixQuat(np.dot(q1i, q2))

    # Reshape scalar comp columnwise by point in q2 (and q1, if applicable)
    sclEqvMis = eqvMis[0, :].reshape(n, p*m).T

    # Find misorientation closest to origin for each n equivalence classes
    #   - fixed quats so garaunteed that sclEqvMis is nonnegative
    qmax = sclEqvMis.max(0)

    # remap indices to use in eqvMis
    qmaxInd = (sclEqvMis == qmax).nonzero()
    qmaxInd = np.c_[qmaxInd[0], qmaxInd[1]]

    eqvMisColInd = np.sort(qmaxInd[:, 0] + qmaxInd[:, 1]*p*m)

    # store Rmin in q
    # mis = eqvMis[np.ix_(list(range(4)), eqvMisColInd)]

    angle = 2 * hexrd_rot.arccosSafe(qmax)
    # end augmented code
    # *************************************************************************
    
    # find the quaternions remapped to be closest to the center_quat
    q2_remap = q2[:, eqvMisColInd]
    
    return q2_remap, angle


def plot_eta_ome_maps(eta_ome_dir, vmin=None, vmax=None, show_hkl_list=None):
    if os.path.exists(eta_ome_dir):
        eta_ome = EtaOmeMaps(eta_ome_dir)
    else:
        raise ValueError('ETA-OME MAP NOT FOUND FOR GRAIN!')
    
    none_thresh = False
    if vmin is None:
        none_thresh = True
    
    if show_hkl_list is not None:
        eta_ome.iHKLList = np.intersect1d(eta_ome.iHKLList, show_hkl_list)
    print(eta_ome.iHKLList)
    for i_ring in range(eta_ome.iHKLList.size):
        cmap = plt.cm.hot
        cmap.set_under('b')
    
        fig, ax = plt.subplots()
        this_map_f = eta_ome.dataStore[i_ring]
        this_map_no_nan = this_map_f[~np.isnan(this_map_f)]
        #this_map_no_nan = this_map_no_nan[this_map_no_nan>0]
        print('eta-ome ring %i: Max=%f, Mean=%f, Median=%f' %(eta_ome.iHKLList[i_ring], 
                                                              np.max(this_map_no_nan), 
                                                              np.mean(this_map_no_nan), 
                                                              np.median(this_map_no_nan)))
        
        if none_thresh:
            vmin = (np.median(this_map_no_nan) + np.mean(this_map_no_nan)) / 2
            #vmin = np.median(this_map_no_nan) 
            #vmin = np.mean(this_map_no_nan)
            print('Min Threshold: %f' %(vmin))
        ax.imshow(this_map_f, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axis('tight')
        plt.draw()
    return 0

if __name__ == "__main__": 
    
    discrete_FR_rod = discretizeFundamentalRegion(phi1_step=5, theta_step=5, phi2_step=5, ret_type='rod')
    print(discrete_FR_rod.shape)
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ori_ax = Axes3D(fig)
    
    ori_ax.scatter(discrete_FR_rod[:, 0], discrete_FR_rod[:, 1], discrete_FR_rod[:, 2])
    ori_ax = PlotFR('cubic', ori_ax)
    
    # label axis and show
    ori_ax.set_xlabel('$R_{x}$')
    ori_ax.set_ylabel('$R_{y}$')
    ori_ax.set_zlabel('$R_{z}$')
    plt.show()
    
    quat = discretizeFundamentalRegion(phi1_step=5, theta_step=5, phi2_step=5, ret_type='quat')
    exp_map = quat2exp_map(quat)
    print(exp_map.shape)
    
    rod = np.array([[0, 0, 0.4], [0, 0, 0.425], [0, 0, 0.45]])
    rod = quat2rod(hexrd_rot.toFundamentalRegion(rod2quat(rod).T).T)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(rod[:, 0], rod[:, 1], rod[:, 2], c='r')
    ax = PlotFR('cubic', ax)

    ret_val = calc_closest_orientaiton_configuration_quats(rod2quat(np.atleast_2d(rod[0, :])).T, rod2quat(rod).T)

    rod_fix = quat2rod(ret_val[0].T)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(rod_fix[:, 0], rod_fix[:, 1], rod_fix[:, 2], c='g')
    ax = PlotFR('cubic', ax)
    plt.show()

















 
