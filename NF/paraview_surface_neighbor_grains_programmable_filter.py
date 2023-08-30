import numpy as np
from scipy.ndimage import morphology

# Region ID to find neighbors for (replace 3 with the desired region ID)
target_grain_id = 782

# Replace 'connected_regions' with your 3D array of connected region IDs
input_data = inputs[0]
gm = input_data.PointData['grain_id']
gm_shape = (400, 180, 400)
gm = np.reshape(gm, gm_shape)

# perform threshold and dilation
grain_gm_ind = (gm == target_grain_id)
dilated_grain_gm_ind = morphology.binary_dilation(grain_gm_ind, structure=np.ones([3, 3, 3]), iterations=1)

# get neighboring grain ids
neighboring_grain_gm_ind = np.zeros(grain_gm_ind.shape)
neighboring_grain_gm_ind[dilated_grain_gm_ind] = 1
neighboring_grain_gm_ind[grain_gm_ind] = 2

neighboring_grain_gm_ind = neighboring_grain_gm_ind.flatten()

for key in input_data.PointData.keys():
    output.PointData.append(input_data.PointData[key], key)

output.PointData.append(neighboring_grain_gm_ind, 'surface_neighbor_grains_ind')

