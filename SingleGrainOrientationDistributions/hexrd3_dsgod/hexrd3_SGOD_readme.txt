# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:52:30 2022

@author: Dalton Shadle
"""

*** First two files might be unnecessary, see notes
hexrd3_GenerateEtaOmeMapsSGOD.py
- Script for generating eta_omega_maps for each individual grain
- HOWEVER, during my testing, I did not see any difference between polar maps
  generated with or without grain_params, so my conclusion is that grain_params
  is unnecessary and eta_omega_maps generated during indexing should be 
  sufficient for generating DSGODs for all grains, in which case a user could
  pass in the path to the indexing eta_omega_map and only run hexrd3_ConstructSGOD.py 
  
hexrd3_instrument.py
- hexrd3 script updated in extract_polar_maps (line 841) to include 
  grain_params for each individual grain
- This is the way Kelly Nygren set it up for here GOE calculations
- Newer versions of hexrd have done away with this extra parameter (grain_params)
  in extract_polar_maps so it was re-included here
- HOWEVER, during my testing, I did not see any difference between polar maps
  generated with or without grain_params, so my conclusion is that grain_params
  is unnecessary and eta_omega_maps generated during indexing should be 
  sufficient for generating DSGODs for all grains, in which case a user could
  pass in the path to the indexing eta_omega_map and only run hexrd3_ConstructSGOD.py 


*** Next two files are necessary
hexrd3_ConstructSGOD.py
- Script for constructing DSGODs from eta_omega_maps for grains in grains.out
- Uses updated hexrd3 script hexrd3_indexer.py to pull intensity information
- Can be post processed using the "SGODAnalysis.py" file

hexrd3_indexer.py
- hexrd3 script updated to pull intensity information from eta_omega_maps when 
  indexing orientations for constructing DSGODs
- Each updated function will have "_dsgod" tagged on to the end
- Added an additional parameter to PaintGrid (dsgod=False) as a flag for switching
  modes from indexing to constructing DSGODs
- THIS FILE WILL NEED TO BE RENAMED TO "indexer.py" AND REPLACE THE INDEXER FILE
  IN YOUR HEXRD3 ENVIRONMENT (note, I'm working on updating so we don't have
  to do this, but this works for now. A simple copy-paste of the code into 
  hexrd3_ConstructSGOD.py may be in order or possibly seeing if the hexrd team
  will add it to their main branch)