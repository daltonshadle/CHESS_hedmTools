#!/bin/bash

MY_PRE="ss718_total_nf_with_ff"

TESR_FN="${MY_PRE}_tesr"
TESS_FN="${MY_PRE}_tess"
MESH_FN="${MY_PRE}_mesh"

TESR_PNG_FN="${MY_PRE}_tesr_png"
TESS_PNG_FN="${MY_PRE}_tess_png"
MESH_PNG_FN="${MY_PRE}_mesh_png"

STATS_FN="${MY_PRE}_stats"

# Visualize tesr
# neper -V example_tesr.tesr -datacellcol id -imagesize 600:600 -cameracoo 2:2:1.5 -cameralookat 0:0:0 -print example_tesr
neper -V "${TESR_FN}.tesr" -datacellcol id -imagesize 1200:1200 -cameraangle 24 -print $TESR_PNG_FN

# Create tessellation
# neper -T -n from_morpho -domain "cube(0.5,0.5,0.5)" -morpho "tesr:file(example_tesr.tesr)" -morphooptiobjective "pts(region=surf,res=7)" -morphooptistop itermax=30 -reg 1 -o example_tess
#neper -T -n from_morpho -domain "cube(0.990,0.990,0.550)" -morpho "tesr:file("${TESR_FN}.tesr")" -morphooptiobjective "pts(region=surf,res=7)" -morphooptistop itermax=50 -reg 1 -o $TESS_FN
neper -T -n from_morpho -domain "cube(0.820,0.990,0.990)" -morpho "tesr:file("${TESR_FN}.tesr")" -morphooptiobjective "tesr:pts(region=surf, res=7)" -morphooptistop itermax=50 -reg 1 -o $TESS_FN

# Visualize tessellation
#neper -V example_tess.tess -datacellcol id -imagesize 600:600 -cameracoo 2:2:1.5 -cameralookat 0:0:0 -print example_tess
neper -V "${TESS_FN}.tess" -datacellcol id -imagesize 1200:1200 -cameraangle 24 -print $TESS_PNG_FN

# Mesh tessellation
# neper -M example_tess.tess -part 32:8 -for msh,fepx -order 2 -faset z0,z1,x0,x1,y0,y1
neper -M "${TESS_FN}.tess" -part 32:8 -for msh,fepx:legacy -rcl 2 -order 2 -faset z0,z1,x0,x1,y0,y1 -o $MESH_FN

# Visualize mesh
# neper -V example_mesh.msh -showelt1d all -dataelset3dcol id -dataelt1drad 0.0025 -dataelt3dedgerad 0.0025 -imagesize 600:600 -cameracoo 3:3:2.2 -cameralookat 0.5:0.5:0.3 -print example_mesh
#neper -V "${MESH_FN}.msh" -showelt1d all -dataelset3dcol id -dataelt1drad 0.0025 -dataelt3dedgerad 0.0025 -imagesize 1200:1200 -cameracoo 8:5:8 -cameralookat 0:0:0 -print $MESH_PNG_FN
neper -V "${TESS_FN}.tess","${MESH_FN}.msh" -dataelsetcol id -imagesize 1200:1200 -cameraangle 24 -print $MESH_PNG_FN

# Get Voronoi statistics
# neper -T -loadtess example_tess.tess -statcell vol,x,y,z -o example_tess
#neper -T -loadtess "${TESS_FN}.tess" -statcell vol,x,y,z -o $STATS_FN


echo "Done!"
exit 0
