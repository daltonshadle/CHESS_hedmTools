#!/bin/bash

MY_PRE="det6_load0MPa"
NUM_GRAINS="728"

TESR_FN="${MY_PRE}_tesr"
TESS_FN="${MY_PRE}_tess"
MESH_FN="${MY_PRE}_mesh"

TESR_PNG_FN="${MY_PRE}_tesr_png"
TESS_PNG_FN="${MY_PRE}_tess_png"
MESH_PNG_FN="${MY_PRE}_mesh_png"

STATS_FN="${MY_PRE}_stats"


# Create tessellation
neper -T -n ${NUM_GRAINS} -dim 3 -domain "cube(1.0,0.3,1.0)" -loadpoint "file("${MY_PRE}.npos"):dim" -oricrysym cubic -reg 1 -mloop 10 -morpho voronoi -morphooptistop itermax=5000 -o $TESS_FN

# Visualize tessellation
neper -V "${TESS_FN}.tess" -datacellcol id -imagesize 1200:1200 -cameracoo 5:3:3 -cameralookat 0:0:0 -print $TESS_PNG_FN

# Mesh tessellation
# neper -M "${TESS_FN}.tess" -part 32:8 -for msh,fepx:legacy -order 2 -faset z0,z1,x0,x1,y0,y1 -o $MESH_FN
# elttype can be tri, quad, quad9, tet, hex, default is tet
neper -M "${TESS_FN}.tess" -elttype tet -rcl 0.9 -for msh,fepx:legacy -order 2 -faset z0,z1,x0,x1,y0,y1 -part 64 -o $MESH_FN

# Visualize mesh
neper -V "${MESH_FN}.msh" -showelt1d all -dataelset3dcol id -dataelt1drad 0.0025 -dataelt3dedgerad 0.0025 -imagesize 1200:1200 -cameracoo 5:3:3 -cameralookat 0:0:0 -print $MESH_PNG_FN

# Get Voronoi statistics
neper -T -loadtess "${TESS_FN}.tess" -statcell vol,x,y,z -o $STATS_FN


echo "Done!"
exit 0