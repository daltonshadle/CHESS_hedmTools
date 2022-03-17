#!/bin/bash

MY_PRE="ss718_ff_sc32"
MICRO_FN_OUT="${MY_PRE}"
MICRO_PNG_FN_OUT="${MY_PRE}_png"



# Create tessellation and tesr
# neper -T -n ${NUM_GRAINS} -dim 3 -domain "cube(1.0,0.3,1.0)" -loadpoint "file("${MY_PRE}.npos"):dim" -oricrysym cubic -reg 1 -mloop 10 -morpho voronoi -morphooptistop itermax=5000 -o $TESS_FN
#neper -T -n 815 -dim 3 -domain "cube(0.990,0.990,0.170)" -loadpoint "file("ss718_sc28_neper_centroid.npos"):dim" -reg 1 -mloop 10 -morpho voronoi -morphooptistop itermax=5000 -format tess,tesr -tesrsize "198:198:34" -tesrformat "ascii" -o $MICRO_FN_OUT
neper -T -n 748 -dim 3 -domain "cube(0.990,0.990,0.170)" -loadpoint "file("ss718_sc32_neper_centroid.npos"):dim" -reg 1 -mloop 10 -morpho voronoi -morphooptistop itermax=5000 -format tess,tesr -tesrsize "198:198:34" -tesrformat "ascii" -o $MICRO_FN_OUT

# Visualize tessellation
neper -V "${MICRO_FN_OUT}.tess" -datacellcol id -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_tess"

# Visualize tesr
neper -V "${MICRO_FN_OUT}.tesr" -datacellcol id -imagesize 1200:1200 -cameraangle 24 -print "${MICRO_PNG_FN_OUT}_tesr"


echo "Done!"
exit 0
