
#usage: dsgod_generate_eta_ome_maps.py [-h] [--select_grain_ids [select_ids]]
#                                      cfg samp_name scan_num grains_out_dir
#                                      frame_cache_dir output_dir

#dsgod_construct_dsgods.py [-h] [--mis_bound [mis_bound]]
#                                 [--mis_spacing [mis_spacing]]
#                                 [--select_grain_ids [select_ids]]
#                                 cfg samp_name scan_num grains_out_dir
#                                 output_dir

path_to_gen_eta_ome="/home/djs522/additional_sw/hexrd3/hexrd3_dsgod/dsgod_generate_eta_ome_maps.py"
path_to_construct_dsgod="/home/djs522/additional_sw/hexrd3/hexrd3_dsgod/dsgod_construct_dsgods.py"

basepath='/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/c65_0/'  
grains_out_dir="${basepath}"
frame_cache_dir="/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/ff/"                              
output_dir="${basepath}dsgod/"

eta_ome_map_file="/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/eta_ome.npz"
eta_ome_select_grain_id_file="/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/GLOBAL_LAYER0_IDS.npy"
dsgod_select_grain_id_file="/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/GLOBAL_LAYER0_IDS.npy"

samp_name="ss718-1"

mis_bound=3.0
mis_space=0.2

start_omega=0.0
end_omega=360.0

scan_array=(1076)
for scan_num in "${scan_array[@]}"
do
    # NOTE: This first function call to generate eta_omega_maps might not be necessary, see readme
    yaml="${basepath}c65_0_sc${scan_num}_dsgod.yml"
    grains_out="${grains_out_dir}${scan_num}/grains.out"
    eta_ome_python_call="python ${path_to_gen_eta_ome} ${yaml} ${samp_name} ${scan_num} ${grains_out} ${frame_cache_dir} ${output_dir} --start_omega ${start_omega} --end_omega ${end_omega} --select_grain_ids ${eta_ome_select_grain_id_file}"
    echo "STARTING ETA-OME MAP GENERATION ${scan_num}"
    #echo $eta_ome_python_call
    $eta_ome_python_call
    #dsgod_python_call="python ${path_to_construct_dsgod} ${yaml} ${samp_name} ${scan_num} ${grains_out} ${output_dir} --select_grain_ids ${select_grain_id_file_1} --mis_bound ${mis_bound} --mis_spacing ${mis_space}"
    dsgod_python_call="python ${path_to_construct_dsgod} ${yaml} ${samp_name} ${scan_num} ${grains_out} ${output_dir} --mis_bound ${mis_bound} --mis_spacing ${mis_space} --start_omega ${start_omega} --end_omega ${end_omega} --select_grain_ids ${dsgod_select_grain_id_file} --eta_omega_map_path ${eta_ome_map_file}"
    echo "STARTING DSGOD CONSTRUCTION ${scan_num}"
    #echo $dsgod_python_call
    $dsgod_python_call
done
