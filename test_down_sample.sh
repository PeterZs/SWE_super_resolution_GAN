#! /bin/bash
#get low resolution data set by gauss filtering the high resolution and interpolation the blur result to low resolution version.
<<EOF
data_path=../SWE_solver/data/data_velocity_4/
index_begin=0
index_end=1
frame_num_per_sim=400
scale=4.0
size_sim_high=256
size_sim_low=64
interp=bilinear #lanczos nearest bicubic bilinear

exe=./down_sample.py
    
python2 ${exe} data_path ${data_path} index_begin ${index_begin} index_end ${index_end} frame_num_per_sim ${frame_num_per_sim} scale ${scale} size_sim_high ${size_sim_high} size_sim_low ${size_sim_low} interp ${interp}


data_path=../SWE_solver/data/data_velocity_5/
index_begin=0
index_end=1
frame_num_per_sim=400
scale=4.0
size_sim_high=256
size_sim_low=64
interp=bilinear #lanczos nearest bicubic bilinear

exe=./down_sample.py
    
python2 ${exe} data_path ${data_path} index_begin ${index_begin} index_end ${index_end} frame_num_per_sim ${frame_num_per_sim} scale ${scale} size_sim_high ${size_sim_high} size_sim_low ${size_sim_low} interp ${interp}

data_path=../SWE_solver/data/data_velocity_6/
index_begin=0
index_end=1
frame_num_per_sim=400
scale=4.0
size_sim_high=256
size_sim_low=64
interp=bilinear #lanczos nearest bicubic bilinear

exe=./down_sample.py
    
python2 ${exe} data_path ${data_path} index_begin ${index_begin} index_end ${index_end} frame_num_per_sim ${frame_num_per_sim} scale ${scale} size_sim_high ${size_sim_high} size_sim_low ${size_sim_low} interp ${interp}
EOF

data_path=../SWE_solver/data/data_velocity_7/
index_begin=0
index_end=1
frame_num_per_sim=400
scale=4.0
size_sim_high=256
size_sim_low=64
interp=bilinear #lanczos nearest bicubic bilinear

exe=./down_sample.py
    
python2 ${exe} data_path ${data_path} index_begin ${index_begin} index_end ${index_end} frame_num_per_sim ${frame_num_per_sim} scale ${scale} size_sim_high ${size_sim_high} size_sim_low ${size_sim_low} interp ${interp} 
