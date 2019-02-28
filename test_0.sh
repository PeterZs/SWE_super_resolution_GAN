#! /bin/bash
<<EOF
#get the 7's result
i=0
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


#get the similar result of the original tempoGAN 
i=1
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}
EOF

<<EOF
i=2
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}



i=3
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF

<<EOF
i=4
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_1/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=5
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-0.00001  #weight of layer_loss term on generator loss
k2_l1=1 #weight of l1 layer_loss term on gen loss
k2_l2=1 #weight of l2 layer_loss term on gen loss
k2_l3=1 #weight of l3 layer_loss term on gen loss
k2_l4=1 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=1
train_index_begin=0
train_index_end=6
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF

<<EOF
i=6
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_no_velocity_1/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_0.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=7
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_no_velocity_1/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1   #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1

exe=./testGAN.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs}


EOF

<<EOF
i=8
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_no_velocity_1/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF
<<EOF
i=9
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_no_velocity_1/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}
EOF

<<EOF
i=10
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_no_velocity_1/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=50
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF

<<EOF
i=11
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_no_velocity_1/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=30
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF
<<EOF


i=12
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_no_velocity_1/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=13
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_no_velocity_1/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=14
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=15
get_model=0
learning_rate_ori=0.00001 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=0.1  #weight of l1 term on generator loss
k2=0.1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1.0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=70
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=1
gen_runs=1
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}



EOF
<<EOF

i=16
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_3.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_3.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=17
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_3.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_3.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

i=18
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_3.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_3.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=19
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_3.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_3.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=20
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_4.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_4.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=21
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_4.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_4.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

i=22
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_4.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_4.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=23
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=50
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_4.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_4.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}
EOF

<<EOF
i=24
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
#python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}



i=25
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
#python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}



i=26
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
#python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}



i=27
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
#python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF

<<EOF
# for residual block, we replace the tf.nn.relu with the tf.nn.leaky_relu to whether the tf.nn.relu selects the feature sparsely.
i=28
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


i=29
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=3 

exe=./testGAN_1.5_modified_gradient_penalty.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_gradient_penalty.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}
EOF


<<EOF
i=30
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=7 

exe=./testGAN_1.5_modified_gradient_penalty_leaky_relu.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_gradient_penalty_leaky_relu.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF

<<EOF

# for residual block, we replace the tf.nn.relu with the tf.nn.leaky_relu to whether the tf.nn.relu selects the feature sparsely.
i=31
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=0  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}




# for residual block, we replace the tf.nn.relu with the tf.nn.leaky_relu to whether the tf.nn.relu selects the feature sparsely.
i=32
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=0
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF


<<EOF
#this result is the same as test 28 
i=33
start_frame=0
end_frame=12
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu_full_image.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu_full_image.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}



i=34
start_frame=6
end_frame=7
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=64
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu_full_image.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu_full_image.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF

<<EOF
#this test has diverged in training process 
i=35
start_frame=0
end_frame=3
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu_full_image.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu_full_image.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF


<<EOF
i=36
start_frame=1
end_frame=2
get_model=0
learning_rate_ori=0.0002 #0.0002 may lead the training process to be diverged 
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=64
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu_full_image_large_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu_full_image_large_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}
EOF


<<EOF
#this test has diverged in training process 
i=37
start_frame=0
end_frame=2
get_model=0
learning_rate_ori=0.0002 #0.0002 may lead the training process 35 to be diverged ,so we decrease the base learning rate from 0.0002 to 0.00001 and redo the test as 37
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu_full_image_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu_full_image_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF

<<EOF
#this test has diverged in training process

#increase the discriminator's trained weight parameter's number
i=38
start_frame=1
end_frame=2
get_model=0
learning_rate_ori=0.0002 #0.0002 may lead the training process 35 to be diverged ,so we decrease the base learning rate from 0.0002 to 0.00001 and redo the test as 37
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu_full_image_3.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
#python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu_full_image_3.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}



i=39
get_model=0
learning_rate_ori=0.0002 #0.0002
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_gradient_penalty_leaky_relu.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
#python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_gradient_penalty_leaky_relu.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
#python ${exe} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF

<<EOF

i=40
start_frame=1
end_frame=2
get_model=0
learning_rate_ori=0.0002 #0.0002 may lead the training process to be diverged 
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_2/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu_full_image_large_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu_full_image_large_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}

EOF

#use the gradient to optimize the GAN asynchronously
i=41
start_frame=5
end_frame=8
get_model=0
learning_rate_ori=0.0002 
decay_lr=0.05
data_path=../SWE_solver/data/data_velocity_3/
log_path=./result/log_${i}/
model_path=./result/model_${i}/
test_path=./result/test_${i}/
k1=5  #weight of l1 term on generator loss
k2=-1  #weight of layer_loss term on generator loss
k2_l1=0.00001 #weight of l1 layer_loss term on gen loss
k2_l2=0.00001 #weight of l2 layer_loss term on gen loss
k2_l3=0.00001 #weight of l3 layer_loss term on gen loss
k2_l4=0.00001 #weight of l4 layer_loss term on gen loss
k_tempo=1  # tempo discriminator loss weight if it equals 1.0 and if 0.0 will disable
epochs=90
use_velocities=1
train_index_begin=0
train_index_end=5
test_index_begin=16
test_index_end=20
frame_num_gen=120
disc_spatial_runs=2
disc_tempo_runs=2
gen_runs=2
size_tile_low=16
size_sim_low=64
test_mode=0

gradient_loss_kind=6

exe=./testGAN_1.5_modified_leaky_relu_full_image_asy_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
#python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}


size_tile_low=64
size_sim_low=64
get_model=1
test_mode=1

exe=./testGAN_1.5_modified_leaky_relu_full_image_asy_2.py

mkdir -p ${log_path}
mkdir -p ${model_path}
mkdir -p ${test_path}
    
python ${exe} start_frame ${start_frame} end_frame ${end_frame} get_model ${get_model} learning_rate_ori ${learning_rate_ori} decay_lr ${decay_lr} data_path ${data_path} log_path ${log_path} model_path ${model_path} test_path ${test_path} k1 ${k1} k2 ${k2} k2_l1 ${k2_l1} k2_l2 ${k2_l2} k2_l3 ${k2_l3} k2_l4 ${k2_l4} k_tempo ${k_tempo} epochs ${epochs} use_velocities ${use_velocities} train_index_begin ${train_index_begin} train_index_end ${train_index_end} test_index_begin ${test_index_begin} test_index_end ${test_index_end} frame_num_gen ${frame_num_gen} disc_spatial_runs ${disc_spatial_runs} disc_tempo_runs ${disc_tempo_runs} gen_runs ${gen_runs} size_tile_low ${size_tile_low} size_sim_low ${size_sim_low} test_mode ${test_mode} gradient_loss_kind ${gradient_loss_kind}
