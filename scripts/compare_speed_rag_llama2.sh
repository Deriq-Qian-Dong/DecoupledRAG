#!/bin/bash
git pull
export CUDA_VISIBLE_DEVICES=0,1,2,3
output_dir=output_ca_rag_llama2_speed
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
sh scripts/set_gpu_num.sh
accelerate launch --main_process_port=29502 --config_file config/default_config.yaml\
    src/main.py config/compare_speed_rag_llama2_config.yaml\
    | tee ${log_dir}/compare_speed_rag_llama2.log  

echo "=================done compare speed rag llama2=================="