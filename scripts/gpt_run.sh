output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
sh scripts/set_gpu_num.sh
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port=29500 --config_file config/default_config.yaml\
    src/main.py config/gpt_config.yaml\
    | tee ${log_dir}/train.log  

echo "=================done train=================="
