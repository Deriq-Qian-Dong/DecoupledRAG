git pull
output_dir=output_sa
export CUDA_VISIBLE_DEVICES=2,3,4,7
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
sh scripts/set_gpu_num.sh
accelerate launch --main_process_port=29509 --config_file config/default_config.yaml\
    src/main.py config/llama_config.yaml\
    | tee ${log_dir}/train.log  

echo "=================done train=================="
