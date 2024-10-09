git pull
output_dir=output_sa_re
export CUDA_VISIBLE_DEVICES=2,3
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
sh scripts/set_gpu_num.sh
accelerate launch --main_process_port=29599 --config_file config/default_config.yaml\
    src/main.py config/llama_re_config.yaml\
    | tee ${log_dir}/train.log  

echo "=================done train=================="
