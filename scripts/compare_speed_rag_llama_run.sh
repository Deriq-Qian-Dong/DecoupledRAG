git pull
export CUDA_VISIBLE_DEVICES=0,1,2,3
output_dir=output_ca
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
sh scripts/set_gpu_num.sh
accelerate launch --main_process_port=29511 --config_file config/default_config.yaml\
    src/main.py config/compare_speed_rag_llama_config.yaml\
    | tee ${log_dir}/train.log  

echo "=================done train=================="
