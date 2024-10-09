git pull
export CUDA_VISIBLE_DEVICES=4,5
output_dir=output_ca_llama2
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
sh scripts/set_gpu_num.sh
accelerate launch --main_process_port=29521 --config_file config/default_config.yaml\
    src/main.py config/rag_llama2_config.yaml\
    | tee ${log_dir}/train.log  

echo "=================done train=================="
