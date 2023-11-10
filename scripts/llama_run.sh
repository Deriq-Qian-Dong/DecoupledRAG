output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port=29500 --config_file config/llama_ds_config.yaml\
    src/main.py config/llama_config.yaml\
    | tee ${log_dir}/train.log  

echo "=================done train=================="
