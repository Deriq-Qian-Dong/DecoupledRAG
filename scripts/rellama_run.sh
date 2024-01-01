config_file=$1
if [ -z "$config_file" ]
then
    echo "Use default config file: config/rellama_config.yaml"
    config_file=config/rellama_config.yaml
fi
echo "The config file is ${config_file}"
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
sh script/set_gpu_num.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port=29500 --config_file config/llama_ds_config.yaml\
    src/main.py ${config_file} \
    | tee ${log_dir}/train.log  

echo "=================done train=================="
