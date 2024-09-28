output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
export CUDA_VISIBLE_DEVICES=6,7
sh scripts/set_gpu_num.sh
yaml_file=$1
# 如果yaml_file为空，则使用默认的配置文件
if [ ! -n "$yaml_file" ];then
    yaml_file=config/retrieval.yaml
fi
echo 'yaml_file: '$yaml_file
accelerate launch --main_process_port=29511 --config_file config/default_config.yaml\
    src/retrieval.py ${yaml_file}\
    | tee ${log_dir}/train.log  

echo "=================done train=================="
