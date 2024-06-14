output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
sh scripts/set_gpu_num.sh
yaml_file=$1
# 如果yaml_file为空，则使用默认的配置文件
if [ ! -n "$yaml_file" ];then
    yaml_file=config/computing_embeddings.yaml
fi
echo 'yaml_file: '$yaml_file
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port=29599 --config_file config/default_config.yaml\
#     src/data_construction/compute_embedding_by_decoder_only_multi_gpu.py ${yaml_file}\
#     | tee ${log_dir}/train.log  
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port=29599 --config_file config/default_config.yaml\
    src/data_construction/compute_embedding_by_dense_retrieval_for_corpus.py \
    | tee ${log_dir}/train.log  

echo "=================done train=================="