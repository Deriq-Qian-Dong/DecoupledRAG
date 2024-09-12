cat config/default_config.yaml.sample > config/default_config.yaml
# num_processes=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# echo "num_processes: "$num_processes >>  config/default_config.yaml
# 获取 CUDA_VISIBLE_DEVICES 环境变量
cuda_devices=${CUDA_VISIBLE_DEVICES:-"Not Set"}
num_processes=$(echo "$cuda_devices" | awk -F',' '{print NF}')
echo "num_processes: $num_processes" >> config/default_config.yaml