cat config/default_config.yaml.sample > config/default_config.yaml
num_processes=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "num_processes: "$num_processes >>  config/default_config.yaml
# 获取 CUDA_VISIBLE_DEVICES 环境变量
cuda_devices=${CUDA_VISIBLE_DEVICES:-"Not Set"}
echo "CUDA_VISIBLE_DEVICES: $cuda_devices" >> config/default_config.yaml