cat config/default_config.yaml.sample > config/default_config.yaml
num_processes=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "num_processes: "$num_processes >>  config/default_config.yaml
