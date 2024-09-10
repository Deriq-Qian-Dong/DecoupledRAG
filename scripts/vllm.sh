#!/bin/bash

# Get the number of GPUs
num_processes=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Create a log directory if it doesn't exist
mkdir -p logs

# Loop over the number of GPUs and submit a job for each
for local_rank in $(seq 0 $((num_processes - 1)))
do
    echo "Running process on GPU $local_rank"
    python src/contrastive_generation_vllm.py $num_processes $local_rank > logs/log_gpu_${local_rank}.log 2>&1 &
done

# Wait for all background processes to finish
wait

echo "All processes completed."
