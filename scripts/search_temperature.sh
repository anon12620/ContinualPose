#!/bin/bash

# Configuration array
configs=(
    'configs/hyperparameters/temperature/lwf/rtmpose-t_8xb256-520e_lwf-0.4-t-0.1-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/temperature/lwf/rtmpose-t_8xb256-520e_lwf-0.4-t-0.5-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/temperature/lwf/rtmpose-t_8xb256-520e_lwf-0.4-t-1.0-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/temperature/lwf/rtmpose-t_8xb256-520e_lwf-0.4-t-2.0-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/temperature/lwf/rtmpose-t_8xb256-520e_lwf-0.4-t-3.0-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/temperature/lwf/rtmpose-t_8xb256-520e_lwf-0.4-t-4.0-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/temperature/lwf/rtmpose-t_8xb256-520e_lwf-0.4-t-5.0-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/temperature/lwf/rtmpose-t_8xb256-520e_lwf-0.4-t-10.0-coco-mpii-crowdpose_256x192.py'
)

# Iterate through each config file
for config in "${configs[@]}"; do
    # Create a screen session named after the config file
    # Here you need to set the correct directory path and environmental variables if needed
    screen_name="${config%.*}"
    screen_name="${screen_name//./_}"
    screen_name="${screen_name#*0e_}"
    echo "Starting experiment: $screen_name"
    screen -dmS $screen_name bash -c "GPUS=1 CPUS=32 ./tools/slurm_train.sh batch,RTX3090,V100-32GB,RTXA6000,RTXA6000-AV,A100-40GB $config"
done