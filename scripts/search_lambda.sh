#!/bin/bash

# Configuration array
configs=(
    'configs/hyperparameters/lambda/ewc-o/rtmpose-t_8xb256-520e_ewc-online-0.1-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-o/rtmpose-t_8xb256-520e_ewc-online-0.2-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-o/rtmpose-t_8xb256-520e_ewc-online-0.3-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-o/rtmpose-t_8xb256-520e_ewc-online-0.4-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-o/rtmpose-t_8xb256-520e_ewc-online-0.5-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-o/rtmpose-t_8xb256-520e_ewc-online-0.6-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-o/rtmpose-t_8xb256-520e_ewc-online-0.7-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-o/rtmpose-t_8xb256-520e_ewc-online-0.8-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-o/rtmpose-t_8xb256-520e_ewc-online-0.9-coco-mpii-crowdpose_256x192.py'

    'configs/hyperparameters/lambda/ewc-s/rtmpose-t_8xb256-520e_ewc-seperate-0.1-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-s/rtmpose-t_8xb256-520e_ewc-seperate-0.2-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-s/rtmpose-t_8xb256-520e_ewc-seperate-0.3-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-s/rtmpose-t_8xb256-520e_ewc-seperate-0.4-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-s/rtmpose-t_8xb256-520e_ewc-seperate-0.5-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-s/rtmpose-t_8xb256-520e_ewc-seperate-0.6-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-s/rtmpose-t_8xb256-520e_ewc-seperate-0.7-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-s/rtmpose-t_8xb256-520e_ewc-seperate-0.8-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/ewc-s/rtmpose-t_8xb256-520e_ewc-seperate-0.9-coco-mpii-crowdpose_256x192.py'

    'configs/hyperparameters/lambda/iwd/rtmpose-t_8xb256-520e_iwd-0.1-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/iwd/rtmpose-t_8xb256-520e_iwd-0.2-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/iwd/rtmpose-t_8xb256-520e_iwd-0.3-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/iwd/rtmpose-t_8xb256-520e_iwd-0.4-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/iwd/rtmpose-t_8xb256-520e_iwd-0.5-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/iwd/rtmpose-t_8xb256-520e_iwd-0.6-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/iwd/rtmpose-t_8xb256-520e_iwd-0.7-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/iwd/rtmpose-t_8xb256-520e_iwd-0.8-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/iwd/rtmpose-t_8xb256-520e_iwd-0.9-coco-mpii-crowdpose_256x192.py'

    'configs/hyperparameters/lambda/lfl/rtmpose-t_8xb256-520e_lfl-0.1-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lfl/rtmpose-t_8xb256-520e_lfl-0.2-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lfl/rtmpose-t_8xb256-520e_lfl-0.3-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lfl/rtmpose-t_8xb256-520e_lfl-0.4-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lfl/rtmpose-t_8xb256-520e_lfl-0.5-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lfl/rtmpose-t_8xb256-520e_lfl-0.6-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lfl/rtmpose-t_8xb256-520e_lfl-0.7-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lfl/rtmpose-t_8xb256-520e_lfl-0.8-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lfl/rtmpose-t_8xb256-520e_lfl-0.9-coco-mpii-crowdpose_256x192.py'

    'configs/hyperparameters/lambda/lwf/rtmpose-t_8xb256-520e_lwf-0.1-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lwf/rtmpose-t_8xb256-520e_lwf-0.2-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lwf/rtmpose-t_8xb256-520e_lwf-0.3-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lwf/rtmpose-t_8xb256-520e_lwf-0.4-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lwf/rtmpose-t_8xb256-520e_lwf-0.5-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lwf/rtmpose-t_8xb256-520e_lwf-0.6-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lwf/rtmpose-t_8xb256-520e_lwf-0.7-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lwf/rtmpose-t_8xb256-520e_lwf-0.8-coco-mpii-crowdpose_256x192.py'
    'configs/hyperparameters/lambda/lwf/rtmpose-t_8xb256-520e_lwf-0.9-coco-mpii-crowdpose_256x192.py'
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