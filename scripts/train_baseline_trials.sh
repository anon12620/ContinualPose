#!/bin/bash

# Configuration array
configs=(
    'configs/baselines/trials/ewc-o/rtmpose-t_8xb256-520e_ewc-o-0.2-trail1-coco-mpii-crowdpose_256x192.py'
    'configs/baselines/trials/ewc-o/rtmpose-t_8xb256-520e_ewc-o-0.2-trail2-coco-mpii-crowdpose_256x192.py'
    'configs/baselines/trials/ewc-o/rtmpose-t_8xb256-520e_ewc-o-0.2-trail3-coco-mpii-crowdpose_256x192.py'
    'configs/baselines/trials/ewc-o/rtmpose-t_8xb256-520e_ewc-o-0.2-trail4-coco-mpii-crowdpose_256x192.py'
    'configs/baselines/trials/ewc-o/rtmpose-t_8xb256-520e_ewc-o-0.2-trail5-coco-mpii-crowdpose_256x192.py'

    configs/baselines/trials/ewc-s/rtmpose-t_8xb256-520e_ewc-s-0.3-trial1-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/ewc-s/rtmpose-t_8xb256-520e_ewc-s-0.3-trial2-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/ewc-s/rtmpose-t_8xb256-520e_ewc-s-0.3-trial3-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/ewc-s/rtmpose-t_8xb256-520e_ewc-s-0.3-trial4-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/ewc-s/rtmpose-t_8xb256-520e_ewc-s-0.3-trial5-coco-mpii-crowdpose_256x192.py

    configs/baselines/trials/lfl/rtmpose-t_8xb256-520e_lfl-0.4-trial1-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/lfl/rtmpose-t_8xb256-520e_lfl-0.4-trial2-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/lfl/rtmpose-t_8xb256-520e_lfl-0.4-trial3-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/lfl/rtmpose-t_8xb256-520e_lfl-0.4-trial4-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/lfl/rtmpose-t_8xb256-520e_lfl-0.4-trial5-coco-mpii-crowdpose_256x192.py

    configs/baselines/trials/lwf/rtmpose-t_8xb256-520e_lwf-0.4-trial1-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/lwf/rtmpose-t_8xb256-520e_lwf-0.4-trial2-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/lwf/rtmpose-t_8xb256-520e_lwf-0.4-trial3-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/lwf/rtmpose-t_8xb256-520e_lwf-0.4-trial4-coco-mpii-crowdpose_256x192.py
    configs/baselines/trials/lwf/rtmpose-t_8xb256-520e_lwf-0.4-trial5-coco-mpii-crowdpose_256x192.py
)

# Iterate through each config file
for config in "${configs[@]}"; do
    # Create a screen session named after the config file
    screen_name=$(echo $config | awk -F/ '{print $NF}')
    screen_name="${screen_name%.*}"
    screen_name="${screen_name//./_}"

    # Run the experiment
    screen -dmS $screen_name bash -c "GPUS=2 CPUS=32 ./tools/slurm_train.sh batch,V100-16GB,RTX3090,V100-32GB,RTXA6000,RTXA6000-AV $config"
done