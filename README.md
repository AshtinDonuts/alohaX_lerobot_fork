
## Installation tips
Follow old_README.md, but on top:
```
pip install numpy==1.26.4
pip install pyserial
pip install av
pip install rerun==0.16.0
```

Sometimes you need to refresh the aloha module if changes don't take place
```
cd <root>
pip install -e .
```
## Postprocessing

### Conversion script
Converting from Trossen Data (hdf5) format into LeRobot V1 format (ours) to save space.
```
python lerobot/scripts/convert_aloha_data_to_lerobotv1.py --raw-dir /mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/khw/green_ball_ssil/ --repo-id my_repo/test_dataset    --local-dir data/lerobot/my_repo-test_dataset
```

## Useful Commands

### Send to Sleep
Default takes 5 seconds to send to sleep pose.
```
python lerobot/scripts/sleep.py \
    --robot-path lerobot/configs/robot/aloha_solo.yaml \
   --arms left_follower
```
### Shut down Torque
```
python lerobot/scripts/disable_torque.py --robot-path lerobot/configs/robot/aloha_solo.yaml
```

### Teleoperation
Teleop:
```
python lerobot/scripts/control_robot.py teleoperate \
   --robot-path lerobot/configs/robot/aloha_solo.yaml
```

Teleop with 30 fps:
```
python lerobot/scripts/control_robot.py teleoperate \
   --robot-path lerobot/configs/robot/aloha_solo.yaml --fps 30
```

Teleop with no cameras:
```
python lerobot/scripts/control_robot.py teleoperate \
   --robot-path lerobot/configs/robot/aloha_solo.yaml \
   --robot-overrides '~cameras'
```

### Recording:
```
python lerobot/scripts/record_eps.py \
   --robot-path lerobot/configs/robot/aloha_solo.yaml \
   --fps 50 \
   --root data \
   --repo-id aloha/test \
   --tags tutorial \
   --warmup-time-s 3 \
   --episode-time-s 5 \
   --reset-time-s 5 \
   --num-episodes 2
```

#### Note on using local repos:

* --root defaults to "data" if not specified
* --repo-id can be a simple name (e.g., my_dataset) or follow the convention username/dataset_name (e.g., lerobot/test)
* The local path will be: {root}/{repo_id}
* For example: --root data --repo-id my_test creates/uses data/my_test/

### Replay:
```
python lerobot/scripts/control_robot.py replay \
   --robot-path lerobot/configs/robot/aloha_solo.yaml \
   --fps 30 \
   --root data \
   --repo-id my_local_repo \
   --episode 0
```

### Calibration:
```
python lerobot/scripts/control_robot.py calibrate \
   --robot-path lerobot/configs/robot/aloha_solo.yaml
```

### Training:
```
DATA_DIR=data python lerobot/scripts/train.py \
   dataset_repo_id=my_local_repo \
   policy=act_aloha_solo_real \  
   env=aloha_solo_real \
   hydra.run.dir=outputs/train/act_aloha_test \
   hydra.job.name=act_aloha_test \
   device=cuda \
   wandb.enable=false
```

### Common Bugs
Unknown encoder 'libsvtav1'
```
wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
tar xvf ffmpeg-master-latest-linux64-gpl.tar.xz
cd ffmpeg-master-latest-linux64-gpl/bin/
./ffmpeg -encoders | grep svtav1
```

To make the change global, inside bashrc do this:
```
export PATH="/home/khw/ffmpeg-master-latest-linux64-gpl/bin:$PATH"
```

### Read more
https://docs.trossenrobotics.com/aloha_docs/2.0/training/lerobot_guide.html#lerobot-x-aloha-solo-user-guide