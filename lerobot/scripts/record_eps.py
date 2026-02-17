"""
Utilities to record robot episodes.

Useful to record a dataset and run a pretrained policy on your robot to record an evaluation dataset.

Examples of usage:

- Record one episode:
```bash
python lerobot/scripts/record_eps.py \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/record_eps.py \
    --fps 30 \
    --root data \
    --repo-id $USER/koch_pick_place_lego \
    --num-episodes 50 \
    --warmup-time-s 2 \
    --episode-time-s 30 \
    --reset-time-s 10
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command twice.
To avoid resuming by deleting the dataset, use `--force-override 1`.

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/record_episode.py \
    --fps 30 \
    --root data \
    --repo-id $USER/eval_act_koch_real \
    --num-episodes 10 \
    --warmup-time-s 2 \
    --episode-time-s 30 \
    --reset-time-s 10 \
    -p outputs/train/act_koch_real/checkpoints/080000/pretrained_model
```
"""

import argparse
import logging
from pathlib import Path
from typing import List

from lerobot.common.datasets.populate_dataset import (
    create_lerobot_dataset,
    delete_current_episode,
    init_dataset,
    save_current_episode,
)
from lerobot.common.robot_devices.control_utils import (
    has_method,
    init_keyboard_listener,
    init_policy,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int


@safe_disconnect
def record(
    robot: Robot,
    root: str,
    repo_id: str,
    pretrained_policy_name_or_path: str | None = None,
    policy_overrides: List[str] | None = None,
    fps: int | None = None,
    warmup_time_s=2,
    episode_time_s=10,
    reset_time_s=5,
    num_episodes=50,
    video=True,
    run_compute_stats=True,
    push_to_hub=True,
    tags=None,
    num_image_writer_processes=0,
    num_image_writer_threads_per_camera=4,
    force_override=False,
    display_cameras=True,
    play_sounds=True,
):
    # TODO(rcadene): Add option to record logs
    listener = None
    events = None
    policy = None
    device = None
    use_amp = None

    # Load pretrained policy
    if pretrained_policy_name_or_path is not None:
        policy, policy_fps, device, use_amp = init_policy(pretrained_policy_name_or_path, policy_overrides)

        if fps is None:
            fps = policy_fps
            logging.warning(f"No fps provided, so using the fps from policy config ({policy_fps}).")
        elif fps != policy_fps:
            logging.warning(
                f"There is a mismatch between the provided fps ({fps}) and the one from policy config ({policy_fps})."
            )

    # Create empty dataset or load existing saved episodes
    sanity_check_dataset_name(repo_id, policy)
    dataset = init_dataset(
        repo_id,
        root,
        force_override,
        fps,
        video,
        write_images=robot.has_camera,
        num_image_writer_processes=num_image_writer_processes,
        num_image_writer_threads=num_image_writer_threads_per_camera * robot.num_cameras,
    )

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("Warmup record", play_sounds)
    warmup_record(robot, events, enable_teleoperation, warmup_time_s, display_cameras, fps)

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    while True:
        if dataset["num_episodes"] >= num_episodes:
            break

        episode_index = dataset["num_episodes"]
        log_say(f"Recording episode {episode_index}", play_sounds)
        record_episode(
            dataset=dataset,
            robot=robot,
            events=events,
            episode_time_s=episode_time_s,
            display_cameras=display_cameras,
            policy=policy,
            device=device,
            use_amp=use_amp,
            fps=fps,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (episode_index < num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", play_sounds)
            reset_environment(robot, events, reset_time_s)

        if events["rerecord_episode"]:
            log_say("Re-record episode", play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            delete_current_episode(dataset)
            continue

        # Increment by one dataset["current_episode_index"]
        save_current_episode(dataset)

        if events["stop_recording"]:
            break

    log_say("Stop recording", play_sounds, blocking=True)
    stop_recording(robot, listener, display_cameras)

    lerobot_dataset = create_lerobot_dataset(dataset, run_compute_stats, push_to_hub, tags, play_sounds)

    log_say("Exiting", play_sounds)
    return lerobot_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/aloha.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser.add_argument(
        "--warmup-time-s",
        type=int,
        default=10,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser.add_argument(
        "--episode-time-s",
        type=int,
        default=60,
        help="Number of seconds for data recording for each episode.",
    )
    parser.add_argument(
        "--reset-time-s",
        type=int,
        default=60,
        help="Number of seconds for resetting the environment after each episode.",
    )
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record.")
    parser.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=int,
        default=0,
        help="Upload dataset to Hugging Face hub.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Add tags to your dataset on the hub.",
    )
    parser.add_argument(
        "--num-image-writer-processes",
        type=int,
        default=0,
        help=(
            "Number of subprocesses handling the saving of frames as PNGs. Set to 0 to use threads only; "
            "set to ≥1 to use subprocesses, each using threads to write images. The best number of processes "
            "and threads depends on your system. We recommend 4 threads per camera with 0 processes. "
            "If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses."
        ),
    )
    parser.add_argument(
        "--num-image-writer-threads-per-camera",
        type=int,
        default=4,
        help=(
            "Number of threads writing the frames as png images on disk, per camera. "
            "Too many threads might cause unstable teleoperation fps due to main thread being blocked. "
            "Not enough threads might cause low camera fps."
        ),
    )
    parser.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="By default, data recording is resumed. When set to 1, delete the local directory and start data recording from scratch.",
    )
    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser.add_argument(
        "--display-cameras",
        type=int,
        default=1,
        help="Display all cameras on screen (set to 1 to display or 0).",
    )
    parser.add_argument(
        "--play-sounds",
        type=int,
        default=1,
        help="Play sounds during recording (set to 1 to play or 0).",
    )
    parser.add_argument(
        "--video",
        type=int,
        default=1,
        help="Record video (set to 1 to record or 0).",
    )

    args = parser.parse_args()

    init_logging()

    robot_path = args.robot_path
    robot_overrides = args.robot_overrides
    kwargs = vars(args)
    del kwargs["robot_path"]
    del kwargs["robot_overrides"]

    # Convert types to match function signature
    kwargs["root"] = str(kwargs["root"])
    kwargs["display_cameras"] = bool(kwargs["display_cameras"])
    kwargs["play_sounds"] = bool(kwargs["play_sounds"])
    kwargs["video"] = bool(kwargs["video"])
    kwargs["force_override"] = bool(kwargs["force_override"])
    kwargs["run_compute_stats"] = bool(kwargs["run_compute_stats"])
    kwargs["push_to_hub"] = bool(kwargs["push_to_hub"])

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    record(robot, **kwargs)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()
