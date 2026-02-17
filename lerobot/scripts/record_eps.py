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
import json
import logging
import time
from pathlib import Path
from typing import List

import torch

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
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.robots.utils import Robot, get_arm_id
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int


def perform_opening_ceremony(
    robot: Robot,
    pose_file: str,
    duration_s: float = 5.0,
    fps: int = 50,
    gripper_open_position: float | None = None,
    gripper_close_threshold: float = 5.0,
    events=None,
):
    """Move robot to rest pose, open gripper, lock joints, and wait for gripper close.
    
    Args:
        robot: Robot instance
        pose_file: Path to JSON file containing joint positions
        duration_s: Duration in seconds to complete the movement (default: 5.0)
        fps: Control frequency in Hz (default: 50)
        gripper_open_position: Target gripper position for "open" state. If None, uses a high value.
        gripper_close_threshold: Threshold for detecting gripper close (position decrease from initial)
        events: Optional events dictionary for keyboard interrupts
    """
    # Check if it's a ManipulatorRobot
    if not isinstance(robot, ManipulatorRobot):
        logging.warning(
            f"Opening ceremony only supports ManipulatorRobot. Got {type(robot).__name__} instead. "
            "Skipping opening ceremony."
        )
        return

    # Determine which TorqueMode to use based on robot type
    if robot.robot_type in ["koch", "koch_bimanual", "aloha"]:
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
    elif robot.robot_type in ["so100", "moss"]:
        from lerobot.common.robot_devices.motors.feetech import TorqueMode
    else:
        logging.warning(f"Unsupported robot type for opening ceremony: {robot.robot_type}. Skipping.")
        return

    # Enable torque on leader arms (all motors except gripper) so they can be moved programmatically
    logging.info("\nEnabling torque on leader arms for movement...")
    for name in robot.leader_arms:
        arm = robot.leader_arms[name]
        for motor_name in arm.motor_names:
            if motor_name != "gripper":
                arm.write("Torque_Enable", TorqueMode.ENABLED.value, motor_name)
                logging.info(f"  {name} leader: Enabled torque on {motor_name}")

    # Load pose file
    pose_path = Path(pose_file)
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    logging.info(f"Loading pose from '{pose_path}'...")
    with open(pose_path, "r") as f:
        pose_data = json.load(f)

    # Verify robot type matches
    if pose_data.get("robot_type") != robot.robot_type:
        logging.warning(
            f"Robot type mismatch. Pose file has '{pose_data.get('robot_type')}', "
            f"but robot is '{robot.robot_type}'"
        )

    # Get all arms (both leader and follower)
    arms = []
    for name in robot.follower_arms:
        arms.append(get_arm_id(name, "follower"))
    for name in robot.leader_arms:
        arms.append(get_arm_id(name, "leader"))

    # Get available arms from pose file
    available_pose_arms = list(pose_data.get("joint_values", {}).keys())
    available_robot_arms = robot.available_arms

    # Validate arms
    unknown_arms = [arm_id for arm_id in arms if arm_id not in available_robot_arms]
    missing_pose_arms = [arm_id for arm_id in arms if arm_id not in available_pose_arms]

    if unknown_arms:
        raise ValueError(
            f"Unknown arms provided ('{', '.join(unknown_arms)}'). "
            f"Available robot arms: {', '.join(available_robot_arms)}"
        )

    if missing_pose_arms:
        raise ValueError(
            f"Arms not found in pose file ('{', '.join(missing_pose_arms)}'). "
            f"Available pose arms: {', '.join(available_pose_arms)}"
        )

    # Get current positions and target positions
    current_positions = {}
    target_positions = {}

    for arm_id in arms:
        # Parse arm_id to get name and type
        parts = arm_id.split("_")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid arm ID format: {arm_id}. Expected format: 'name_type' (e.g., 'left_follower')"
            )

        arm_type = parts[-1]  # "follower" or "leader"
        arm_name = "_".join(parts[:-1])  # "left" or "right" or "main", etc.

        # Get the appropriate arm
        if arm_type == "follower":
            if arm_name not in robot.follower_arms:
                raise ValueError(
                    f"Follower arm '{arm_name}' not found. Available: {list(robot.follower_arms.keys())}"
                )
            arm = robot.follower_arms[arm_name]
        elif arm_type == "leader":
            if arm_name not in robot.leader_arms:
                raise ValueError(
                    f"Leader arm '{arm_name}' not found. Available: {list(robot.leader_arms.keys())}"
                )
            arm = robot.leader_arms[arm_name]
        else:
            raise ValueError(f"Invalid arm type '{arm_type}'. Must be 'follower' or 'leader'")

        # Read current position
        current_pos = arm.read("Present_Position")
        current_positions[arm_id] = torch.from_numpy(current_pos)

        # Get target position from pose file
        arm_pose_data = pose_data["joint_values"][arm_id]
        target_positions_list = arm_pose_data["positions"]
        target_positions[arm_id] = torch.tensor(target_positions_list, dtype=torch.float32)

        logging.info(f"  {arm_id}: {len(arm.motor_names)} joints")

    # Calculate number of steps
    num_steps = int(duration_s * fps)

    logging.info(f"\nMoving robot to rest position over {duration_s} seconds ({num_steps} steps)...")
    logging.info(f"Moving {len(arms)} arm(s): {', '.join(arms)}")

    # Interpolate and move to rest position
    for step in range(num_steps + 1):
        # Check for keyboard interrupts
        if events and events.get("stop_recording", False):
            logging.info("\nOpening ceremony interrupted by user.")
            return

        alpha = step / num_steps  # 0 to 1

        # Build action tensor by concatenating all follower arm positions
        action_parts = []
        for arm_id in arms:
            # Parse arm_id
            parts = arm_id.split("_")
            arm_type = parts[-1]
            arm_name = "_".join(parts[:-1])

            # Only send actions to follower arms (leader arms are controlled manually)
            if arm_type == "follower":
                current = current_positions[arm_id]
                target = target_positions[arm_id]
                # Linear interpolation
                interpolated = current + alpha * (target - current)
                action_parts.append(interpolated)

        if len(action_parts) > 0:
            action = torch.cat(action_parts)

            # Send action with safety limit temporarily disabled for smooth movement
            original_max_relative = robot.config.max_relative_target
            try:
                robot.config.max_relative_target = None
                robot.send_action(action)
            finally:
                robot.config.max_relative_target = original_max_relative

        # Also move leader arms directly
        for arm_id in arms:
            parts = arm_id.split("_")
            arm_type = parts[-1]
            arm_name = "_".join(parts[:-1])

            if arm_type == "leader":
                current = current_positions[arm_id]
                target = target_positions[arm_id]
                interpolated = current + alpha * (target - current)
                goal_pos = interpolated.numpy().astype("int32")
                robot.leader_arms[arm_name].write("Goal_Position", goal_pos)

        # Wait for next step
        if step < num_steps:
            busy_wait(1.0 / fps)

    logging.info("✓ Robot moved to rest position")

    # Set gripper positions: close follower gripper, open leader gripper
    logging.info("\nSetting gripper positions...")
    for arm_id in arms:
        parts = arm_id.split("_")
        arm_type = parts[-1]
        arm_name = "_".join(parts[:-1])

        if arm_type == "follower":
            arm = robot.follower_arms[arm_name]
            # Close follower gripper
            if robot.robot_type == "aloha":
                closed_pos = 0.0  # 0 = fully closed for Aloha
            else:
                closed_pos = 0.0  # For other robots, use low value for closed
            
            # Enable torque on follower gripper temporarily so we can move it
            arm.write("Torque_Enable", TorqueMode.ENABLED.value, "gripper")
            logging.info(f"  {arm_id}: Enabled torque on gripper (temporarily)")
            
            # Set gripper to closed position
            arm.write("Goal_Position", int(closed_pos), "gripper")
            logging.info(f"  {arm_id}: Set gripper to {closed_pos} (closed)")
        else:
            # Leader arm - open the gripper
            arm = robot.leader_arms[arm_name]
            
            # Determine gripper open position
            if gripper_open_position is None:
                # Use a high value for open (assuming higher = more open for Aloha)
                # For Aloha, gripper range is typically 0-100, so use 80-90 for open
                if robot.robot_type == "aloha":
                    open_pos = 85.0
                else:
                    # For other robots, use a reasonable open position
                    open_pos = 80.0
            else:
                open_pos = gripper_open_position

            # For leader arms, enable torque on gripper temporarily so we can move it
            arm.write("Torque_Enable", TorqueMode.ENABLED.value, "gripper")
            logging.info(f"  {arm_id}: Enabled torque on gripper (temporarily)")

            # Set gripper to open position
            arm.write("Goal_Position", int(open_pos), "gripper")
            logging.info(f"  {arm_id}: Set gripper to {open_pos} (open)")

    # Wait a bit for gripper to move
    time.sleep(1.0)

    # Lock all joints except gripper by enabling torque
    logging.info("\nLocking joints (enabling torque on all motors except gripper)...")
    for arm_id in arms:
        parts = arm_id.split("_")
        arm_type = parts[-1]
        arm_name = "_".join(parts[:-1])

        if arm_type == "follower":
            arm = robot.follower_arms[arm_name]
        else:
            arm = robot.leader_arms[arm_name]

        # First, explicitly disable torque on gripper to ensure it's free
        if "gripper" in arm.motor_names:
            arm.write("Torque_Enable", TorqueMode.DISABLED.value, "gripper")
            logging.info(f"  {arm_id}: Disabled torque on gripper")

        # Enable torque on all motors except gripper
        for motor_name in arm.motor_names:
            if motor_name != "gripper":
                arm.write("Torque_Enable", TorqueMode.ENABLED.value, motor_name)
                logging.info(f"  {arm_id}: Enabled torque on {motor_name}")

    logging.info("✓ All joints locked (except gripper)")

    # Monitor leader arm gripper position
    logging.info("\nMonitoring leader arm gripper position...")
    logging.info("Close the leader gripper to start recording.")
    if events:
        logging.info("Press Ctrl+C or ESC to exit early.\n")
    else:
        logging.info("Press Ctrl+C to exit early.\n")

    # Get initial gripper positions for all leader arms
    initial_gripper_positions = {}
    for arm_id in arms:
        parts = arm_id.split("_")
        arm_type = parts[-1]
        arm_name = "_".join(parts[:-1])

        if arm_type == "leader":
            arm = robot.leader_arms[arm_name]
            gripper_idx = arm.motor_names.index("gripper")
            pos = arm.read("Present_Position")
            initial_gripper_positions[arm_id] = float(pos[gripper_idx])
            logging.info(f"  {arm_id} initial gripper position: {initial_gripper_positions[arm_id]}")

    # Monitor loop
    try:
        while True:
            # Check for keyboard interrupts
            if events and events.get("stop_recording", False):
                logging.info("\nOpening ceremony interrupted by user.")
                return

            gripper_closed = False

            for arm_id in arms:
                parts = arm_id.split("_")
                arm_type = parts[-1]
                arm_name = "_".join(parts[:-1])

                if arm_type == "leader":
                    arm = robot.leader_arms[arm_name]
                    gripper_idx = arm.motor_names.index("gripper")
                    current_pos = arm.read("Present_Position")
                    current_gripper_pos = float(current_pos[gripper_idx])

                    # Check if gripper has closed (position decreased significantly)
                    if current_gripper_pos < (initial_gripper_positions[arm_id] - gripper_close_threshold):
                        logging.info(f"\n✓ {arm_id} gripper closed detected!")
                        logging.info(f"  Initial: {initial_gripper_positions[arm_id]}, Current: {current_gripper_pos}")
                        gripper_closed = True
                        break

            if gripper_closed:
                break

            # Small delay to avoid busy waiting
            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("\n\nInterrupted by user.")
        if events:
            events["stop_recording"] = True
        return

    # Disable torque on leader arms (except gripper) to allow manual movement
    # Keep torque enabled on follower arms so they can follow the leader
    # Enable torque on ALL follower arm motors (including gripper) so they can follow
    logging.info("\nPreparing for recording...")
    logging.info("Disabling torque on leader arms (except gripper) to allow manual movement...")
    logging.info("Ensuring follower arms have torque enabled on all motors for teleoperation...")
    for arm_id in arms:
        parts = arm_id.split("_")
        arm_type = parts[-1]
        arm_name = "_".join(parts[:-1])

        if arm_type == "leader":
            arm = robot.leader_arms[arm_name]
            # Disable torque on all leader motors except gripper (gripper already disabled)
            for motor_name in arm.motor_names:
                if motor_name != "gripper":
                    arm.write("Torque_Enable", TorqueMode.DISABLED.value, motor_name)
                    logging.info(f"  {arm_id}: Disabled torque on {motor_name}")
        else:
            # Follower arm - enable torque on ALL motors (including gripper) so it can follow leader
            arm = robot.follower_arms[arm_name]
            for motor_name in arm.motor_names:
                arm.write("Torque_Enable", TorqueMode.ENABLED.value, motor_name)
                logging.info(f"  {arm_id}: Enabled torque on {motor_name} for teleoperation")

    logging.info("\n✓ Leader arms are now free to move (except gripper).")
    logging.info("✓ Follower arms have torque enabled on all motors and will follow leader movements.")
    logging.info("\nStarting recording...\n")


def enable_torque_for_reset(robot: Robot):
    """Enable torque on all leader and follower arm motors for reset.
    
    This allows both leader and follower arms to be moved during reset.
    
    Args:
        robot: Robot instance
    """
    # Check if it's a ManipulatorRobot
    if not isinstance(robot, ManipulatorRobot):
        logging.warning(
            f"enable_torque_for_reset only supports ManipulatorRobot. Got {type(robot).__name__} instead. "
            "Skipping torque enable."
        )
        return

    # Determine which TorqueMode to use based on robot type
    if robot.robot_type in ["koch", "koch_bimanual", "aloha"]:
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
    elif robot.robot_type in ["so100", "moss"]:
        from lerobot.common.robot_devices.motors.feetech import TorqueMode
    else:
        logging.warning(f"Unsupported robot type for reset torque enable: {robot.robot_type}. Skipping.")
        return

    logging.info("\nEnabling torque on all arms for reset...")
    
    # Enable torque on all follower arm motors
    for name in robot.follower_arms:
        arm = robot.follower_arms[name]
        for motor_name in arm.motor_names:
            arm.write("Torque_Enable", TorqueMode.ENABLED.value, motor_name)
            logging.info(f"  {name} follower: Enabled torque on {motor_name}")
    
    # Enable torque on all leader arm motors
    for name in robot.leader_arms:
        arm = robot.leader_arms[name]
        for motor_name in arm.motor_names:
            arm.write("Torque_Enable", TorqueMode.ENABLED.value, motor_name)
            logging.info(f"  {name} leader: Enabled torque on {motor_name}")
    
    logging.info("✓ All arms have torque enabled for reset.")


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
    use_opening_ceremony=False,
    pose_file=".cache/poses/ceremony_pose.json",
    ceremony_duration_s=5.0,
    ceremony_fps=50,
    gripper_open_position=None,
    gripper_close_threshold=5.0,
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

    # Execute warmup if not using opening ceremony (opening ceremony will be done before each episode)
    if not use_opening_ceremony:
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
        
        # Perform opening ceremony before each episode if enabled
        if use_opening_ceremony:
            log_say(f"Opening ceremony for episode {episode_index}", play_sounds)
            try:
                perform_opening_ceremony(
                    robot=robot,
                    pose_file=pose_file,
                    duration_s=ceremony_duration_s,
                    fps=ceremony_fps,
                    gripper_open_position=gripper_open_position,
                    gripper_close_threshold=gripper_close_threshold,
                    events=events,
                )
                # Check if user interrupted during opening ceremony
                if events and events.get("stop_recording", False):
                    log_say("Stop recording", play_sounds, blocking=True)
                    stop_recording(robot, listener, display_cameras)
                    return None
            except (ValueError, FileNotFoundError) as e:
                logging.warning(f"Opening ceremony failed: {e}. Skipping to recording.")
                # Continue to recording even if ceremony fails
        
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
            # Enable torque on all arms (leader and follower) for reset
            enable_torque_for_reset(robot)
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
    parser.add_argument(
        "--use-opening-ceremony",
        type=int,
        default=1,
        help="Use opening ceremony instead of warmup (set to 1 to enable or 0 to disable, default: 1).",
    )
    parser.add_argument(
        "--pose-file",
        type=str,
        default=".cache/poses/ceremony_pose.json",
        help="Path to JSON file containing the ceremony pose joint positions.",
    )
    parser.add_argument(
        "--ceremony-duration-s",
        type=float,
        default=5.0,
        help="Duration in seconds to complete the opening ceremony movement (default: 5.0).",
    )
    parser.add_argument(
        "--ceremony-fps",
        type=int,
        default=50,
        help="Control frequency in Hz for the opening ceremony movement (default: 50).",
    )
    parser.add_argument(
        "--gripper-open-position",
        type=float,
        default=None,
        help="Target gripper position for 'open' state. If not specified, uses default based on robot type.",
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=5.0,
        help="Threshold for detecting gripper close (position decrease from initial, default: 5.0).",
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
    kwargs["use_opening_ceremony"] = bool(kwargs["use_opening_ceremony"])

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    record(robot, **kwargs)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()
