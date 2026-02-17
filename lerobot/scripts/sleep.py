"""
Script to move the robot to its rest position (sleep pose).

This script loads joint positions from a JSON file (recorded using record_joint_values.py)
and slowly moves the robot to that position. By default, only moves follower arms.

Example of usage:
```bash
python lerobot/scripts/sleep.py --robot-path lerobot/configs/robot/aloha_solo.yaml --pose-file .cache/poses/rest_pose.json
```
"""

import argparse
import json
from pathlib import Path

import torch

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import init_hydra_config


@safe_disconnect
def move_to_rest_position(
    robot_path: str,
    pose_file: str,
    duration_s: float = 5.0,
    fps: int = 60,
    arms: list[str] | None = None,
    robot_overrides: list[str] | None = None,
):
    """Slowly move the robot to its rest position loaded from a JSON file.
    
    Args:
        robot_path: Path to robot yaml configuration file
        pose_file: Path to JSON file containing joint positions
        duration_s: Duration in seconds to complete the movement (default: 5.0)
        fps: Control frequency in Hz (default: 30)
        arms: List of arm IDs to move. If None, moves only follower arms.
        robot_overrides: Optional robot config overrides
    """
    # Load robot configuration
    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    # Check if it's a ManipulatorRobot
    if not isinstance(robot, ManipulatorRobot):
        raise ValueError(
            f"This script only supports ManipulatorRobot. Got {type(robot).__name__} instead."
        )

    # Connect to the robot
    if not robot.is_connected:
        print("Connecting to robot...")
        robot.connect()
        print("Connected.")

    # Load pose file
    pose_path = Path(pose_file)
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    print(f"Loading pose from '{pose_path}'...")
    with open(pose_path, "r") as f:
        pose_data = json.load(f)

    # Verify robot type matches
    if pose_data.get("robot_type") != robot.robot_type:
        print(
            f"Warning: Robot type mismatch. Pose file has '{pose_data.get('robot_type')}', "
            f"but robot is '{robot.robot_type}'"
        )

    # Determine which arms to move
    if arms is None:
        # Default: only follower arms
        arms = [get_arm_id(name, "follower") for name in robot.follower_arms]
        print(f"No arms specified, defaulting to follower arms: {arms}")

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

        # Verify motor names match
        pose_motor_names = arm_pose_data.get("motor_names", [])
        if pose_motor_names != arm.motor_names:
            print(
                f"Warning: Motor names mismatch for {arm_id}.\n"
                f"  Pose file: {pose_motor_names}\n"
                f"  Robot: {arm.motor_names}\n"
                f"  Proceeding with assumption that order matches..."
            )

        print(f"  {arm_id}: {len(arm.motor_names)} joints")
        print(f"    Current: {dict(zip(arm.motor_names, current_pos.tolist()))}")
        print(f"    Target:  {dict(zip(arm.motor_names, target_positions_list))}")

    # Calculate number of steps
    num_steps = int(duration_s * fps)

    print(f"\nMoving robot to rest position over {duration_s} seconds ({num_steps} steps)...")
    print(f"Moving {len(arms)} arm(s): {', '.join(arms)}")

    # Interpolate and move
    for step in range(num_steps + 1):
        alpha = step / num_steps  # 0 to 1

        # Build action tensor by concatenating all follower arm positions
        # Note: send_action expects only follower arms, so we'll send positions directly
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

        if len(action_parts) == 0:
            print("Warning: No follower arms to move. Skipping action.")
            break

        action = torch.cat(action_parts)

        # Send action with safety limit temporarily disabled for smooth movement
        original_max_relative = robot.config.max_relative_target
        try:
            # Temporarily set a high limit to allow smooth movement
            # The interpolation already ensures small steps
            robot.config.max_relative_target = None
            robot.send_action(action)
        finally:
            robot.config.max_relative_target = original_max_relative

        # Wait for next step
        if step < num_steps:
            busy_wait(1.0 / fps)

    print("✓ Robot moved to rest position")

    # Verify final position
    print("\nFinal positions:")
    for arm_id in arms:
        parts = arm_id.split("_")
        arm_type = parts[-1]
        arm_name = "_".join(parts[:-1])

        if arm_type == "follower":
            arm = robot.follower_arms[arm_name]
        else:
            arm = robot.leader_arms[arm_name]

        final_pos = arm.read("Present_Position")
        print(f"  {arm_id}: {dict(zip(arm.motor_names, final_pos.tolist()))}")


def main():
    parser = argparse.ArgumentParser(
        description="Move robot to rest position (sleep pose) from a recorded pose file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/aloha.yaml",
        help="Path to robot yaml file used to instantiate the robot.",
    )
    parser.add_argument(
        "--pose-file",
        type=str,
        default=".cache/poses/rest_pose.json",
        help="Path to JSON file containing the rest pose joint positions.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=5.0,
        help="Duration in seconds to complete the movement (default: 5.0).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Control frequency in Hz for the movement (default: 50).",
    )
    parser.add_argument(
        "--arms",
        type=str,
        nargs="*",
        default=None,
        help="List of arm IDs to move (e.g., 'left_follower right_follower'). "
        "If not specified, moves only follower arms by default.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        default=None,
        help="Override robot config parameters. Example: 'robot.port=/dev/ttyUSB0'",
    )

    args = parser.parse_args()
    move_to_rest_position(
        args.robot_path,
        args.pose_file,
        args.duration_s,
        args.fps,
        args.arms,
        args.robot_overrides,
    )


if __name__ == "__main__":
    main()
