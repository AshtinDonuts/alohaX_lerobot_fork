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
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import init_hydra_config


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
        fps: Control frequency in Hz (default: 50)
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

    # Connect to the robot. This function receives a config path rather than a
    # robot object, so the generic @safe_disconnect decorator cannot be used.
    # Keep cleanup local to ensure connection failures preserve their original
    # exception and successfully connected buses are always closed.
    try:
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

        if duration_s <= 0:
            raise ValueError("duration_s must be greater than zero")
        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        move_arms = {}
        for arm_id in arms:
            arm_name, separator, arm_type = arm_id.rpartition("_")
            if not separator or arm_type != "follower" or arm_name not in robot.follower_arms:
                raise ValueError(
                    f"Only follower arms can be moved. Got '{arm_id}'. "
                    f"Available follower arms: "
                    f"{', '.join(get_arm_id(name, 'follower') for name in robot.follower_arms)}"
                )
            move_arms[arm_id] = robot.follower_arms[arm_name]

    # Get available arms from pose file
        available_pose_arms = list(pose_data.get("joint_values", {}).keys())

    # Validate arms
        missing_pose_arms = [arm_id for arm_id in arms if arm_id not in available_pose_arms]
        if missing_pose_arms:
            raise ValueError(
                f"Arms not found in pose file ('{', '.join(missing_pose_arms)}'). "
                f"Available pose arms: {', '.join(available_pose_arms)}"
            )

    # Get current positions and target positions
        current_positions = {}
        target_positions = {}

        for arm_id, arm in move_arms.items():

        # Read current position
            current_pos = arm.read("Present_Position")
            current_positions[arm_id] = torch.from_numpy(current_pos)

        # Get target position from pose file
            arm_pose_data = pose_data["joint_values"][arm_id]
            target_positions_list = arm_pose_data["positions"]
            target_positions[arm_id] = torch.tensor(target_positions_list, dtype=torch.float32)
            if len(target_positions_list) != len(arm.motor_names):
                raise ValueError(
                    f"Pose for {arm_id} has {len(target_positions_list)} positions, "
                    f"but the arm has {len(arm.motor_names)} motors"
                )

        # Verify motor names match
            pose_motor_names = arm_pose_data.get("motor_names", [])
            if pose_motor_names != arm.motor_names:
                raise ValueError(
                    f"Motor names for {arm_id} do not match.\n"
                    f"Pose file: {pose_motor_names}\nRobot: {arm.motor_names}"
                )

            print(f"  {arm_id}: {len(arm.motor_names)} joints")
            print(f"    Current: {dict(zip(arm.motor_names, current_pos.tolist()))}")
            print(f"    Target:  {dict(zip(arm.motor_names, target_positions_list))}")

    # Calculate number of steps
        num_steps = max(1, round(duration_s * fps))

        print(f"\nMoving robot to rest position over {duration_s} seconds ({num_steps} steps)...")
        print(f"Moving {len(arms)} arm(s): {', '.join(arms)}")

    # Interpolate and move
        for step in range(1, num_steps + 1):
            alpha = step / num_steps
            for arm_id, arm in move_arms.items():
                current = current_positions[arm_id]
                target = target_positions[arm_id]
                interpolated = current + alpha * (target - current)
                arm.write("Goal_Position", interpolated.numpy().astype("int32"))

        # Wait for next step
            if step < num_steps:
                busy_wait(1.0 / fps)

        print("✓ Robot moved to rest position")

    # Verify final position
        print("\nFinal positions:")
        for arm_id, arm in move_arms.items():
            final_pos = arm.read("Present_Position")
            print(f"  {arm_id}: {dict(zip(arm.motor_names, final_pos.tolist()))}")
    finally:
        if robot.is_connected:
            print("Disconnecting from robot...")
            robot.disconnect()


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
