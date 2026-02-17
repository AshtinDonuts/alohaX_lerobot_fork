"""
Script to record current joint values from a robot.

This script connects to the robot and records the current joint positions from all
leader and follower arms, saving them to a JSON file. This is useful for saving
specific poses or configurations.

Example of usage:
```bash
python lerobot/scripts/record_joint_values.py --robot-path lerobot/configs/robot/aloha_solo.yaml --output poses/my_pose.json
```

Alternatively, you can use the default robot config and output path:
```bash
python lerobot/scripts/record_joint_values.py
```

You can also specify which arms to record:
```bash
python lerobot/scripts/record_joint_values.py --arms left_follower right_follower
```
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.utils.utils import init_hydra_config


def record_joint_values(
    robot_path: str,
    output_path: str | None = None,
    arms: list[str] | None = None,
    robot_overrides: list[str] | None = None,
):
    """Record current joint values from the robot.
    
    Args:
        robot_path: Path to robot yaml configuration file
        output_path: Path to output JSON file. If None, uses default location.
        arms: List of arm IDs to record. If None, records all available arms.
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

    # Determine which arms to record
    if arms is None:
        arms = robot.available_arms

    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    available_arms_str = " ".join(robot.available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    if len(arms) == 0:
        raise ValueError(
            "No arms to record. Use `--arms` to specify which arms to record.\n"
            f"For instance: `--arms {available_arms_str}`"
        )

    # Record joint values for each arm
    joint_values = {}
    
    print(f"\nRecording joint values for {len(arms)} arm(s)...")
    
    for arm_id in arms:
        # Parse arm_id to get name and type (e.g., "left_follower" -> name="left", type="follower")
        parts = arm_id.split("_")
        if len(parts) < 2:
            raise ValueError(f"Invalid arm ID format: {arm_id}. Expected format: 'name_type' (e.g., 'left_follower')")
        
        arm_type = parts[-1]  # "follower" or "leader"
        arm_name = "_".join(parts[:-1])  # "left" or "right" or "main", etc.
        
        # Get the appropriate arm dictionary
        if arm_type == "follower":
            if arm_name not in robot.follower_arms:
                raise ValueError(f"Follower arm '{arm_name}' not found. Available: {list(robot.follower_arms.keys())}")
            arm = robot.follower_arms[arm_name]
        elif arm_type == "leader":
            if arm_name not in robot.leader_arms:
                raise ValueError(f"Leader arm '{arm_name}' not found. Available: {list(robot.leader_arms.keys())}")
            arm = robot.leader_arms[arm_name]
        else:
            raise ValueError(f"Invalid arm type '{arm_type}'. Must be 'follower' or 'leader'")
        
        # Read current positions
        print(f"  Reading positions from {arm_id}...")
        positions = arm.read("Present_Position")
        
        # Convert numpy array to list for JSON serialization
        positions_list = positions.tolist()
        
        # Store with motor names for clarity
        joint_values[arm_id] = {
            "motor_names": arm.motor_names,
            "positions": positions_list,
            "positions_dict": dict(zip(arm.motor_names, positions_list)),
        }
        
        print(f"    ✓ Recorded {len(arm.motor_names)} joints from {arm_id}")
        print(f"      Values: {dict(zip(arm.motor_names, positions_list))}")

    # Create output data structure
    output_data = {
        "robot_type": robot.robot_type,
        "timestamp": datetime.now().isoformat(),
        "arms_recorded": arms,
        "joint_values": joint_values,
    }

    # Determine output path
    if output_path is None:
        # Default: save to .cache/poses/ directory
        output_dir = Path(".cache/poses")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{robot.robot_type}_{timestamp_str}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to file
    print(f"\nSaving joint values to '{output_path}'...")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Joint values saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Arms recorded: {', '.join(arms)}")

    # Disconnect from the robot
    print("\nDisconnecting from robot...")
    robot.disconnect()
    print("Disconnected.")


def main():
    parser = argparse.ArgumentParser(
        description="Record current joint values from a robot.",
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
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file. If not specified, saves to .cache/poses/ with timestamp.",
    )
    parser.add_argument(
        "--arms",
        type=str,
        nargs="*",
        default=None,
        help="List of arm IDs to record (e.g., 'left_follower right_follower'). If not specified, records all available arms.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        default=None,
        help="Override robot config parameters. Example: 'robot.port=/dev/ttyUSB0'",
    )

    args = parser.parse_args()
    record_joint_values(
        args.robot_path,
        args.output,
        args.arms,
        args.robot_overrides,
    )


if __name__ == "__main__":
    main()
