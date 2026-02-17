"""
Script to move robot to rest pose, open gripper, lock joints with torque (except gripper),
and release when leader gripper closes.

This script:
1. Moves both leader and follower arms to the rest pose from a JSON file
2. Opens the gripper
3. Locks all joint positions by enabling torque, except the gripper
4. Monitors the leader arm gripper position
5. When the leader gripper closes, disables torque on all motors to free the arms

Example of usage:
```bash
python lerobot/scripts/opening_ceremony.py --robot-path lerobot/configs/robot/aloha_solo.yaml --pose-file .cache/poses/ceremony_pose.json
```

Alternatively, you can use the default robot config and pose file:
```bash
python lerobot/scripts/opening_ceremony.py
```
"""

import argparse
import json
import time
from pathlib import Path

import torch

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import init_hydra_config


def rest_pose_with_torque_lock(
    robot_path: str,
    pose_file: str,
    duration_s: float = 5.0,
    fps: int = 50,
    gripper_open_position: float | None = None,
    gripper_close_threshold: float = 5.0,
    teleop_fps: int = 200,
    robot_overrides: list[str] | None = None,
):
    """Move robot to rest pose, open gripper, lock joints, and release on gripper close.
    
    Args:
        robot_path: Path to robot yaml configuration file
        pose_file: Path to JSON file containing joint positions
        duration_s: Duration in seconds to complete the movement (default: 5.0)
        fps: Control frequency in Hz (default: 30)
        gripper_open_position: Target gripper position for "open" state. If None, uses a high value.
        gripper_close_threshold: Threshold for detecting gripper close (position decrease from initial)
        robot_overrides: Optional robot config overrides
    """
    # Load robot configuration
    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)
    
    try:
        # Check if it's a ManipulatorRobot
        if not isinstance(robot, ManipulatorRobot):
            raise ValueError(
                f"This script only supports ManipulatorRobot. Got {type(robot).__name__} instead."
            )

        # Determine which TorqueMode to use based on robot type
        if robot.robot_type in ["koch", "koch_bimanual", "aloha"]:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        elif robot.robot_type in ["so100", "moss"]:
            from lerobot.common.robot_devices.motors.feetech import TorqueMode
        else:
            raise ValueError(f"Unsupported robot type: {robot.robot_type}")

        # Connect to the robot
        if not robot.is_connected:
            print("Connecting to robot...")
            robot.connect()
            print("Connected.")

        # Enable torque on leader arms (all motors except gripper) so they can be moved programmatically
        print("\nEnabling torque on leader arms for movement...")
        for name in robot.leader_arms:
            arm = robot.leader_arms[name]
            for motor_name in arm.motor_names:
                if motor_name != "gripper":
                    arm.write("Torque_Enable", TorqueMode.ENABLED.value, motor_name)
                    print(f"  {name} leader: Enabled torque on {motor_name}")

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

            print(f"  {arm_id}: {len(arm.motor_names)} joints")
            print(f"    Current: {dict(zip(arm.motor_names, current_pos.tolist()))}")
            print(f"    Target:  {dict(zip(arm.motor_names, target_positions_list))}")

        # Calculate number of steps
        num_steps = int(duration_s * fps)

        print(f"\nMoving robot to rest position over {duration_s} seconds ({num_steps} steps)...")
        print(f"Moving {len(arms)} arm(s): {', '.join(arms)}")

        # Interpolate and move to rest position
        for step in range(num_steps + 1):
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

        print("✓ Robot moved to rest position")

        # Set gripper positions: close follower gripper, open leader gripper
        print("\nSetting gripper positions...")
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
                print(f"  {arm_id}: Enabled torque on gripper (temporarily)")
                
                # Set gripper to closed position
                arm.write("Goal_Position", int(closed_pos), "gripper")
                print(f"  {arm_id}: Set gripper to {closed_pos} (closed)")
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
                print(f"  {arm_id}: Enabled torque on gripper (temporarily)")

                # Set gripper to open position
                arm.write("Goal_Position", int(open_pos), "gripper")
                print(f"  {arm_id}: Set gripper to {open_pos} (open)")

        # Wait a bit for gripper to move
        time.sleep(1.0)

        # Lock all joints except gripper by enabling torque
        print("\nLocking joints (enabling torque on all motors except gripper)...")
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
                print(f"  {arm_id}: Disabled torque on gripper")

            # Enable torque on all motors except gripper
            for motor_name in arm.motor_names:
                if motor_name != "gripper":
                    arm.write("Torque_Enable", TorqueMode.ENABLED.value, motor_name)
                    print(f"  {arm_id}: Enabled torque on {motor_name}")

        print("✓ All joints locked (except gripper)")

        # Monitor leader arm gripper position
        print("\nMonitoring leader arm gripper position...")
        print("Close the leader gripper to release torque on all motors.")
        print("Press Ctrl+C to exit early.\n")

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
                print(f"  {arm_id} initial gripper position: {initial_gripper_positions[arm_id]}")

        # Monitor loop
        try:
            while True:
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
                            print(f"\n✓ {arm_id} gripper closed detected!")
                            print(f"  Initial: {initial_gripper_positions[arm_id]}, Current: {current_gripper_pos}")
                            gripper_closed = True
                            break

                if gripper_closed:
                    break

                # Small delay to avoid busy waiting
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")

        # Disable torque on leader arms (except gripper) to allow manual movement
        # Keep torque enabled on follower arms so they can follow the leader
        # Enable torque on follower gripper so it can follow leader gripper
        print("\nPreparing for teleoperation...")
        print("Disabling torque on leader arms (except gripper) to allow manual movement...")
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
                        print(f"  {arm_id}: Disabled torque on {motor_name}")
            else:
                # Follower arm - enable torque on gripper so it can follow leader gripper
                arm = robot.follower_arms[arm_name]
                if "gripper" in arm.motor_names:
                    arm.write("Torque_Enable", TorqueMode.ENABLED.value, "gripper")
                    print(f"  {arm_id}: Enabled torque on gripper for teleoperation")

        print("\n✓ Leader arms are now free to move (except gripper).")
        print("✓ Follower arms have torque enabled and will follow leader movements.")
        print("\nStarting teleoperation...")
        print("The follower arm will now follow the leader arm movements.")
        print("Press Ctrl+C to stop teleoperation.\n")

        # Start teleoperation loop
        try:
            while True:
                start_loop_t = time.perf_counter()
                
                # Perform one teleoperation step
                robot.teleop_step(record_data=False)
                
                # Maintain desired fps
                if teleop_fps is not None:
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1.0 / teleop_fps - dt_s)
                    
        except KeyboardInterrupt:
            print("\n\nTeleoperation stopped by user.")
    finally:
        # Disconnect from the robot
        if robot.is_connected:
            print("\nDisconnecting from robot...")
            robot.disconnect()
            print("Disconnected.")


def main():
    parser = argparse.ArgumentParser(
        description="Move robot to rest pose, lock joints with torque, and release on gripper close.",
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
        default=".cache/poses/ceremony_pose.json",
        help="Path to JSON file containing the ceremony pose joint positions.",
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
        help="Control frequency in Hz for the movement (default: 30).",
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
    parser.add_argument(
        "--teleop-fps",
        type=int,
        default=200,
        help="Control frequency in Hz for teleoperation after leader gripper closes (default: 200).",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        default=None,
        help="Override robot config parameters. Example: 'robot.port=/dev/ttyUSB0'",
    )

    args = parser.parse_args()
    rest_pose_with_torque_lock(
        args.robot_path,
        args.pose_file,
        args.duration_s,
        args.fps,
        args.gripper_open_position,
        args.gripper_close_threshold,
        args.teleop_fps,
        args.robot_overrides,
    )


if __name__ == "__main__":
    main()
