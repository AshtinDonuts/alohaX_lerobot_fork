"""
Script to disable torque on all motors of a robot.

This script connects to the robot and disables torque on all leader and follower arms,
allowing the robot to be moved freely.

Example of usage:
```bash
python lerobot/scripts/disable_torque.py --robot-path lerobot/configs/robot/aloha_solo.yaml
```

Alternatively, you can use the default robot config:
```bash
python lerobot/scripts/disable_torque.py
```
"""

import argparse

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.utils.utils import init_hydra_config


def disable_torque(robot_path: str, robot_overrides: list[str] | None = None):
    """Disable torque on all motors of the robot."""
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

    # Determine which TorqueMode to use based on robot type
    if robot.robot_type in ["koch", "koch_bimanual", "aloha"]:
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
    elif robot.robot_type in ["so100", "moss"]:
        from lerobot.common.robot_devices.motors.feetech import TorqueMode
    else:
        raise ValueError(f"Unsupported robot type: {robot.robot_type}")

    # Disable torque on all follower arms
    print("Disabling torque on follower arms...")
    for name in robot.follower_arms:
        print(f"  Disabling torque on {name} follower arm...")
        robot.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        print(f"  ✓ Torque disabled on {name} follower arm")

    # Disable torque on all leader arms
    print("Disabling torque on leader arms...")
    for name in robot.leader_arms:
        print(f"  Disabling torque on {name} leader arm...")
        robot.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        print(f"  ✓ Torque disabled on {name} leader arm")

    print("\n✓ All torque has been disabled.")
    print("⚠️  WARNING: The robot is now free to move. Make sure to hold it if needed!")

    # Disconnect from the robot
    print("\nDisconnecting from robot...")
    robot.disconnect()
    print("Disconnected.")


def main():
    parser = argparse.ArgumentParser(
        description="Disable torque on all motors of a robot.",
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
        "--robot-overrides",
        type=str,
        nargs="*",
        default=None,
        help="Override robot config parameters. Example: 'robot.port=/dev/ttyUSB0'",
    )

    args = parser.parse_args()
    disable_torque(args.robot_path, args.robot_overrides)


if __name__ == "__main__":
    main()
