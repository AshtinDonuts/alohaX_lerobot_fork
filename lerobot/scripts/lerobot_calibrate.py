"""
Helper to (re)calibrate your robot.

Two modes are available via the ``--mode`` flag:

  legacy  (default)
      The original two-step flow: move to zero position, then to a 90-degree
      rotated position.  Drive mode and homing offsets are inferred from those
      two snapshots.  Reference webp images are shown as URLs in the prompts.

  range
      Newer range-of-motion flow (no reference images):
        1. Move the arm to its anatomical zero and press Enter.
        2. Sweep every joint through its full range while a live
           MIN | POS | MAX table streams to the terminal.
        3. Drive mode is inferred automatically from which raw direction each
           motor traveled further from zero.
      Produces the same JSON format as ``legacy``.

Examples:

```bash
# Range-of-motion calibration of all arms:
python lerobot/scripts/lerobot_calibrate.py \\
  --robot-path lerobot/configs/robot/aloha.yaml \\
  --mode range

# Recalibrate only specific arms (range mode):
python lerobot/scripts/lerobot_calibrate.py \\
  --robot-path lerobot/configs/robot/aloha.yaml \\
  --mode range \\
  --arms left_follower right_follower

# Legacy two-step calibration:
python lerobot/scripts/lerobot_calibrate.py \\
  --robot-path lerobot/configs/robot/aloha.yaml \\
  --mode legacy
```
"""

from __future__ import annotations

import argparse
import json
import logging

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.robots.utils import Robot, get_arm_id
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.utils.utils import init_hydra_config, init_logging


def _validate_arms(robot: Robot, arms: list[str] | None) -> list[str]:
    if arms is None:
        arms = list(getattr(robot, "available_arms"))

    if len(arms) == 0:
        available_arms_str = " ".join(getattr(robot, "available_arms"))
        raise ValueError(
            "No arm provided. Use `--arms` with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    available_arms = set(getattr(robot, "available_arms"))
    unknown_arms = [arm_id for arm_id in arms if arm_id not in available_arms]
    if unknown_arms:
        raise ValueError(
            f"Unknown arms: {unknown_arms}. Available: {sorted(available_arms)}"
        )
    return arms


def _get_arm_bus(robot: ManipulatorRobot, arm_id: str):
    """Return (bus, name, arm_type) for arm_id like 'left_follower'."""
    for name in robot.follower_arms:
        if get_arm_id(name, "follower") == arm_id:
            return robot.follower_arms[name], name, "follower"
    for name in robot.leader_arms:
        if get_arm_id(name, "leader") == arm_id:
            return robot.leader_arms[name], name, "leader"
    raise ValueError(f"Arm '{arm_id}' not found on robot.")


@safe_disconnect
def _calibrate_legacy(robot: ManipulatorRobot, arms: list[str]) -> None:
    """Original two-step (zero + rotated) calibration."""
    calibration_dir = robot.calibration_dir

    for arm_id in arms:
        arm_calib_path = calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            logging.info(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()

    # Connecting with missing calibration files triggers the interactive flow
    # already implemented in ManipulatorRobot.activate_calibration().
    robot.connect()
    robot.disconnect()
    logging.info("Calibration done. You can now teleoperate and record datasets.")


@safe_disconnect
def _calibrate_range(robot: ManipulatorRobot, arms: list[str]) -> None:
    """Range-of-motion calibration producing the same JSON format."""
    if robot.robot_type in ["koch", "koch_bimanual", "aloha"]:
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        from lerobot.common.robot_devices.robots.dynamixel_calibration import (
            run_arm_range_calibration,
        )
    elif robot.robot_type in ["so100", "moss"]:
        raise NotImplementedError(
            "Range calibration for Feetech (so100/moss) is not yet implemented. "
            "Use --mode legacy for now."
        )
    else:
        raise ValueError(f"Unsupported robot type: {robot.robot_type}")

    calibration_dir = robot.calibration_dir
    calibration_dir.mkdir(parents=True, exist_ok=True)

    # Connect arms (buses) without triggering the existing calibration flow.
    for name in robot.follower_arms:
        robot.follower_arms[name].connect()
    for name in robot.leader_arms:
        robot.leader_arms[name].connect()

    # Disable torque on all arms so joints are free to move.
    for name in robot.follower_arms:
        robot.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
    for name in robot.leader_arms:
        robot.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

    robot.is_connected = True

    for arm_id in arms:
        arm, arm_name, arm_type = _get_arm_bus(robot, arm_id)

        calib_data = run_arm_range_calibration(arm, robot.robot_type, arm_name, arm_type)

        arm_calib_path = calibration_dir / f"{arm_id}.json"
        with open(arm_calib_path, "w") as f:
            json.dump(calib_data, f)
        print(f"Calibration saved to '{arm_calib_path}'")

    robot.disconnect()
    logging.info("Range calibration done. You can now teleoperate and record datasets.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate robot arms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/aloha.yaml",
        help="Path to robot yaml file.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        default=None,
        help="Override robot config parameters.",
    )
    parser.add_argument(
        "--arms",
        type=str,
        nargs="*",
        default=None,
        help="Arms to calibrate (e.g. --arms left_follower right_follower). "
             "If omitted, calibrates all available arms.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["legacy", "range"],
        default="range",
        help="'range' = live range-of-motion (no reference images, default). "
             "'legacy' = original zero+rotated two-step flow.",
    )
    args = parser.parse_args()

    init_logging()

    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
    robot = make_robot(robot_cfg)

    if not isinstance(robot, ManipulatorRobot):
        raise ValueError(
            f"Only ManipulatorRobot is supported by this script. Got {type(robot).__name__}."
        )

    arms = _validate_arms(robot, args.arms)

    if args.mode == "range":
        _calibrate_range(robot, arms)
    else:
        _calibrate_legacy(robot, arms)


if __name__ == "__main__":
    main()
