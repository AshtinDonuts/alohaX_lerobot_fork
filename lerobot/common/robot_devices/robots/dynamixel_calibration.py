"""Logic to calibrate a robot arm built with dynamixel motors"""
# TODO(rcadene, aliberts): move this logic into the robot code when refactoring

import select
import sys
import time

import numpy as np

from lerobot.common.robot_devices.motors.dynamixel import (
    CalibrationMode,
    TorqueMode,
    convert_degrees_to_steps,
)
from lerobot.common.robot_devices.motors.utils import MotorsBus

URL_TEMPLATE = (
    "https://raw.githubusercontent.com/huggingface/lerobot/main/media/{robot}/{arm}_{position}.webp"
)

_RANGE_POLL_HZ = 20  # how often to re-read positions during range recording


def _enter_pressed() -> bool:
    """Non-blocking check: returns True if the user pressed Enter."""
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False

# The following positions are provided in nominal degree range ]-180, +180[
# For more info on these constants, see comments in the code where they get used.
ZERO_POSITION_DEGREE = 0
ROTATED_POSITION_DEGREE = 90


def assert_drive_mode(drive_mode):
    # `drive_mode` is in [0,1] with 0 means original rotation direction for the motor, and 1 means inverted.
    if not np.all(np.isin(drive_mode, [0, 1])):
        raise ValueError(f"`drive_mode` contains values other than 0 or 1: ({drive_mode})")


def apply_drive_mode(position, drive_mode):
    assert_drive_mode(drive_mode)
    # Convert `drive_mode` from [0, 1] with 0 indicates original rotation direction and 1 inverted,
    # to [-1, 1] with 1 indicates original rotation direction and -1 inverted.
    signed_drive_mode = -(drive_mode * 2 - 1)
    position *= signed_drive_mode
    return position


def compute_nearest_rounded_position(position, models):
    delta_turn = convert_degrees_to_steps(ROTATED_POSITION_DEGREE, models)
    nearest_pos = np.round(position.astype(float) / delta_turn) * delta_turn
    return nearest_pos.astype(position.dtype)


def run_arm_calibration(arm: MotorsBus, robot_type: str, arm_name: str, arm_type: str):
    """This function ensures that a neural network trained on data collected on a given robot
    can work on another robot. For instance before calibration, setting a same goal position
    for each motor of two different robots will get two very different positions. But after calibration,
    the two robots will move to the same position.To this end, this function computes the homing offset
    and the drive mode for each motor of a given robot.

    Homing offset is used to shift the motor position to a ]-2048, +2048[ nominal range (when the motor uses 2048 steps
    to complete a half a turn). This range is set around an arbitrary "zero position" corresponding to all motor positions
    being 0. During the calibration process, you will need to manually move the robot to this "zero position".

    Drive mode is used to invert the rotation direction of the motor. This is useful when some motors have been assembled
    in the opposite orientation for some robots. During the calibration process, you will need to manually move the robot
    to the "rotated position".

    After calibration, the homing offsets and drive modes are stored in a cache.

    Example of usage:
    ```python
    run_arm_calibration(arm, "koch", "left", "follower")
    ```
    """
    if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
        raise ValueError("To run calibration, the torque must be disabled on all motors.")

    print(f"\nRunning calibration of {robot_type} {arm_name} {arm_type}...")

    print("\nMove arm to zero position")
    print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="zero"))
    input("Press Enter to continue...")

    # We arbitrarily chose our zero target position to be a straight horizontal position with gripper upwards and closed.
    # It is easy to identify and all motors are in a "quarter turn" position. Once calibration is done, this position will
    # correspond to every motor angle being 0. If you set all 0 as Goal Position, the arm will move in this position.
    zero_target_pos = convert_degrees_to_steps(ZERO_POSITION_DEGREE, arm.motor_models)

    # Compute homing offset so that `present_position + homing_offset ~= target_position`.
    zero_pos = arm.read("Present_Position")
    zero_nearest_pos = compute_nearest_rounded_position(zero_pos, arm.motor_models)
    homing_offset = zero_target_pos - zero_nearest_pos

    print("\n=== Zero Position Results ===")
    print(f"Target position (steps): {dict(zip(arm.motor_names, zero_target_pos))}")
    print(f"Present position (steps): {dict(zip(arm.motor_names, zero_pos))}")
    print(f"Nearest rounded position (steps): {dict(zip(arm.motor_names, zero_nearest_pos))}")
    print(f"Initial homing offset (steps): {dict(zip(arm.motor_names, homing_offset))}")

    # The rotated target position corresponds to a rotation of a quarter turn from the zero position.
    # This allows to identify the rotation direction of each motor.
    # For instance, if the motor rotates 90 degree, and its value is -90 after applying the homing offset, then we know its rotation direction
    # is inverted. However, for the calibration being successful, we need everyone to follow the same target position.
    # Sometimes, there is only one possible rotation direction. For instance, if the gripper is closed, there is only one direction which
    # corresponds to opening the gripper. When the rotation direction is ambiguous, we arbitrarely rotate clockwise from the point of view
    # of the previous motor in the kinetic chain.
    print("\nMove arm to rotated target position")
    print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="rotated"))
    input("Press Enter to continue...")

    rotated_target_pos = convert_degrees_to_steps(ROTATED_POSITION_DEGREE, arm.motor_models)

    # Find drive mode by rotating each motor by a quarter of a turn.
    # Drive mode indicates if the motor rotation direction should be inverted (=1) or not (=0).
    rotated_pos = arm.read("Present_Position")
    drive_mode = (rotated_pos < zero_pos).astype(np.int32)

    # Re-compute homing offset to take into account drive mode
    rotated_drived_pos = apply_drive_mode(rotated_pos, drive_mode)
    rotated_nearest_pos = compute_nearest_rounded_position(rotated_drived_pos, arm.motor_models)
    homing_offset = rotated_target_pos - rotated_nearest_pos

    print("\n=== Rotated Position Results ===")
    print(f"Target position (steps): {dict(zip(arm.motor_names, rotated_target_pos))}")
    print(f"Present position (steps): {dict(zip(arm.motor_names, rotated_pos))}")
    print(f"Drive mode (0=normal, 1=inverted): {dict(zip(arm.motor_names, drive_mode))}")
    print(f"Position after drive mode applied (steps): {dict(zip(arm.motor_names, rotated_drived_pos))}")
    print(f"Nearest rounded position (steps): {dict(zip(arm.motor_names, rotated_nearest_pos))}")
    print(f"Final homing offset (steps): {dict(zip(arm.motor_names, homing_offset))}")

    print("\nMove arm to rest position")
    print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="rest"))
    input("Press Enter to continue...")

    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    calib_mode = [CalibrationMode.DEGREE.name] * len(arm.motor_names)

    # TODO(rcadene): make type of joints (DEGREE or LINEAR) configurable from yaml?
    if robot_type in ["aloha"] and "gripper" in arm.motor_names:
        # Joints with linear motions (like gripper of Aloha) are experessed in nominal range of [0, 100]
        calib_idx = arm.motor_names.index("gripper")
        calib_mode[calib_idx] = CalibrationMode.LINEAR.name

    calib_data = {
        "homing_offset": homing_offset.tolist(),
        "drive_mode": drive_mode.tolist(),
        "start_pos": zero_pos.tolist(),
        "end_pos": rotated_pos.tolist(),
        "calib_mode": calib_mode,
        "motor_names": arm.motor_names,
    }

    print("\n=== Final Calibration Results ===")
    print(f"Calibration mode: {dict(zip(arm.motor_names, calib_mode))}")
    print(f"Homing offset (steps): {dict(zip(arm.motor_names, homing_offset))}")
    print(f"Drive mode (0=normal, 1=inverted): {dict(zip(arm.motor_names, drive_mode))}")
    print(f"Start position (zero, steps): {dict(zip(arm.motor_names, zero_pos))}")
    print(f"End position (rotated, steps): {dict(zip(arm.motor_names, rotated_pos))}")
    print()

    return calib_data


def run_arm_range_calibration(arm: MotorsBus, robot_type: str, arm_name: str, arm_type: str):
    """Range-of-motion calibration (newer-style, no reference images).

    This is a drop-in replacement for ``run_arm_calibration`` that produces the same
    JSON format while using a different interaction model:

    1. **Zero step** – user moves the arm to its anatomical zero (straight/rest
       position) and presses Enter.  Homing offsets are computed so that position
       becomes 0 degrees.

    2. **Range step** – a live table streams ``MIN | POS | MAX`` raw encoder counts
       for every joint while the user sweeps each joint through its full range of
       motion with torque off.  Press Enter to finish.

    3. **Drive-mode inference** – for each DEGREE joint the script checks whether
       the motor traveled further in the raw-negative direction from zero (→ the
       encoder is "upside-down", drive_mode=1) or further in the raw-positive
       direction (drive_mode=0).  This relies on the zero position being near the
       anatomical mid-range of each joint, which holds for Aloha/Koch-style arms.

    LINEAR joints (e.g. Aloha gripper) use ``range_min`` as ``start_pos`` (0 %)
    and ``range_max`` as ``end_pos`` (100 %).
    """
    if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
        raise ValueError("To run calibration, the torque must be disabled on all motors.")

    print(f"\nRunning range-of-motion calibration of {robot_type} {arm_name} {arm_type}...")

    # ── Phase 1: zero position ────────────────────────────────────────────────
    print("\n[Phase 1] Move the arm to its ZERO / rest position.")
    print("  All joints should be in a neutral, straight position.")
    input("Press Enter when ready...")

    zero_pos = arm.read("Present_Position")
    zero_target_pos = convert_degrees_to_steps(ZERO_POSITION_DEGREE, arm.motor_models)
    zero_nearest_pos = compute_nearest_rounded_position(zero_pos, arm.motor_models)

    print("\n=== Zero Position ===")
    for name, pos in zip(arm.motor_names, zero_pos):
        print(f"  {name:<20} raw={pos}")

    # ── Phase 2: range of motion ──────────────────────────────────────────────
    print("\n[Phase 2] Slowly move EACH JOINT through its FULL range of motion")
    print("  (both directions from zero).  The table below updates live.")
    print("  Press Enter when you have swept all joints to their limits.\n")

    # Flush any pending newlines in stdin before the non-blocking loop.
    if select.select([sys.stdin], [], [], 0.0)[0]:
        sys.stdin.readline()

    range_min = zero_pos.copy()
    range_max = zero_pos.copy()

    col_w = max(len(n) for n in arm.motor_names) + 2
    header = f"{'NAME':<{col_w}} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}"

    while True:
        positions = arm.read("Present_Position")
        range_min = np.minimum(range_min, positions)
        range_max = np.maximum(range_max, positions)

        # Overwrite the previous table lines in-place.
        lines = len(arm.motor_names) + 2
        sys.stdout.write(f"\033[{lines}A")  # move cursor up
        print(header)
        print("-" * len(header))
        for name, mn, pos, mx in zip(arm.motor_names, range_min, positions, range_max):
            print(f"{name:<{col_w}} | {mn:>6} | {pos:>6} | {mx:>6}")

        if _enter_pressed():
            break

        time.sleep(1.0 / _RANGE_POLL_HZ)

    end_pos = arm.read("Present_Position")

    print("\n=== Range of Motion Recorded ===")
    for name, mn, mx in zip(arm.motor_names, range_min, range_max):
        print(f"  {name:<20} min={mn}  max={mx}  span={mx - mn}")

    # ── Phase 3: derive drive_mode and homing_offset ──────────────────────────
    # drive_mode=1 when the motor has more raw range in the negative direction
    # from zero (i.e. the encoder decreases for positive physical motion).
    neg_excursion = zero_pos - range_min   # how far below zero we reached
    pos_excursion = range_max - zero_pos   # how far above zero we reached
    drive_mode = (neg_excursion > pos_excursion).astype(np.int32)

    # Recompute homing_offset accounting for drive_mode (same formula as old calibration).
    driven_zero = apply_drive_mode(zero_pos.copy(), drive_mode)
    driven_nearest = compute_nearest_rounded_position(driven_zero, arm.motor_models)
    homing_offset = zero_target_pos - driven_nearest

    # ── calib_mode per joint ──────────────────────────────────────────────────
    calib_mode = [CalibrationMode.DEGREE.name] * len(arm.motor_names)
    if robot_type in ["aloha"] and "gripper" in arm.motor_names:
        calib_idx = arm.motor_names.index("gripper")
        calib_mode[calib_idx] = CalibrationMode.LINEAR.name

    # For LINEAR joints override start/end with the observed range extremes.
    start_pos = zero_pos.copy().tolist()
    stored_end_pos = end_pos.tolist()
    for i, (mode, name) in enumerate(zip(calib_mode, arm.motor_names)):
        if mode == CalibrationMode.LINEAR.name:
            start_pos[i] = int(range_min[i])
            stored_end_pos[i] = int(range_max[i])

    calib_data = {
        "homing_offset": homing_offset.tolist(),
        "drive_mode": drive_mode.tolist(),
        "start_pos": start_pos,
        "end_pos": stored_end_pos,
        "calib_mode": calib_mode,
        "motor_names": arm.motor_names,
    }

    print("\n=== Final Calibration ===")
    print(f"  {'Motor':<20} {'drive_mode':>10} {'homing_offset':>14} {'calib_mode':>10}")
    print(f"  {'-'*56}")
    for name, dm, ho, cm in zip(arm.motor_names, drive_mode, homing_offset, calib_mode):
        print(f"  {name:<20} {dm:>10} {ho:>14} {cm:>10}")
    print()

    return calib_data
