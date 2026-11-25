"""Sweep geometry and acquisition measurements for the robustness experiments."""

from copy import copy

import numpy as np


def append_transverse_lanes(waypoints, region, lane_offsets):
    transverse = [pose for phase, pose in waypoints if phase == "transverse"]
    template_x, _, angle = transverse[0]
    cranial = sorted(z for x, z, _ in transverse if x == template_x)
    result = list(waypoints)
    for i, lateral_offset in enumerate(lane_offsets):
        direction = cranial[::-1] if i % 2 == 0 else cranial
        result.extend(("transverse", (region.home_x + lateral_offset, z, angle)) for z in direction)
    return result


def nominal_probe_transform(sensor, waypoint):
    import torch

    planner = copy(sensor)
    # The surface planner writes its target buffers in place.
    for key in ("target_rot_mat", "target_position", "target_z_axis", "target_x_axis_proj"):
        setattr(planner, key, getattr(sensor, key).clone())
    planner.current_x_z_x_angle_cmd = sensor.current_x_z_x_angle_cmd.new_tensor([waypoint])
    identity = torch.tensor([[1., 0., 0., 0.]], device=sensor.device)
    origin = torch.zeros((1, 3), device=sensor.device)
    return planner.compute_world_ee_pose_from_cmd(origin, identity)


def transform_error(target_position, target_quaternion, position, quaternion):
    a, b = np.array(target_quaternion, dtype=float), np.array(quaternion, dtype=float)
    cosine = np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0., 1.)
    return (float(1000 * np.linalg.norm(np.asarray(target_position) - position)),
            float(np.rad2deg(2 * np.arccos(cosine))))


def hold_probe(env, sensor, actions, steps):
    import torch

    action = torch.zeros((1, 3), device=env.unwrapped.device)
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(action)
        valid = not (bool(terminated[0]) or bool(truncated[0]) or bool(sensor.no_collide[0]))
        actions.append({
            "finite": valid, "command_position_clamped": False, "command_mapping_valid": True,
            "surface_proxy_ok": valid, "terminated": bool(terminated[0]), "truncated": bool(truncated[0]),
        })
        if not valid:
            raise TimeoutError("contact or episode failed during stabilization")
