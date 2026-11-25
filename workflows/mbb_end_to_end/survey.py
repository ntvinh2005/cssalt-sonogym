"""Append public Stage 1-relative oblique views to the matched 102-waypoint survey."""

import hashlib
import json
import math


def append_oblique(waypoints, region, supplement):
    encoded = (json.dumps(waypoints, indent=2) + "\n").encode()
    if (len(waypoints) != 102 or
            hashlib.sha256(encoded).hexdigest() != supplement["original_waypoints_sha256"]):
        raise ValueError("original waypoints differ from the sealed recordings")
    result = list(waypoints)
    for yaw in supplement["yaw_radians"]:
        for lane, lateral in enumerate(supplement["right_offsets_index"]):
            offsets = supplement["cranial_offsets_index"]
            for cranial in (offsets if lane % 2 == 0 else offsets[::-1]):
                pose = (float(region.home_x + lateral), float(region.l5_z + cranial), float(yaw))
                if not all(map(math.isfinite, pose)):
                    raise ValueError("nonfinite Stage 1-relative supplemental pose")
                result.append(("oblique", pose))
    count = len(result) - len(waypoints)
    if not 20 <= count <= 40 or count != supplement["additional_waypoint_count"]:
        raise ValueError("supplemental waypoint budget exceeded or inconsistent")
    return result
