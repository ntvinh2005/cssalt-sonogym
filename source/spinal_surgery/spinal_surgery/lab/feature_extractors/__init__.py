# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

try:
    from . import sb3_cnn_extractor
except ModuleNotFoundError as exc:
    if exc.name != "stable_baselines3":
        raise
