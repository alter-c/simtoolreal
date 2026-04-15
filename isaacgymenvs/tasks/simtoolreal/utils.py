# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import Tuple

from torch import Tensor


def populate_dof_properties(hand_arm_dof_props, arm_dofs: int, hand_dofs: int) -> None:
    assert len(hand_arm_dof_props["stiffness"]) == arm_dofs + hand_dofs

    import numpy as np

    # G1 left arm PD controller parameters (scaled from Kuka values proportionally)
    # Kuka arm efforts were 300Nm for shoulder/elbow, scaled to G1 25Nm shoulder = 25/300 ratio
    # Shoulder joints: 25Nm effort, elbow: 25Nm, wrist: 5Nm
    # Stiffness/damping scaled similarly using the same ratio (~0.083x)
    arm_efforts = [25.0, 25.0, 25.0, 25.0, 5.0, 25.0, 5.0]
    arm_stiffnesses = [50.0, 50.0, 42.0, 33.0, 4.0, 17.0, 4.0]
    arm_dampings = [
        4.0,
        4.0,
        3.5,
        3.0,
        0.5,
        1.5,
        0.5,
    ]
    arm_armatures = [
        0.1,
        0.1,
        0.08,
        0.08,
        0.01,
        0.04,
        0.01,
    ]

    assert (
        len(arm_stiffnesses)
        == len(arm_dampings)
        == len(arm_armatures)
        == len(arm_efforts)
        == arm_dofs
    ), (
        f"{len(arm_stiffnesses)} != {len(arm_dampings)} != {len(arm_armatures)} != {len(arm_efforts)} != {arm_dofs}"
    )

    hand_arm_dof_props["stiffness"][0:arm_dofs] = arm_stiffnesses
    hand_arm_dof_props["damping"][0:arm_dofs] = arm_dampings
    hand_arm_dof_props["armature"][0:arm_dofs] = arm_armatures
    hand_arm_dof_props["effort"][0:arm_dofs] = arm_efforts

    # LinkerHand O6 left hand PD parameters
    # 11 finger joints: lh_thumb_cmc_yaw, lh_thumb_cmc_pitch, lh_thumb_ip,
    #   lh_index_mcp_pitch, lh_index_dip, lh_middle_mcp_pitch, lh_middle_dip,
    #   lh_ring_mcp_pitch, lh_ring_dip, lh_pinky_mcp_pitch, lh_pinky_dip
    hand_stiffnesses = [
        8.0,
        10.0,
        3.0,
        6.0,
        2.0,
        6.0,
        2.0,
        6.0,
        2.0,
        6.0,
        2.0,
    ]
    hand_dampings = [
        0.4,
        0.5,
        0.15,
        0.3,
        0.1,
        0.3,
        0.1,
        0.3,
        0.1,
        0.3,
        0.1,
    ]
    hand_armatures = [
        0.001,
        0.001,
        0.0005,
        0.001,
        0.0003,
        0.001,
        0.0003,
        0.001,
        0.0003,
        0.001,
        0.0003,
    ]
    hand_frictions = [
        0.05,
        0.05,
        0.02,
        0.05,
        0.02,
        0.05,
        0.02,
        0.05,
        0.02,
        0.05,
        0.02,
    ]
    assert (
        len(hand_stiffnesses)
        == len(hand_dampings)
        == len(hand_armatures)
        == len(hand_frictions)
        == hand_dofs
    ), (
        f"{len(hand_stiffnesses)} != {len(hand_dampings)} != {len(hand_armatures)} != {len(hand_frictions)} != {hand_dofs}"
    )
    hand_arm_dof_props["stiffness"][arm_dofs:] = hand_stiffnesses
    hand_arm_dof_props["damping"][arm_dofs:] = hand_dampings
    hand_arm_dof_props["armature"][arm_dofs:] = hand_armatures
    hand_arm_dof_props["friction"][arm_dofs:] = hand_frictions


def tolerance_curriculum(
    last_curriculum_update: int,
    frames_since_restart: int,
    curriculum_interval: int,
    prev_episode_successes: Tensor,
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    tolerance_curriculum_increment: float,
) -> Tuple[float, int]:
    """
    Returns: new tolerance, new last_curriculum_update
    """
    if frames_since_restart - last_curriculum_update < curriculum_interval:
        return success_tolerance, last_curriculum_update

    mean_successes_per_episode = prev_episode_successes.mean()
    if mean_successes_per_episode < 3.0:
        # this policy is not good enough with the previous tolerance value, keep training for now...
        return success_tolerance, last_curriculum_update

    # decrease the tolerance now
    success_tolerance *= tolerance_curriculum_increment
    success_tolerance = min(success_tolerance, initial_tolerance)
    success_tolerance = max(success_tolerance, target_tolerance)

    print(
        f"Prev episode successes: {mean_successes_per_episode}, success tolerance: {success_tolerance}"
    )

    last_curriculum_update = frames_since_restart
    return success_tolerance, last_curriculum_update


def interp_0_1(x_curr: float, x_initial: float, x_target: float) -> float:
    """
    Outputs 1 when x_curr == x_target (curriculum completed)
    Outputs 0 when x_curr == x_initial (just started training)
    Interpolates value in between.
    """
    span = x_initial - x_target
    return (x_initial - x_curr) / span


def tolerance_successes_objective(
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    successes: Tensor,
) -> Tensor:
    """
    Objective for the PBT. This basically prioritizes tolerance over everything else when we
    execute the curriculum, after that it's just #successes.
    """
    # this grows from 0 to 1 as we reach the target tolerance
    if initial_tolerance > target_tolerance:
        # makeshift unit tests:
        eps = 1e-5
        assert (
            abs(interp_0_1(initial_tolerance, initial_tolerance, target_tolerance))
            < eps
        )
        assert (
            abs(interp_0_1(target_tolerance, initial_tolerance, target_tolerance) - 1.0)
            < eps
        )
        mid_tolerance = (initial_tolerance + target_tolerance) / 2
        assert (
            abs(interp_0_1(mid_tolerance, initial_tolerance, target_tolerance) - 0.5)
            < eps
        )

        tolerance_objective = interp_0_1(
            success_tolerance, initial_tolerance, target_tolerance
        )
    else:
        tolerance_objective = 1.0

    if success_tolerance > target_tolerance:
        # add succeses with a small coefficient to differentiate between policies at the beginning of training
        # increment in tolerance improvement should always give higher value than higher successes with the
        # previous tolerance, that's why this coefficient is very small
        true_objective = (successes * 0.01) + tolerance_objective
    else:
        # basically just the successes + tolerance objective so that true_objective never decreases when we cross
        # the threshold
        true_objective = successes + tolerance_objective

    return true_objective
