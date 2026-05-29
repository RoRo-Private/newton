# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Legacy-style coupler scaffold.

This module mirrors the LegacyCoupler flow:

- ``build()``
- ``preprocess(substep_index)``
- ``couple(substep_index)``

Engine-specific coupling logic is delegated to an adapter object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class CouplerOptions:
    rigid_xpbd: bool = True
    eps: float = 1e-12


@dataclass
class RigidCouplingMaterial:
    needs_coup: bool = True
    coup_friction: float = 0.1
    coup_softness: float = 0.002
    coup_restitution: float = 0.0


@dataclass
class CollisionSample:
    valid: bool = False
    signed_distance: float = 0.0
    normal_world: np.ndarray | None = None
    rigid_body_id: int = -1
    material: RigidCouplingMaterial = field(default_factory=RigidCouplingMaterial)


class CouplerAdapter(Protocol):
    def is_rigid_active(self) -> bool: ...

    def is_xpbd_active(self) -> bool: ...

    def substep_dt(self) -> float: ...

    def couple_xpbd_with_rigid(self) -> None: ...

    def query_rigid_collision(self, pos_world: np.ndarray) -> CollisionSample: ...

    def rigid_velocity_at_point(self, rigid_body_id: int, pos_world: np.ndarray) -> np.ndarray: ...

    def apply_coupling_force(self, rigid_body_id: int, force_world: np.ndarray, at_pos_world: np.ndarray) -> None: ...


class LegacyLikeCoupler:
    def __init__(self, adapter: CouplerAdapter, options: CouplerOptions | None = None):
        self._adapter = adapter
        self._options = options or CouplerOptions()
        self._rigid_xpbd = False

    def build(self) -> None:
        self._rigid_xpbd = (
            self._adapter.is_rigid_active() and self._adapter.is_xpbd_active() and self._options.rigid_xpbd
        )

    def preprocess(self, _substep_index: int) -> None:
        # Reserved hook for cached normals/neighbors.
        return

    def couple(self, _substep_index: int) -> None:
        if self._rigid_xpbd:
            self._adapter.couple_xpbd_with_rigid()

    def resolve_rigid_collision(
        self,
        pos_world: np.ndarray,
        vel: np.ndarray,
        mass: float,
        sample: CollisionSample,
    ) -> np.ndarray:
        if not sample.valid or not sample.material.needs_coup:
            return vel

        influence = self._influence_from_signed_distance(sample.signed_distance, sample.material.coup_softness)
        if influence <= 0.1:
            return vel

        normal = self._normalize(sample.normal_world)
        vel_rigid = self._adapter.rigid_velocity_at_point(sample.rigid_body_id, pos_world)

        rvel = vel - vel_rigid
        rvel_n_mag = float(np.dot(rvel, normal))
        if rvel_n_mag >= 0.0:
            return vel

        rvel_tan = rvel - normal * rvel_n_mag
        tan_norm = max(float(np.linalg.norm(rvel_tan)), self._options.eps)
        tan_after = max(0.0, tan_norm + rvel_n_mag * sample.material.coup_friction)
        rvel_tan *= tan_after / tan_norm

        rvel_normal = normal * (-rvel_n_mag * sample.material.coup_restitution)
        rvel_new = rvel_tan + rvel_normal
        new_vel = vel_rigid + rvel_new * influence + rvel * (1.0 - influence)

        delta_mv = (new_vel - vel) * mass
        inv_dt = 1.0 / max(self._adapter.substep_dt(), self._options.eps)
        reaction_force = -delta_mv * inv_dt
        self._adapter.apply_coupling_force(sample.rigid_body_id, reaction_force, pos_world)

        return new_vel

    def _influence_from_signed_distance(self, signed_distance: float, coup_softness: float) -> float:
        softness = max(coup_softness, 1e-10)
        return min(math.exp(-signed_distance / softness), 1.0)

    def _normalize(self, v: np.ndarray | None) -> np.ndarray:
        if v is None:
            return np.zeros(3, dtype=np.float64)
        n = float(np.linalg.norm(v))
        if n < self._options.eps:
            return np.zeros(3, dtype=np.float64)
        return v / n


__all__ = [
    "CollisionSample",
    "CouplerAdapter",
    "CouplerOptions",
    "LegacyLikeCoupler",
    "RigidCouplingMaterial",
]
