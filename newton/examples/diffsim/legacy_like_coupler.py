# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible re-export for LegacyLikeCoupler symbols.

Prefer importing from ``newton.solvers``.
"""

from newton.solvers import CollisionSample, CouplerAdapter, CouplerOptions, LegacyLikeCoupler, RigidCouplingMaterial

__all__ = [
    "CollisionSample",
    "CouplerAdapter",
    "CouplerOptions",
    "LegacyLikeCoupler",
    "RigidCouplingMaterial",
]
