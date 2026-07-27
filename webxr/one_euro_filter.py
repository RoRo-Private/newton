# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""One Euro Filter for VR controller poses.

Implements the One Euro Filter [Casiez et al., CHI 2012] for 3-D position
and unit quaternion orientation.  The filter adapts its cutoff frequency to
signal speed: slow / stationary → strong smoothing; fast motion → low lag.

Typical usage::

    from webxr.one_euro_filter import ControllerFilter

    filt = ControllerFilter(min_cutoff=1.0, beta=0.1, d_cutoff=1.0)

    # each step (dt in seconds):
    pos_f, quat_f = filt.update(position, quaternion_xyzw, dt)

References:
    Casiez, G., Roussel, N., & Vogel, D. (2012). 1€ filter: A simple
    speed-based low-pass filter for noisy input in interactive systems.
    CHI 2012. https://doi.org/10.1145/2207676.2208639
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _alpha(cutoff: float, dt: float) -> float:
    """Compute EMA coefficient from cutoff frequency [Hz] and time step [s]."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions q0 and q1."""
    dot = float(np.dot(q0, q1))
    # Ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        # Nearly identical — linear interpolation is fine
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


# ---------------------------------------------------------------------------
# Scalar / vector One Euro Filter
# ---------------------------------------------------------------------------

class VectorOneEuroFilter:
    """One Euro Filter for an N-dimensional real-valued signal.

    Args:
        min_cutoff: Minimum cutoff frequency [Hz].  Lower → smoother but more lag
            when stationary.
        beta: Speed coefficient.  Higher → faster response during fast motion.
        d_cutoff: Cutoff frequency [Hz] for the derivative low-pass filter.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None

    def update(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Filter one sample.

        Args:
            x: New raw sample, shape (N,).
            dt: Time since last sample [s].  Ignored on the first call.

        Returns:
            Filtered sample, same shape as *x*.
        """
        x = np.asarray(x, dtype=float)

        if self._x_prev is None or dt <= 0.0:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()

        # Derivative estimate
        dx = (x - self._x_prev) / dt

        # Low-pass filter the derivative
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff based on speed
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed

        # Low-pass filter the signal
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


# ---------------------------------------------------------------------------
# Quaternion One Euro Filter
# ---------------------------------------------------------------------------

class QuaternionOneEuroFilter:
    """One Euro Filter for unit quaternions (xyzw convention).

    Uses SLERP for the low-pass step and angular velocity for speed
    estimation, so the filter works correctly on SO(3).

    Args:
        min_cutoff: Minimum cutoff frequency [Hz].
        beta: Speed coefficient (scales with angular velocity [rad/s]).
        d_cutoff: Cutoff for the derivative (angular velocity) filter [Hz].
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._q_prev: np.ndarray | None = None
        self._dq_prev: float = 0.0  # angular speed [rad/s]

    def reset(self) -> None:
        self._q_prev = None
        self._dq_prev = 0.0

    def update(self, q: np.ndarray, dt: float) -> np.ndarray:
        """Filter one quaternion sample.

        Args:
            q: Raw quaternion (x, y, z, w), shape (4,).
            dt: Time since last sample [s].

        Returns:
            Filtered unit quaternion, shape (4,).
        """
        q = np.asarray(q, dtype=float)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        if self._q_prev is None or dt <= 0.0:
            self._q_prev = q.copy()
            return q.copy()

        # Shortest-path dot for angular distance
        dot = float(np.dot(self._q_prev, q))
        if dot < 0.0:
            q_cmp = -q
            dot = -dot
        else:
            q_cmp = q
        dot = min(dot, 1.0)

        # Angular speed [rad/s]
        angular_dist = 2.0 * math.acos(dot)
        dq = angular_dist / dt

        # Low-pass filter angular speed
        a_d = _alpha(self.d_cutoff, dt)
        dq_hat = a_d * dq + (1.0 - a_d) * self._dq_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * dq_hat

        # SLERP low-pass filter
        a = _alpha(cutoff, dt)
        q_hat = _slerp(self._q_prev, q_cmp, a)

        self._q_prev = q_hat
        self._dq_prev = dq_hat
        return q_hat


# ---------------------------------------------------------------------------
# Convenience wrapper for a single controller
# ---------------------------------------------------------------------------

class ControllerFilter:
    """Combined position + quaternion One Euro Filter for one controller.

    Args:
        min_cutoff: Minimum cutoff frequency [Hz] for both sub-filters.
        beta: Speed coefficient for both sub-filters.
        d_cutoff: Derivative cutoff [Hz] for both sub-filters.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._pos = VectorOneEuroFilter(min_cutoff, beta, d_cutoff)
        self._rot = QuaternionOneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self._pos.reset()
        self._rot.reset()

    def update(
        self,
        position: tuple[float, float, float] | np.ndarray,
        quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply filter to position and quaternion.

        Args:
            position: Raw position (x, y, z) [m].
            quaternion_xyzw: Raw orientation (x, y, z, w).
            dt: Time elapsed since previous call [s].

        Returns:
            Tuple of (filtered_position, filtered_quaternion_xyzw).
        """
        pos_f = self._pos.update(np.asarray(position, dtype=float), dt)
        quat_f = self._rot.update(np.asarray(quaternion_xyzw, dtype=float), dt)
        return pos_f, quat_f
