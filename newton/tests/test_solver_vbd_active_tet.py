# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.particle_vbd_kernels import (
    evaluate_active_tet_contraction_force_and_hessian_kernel,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _active_energy(rest_positions, positions, director, activation, stiffness):
    dm = np.stack(
        (
            rest_positions[1] - rest_positions[0],
            rest_positions[2] - rest_positions[0],
            rest_positions[3] - rest_positions[0],
        ),
        axis=1,
    )
    ds = np.stack(
        (
            positions[1] - positions[0],
            positions[2] - positions[0],
            positions[3] - positions[0],
        ),
        axis=1,
    )
    director_norm = np.linalg.norm(director)
    if activation <= 0.0 or stiffness <= 0.0 or director_norm < 1.0e-8:
        return 0.0
    fiber = ds @ np.linalg.inv(dm) @ (director / director_norm)
    rest_volume = np.linalg.det(dm) / 6.0
    return 0.5 * rest_volume * stiffness * activation * np.dot(fiber, fiber)


def _evaluate_active_tet(
    device,
    rest_positions,
    positions,
    director,
    activation,
    stiffness,
    v_order,
    dm_inv_override=None,
):
    dm = np.stack(
        (
            rest_positions[1] - rest_positions[0],
            rest_positions[2] - rest_positions[0],
            rest_positions[3] - rest_positions[0],
        ),
        axis=1,
    )
    tet_indices = wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), dtype=int, device=device)
    dm_inv = np.linalg.inv(dm) if dm_inv_override is None else dm_inv_override
    tet_poses = wp.array(np.asarray(dm_inv)[None, ...], dtype=wp.mat33, device=device)
    pos = wp.array(positions, dtype=wp.vec3, device=device)
    directors = wp.array(np.asarray(director, dtype=np.float32)[None, ...], dtype=wp.vec3, device=device)
    activations = wp.array([activation], dtype=float, device=device)
    active_stiffness = wp.array([stiffness], dtype=float, device=device)
    force = wp.empty(1, dtype=wp.vec3, device=device)
    hessian = wp.empty(1, dtype=wp.mat33, device=device)

    wp.launch(
        kernel=evaluate_active_tet_contraction_force_and_hessian_kernel,
        dim=1,
        inputs=[
            v_order,
            pos,
            tet_indices,
            tet_poses,
            directors,
            activations,
            active_stiffness,
        ],
        outputs=[force, hessian],
        device=device,
    )
    return force.numpy()[0], hessian.numpy()[0]


def _build_active_grid(device, *, fix_left, activation, active_stiffness, steps=180):
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=6,
        dim_y=2,
        dim_z=2,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        density=1000.0,
        k_mu=1.0e5,
        k_lambda=1.0e5,
        k_damp=1.0e3,
        fix_left=fix_left,
        particle_radius=0.0,
        add_surface_mesh_edges=False,
    )
    builder.color()
    model = builder.finalize(device=device)
    model.gravity.zero_()

    solver = newton.solvers.SolverVBD(
        model,
        iterations=20,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
    )
    solver.set_tet_active_contraction(
        directors=np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (model.tet_count, 1)),
        activations=np.full(model.tet_count, activation, dtype=np.float32),
        stiffness=np.full(model.tet_count, active_stiffness, dtype=np.float32),
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    initial_positions = state_0.particle_q.numpy().copy()
    dt = 1.0 / 600.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    return initial_positions, state_0.particle_q.numpy().copy()


def _extent_ratios(initial_positions, final_positions):
    initial_extents = np.ptp(initial_positions, axis=0)
    final_extents = np.ptp(final_positions, axis=0)
    return final_extents / initial_extents


def test_active_tet_finite_difference(test, device):
    """Verify active tet force and Hessian against finite differences."""
    rest_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [0.1, 0.9, 0.1],
            [0.0, 0.2, 1.1],
        ],
        dtype=np.float64,
    )
    positions = (
        rest_positions
        @ np.array(
            [
                [0.92, 0.04, -0.02],
                [0.03, 1.05, 0.06],
                [0.01, -0.03, 0.97],
            ],
            dtype=np.float64,
        ).T
    )
    cases = (
        np.array([1.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    )
    activation = 0.65
    stiffness = 2.5e4
    epsilon = 1.0e-4

    for director in cases:
        for v_order in range(4):
            with test.subTest(director=director.tolist(), v_order=v_order):
                force, hessian = _evaluate_active_tet(
                    device,
                    rest_positions,
                    positions,
                    director,
                    activation,
                    stiffness,
                    v_order,
                )
                energy_gradient = np.zeros(3)
                force_jacobian = np.zeros((3, 3))
                for axis in range(3):
                    positions_plus = positions.copy()
                    positions_minus = positions.copy()
                    positions_plus[v_order, axis] += epsilon
                    positions_minus[v_order, axis] -= epsilon
                    energy_plus = _active_energy(rest_positions, positions_plus, director, activation, stiffness)
                    energy_minus = _active_energy(rest_positions, positions_minus, director, activation, stiffness)
                    energy_gradient[axis] = (energy_plus - energy_minus) / (2.0 * epsilon)

                    force_plus, _ = _evaluate_active_tet(
                        device,
                        rest_positions,
                        positions_plus,
                        director,
                        activation,
                        stiffness,
                        v_order,
                    )
                    force_minus, _ = _evaluate_active_tet(
                        device,
                        rest_positions,
                        positions_minus,
                        director,
                        activation,
                        stiffness,
                        v_order,
                    )
                    force_jacobian[:, axis] = (force_plus - force_minus) / (2.0 * epsilon)

                np.testing.assert_allclose(force, -energy_gradient, rtol=2.0e-4, atol=2.0e-3)
                np.testing.assert_allclose(hessian, -force_jacobian, rtol=2.0e-3, atol=2.0)
                np.testing.assert_allclose(hessian, hessian.T, rtol=0.0, atol=1.0e-6)
                test.assertGreaterEqual(np.linalg.eigvalsh(hessian).min(), -1.0e-5)


def test_active_tet_disabled_inputs(test, device):
    """Verify zero activation, stiffness, and director disable active energy."""
    rest_positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    for director, activation, stiffness in (
        (np.array([1.0, 0.0, 0.0]), 0.0, 1.0e5),
        (np.array([1.0, 0.0, 0.0]), 0.5, 0.0),
        (np.array([0.0, 0.0, 0.0]), 0.5, 1.0e5),
    ):
        force, hessian = _evaluate_active_tet(
            device,
            rest_positions,
            rest_positions,
            director,
            activation,
            stiffness,
            0,
        )
        np.testing.assert_array_equal(force, np.zeros(3))
        np.testing.assert_array_equal(hessian, np.zeros((3, 3)))

    force, hessian = _evaluate_active_tet(
        device,
        rest_positions,
        rest_positions,
        np.array([1.0, 0.0, 0.0]),
        0.5,
        1.0e5,
        0,
        dm_inv_override=np.zeros((3, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(force, np.zeros(3))
    np.testing.assert_array_equal(hessian, np.zeros((3, 3)))


def test_active_tet_setter_validation(test, device):
    """Verify setter shape checks, clamping, and in-place storage."""
    builder = newton.ModelBuilder()
    for position in (
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
    ):
        builder.add_particle(position, wp.vec3(0.0), mass=1.0, radius=0.0)
    builder.add_tetrahedron(0, 1, 2, 3, k_mu=1.0e5, k_lambda=1.0e5, k_damp=0.0)
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, particle_enable_tile_solve=False)

    director_storage = solver.tet_active_directors
    activation_storage = solver.tet_active_activations
    stiffness_storage = solver.tet_active_stiffness
    solver.set_tet_active_contraction(
        directors=np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
        activations=np.array([1.5], dtype=np.float32),
        stiffness=np.array([2.0e5], dtype=np.float32),
    )

    test.assertIs(solver.tet_active_directors, director_storage)
    test.assertIs(solver.tet_active_activations, activation_storage)
    test.assertIs(solver.tet_active_stiffness, stiffness_storage)
    np.testing.assert_array_equal(solver.tet_active_directors.numpy(), np.array([[2.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(solver.tet_active_activations.numpy(), np.array([1.0]))
    np.testing.assert_array_equal(solver.tet_active_stiffness.numpy(), np.array([2.0e5]))

    with test.assertRaisesRegex(ValueError, "directors must have shape"):
        solver.set_tet_active_contraction(
            directors=np.zeros((2, 3), dtype=np.float32),
            activations=np.ones(1, dtype=np.float32),
            stiffness=np.ones(1, dtype=np.float32),
        )


def test_active_tet_zero_activation_regression(test, device):
    """Verify zero activation preserves an unloaded VBD soft body."""
    initial, final = _build_active_grid(
        device,
        fix_left=True,
        activation=0.0,
        active_stiffness=1.0e5,
        steps=60,
    )
    np.testing.assert_allclose(final, initial, rtol=0.0, atol=1.0e-6)


def test_active_tet_directional_contraction(test, device):
    """Verify x-directed activation contracts primarily along x."""
    initial, final = _build_active_grid(
        device,
        fix_left=True,
        activation=1.0,
        active_stiffness=1.0e5,
    )
    ratios = _extent_ratios(initial, final)
    test.assertLess(ratios[0], 0.98)
    test.assertGreater(1.0 - ratios[0], max(abs(1.0 - ratios[1]), abs(1.0 - ratios[2])))


def test_active_tet_activation_monotonicity(test, device):
    """Verify contraction generally increases with activation."""
    lengths = []
    for activation in (0.0, 0.25, 0.5, 0.75, 1.0):
        initial, final = _build_active_grid(
            device,
            fix_left=True,
            activation=activation,
            active_stiffness=1.0e5,
            steps=120,
        )
        lengths.append(_extent_ratios(initial, final)[0])
    test.assertTrue(np.all(np.diff(lengths) <= 2.0e-3), msg=f"length ratios are not monotonic: {lengths}")


def test_active_tet_stiffness_monotonicity(test, device):
    """Verify contraction generally increases with active stiffness."""
    lengths = []
    for stiffness in (2.5e4, 5.0e4, 1.0e5, 2.0e5):
        initial, final = _build_active_grid(
            device,
            fix_left=True,
            activation=0.75,
            active_stiffness=stiffness,
            steps=120,
        )
        lengths.append(_extent_ratios(initial, final)[0])
    test.assertTrue(np.all(np.diff(lengths) <= 2.0e-3), msg=f"length ratios are not monotonic: {lengths}")


def test_active_tet_free_body_com(test, device):
    """Verify uniform active contraction preserves free-body center of mass."""
    initial, final = _build_active_grid(
        device,
        fix_left=False,
        activation=1.0,
        active_stiffness=1.0e5,
    )
    com_drift = np.linalg.norm(np.mean(final, axis=0) - np.mean(initial, axis=0))
    test.assertLess(com_drift, 1.0e-4)


class TestSolverVBDActiveTet(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestSolverVBDActiveTet, "test_active_tet_finite_difference", test_active_tet_finite_difference, devices=devices
)
add_function_test(
    TestSolverVBDActiveTet, "test_active_tet_disabled_inputs", test_active_tet_disabled_inputs, devices=devices
)
add_function_test(
    TestSolverVBDActiveTet,
    "test_active_tet_setter_validation",
    test_active_tet_setter_validation,
    devices=devices,
)
add_function_test(
    TestSolverVBDActiveTet,
    "test_active_tet_zero_activation_regression",
    test_active_tet_zero_activation_regression,
    devices=devices,
)
add_function_test(
    TestSolverVBDActiveTet,
    "test_active_tet_directional_contraction",
    test_active_tet_directional_contraction,
    devices=devices,
)
add_function_test(
    TestSolverVBDActiveTet,
    "test_active_tet_activation_monotonicity",
    test_active_tet_activation_monotonicity,
    devices=devices,
)
add_function_test(
    TestSolverVBDActiveTet,
    "test_active_tet_stiffness_monotonicity",
    test_active_tet_stiffness_monotonicity,
    devices=devices,
)
add_function_test(
    TestSolverVBDActiveTet, "test_active_tet_free_body_com", test_active_tet_free_body_com, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
