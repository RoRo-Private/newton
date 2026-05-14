# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Helper-centric alias for the cloth helper + spring rollout example."""

import newton.examples
from newton.examples.diffsim.example_diffsim_cloth_phystwin_interact_spring import Example as BaseExample


class Example(BaseExample):
    """Alias entrypoint with a helper-centric name."""

    @staticmethod
    def create_parser():
        parser = BaseExample.create_parser()
        parser.set_defaults(
            collision_minimal=True,
            material_preset="paper_like",
            proxy_voxel_size=0.015,
            phase_machine_proxy_voxel_size=0.03,
            with_ground=False,
            spring_from_faces="none",
            cloth_z_offset=0.08,
            settle_frames=0,
            gravity=0.0,
            sim_substeps=48,
            iterations=16,
            anchor_mode="reference_template",
            policy_mode="template_randomized",
            control_mode="template_preview",
            preview_displacement_scale=1.0,
            sequence_loop=False,
            sequence_align_endpoints=True,
            disable_grounded_pull=False,
            force_ramp_time=0.75,
            density=0.05,
            tri_ke=6.0e2,
            tri_ka=6.0e2,
            tri_kd=3.0,
            edge_ke=8.0,
            edge_kd=6.0,
            contact_ke=2.0e1,
            contact_kd=8.0,
            contact_mu=0.05,
            left_arm_grip_mode="cuff_patch",
            hidden_grip_count=6,
            left_arm_max_control_count=32,
            min_control_count_after_settle=4,
            phase_machine_min_patch_count=16,
            anchor_pull_ke=0.3,
            anchor_max_pull_force=0.0015,
            pull_ke=1.5,
            pull_kd=4.0,
            max_pull_force=0.01,
            global_drag_kd=2.0,
            max_anchor_speed=0.008,
            max_control_speed=0.004,
            unified_patch_scale=1.2,
            unified_weight_floor=0.85,
            unified_amp_x_scale=0.35,
            unified_amp_y_scale=0.35,
            unified_amp_z_scale=0.5,
            unified_period_scale=1.4,
            unified_patch_scale_factor=0.75,
            template_amp_scale_min=0.35,
            template_amp_scale_max=0.45,
            template_z_scale_min=0.35,
            template_z_scale_max=0.45,
            template_period_scale_min=1.0,
            template_period_scale_max=1.35,
            template_patch_scale_min=0.45,
            template_patch_scale_max=0.65,
            template_phase_jitter=0.02,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
