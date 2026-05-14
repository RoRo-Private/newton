# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os

import newton.examples
from newton.examples.diffsim.example_diffsim_cloth_phystwin_interact import CLOTH_DATA_ROOT, Example as BaseExample


DEFAULT_DATASET_PATH = os.path.join(CLOTH_DATA_ROOT, "cloth_2_2", "newton_cloth", "cloth_export.npz")
DEFAULT_META_PATH = os.path.join(CLOTH_DATA_ROOT, "cloth_2_2", "newton_cloth", "meta.json")


class Example(BaseExample):
    @staticmethod
    def create_parser():
        parser = BaseExample.create_parser()
        parser.set_defaults(dataset=DEFAULT_DATASET_PATH, meta=DEFAULT_META_PATH)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
