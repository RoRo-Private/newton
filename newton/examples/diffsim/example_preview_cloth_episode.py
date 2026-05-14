# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Alias entrypoint for previewing exported cloth preview episodes."""

from newton.examples.diffsim.preview_cloth_episode import Example


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = __import__("newton.examples", fromlist=["init"]).init(parser)

    example = Example(viewer, args)
    __import__("newton.examples", fromlist=["run"]).run(example, args)
