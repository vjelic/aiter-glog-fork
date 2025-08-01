# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from ..jit.core import (
    compile_ops,
)


@compile_ops("module_gemm_common")
def get_padded_m(M: int, N: int, K: int, gl: int) -> int: ...
