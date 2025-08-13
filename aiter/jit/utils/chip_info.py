# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import functools
import os
import re
import subprocess
from torch_guard import torch_compile_guard

from cpp_extension import executable_path

GFX_MAP = {
    0: "native",
    1: "gfx90a",
    2: "gfx908",
    3: "gfx940",
    4: "gfx941",
    5: "gfx942",
    6: "gfx945",
    7: "gfx1100",
    8: "gfx950",
}


@functools.lru_cache(maxsize=1)
def _detect_native() -> list[str]:
    try:
        rocminfo = executable_path("rocminfo")
        result = subprocess.run(
            [rocminfo],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "gfx" in line.lower():
                return [line.split(":", 1)[-1].strip()]
    except Exception as e:
        raise RuntimeError(f"Get GPU arch from rocminfo failed: {e}") from e
    raise RuntimeError("No gfx arch found in rocminfo output.")


@torch_compile_guard()
def get_gfx_custom_op() -> int:
    gfx_mapping = {v: k for k, v in GFX_MAP.items()}
    gfx = os.getenv("GPU_ARCHS", "native")
    if gfx == "native":
        try:
            rocminfo = executable_path("rocminfo")
            result = subprocess.run(
                [rocminfo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout
            for line in output.split("\n"):
                if "gfx" in line.lower():
                    try:
                        return gfx_mapping[line.split(":")[-1].strip()]
                    except KeyError:
                        raise KeyError(
                            f'Unknown GPU architecture: {line.split(":")[-1].strip()}. '
                            f"Supported architectures: {list(gfx_mapping.values())}"
                        )

        except Exception as e:
            raise RuntimeError(f"Get GPU arch from rocminfo failed {str(e)}")
    elif ";" in gfx:
        gfx = gfx.split(";")[-1]
    try:
        return gfx_mapping[gfx]
    except KeyError:
        raise KeyError(
            f"Unknown GPU architecture: {gfx}. "
            f"Supported architectures: {list(gfx_mapping.values())}"
        )


@functools.lru_cache(maxsize=1)
def get_gfx():
    gfx_num = get_gfx_custom_op()
    return GFX_MAP.get(gfx_num, "unknown")


@functools.lru_cache(maxsize=1)
def get_gfx_list() -> list[str]:

    gfx_env = os.getenv("GPU_ARCHS", "native")
    if gfx_env == "native":
        return _detect_native()

    return [g.strip() for g in gfx_env.split(";") if g.strip()]


@torch_compile_guard()
def get_cu_num_custom_op() -> int:
    cu_num = int(os.getenv("CU_NUM", 0))
    if cu_num == 0:
        try:
            rocminfo = executable_path("rocminfo")
            result = subprocess.run(
                [rocminfo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout
            devices = re.split(r"Agent\s*\d+", output)
            gpu_compute_units = []
            for device in devices:
                for line in device.split("\n"):
                    if "Device Type" in line and line.find("GPU") != -1:
                        match = re.search(r"Compute Unit\s*:\s*(\d+)", device)
                        if match:
                            gpu_compute_units.append(int(match.group(1)))
                        break
        except Exception as e:
            raise RuntimeError(f"Get GPU Compute Unit from rocminfo failed {str(e)}")
        assert len(set(gpu_compute_units)) == 1
        cu_num = gpu_compute_units[0]
    return cu_num


@functools.lru_cache(maxsize=1)
def get_cu_num():
    cu_num = get_cu_num_custom_op()
    return cu_num


def get_device_name():
    gfx = get_gfx()

    if gfx == "gfx942":
        cu = get_cu_num()
        if cu == 304:
            return "MI300"
        elif cu == 80 or cu == 64:
            return "MI308"
    elif gfx == "gfx950":
        return "MI350"
    else:
        raise RuntimeError("Unsupported gfx")
