# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_CORE_DIR = os.path.abspath(f"{this_dir}/../../")
if os.path.exists(os.path.join(AITER_CORE_DIR, "aiter_meta")):
    AITER_CORE_DIR = os.path.join(AITER_CORE_DIR, "aiter/jit/utils")  # pip install mode
else:
    AITER_CORE_DIR = os.path.abspath(
        f"{this_dir}/../../aiter/jit/utils"
    )  # develop mode
sys.path.insert(0, AITER_CORE_DIR)

from chip_info import get_gfx_list  # noqa: E402

GEN_DIR = ""  # in Cmake, have to generate files in same folder

AITER_API_FILENAME = "mha_fwd.cpp"

AITER_CPP_API = """#include "mha_fwd.h"
#include <iostream>

namespace aiter {{
mha_fwd_traits get_mha_fwd_traits(int head_size_q,
                                  int head_size_v,
                                  std::string dtype,
                                  bool is_group_mode,
                                  bool has_logits_soft_cap,
                                  mask_enum mask_type,
                                  bias_enum bias_type,
                                  bool has_lse,
                                  bool has_dropout,
                                  bool use_ext_asm,
                                  bool skip_min_seqlen_q = false)
{{
    return mha_fwd_traits(head_size_q,
                          head_size_v,
                          dtype,
                          is_group_mode,
                          has_logits_soft_cap,
                          mask_type,
                          bias_type,
                          has_lse,
                          has_dropout,
                          use_ext_asm,
                          skip_min_seqlen_q);
}}

mha_fwd_splitkv_traits get_mha_fwd_splitkv_traits(int head_size_q,
                                                  int head_size_v,
                                                  std::string dtype,
                                                  bool is_group_mode,
                                                  bool has_logits_soft_cap,
                                                  mask_enum mask_type,
                                                  bias_enum bias_type,
                                                  bool has_lse)
{{
    return mha_fwd_splitkv_traits(head_size_q,
                                  head_size_v,
                                  dtype,
                                  is_group_mode,
                                  has_logits_soft_cap,
                                  mask_type,
                                  bias_type,
                                  has_lse);
}}
{F_dispatch}

}} // namespace aiter

"""

FMHA_FWD_API = """
float mha_fwd(mha_fwd_args args,
              const ck_tile::stream_config& stream_config,
              std::string q_dtype_str,
              bool is_group_mode,
              mask_enum mask_type,
              bias_enum bias_type,
              bool has_lse,
              bool use_ext_asm)
{{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    bool has_dropout = args.p_drop > 0.f;
    auto traits = get_mha_fwd_traits(head_size_q,
                                     head_size_v,
                                     q_dtype_str,
                                     is_group_mode,
                                     args.logits_soft_cap > 0.f,
                                     mask_type,
                                     bias_type,
                                     has_lse,
                                     has_dropout,
                                     use_ext_asm,
                                     args.min_seqlen_q != 0);
    float t = -1;
    {F_inner_dispatch}
    return t;
}}"""

FMHA_FWD_SPLITKV_API = """
float mha_fwd_splitkv(mha_fwd_splitkv_args args,
                      const ck_tile::stream_config& stream_config,
                      std::string q_dtype_str,
                      bool is_group_mode,
                      mask_enum mask_type,
                      bias_enum bias_type,
                      bool has_lse)
{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    auto traits = get_mha_fwd_splitkv_traits(head_size_q,
                                             head_size_v,
                                             q_dtype_str,
                                             is_group_mode,
                                             args.logits_soft_cap > 0.f,
                                             mask_type,
                                             bias_type,
                                             has_lse);
    return fmha_fwd_splitkv(traits, args, stream_config);
}"""

FMHA_BATCH_PREFILL_API = """
float mha_batch_prefill(mha_batch_prefill_args args,
              const ck_tile::stream_config& stream_config,
              std::string q_dtype_str,
              bool is_group_mode,
              mask_enum mask_type,
              bias_enum bias_type,
              bool has_lse,
              bool use_ext_asm)
{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    bool has_dropout = args.p_drop > 0.f;
    auto traits = get_mha_fwd_traits(head_size_q,
                                     head_size_v,
                                     q_dtype_str,
                                     is_group_mode,
                                     args.logits_soft_cap > 0.f,
                                     mask_type,
                                     bias_type,
                                     has_lse,
                                     has_dropout,
                                     use_ext_asm);
    return fmha_batch_prefill(traits, args, stream_config);
}"""

V2_API = """t = fmha_fwd(traits, args, stream_config);"""

V3_MULTI_TARGET_API = """
    if (get_gpu_arch() == "gfx942") {
        t = gfx942::fmha_fwd_v3(traits, args, stream_config);
    } else if (get_gpu_arch() == "gfx950") {
        t = gfx950::fmha_fwd_v3(traits, args, stream_config);
    } else {
        std::cout << "No supported GPU arch found!" << std::endl;
        return -1;
    }
"""


def get_v3_api():
    gfx_list = get_gfx_list()
    if len(gfx_list) == 1:
        return f"t = {gfx_list[0]}::fmha_fwd_v3(traits, args, stream_config);"
    else:
        return V3_MULTI_TARGET_API


V3_API = get_v3_api()

COMBINED_API = (
    V3_API
    + """
    if (t == -1) { t = fmha_fwd(traits, args, stream_config); }
"""
)

API_MAP = {
    1: FMHA_FWD_API.format(F_inner_dispatch=V3_API),
    2: FMHA_FWD_API.format(F_inner_dispatch=V2_API),
    3: FMHA_FWD_API.format(F_inner_dispatch=V2_API) + FMHA_FWD_SPLITKV_API,
    4: FMHA_BATCH_PREFILL_API,
    5: FMHA_FWD_API.format(F_inner_dispatch=COMBINED_API)
    + FMHA_FWD_SPLITKV_API
    + FMHA_BATCH_PREFILL_API,
    6: FMHA_FWD_API.format(F_inner_dispatch=COMBINED_API) + FMHA_FWD_SPLITKV_API,
}


def write_blobs(output_dir: Optional[str], receipt) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    api = AITER_CPP_API.format(F_dispatch=API_MAP[receipt])
    (output_dir / AITER_API_FILENAME).write_text(api)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK fmha kernel",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory",
    )
    parser.add_argument(
        "-r",
        "--receipt",
        default=0,
        required=False,
        help="codegen receipt. 1: generate mha_fwd asm c++ api\n"
        + "  2: generate mha_fwd v2(ck) c++ api\n"
        + "  3: generate fmha varlen fwd c++ api\n"
        + "  4: generate mha_batch_prefill c++ api\n"
        + "  5: generate all fmha fwd c++ api, also can be use for PREBUILD",
    )

    args = parser.parse_args()

    write_blobs(args.output_dir, int(args.receipt))
