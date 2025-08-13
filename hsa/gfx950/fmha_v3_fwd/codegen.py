# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
import os
from pathlib import Path
from typing import Optional

this_dir = os.path.dirname(os.path.abspath(__file__))

GEN_DIR = ""  # in Cmake, have to generate files in same folder

FMHA_FWD_API_FILENAME = "asm_fmha_fwd_v3_gfx950.cpp"

FMHA_FWD_KERNEL_HEADER = """// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.\n
"""

FMHA_FWD_API = """#include <hip/hip_fp16.h>
#include "mha_fwd.h"

namespace aiter {{

// ######################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad | kStoreLSE | GPUArch
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16"; }};
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16"; }};
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16_causal"; }};
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16_causal"; }};
// template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_fp16"; }};
// template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_fp16"; }};
// template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_fp16_causal"; }};
// template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_fp16_causal"; }};

// #####################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad | kStoreLSE | GPUArch
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16.co"; }};
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16.co"; }};
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16_causal.co"; }};
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16_causal.co"; }};
// template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_buf = "fwd_hd128_fp16.co"; }};
// template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_buf = "fwd_hd128_fp16.co"; }};
// template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_buf = "fwd_hd128_fp16_causal.co"; }};
// template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr const char * fwd_v3_buf = "fwd_hd128_fp16_causal.co"; }};

// ####################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad | kStoreLSE | GPUArch
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; }};
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; }};
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; }};
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; }};
// template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; }};
// template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; }};
// template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> {{ static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; }};
// template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> {{ static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; }};

namespace gfx950{{
class fmha_fwd_v3_kernel
{{
    public:
    fmha_fwd_v3_kernel(const char *name, const char *hsaco)
    {{
        int length = strlen(name);
        std::string kernel_func_name = "_ZN5aiter" + std::to_string(length) + name + "E";
        std::string AITER_ASM_DIR = "{F_AITER_ASM_DIR}";
        HIP_CALL(hipModuleLoad(&module, (AITER_ASM_DIR + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_func_name.c_str()));
    }}

    void
    launch_kernel(fmha_fwd_v3_traits fmha_v3_traits, fmha_fwd_v3_args args, const ck_tile::stream_config& s) const
    {{
        size_t arg_size = sizeof(args);
        void* config[]  = {{HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END}};

        int tg_div = (fmha_v3_traits.mask != 0) ? 2 : 1;

        int bdx = 512;
        int gdx = ((fmha_v3_traits.s + fmha_v3_traits.ts_qo - 1) / fmha_v3_traits.ts_qo + tg_div - 1) / tg_div;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx,
                                       gdy,
                                       gdz,
                                       bdx,
                                       1,
                                       1,
                                       0,
                                       s.stream_id_,
                                       NULL,
                                       reinterpret_cast<void**>(&config)));
    }}

    private:
    hipModule_t module;
    hipFunction_t kernel_func;
}};

template <typename fmha_fwd_kernel_selector>
float fmha_fwd_v3_dispatcher(const ck_tile::stream_config& s, fmha_fwd_args a)
{{
    if(s.log_level_ > 0)
        std::cout << ", " << FmhaFwdV3Name<fmha_fwd_kernel_selector>::fwd_v3_name << std::flush;

    int tune_opt = 5;
    if (a.mask_type != 0 && ((a.nhead_q % 8 != 0) || (a.seqlen_q > 16384))) //if num_head is not 8N, or seqlen is bigger than 16K, downgrade to 2and3
    {{
        tune_opt -= 2;
    }}

    fmha_fwd_v3_args args;
    args.ptr_o   = a.o_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_lse = a.lse_ptr;

    args.scalar  = a.scale_s;
    args.seq_len = a.seqlen_q;
    args.Seqs    = a.stride_q * 2;
    args.Ts      = FmhaFwdV3Ts<fmha_fwd_kernel_selector>::ts_qo * a.stride_q * 2;
    args.Hs      = a.nhead_stride_q * 2;
    args.BAs     = a.batch_stride_q * 2;
    args.gqa      = a.nhead_q / a.nhead_k;
    args.Seqs_kv  = a.stride_k * 2;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.opt      = tune_opt;
    args.s_lse    = fmha_fwd_kernel_selector::kStoreLSE;

    auto traits = fmha_fwd_v3_traits{{a.batch,
                                     a.nhead_q,
                                     a.seqlen_q,
                                     a.hdim_q,
                                     a.mask_type,
                                     FmhaFwdV3Ts<fmha_fwd_kernel_selector>::ts_qo,
                                     FmhaFwdV3Ts<fmha_fwd_kernel_selector>::ts_kv}};

    static thread_local fmha_fwd_v3_kernel impl(FmhaFwdV3Name<fmha_fwd_kernel_selector>::fwd_v3_name, FmhaFwdV3Buf<fmha_fwd_kernel_selector>::fwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){{ impl.launch_kernel(traits, args, s_); }}
    );
}}

float fmha_fwd_v3(mha_fwd_traits t, fmha_fwd_args a, const ck_tile::stream_config& s){{
    float r = -1;
    if (t.use_ext_asm == true) {{
        if (t.data_type.compare("bf16") == 0) {{
            if ((t.bias_type == bias_enum::no_bias) && (t.has_dropout == false) &&
                        (a.seqlen_q == a.seqlen_k) && (a.seqlen_q >= 384) &&
                        (a.batch_stride_lse >= a.nhead_stride_lse) && (a.hdim_q == a.hdim_v) &&
                        (a.hdim_q == 128) && (a.stride_q == a.stride_o) && (a.nhead_stride_q == a.nhead_stride_o) &&
                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) &&
                        (a.batch_stride_q == a.batch_stride_o) && (a.batch_stride_q >= a.stride_q) && (a.batch_stride_o >= a.stride_o)) {{
                if (t.has_lse == true) {{
                    if (t.mask_type == mask_enum::no_mask) {{
                        using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 0, false, false, 1, GPUArch::gfx950>;
                        r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a);
                    }}
                    else if ((t.mask_type == mask_enum::mask_top_left) || (t.mask_type == mask_enum::mask_bottom_right)) {{
                        using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 1, false, false, 1, GPUArch::gfx950>;
                        r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a);
                    }}
                }}
            }}
        }}
    }}
    return r;
}}
}}
}} // namespace aiter
"""


def write_blobs(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    forward_kernel = FMHA_FWD_KERNEL_HEADER + FMHA_FWD_API.format(
        F_AITER_ASM_DIR=this_dir + "/",
    )

    (output_dir / FMHA_FWD_API_FILENAME).write_text(forward_kernel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen fmha fwd asm kernel API",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory",
    )

    args = parser.parse_args()
    write_blobs(args.output_dir)
