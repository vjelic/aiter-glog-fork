// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#define ACTIVATION_PYBIND                                                               \
    m.def("silu_and_mul", &aiter::silu_and_mul, "Activation function used in SwiGLU."); \
    m.def("scaled_silu_and_mul",                                                        \
          &aiter::scaled_silu_and_mul,                                                  \
          "Activation function used in scaled SwiGLU.");                                \
    m.def("gelu_and_mul", &aiter::gelu_and_mul, "Activation function used in GELU.");   \
    m.def("gelu_tanh_and_mul", &aiter::gelu_tanh_and_mul, "Activation function used in GELU tanh.");

#define AITER_OPERATOR_PYBIND                                                   \
    m.def("add", &aiter_add, "apply for add with transpose and broadcast.");    \
    m.def("mul", &aiter_mul, "apply for mul with transpose and broadcast.");    \
    m.def("sub", &aiter_sub, "apply for sub with transpose and broadcast.");    \
    m.def("div", &aiter_div, "apply for div with transpose and broadcast.");    \
    m.def("add_", &aiter_add_, "apply for add_ with transpose and broadcast."); \
    m.def("mul_", &aiter_mul_, "apply for mul_ with transpose and broadcast."); \
    m.def("sub_", &aiter_sub_, "apply for sub_ with transpose and broadcast."); \
    m.def("div_", &aiter_div_, "apply for div_ with transpose and broadcast.");
#define AITER_UNARY_PYBIND                                  \
    m.def("sigmoid", &aiter_sigmoid, "apply for sigmoid."); \
    m.def("tanh", &aiter_tanh, "apply for tanh.");

#define ATTENTION_ASM_MLA_PYBIND        \
    m.def("mla_decode_stage1_asm_fwd",  \
          &mla_decode_stage1_asm_fwd,   \
          "mla_decode_stage1_asm_fwd",  \
          py::arg("Q"),                 \
          py::arg("KV"),                \
          py::arg("qo_indptr"),         \
          py::arg("kv_indptr"),         \
          py::arg("kv_page_indices"),   \
          py::arg("kv_last_page_lens"), \
          py::arg("max_seqlen_q"),      \
          py::arg("softmax_scale"),     \
          py::arg("splitData"),         \
          py::arg("splitLse"));         \
    m.def("mla_prefill_asm_fwd",        \
          &mla_prefill_asm_fwd,         \
          "mla_prefill_asm_fwd",        \
          py::arg("Q"),                 \
          py::arg("KV"),                \
          py::arg("qo_indptr"),         \
          py::arg("kv_indptr"),         \
          py::arg("kv_page_indices"),   \
          py::arg("kv_last_page_lens"), \
          py::arg("max_seqlen_q"),      \
          py::arg("softmax_scale"),     \
          py::arg("splitData"),         \
          py::arg("splitLse"));

#define ATTENTION_ASM_PYBIND                        \
    m.def("pa_fwd_asm",                             \
          &pa_fwd,                                  \
          "pa_fwd",                                 \
          py::arg("Q"),                             \
          py::arg("K"),                             \
          py::arg("V"),                             \
          py::arg("block_tables"),                  \
          py::arg("context_lens"),                  \
          py::arg("max_num_blocks"),                \
          py::arg("max_qlen")       = 1,            \
          py::arg("K_QScale")       = std::nullopt, \
          py::arg("V_QScale")       = std::nullopt, \
          py::arg("out_")           = std::nullopt, \
          py::arg("qo_indptr")      = std::nullopt, \
          py::arg("high_precision") = 1,            \
          py::arg("kernelName")     = "");

#define ATTENTION_CK_PYBIND            \
    m.def("pa_fwd_naive",              \
          &pa_fwd_naive,               \
          "pa_fwd_naive",              \
          py::arg("Q"),                \
          py::arg("K"),                \
          py::arg("V"),                \
          py::arg("block_tables"),     \
          py::arg("context_lens"),     \
          py::arg("k_dequant_scales"), \
          py::arg("v_dequant_scales"), \
          py::arg("max_seq_len"),      \
          py::arg("num_kv_heads"),     \
          py::arg("scale_s"),          \
          py::arg("scale_k"),          \
          py::arg("scale_v"),          \
          py::arg("block_size"),       \
          py::arg("quant_algo"),       \
          py::arg("out_") = std::nullopt);

#define ATTENTION_PYBIND                                          \
    m.def("paged_attention_rocm",                                 \
          &paged_attention,                                       \
          "paged_attention_rocm(Tensor! out, Tensor exp_sums,"    \
          "                Tensor max_logits, Tensor tmp_out,"    \
          "                Tensor query, Tensor key_cache,"       \
          "                Tensor value_cache, int num_kv_heads," \
          "                float scale, Tensor block_tables,"     \
          "                Tensor context_lens, int block_size,"  \
          "                int max_context_len,"                  \
          "                Tensor? alibi_slopes,"                 \
          "                str kv_cache_dtype,"                   \
          "                float k_scale, float v_scale) -> ()");

#define ATTENTION_RAGGED_PYBIND                                   \
    m.def("paged_attention_ragged",                               \
          &paged_attention_ragged,                                \
          "paged_attention_ragged(Tensor! out, Tensor exp_sums,"  \
          "                Tensor max_logits, Tensor tmp_out,"    \
          "                Tensor query, Tensor key_cache,"       \
          "                Tensor value_cache, int num_kv_heads," \
          "                float scale, Tensor block_tables,"     \
          "                Tensor context_lens, int block_size,"  \
          "                int max_context_len,"                  \
          "                Tensor? alibi_slopes,"                 \
          "                str kv_cache_dtype,"                   \
          "                float k_scale, float v_scale) -> ()");

#define ATTENTION_V1_PYBIND                                       \
    m.def("paged_attention_v1",                                   \
          &paged_attention_v1,                                    \
          "paged_attention_v1(Tensor! out, Tensor exp_sums,"      \
          "                Tensor max_logits, Tensor tmp_out,"    \
          "                Tensor query, Tensor key_cache,"       \
          "                Tensor value_cache, int num_kv_heads," \
          "                float scale, Tensor block_tables,"     \
          "                Tensor context_lens, int block_size,"  \
          "                int max_context_len,"                  \
          "                Tensor? alibi_slopes,"                 \
          "                str kv_cache_dtype,"                   \
          "                float k_scale, float v_scale) -> ()");

#define BATCHED_GEMM_A8W8_PYBIND            \
    m.def("batched_gemm_a8w8",              \
          &batched_gemm_a8w8,               \
          "batched_gemm_a8w8",              \
          py::arg("XQ"),                    \
          py::arg("WQ"),                    \
          py::arg("x_scale"),               \
          py::arg("w_scale"),               \
          py::arg("Out"),                   \
          py::arg("bias")   = std::nullopt, \
          py::arg("splitK") = 0);

#define BATCHED_GEMM_A8W8_TUNE_PYBIND \
    m.def("batched_gemm_a8w8_tune",   \
          &batched_gemm_a8w8_tune,    \
          "batched_gemm_a8w8_tune",   \
          py::arg("XQ"),              \
          py::arg("WQ"),              \
          py::arg("x_scale"),         \
          py::arg("w_scale"),         \
          py::arg("Out"),             \
          py::arg("kernelId") = 0,    \
          py::arg("splitK")   = 0);

#define CACHE_PYBIND                                                                \
    m.def("swap_blocks",                                                            \
          &aiter::swap_blocks,                                                      \
          "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");      \
    m.def("copy_blocks",                                                            \
          &aiter::copy_blocks,                                                      \
          "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "        \
          "Tensor block_mapping) -> ()");                                           \
                                                                                    \
    m.def("reshape_and_cache",                                                      \
          &aiter::reshape_and_cache,                                                \
          "reshape_and_cache",                                                      \
          py::arg("key"),                                                           \
          py::arg("value"),                                                         \
          py::arg("key_cache"),                                                     \
          py::arg("value_cache"),                                                   \
          py::arg("slot_mapping"),                                                  \
          py::arg("kv_cache_dtype"),                                                \
          py::arg("k_scale")    = std::nullopt,                                     \
          py::arg("v_scale")    = std::nullopt,                                     \
          py::arg("asm_layout") = false);                                           \
    m.def("reshape_and_cache_flash",                                                \
          &aiter::reshape_and_cache_flash,                                          \
          "reshape_and_cache_flash(Tensor key, Tensor value,"                       \
          "                        Tensor! key_cache,"                              \
          "                        Tensor! value_cache,"                            \
          "                        Tensor slot_mapping,"                            \
          "                        str kv_cache_dtype,"                             \
          "                        float k_scale, float v_scale) -> ()");           \
    m.def("reshape_and_cache_with_pertoken_quant",                                  \
          &aiter::reshape_and_cache_with_pertoken_quant,                            \
          "reshape_and_cache_with_pertoken_quant(Tensor key, Tensor value,"         \
          "                        Tensor! key_cache,"                              \
          "                        Tensor! value_cache,"                            \
          "                        Tensor! k_dequant_scales,"                       \
          "                        Tensor! v_dequant_scales,"                       \
          "                        Tensor slot_mapping) -> ()");                    \
    m.def("reshape_and_cache_with_block_quant",                                     \
          &aiter::reshape_and_cache_with_block_quant,                               \
          "reshape_and_cache_with_block_quant(Tensor key, Tensor value,"            \
          "                        Tensor! key_cache,"                              \
          "                        Tensor! value_cache,"                            \
          "                        Tensor! k_dequant_scales,"                       \
          "                        Tensor! v_dequant_scales,"                       \
          "                        Tensor slot_mapping,"                            \
          "                        const bool asm_layout) -> ()");                  \
    m.def("reshape_and_cache_with_block_quant_for_asm_pa",                          \
          &aiter::reshape_and_cache_with_block_quant_for_asm_pa,                    \
          "reshape_and_cache_with_block_quant_for_asm_pa(Tensor key, Tensor value," \
          "                        Tensor! key_cache,"                              \
          "                        Tensor! value_cache,"                            \
          "                        Tensor! k_dequant_scales,"                       \
          "                        Tensor! v_dequant_scales,"                       \
          "                        Tensor slot_mapping,"                            \
          "                        const bool asm_layout,"                          \
          "                        const int ori_block_size) -> ()",                \
          py::arg("key"),                                                           \
          py::arg("value"),                                                         \
          py::arg("key_cache"),                                                     \
          py::arg("value_cache"),                                                   \
          py::arg("k_dequant_scales"),                                              \
          py::arg("v_dequant_scales"),                                              \
          py::arg("slot_mapping"),                                                  \
          py::arg("asm_layout"),                                                    \
          py::arg("ori_block_size") = 128);

#define CUSTOM_ALL_REDUCE_PYBIND                                                               \
    m.def("init_custom_ar",                                                                    \
          &aiter::init_custom_ar,                                                              \
          "init_custom_ar(Tensor meta, Tensor rank_data, "                                     \
          "str[] handles, int[] offsets, int rank, "                                           \
          "bool full_nvlink) -> int",                                                          \
          py::arg("meta"),                                                                     \
          py::arg("rank_data"),                                                                \
          py::arg("handles"),                                                                  \
          py::arg("offsets"),                                                                  \
          py::arg("rank"),                                                                     \
          py::arg("full_nvlink"));                                                             \
    m.def("all_reduce_reg",                                                                    \
          &aiter::all_reduce_reg,                                                              \
          "all_reduce_reg(int fa, Tensor inp, Tensor! out, bool open_fp8_quant) -> ()",        \
          py::arg("_fa"),                                                                      \
          py::arg("inp"),                                                                      \
          py::arg("out"),                                                                      \
          py::arg("open_fp8_quant"));                                                          \
    m.def("all_reduce_unreg",                                                                  \
          &aiter::all_reduce_unreg,                                                            \
          "all_reduce_unreg(int fa, Tensor inp, Tensor reg_buffer, Tensor! out) -> ()",        \
          py::arg("_fa"),                                                                      \
          py::arg("inp"),                                                                      \
          py::arg("reg_buffer"),                                                               \
          py::arg("out"));                                                                     \
    m.def("all_reduce_asm_", &all_reduce_asm, "");                                             \
    m.def("all_reduce_rmsnorm_", &all_reduce_rmsnorm, "all_reduce_rmsnorm");                   \
    m.def("all_reduce_rmsnorm_quant_", &all_reduce_rmsnorm_quant, "all_reduce_rmsnorm_quant"); \
    m.def("dispose", &aiter::dispose, py::arg("_fa"));                                         \
    m.def("meta_size", &aiter::meta_size);                                                     \
    m.def("register_buffer",                                                                   \
          &aiter::register_buffer,                                                             \
          "register_buffer(int fa, Tensor t, str[] handles, int[] offsets) -> ()",             \
          py::arg("_fa"),                                                                      \
          py::arg("t"),                                                                        \
          py::arg("handles"),                                                                  \
          py::arg("offsets"));                                                                 \
    m.def("get_graph_buffer_ipc_meta", &aiter::get_graph_buffer_ipc_meta, py::arg("_fa"));     \
    m.def("register_graph_buffers",                                                            \
          &aiter::register_graph_buffers,                                                      \
          py::arg("_fa"),                                                                      \
          py::arg("handles"),                                                                  \
          py::arg("offsets"));                                                                 \
    m.def("allocate_meta_buffer", &aiter::allocate_meta_buffer, py::arg("size"));              \
    m.def("get_meta_buffer_ipc_handle", &aiter::get_meta_buffer_ipc_handle, py::arg("inp"));

#define CUSTOM_PYBIND                                                                           \
    m.def("wvSpltK",                                                                            \
          &aiter::wvSpltK,                                                                      \
          "wvSpltK(Tensor in_a, Tensor in_b, Tensor! out_c, int N_in,"                          \
          "        int CuCount) -> ()");                                                        \
    m.def("wv_splitk_small_fp16_bf16",                                                          \
          &aiter::wv_splitk_small_fp16_bf16_wrapper,                                            \
          "wv_splitk_small_fp16_bf16(Tensor in_a, Tensor in_b, Tensor! out_c, int N_in,"        \
          "        int CuCount) -> ()");                                                        \
    m.def("LLMM1",                                                                              \
          &aiter::LLMM1,                                                                        \
          "LLMM1(Tensor in_a, Tensor in_b, Tensor! out_c, int rows_per_block) -> "              \
          "()");                                                                                \
    m.def("wvSplitKQ",                                                                          \
          &aiter::wvSplitKQ,                                                                    \
          "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor! out_c, Tensor scale_a, Tensor scale_b, " \
          "int CuCount) -> ()");

#define GEMM_A8W8_ASM_PYBIND                                            \
    m.def("gemm_a8w8_asm",                                              \
          &gemm_a8w8_asm,                                               \
          "Asm gemm a8w8 ,  weight should be shuffle to layout(32,16)", \
          py::arg("XQ"),                                                \
          py::arg("WQ"),                                                \
          py::arg("x_scale"),                                           \
          py::arg("w_scale"),                                           \
          py::arg("Out"),                                               \
          py::arg("bias"),                                              \
          py::arg("sub_m")  = 128,                                      \
          py::arg("sub_n")  = 128,                                      \
          py::arg("pad_a")  = 0,                                        \
          py::arg("pad_b")  = 0,                                        \
          py::arg("pad_c")  = 0,                                        \
          py::arg("splitK") = 0);

#define GEMM_A4W4_ASM_PYBIND                      \
    m.def("gemm_a4w4_asm",                        \
          &gemm_a4w4_asm,                         \
          "Asm gemm a4w4",                        \
          py::arg("A"),                           \
          py::arg("B"),                           \
          py::arg("A_scale"),                     \
          py::arg("B_scale"),                     \
          py::arg("out"),                         \
          py::arg("kernelName"),                  \
          py::arg("bias")         = std::nullopt, \
          py::arg("alpha")        = 1.0,          \
          py::arg("beta")         = 0.0,          \
          py::arg("bpreshuffle")  = true,         \
          py::arg("log2_k_split") = std::nullopt);

#define GEMM_A4W4_BLOCKSCALE_PYBIND \
    m.def("gemm_a4w4_blockscale",   \
          &gemm_a4w4_blockscale,    \
          "fp4 blockscale gemm",    \
          py::arg("XQ"),            \
          py::arg("WQ"),            \
          py::arg("x_scale"),       \
          py::arg("w_scale"),       \
          py::arg("Out"),           \
          py::arg("splitK") = 0);

#define GEMM_A8W8_BLOCKSCALE_PYBIND \
    m.def("gemm_a8w8_blockscale",   \
          &gemm_a8w8_blockscale,    \
          "fp8 blockscale gemm",    \
          py::arg("XQ"),            \
          py::arg("WQ"),            \
          py::arg("x_scale"),       \
          py::arg("w_scale"),       \
          py::arg("Out"));

#define GEMM_A8W8_BLOCKSCALE_TUNE_PYBIND \
    m.def("gemm_a8w8_blockscale_tune",   \
          &gemm_a8w8_blockscale_tune,    \
          "gemm_a8w8_blockscale_tune",   \
          py::arg("XQ"),                 \
          py::arg("WQ"),                 \
          py::arg("x_scale"),            \
          py::arg("w_scale"),            \
          py::arg("Out"),                \
          py::arg("kernelId") = 0,       \
          py::arg("splitK")   = 0);

#define GEMM_A4W4_BLOCKSCALE_TUNE_PYBIND \
    m.def("gemm_a4w4_blockscale_tune",   \
          &gemm_a4w4_blockscale_tune,    \
          "gemm_a4w4_blockscale_tune",   \
          py::arg("XQ"),                 \
          py::arg("WQ"),                 \
          py::arg("x_scale"),            \
          py::arg("w_scale"),            \
          py::arg("Out"),                \
          py::arg("kernelId") = 0,       \
          py::arg("splitK")   = 0);

#define GEMM_A8W8_PYBIND                    \
    m.def("gemm_a8w8",                      \
          &gemm_a8w8,                       \
          "gemm_a8w8",                      \
          py::arg("XQ"),                    \
          py::arg("WQ"),                    \
          py::arg("x_scale"),               \
          py::arg("w_scale"),               \
          py::arg("Out"),                   \
          py::arg("bias")   = std::nullopt, \
          py::arg("splitK") = 0);

#define GEMM_A8W8_TUNE_PYBIND      \
    m.def("gemm_a8w8_tune",        \
          &gemm_a8w8_tune,         \
          "gemm_a8w8_tune",        \
          py::arg("XQ"),           \
          py::arg("WQ"),           \
          py::arg("x_scale"),      \
          py::arg("w_scale"),      \
          py::arg("Out"),          \
          py::arg("kernelId") = 0, \
          py::arg("splitK")   = 0);
#define GEMM_A8W8_BPRESHUFFLE_PYBIND \
    m.def("gemm_a8w8_bpreshuffle",   \
          &gemm_a8w8_bpreshuffle,    \
          "gemm_a8w8_bpreshuffle",   \
          py::arg("XQ"),             \
          py::arg("WQ"),             \
          py::arg("x_scale"),        \
          py::arg("w_scale"),        \
          py::arg("Out"));

#define GEMM_A8W8_BPRESHUFFLE_TUNE_PYBIND \
    m.def("gemm_a8w8_bpreshuffle_tune",   \
          &gemm_a8w8_bpreshuffle_tune,    \
          "gemm_a8w8_bpreshuffle_tune",   \
          py::arg("XQ"),                  \
          py::arg("WQ"),                  \
          py::arg("x_scale"),             \
          py::arg("w_scale"),             \
          py::arg("Out"),                 \
          py::arg("kernelId") = 0,        \
          py::arg("splitK")   = 0);
#define MHA_BWD_ASM_PYBIND                        \
    m.def("fmha_v3_bwd",                          \
          &aiter::torch_itfs::fmha_v3_bwd,        \
          py::arg("dout"),                        \
          py::arg("q"),                           \
          py::arg("k"),                           \
          py::arg("v"),                           \
          py::arg("out"),                         \
          py::arg("softmax_lse"),                 \
          py::arg("dropout_p"),                   \
          py::arg("softmax_scale"),               \
          py::arg("is_causal"),                   \
          py::arg("window_size_left"),            \
          py::arg("window_size_right"),           \
          py::arg("deterministic"),               \
          py::arg("is_v3_atomic_fp32"),           \
          py::arg("how_v3_bf16_cvt"),             \
          py::arg("dq")           = std::nullopt, \
          py::arg("dk")           = std::nullopt, \
          py::arg("dv")           = std::nullopt, \
          py::arg("alibi_slopes") = std::nullopt, \
          py::arg("rng_state")    = std::nullopt, \
          py::arg("gen")          = std::nullopt);

#define MHA_VARLEN_BWD_ASM_PYBIND                 \
    m.def("fmha_v3_varlen_bwd",                   \
          &aiter::torch_itfs::fmha_v3_varlen_bwd, \
          py::arg("dout"),                        \
          py::arg("q"),                           \
          py::arg("k"),                           \
          py::arg("v"),                           \
          py::arg("out"),                         \
          py::arg("softmax_lse"),                 \
          py::arg("cu_seqlens_q"),                \
          py::arg("cu_seqlens_k"),                \
          py::arg("max_seqlen_q"),                \
          py::arg("max_seqlen_k"),                \
          py::arg("dropout_p"),                   \
          py::arg("softmax_scale"),               \
          py::arg("zero_tensors"),                \
          py::arg("is_causal"),                   \
          py::arg("window_size_left"),            \
          py::arg("window_size_right"),           \
          py::arg("deterministic"),               \
          py::arg("is_v3_atomic_fp32"),           \
          py::arg("how_v3_bf16_cvt"),             \
          py::arg("dq")           = std::nullopt, \
          py::arg("dk")           = std::nullopt, \
          py::arg("dv")           = std::nullopt, \
          py::arg("alibi_slopes") = std::nullopt, \
          py::arg("rng_state")    = std::nullopt, \
          py::arg("gen")          = std::nullopt);

#define MHA_BWD_PYBIND                            \
    m.def("mha_bwd",                              \
          &aiter::torch_itfs::mha_bwd,            \
          py::arg("dout"),                        \
          py::arg("q"),                           \
          py::arg("k"),                           \
          py::arg("v"),                           \
          py::arg("out"),                         \
          py::arg("softmax_lse"),                 \
          py::arg("dropout_p"),                   \
          py::arg("softmax_scale"),               \
          py::arg("is_causal"),                   \
          py::arg("window_size_left"),            \
          py::arg("window_size_right"),           \
          py::arg("deterministic"),               \
          py::arg("dq")           = std::nullopt, \
          py::arg("dk")           = std::nullopt, \
          py::arg("dv")           = std::nullopt, \
          py::arg("dbias")        = std::nullopt, \
          py::arg("bias")         = std::nullopt, \
          py::arg("alibi_slopes") = std::nullopt, \
          py::arg("rng_state")    = std::nullopt, \
          py::arg("gen")          = std::nullopt);

#define MHA_FWD_ASM_PYBIND                        \
    m.def("fmha_v3_fwd",                          \
          &aiter::torch_itfs::fmha_v3_fwd,        \
          py::arg("q"),                           \
          py::arg("k"),                           \
          py::arg("v"),                           \
          py::arg("dropout_p"),                   \
          py::arg("softmax_scale"),               \
          py::arg("is_causal"),                   \
          py::arg("window_size_left"),            \
          py::arg("window_size_right"),           \
          py::arg("return_softmax_lse"),          \
          py::arg("return_dropout_randval"),      \
          py::arg("out")          = std::nullopt, \
          py::arg("bias")         = std::nullopt, \
          py::arg("alibi_slopes") = std::nullopt, \
          py::arg("gen")          = std::nullopt);

#define MHA_FWD_PYBIND                            \
    m.def("mha_fwd",                              \
          &aiter::torch_itfs::mha_fwd,            \
          py::arg("q"),                           \
          py::arg("k"),                           \
          py::arg("v"),                           \
          py::arg("dropout_p"),                   \
          py::arg("softmax_scale"),               \
          py::arg("is_causal"),                   \
          py::arg("window_size_left"),            \
          py::arg("window_size_right"),           \
          py::arg("return_softmax_lse"),          \
          py::arg("return_dropout_randval"),      \
          py::arg("out")          = std::nullopt, \
          py::arg("bias")         = std::nullopt, \
          py::arg("alibi_slopes") = std::nullopt, \
          py::arg("gen")          = std::nullopt);

#define MHA_VARLEN_BWD_PYBIND                     \
    m.def("mha_varlen_bwd",                       \
          &aiter::torch_itfs::mha_varlen_bwd,     \
          py::arg("dout"),                        \
          py::arg("q"),                           \
          py::arg("k"),                           \
          py::arg("v"),                           \
          py::arg("out"),                         \
          py::arg("softmax_lse"),                 \
          py::arg("cu_seqlens_q"),                \
          py::arg("cu_seqlens_k"),                \
          py::arg("max_seqlen_q"),                \
          py::arg("max_seqlen_k"),                \
          py::arg("dropout_p"),                   \
          py::arg("softmax_scale"),               \
          py::arg("zero_tensors"),                \
          py::arg("is_causal"),                   \
          py::arg("window_size_left"),            \
          py::arg("window_size_right"),           \
          py::arg("deterministic"),               \
          py::arg("dq")           = std::nullopt, \
          py::arg("dk")           = std::nullopt, \
          py::arg("dv")           = std::nullopt, \
          py::arg("alibi_slopes") = std::nullopt, \
          py::arg("rng_state")    = std::nullopt, \
          py::arg("gen")          = std::nullopt);

#define MOE_CK_2STAGES_PYBIND                        \
    m.def("ck_moe_stage1",                           \
          &ck_moe_stage1,                            \
          py::arg("hidden_states"),                  \
          py::arg("w1"),                             \
          py::arg("w2"),                             \
          py::arg("sorted_token_ids"),               \
          py::arg("sorted_expert_ids"),              \
          py::arg("num_valid_ids"),                  \
          py::arg("out"),                            \
          py::arg("topk"),                           \
          py::arg("kernelName"),                     \
          py::arg("w1_scale")       = std::nullopt,  \
          py::arg("a1_scale")       = std::nullopt,  \
          py::arg("block_m")        = 32,            \
          py::arg("sorted_weights") = std::nullopt); \
                                                     \
    m.def("ck_moe_stage2",                           \
          &ck_moe_stage2,                            \
          py::arg("inter_states"),                   \
          py::arg("w1"),                             \
          py::arg("w2"),                             \
          py::arg("sorted_token_ids"),               \
          py::arg("sorted_expert_ids"),              \
          py::arg("num_valid_ids"),                  \
          py::arg("out"),                            \
          py::arg("topk"),                           \
          py::arg("kernelName"),                     \
          py::arg("w2_scale")       = std::nullopt,  \
          py::arg("a2_scale")       = std::nullopt,  \
          py::arg("block_m")        = 32,            \
          py::arg("sorted_weights") = std::nullopt);

#define MHA_VARLEN_FWD_PYBIND                     \
    m.def("mha_varlen_fwd",                       \
          &aiter::torch_itfs::mha_varlen_fwd,     \
          py::arg("q"),                           \
          py::arg("k"),                           \
          py::arg("v"),                           \
          py::arg("cu_seqlens_q"),                \
          py::arg("cu_seqlens_k"),                \
          py::arg("max_seqlen_q"),                \
          py::arg("max_seqlen_k"),                \
          py::arg("min_seqlen_q"),                \
          py::arg("dropout_p"),                   \
          py::arg("softmax_scale"),               \
          py::arg("logits_soft_cap"),             \
          py::arg("zero_tensors"),                \
          py::arg("is_causal"),                   \
          py::arg("window_size_left"),            \
          py::arg("window_size_right"),           \
          py::arg("return_softmax_lse"),          \
          py::arg("return_dropout_randval"),      \
          py::arg("out")          = std::nullopt, \
          py::arg("block_table")  = std::nullopt, \
          py::arg("bias")         = std::nullopt, \
          py::arg("alibi_slopes") = std::nullopt, \
          py::arg("gen")          = std::nullopt);

#define MHA_BATCH_PREFILL_PYBIND                  \
    m.def("mha_batch_prefill",                    \
          &aiter::torch_itfs::mha_batch_prefill,  \
          py::arg("q"),                           \
          py::arg("k"),                           \
          py::arg("v"),                           \
          py::arg("cu_seqlens_q"),                \
          py::arg("kv_indptr"),                   \
          py::arg("kv_page_indices"),             \
          py::arg("max_seqlen_q"),                \
          py::arg("max_seqlen_k"),                \
          py::arg("dropout_p"),                   \
          py::arg("softmax_scale"),               \
          py::arg("logits_soft_cap"),             \
          py::arg("zero_tensors"),                \
          py::arg("is_causal"),                   \
          py::arg("window_size_left"),            \
          py::arg("window_size_right"),           \
          py::arg("return_softmax_lse"),          \
          py::arg("return_dropout_randval"),      \
          py::arg("out")          = std::nullopt, \
          py::arg("bias")         = std::nullopt, \
          py::arg("alibi_slopes") = std::nullopt, \
          py::arg("gen")          = std::nullopt);

#define MOE_OP_PYBIND                                                                         \
    m.def("topk_softmax", &aiter::topk_softmax, "Apply topk softmax to the gating outputs."); \
    m.def("grouped_topk",                                                                     \
          &grouped_topk,                                                                      \
          py::arg("gating_output"),                                                           \
          py::arg("topk_weights"),                                                            \
          py::arg("topk_ids"),                                                                \
          py::arg("num_expert_group"),                                                        \
          py::arg("topk_grp"),                                                                \
          py::arg("need_renorm"),                                                             \
          py::arg("scoring_func")          = "softmax",                                       \
          py::arg("routed_scaling_factor") = 1.0f,                                            \
          "Apply grouped topk softmax/sigmodd to the gating outputs.");                       \
    m.def("biased_grouped_topk",                                                              \
          &biased_grouped_topk,                                                               \
          py::arg("gating_output"),                                                           \
          py::arg("correction_bias"),                                                         \
          py::arg("topk_weights"),                                                            \
          py::arg("topk_ids"),                                                                \
          py::arg("num_expert_group"),                                                        \
          py::arg("topk_grp"),                                                                \
          py::arg("need_renorm"),                                                             \
          py::arg("routed_scaling_factor") = 1.0f,                                            \
          "Apply biased grouped topk softmax to the gating outputs.");                        \
    m.def("moe_fused_gate",                                                                   \
          &moe_fused_gate,                                                                    \
          py::arg("input"),                                                                   \
          py::arg("bias"),                                                                    \
          py::arg("topk_weights"),                                                            \
          py::arg("topk_ids"),                                                                \
          py::arg("num_expert_group"),                                                        \
          py::arg("topk_group"),                                                              \
          py::arg("topk"),                                                                    \
          py::arg("n_share_experts_fusion"),                                                  \
          py::arg("routed_scaling_factor") = 1.0,                                             \
          "Apply biased grouped topk softmax to the gating outputs.");                        \
    m.def("moe_align_block_size",                                                             \
          &aiter::moe_align_block_size,                                                       \
          "Aligning the number of tokens to be processed by each expert such "                \
          "that it is divisible by the block size.");                                         \
    m.def("fmoe", &fmoe);                                                                     \
    m.def("fmoe_int8_g1u0",                                                                   \
          &fmoe_int8_g1u0,                                                                    \
          py::arg("out"),                                                                     \
          py::arg("input"),                                                                   \
          py::arg("gate"),                                                                    \
          py::arg("down"),                                                                    \
          py::arg("sorted_token_ids"),                                                        \
          py::arg("sorted_weights"),                                                          \
          py::arg("sorted_expert_ids"),                                                       \
          py::arg("num_valid_ids"),                                                           \
          py::arg("topk"),                                                                    \
          py::arg("input_scale"),                                                             \
          py::arg("fc1_scale"),                                                               \
          py::arg("fc2_scale"),                                                               \
          py::arg("fc2_smooth_scale") = std::nullopt,                                         \
          py::arg("activation")       = ActivationType::Silu);                                      \
    m.def("fmoe_g1u1",                                                                        \
          &fmoe_g1u1,                                                                         \
          py::arg("out"),                                                                     \
          py::arg("input"),                                                                   \
          py::arg("gate"),                                                                    \
          py::arg("down"),                                                                    \
          py::arg("sorted_token_ids"),                                                        \
          py::arg("sorted_weights"),                                                          \
          py::arg("sorted_expert_ids"),                                                       \
          py::arg("num_valid_ids"),                                                           \
          py::arg("topk"),                                                                    \
          py::arg("input_scale"),                                                             \
          py::arg("fc1_scale"),                                                               \
          py::arg("fc2_scale"),                                                               \
          py::arg("fc2_smooth_scale") = std::nullopt,                                         \
          py::arg("activation")       = ActivationType::Silu);                                      \
    m.def("fmoe_g1u1_tkw1",                                                                   \
          &fmoe_g1u1_tkw1,                                                                    \
          py::arg("out"),                                                                     \
          py::arg("input"),                                                                   \
          py::arg("gate"),                                                                    \
          py::arg("down"),                                                                    \
          py::arg("sorted_token_ids"),                                                        \
          py::arg("sorted_weights"),                                                          \
          py::arg("sorted_expert_ids"),                                                       \
          py::arg("num_valid_ids"),                                                           \
          py::arg("topk"),                                                                    \
          py::arg("input_scale"),                                                             \
          py::arg("fc1_scale"),                                                               \
          py::arg("fc2_scale"),                                                               \
          py::arg("fc2_smooth_scale") = std::nullopt,                                         \
          py::arg("activation")       = ActivationType::Silu);                                      \
    m.def("fmoe_int8_g1u0_a16", &fmoe_int8_g1u0_a16);                                         \
    m.def("fmoe_g1u1_a16",                                                                    \
          &fmoe_g1u1_a16,                                                                     \
          py::arg("out"),                                                                     \
          py::arg("input"),                                                                   \
          py::arg("gate"),                                                                    \
          py::arg("down"),                                                                    \
          py::arg("sorted_token_ids"),                                                        \
          py::arg("sorted_weights"),                                                          \
          py::arg("sorted_expert_ids"),                                                       \
          py::arg("num_valid_ids"),                                                           \
          py::arg("topk"),                                                                    \
          py::arg("fc1_scale"),                                                               \
          py::arg("fc2_scale"),                                                               \
          py::arg("fc1_smooth_scale"),                                                        \
          py::arg("fc2_smooth_scale"),                                                        \
          py::arg("activation") = ActivationType::Silu);                                      \
    m.def("fmoe_fp8_blockscale_g1u1",                                                         \
          &fmoe_fp8_blockscale_g1u1,                                                          \
          py::arg("out"),                                                                     \
          py::arg("input"),                                                                   \
          py::arg("gate"),                                                                    \
          py::arg("down"),                                                                    \
          py::arg("sorted_token_ids"),                                                        \
          py::arg("sorted_weights"),                                                          \
          py::arg("sorted_expert_ids"),                                                       \
          py::arg("num_valid_ids"),                                                           \
          py::arg("topk"),                                                                    \
          py::arg("input_scale"),                                                             \
          py::arg("fc1_scale"),                                                               \
          py::arg("fc2_scale"),                                                               \
          py::arg("fc_scale_blkn")    = 128,                                                  \
          py::arg("fc_scale_blkk")    = 128,                                                  \
          py::arg("fc2_smooth_scale") = std::nullopt,                                         \
          py::arg("activation")       = ActivationType::Silu);                                      \
    m.def("moe_stage1_g1u1",                                                                  \
          &moe_stage1_g1u1,                                                                   \
          py::arg("input"),                                                                   \
          py::arg("w1"),                                                                      \
          py::arg("w2"),                                                                      \
          py::arg("sorted_token_ids"),                                                        \
          py::arg("sorted_expert_ids"),                                                       \
          py::arg("num_valid_ids"),                                                           \
          py::arg("out"),                                                                     \
          py::arg("inter_dim"),                                                               \
          py::arg("kernelName"),                                                              \
          py::arg("block_m"),                                                                 \
          py::arg("ksplit")         = 0,                                                      \
          py::arg("activation")     = ActivationType::Silu,                                   \
          py::arg("quant_type")     = QuantType::No,                                          \
          py::arg("a1_scale")       = std::nullopt,                                           \
          py::arg("w1_scale")       = std::nullopt,                                           \
          py::arg("sorted_weights") = std::nullopt);                                          \
    m.def("moe_sum", &aiter::moe_sum, "moe_sum(Tensor! input, Tensor output) -> ()");

#define MOE_SORTING_PYBIND                             \
    m.def("moe_sorting_fwd",                           \
          &moe_sorting_fwd,                            \
          py::arg("topk_ids"),                         \
          py::arg("topk_weights"),                     \
          py::arg("sorted_token_ids"),                 \
          py::arg("sorted_weights"),                   \
          py::arg("sorted_expert_ids"),                \
          py::arg("num_valid_ids"),                    \
          py::arg("moe_buf"),                          \
          py::arg("num_experts"),                      \
          py::arg("unit_size"),                        \
          py::arg("local_expert_mask") = std::nullopt, \
          py::arg("num_local_tokens")  = std::nullopt, \
          py::arg("dispatch_policy")   = 0);

#define NORM_PYBIND                                               \
    m.def("layernorm2d_fwd",                                      \
          &layernorm2d,                                           \
          py::arg("input"),                                       \
          py::arg("weight"),                                      \
          py::arg("bias"),                                        \
          py::arg("epsilon"),                                     \
          py::arg("x_bias") = std::nullopt);                      \
    m.def("layernorm2d_fwd_with_add",                             \
          &layernorm2d_with_add,                                  \
          py::arg("out"),                                         \
          py::arg("input"),                                       \
          py::arg("residual_in"),                                 \
          py::arg("residual_out"),                                \
          py::arg("weight"),                                      \
          py::arg("bias"),                                        \
          py::arg("epsilon"),                                     \
          py::arg("x_bias") = std::nullopt);                      \
    m.def("layernorm2d_fwd_with_smoothquant",                     \
          &layernorm2d_with_smoothquant,                          \
          py::arg("out"),                                         \
          py::arg("input"),                                       \
          py::arg("xscale"),                                      \
          py::arg("yscale"),                                      \
          py::arg("weight"),                                      \
          py::arg("bias"),                                        \
          py::arg("epsilon"),                                     \
          py::arg("x_bias") = std::nullopt);                      \
    m.def("layernorm2d_fwd_with_add_smoothquant",                 \
          &layernorm2d_with_add_smoothquant,                      \
          py::arg("out"),                                         \
          py::arg("input"),                                       \
          py::arg("residual_in"),                                 \
          py::arg("residual_out"),                                \
          py::arg("xscale"),                                      \
          py::arg("yscale"),                                      \
          py::arg("weight"),                                      \
          py::arg("bias"),                                        \
          py::arg("epsilon"),                                     \
          py::arg("x_bias") = std::nullopt);                      \
    m.def("layernorm2d_fwd_with_dynamicquant",                    \
          &layernorm2d_with_dynamicquant,                         \
          py::arg("out"),                                         \
          py::arg("input"),                                       \
          py::arg("yscale"),                                      \
          py::arg("weight"),                                      \
          py::arg("bias"),                                        \
          py::arg("epsilon"),                                     \
          py::arg("x_bias") = std::nullopt);                      \
    m.def("layernorm2d_fwd_with_add_dynamicquant",                \
          &layernorm2d_with_add_dynamicquant,                     \
          py::arg("out"),                                         \
          py::arg("input"),                                       \
          py::arg("residual_in"),                                 \
          py::arg("residual_out"),                                \
          py::arg("yscale"),                                      \
          py::arg("weight"),                                      \
          py::arg("bias"),                                        \
          py::arg("epsilon"),                                     \
          py::arg("x_bias") = std::nullopt);                      \
    m.def("layernorm2d_with_add_asm", &layernorm2d_with_add_asm); \
    m.def("layernorm2d_with_add_smoothquant_asm", &layernorm2d_with_add_smoothquant_asm);

#define POS_ENCODING_PYBIND                                               \
    m.def("rotary_embedding_fwd", &rotary_embedding, "rotary_embedding"); \
    m.def("batched_rotary_embedding", &batched_rotary_embedding, "batched_rotary_embedding");

#define QUANT_PYBIND                                                     \
    m.def("static_per_tensor_quant", &aiter::static_per_tensor_quant);   \
    m.def("dynamic_per_tensor_quant", &aiter::dynamic_per_tensor_quant); \
    m.def("dynamic_per_token_scaled_quant",                              \
          &aiter::dynamic_per_token_scaled_quant,                        \
          py::arg("out"),                                                \
          py::arg("input"),                                              \
          py::arg("scales"),                                             \
          py::arg("scale_ub")        = std::nullopt,                     \
          py::arg("shuffle_scale")   = false,                            \
          py::arg("num_rows")        = std::nullopt,                     \
          py::arg("num_rows_factor") = 1);                               \
    m.def("dynamic_per_group_scaled_quant_fp4",                          \
          &aiter::dynamic_per_group_scaled_quant_fp4,                    \
          py::arg("out"),                                                \
          py::arg("input"),                                              \
          py::arg("scales"),                                             \
          py::arg("group_size")      = 32,                               \
          py::arg("shuffle_scale")   = true,                             \
          py::arg("num_rows")        = std::nullopt,                     \
          py::arg("num_rows_factor") = 1);                               \
    m.def("partial_transpose",                                           \
          &aiter::partial_transpose,                                     \
          py::arg("out"),                                                \
          py::arg("input"),                                              \
          py::arg("num_rows"));

#define RMSNORM_PYBIND                                                                             \
    m.def("rms_norm_cu",                                                                           \
          &rms_norm,                                                                               \
          "Apply Root Mean Square (RMS) Normalization to the input tensor.");                      \
    m.def(                                                                                         \
        "fused_add_rms_norm_cu", &fused_add_rms_norm, "In-place fused Add and RMS Normalization"); \
    m.def("rmsnorm2d_fwd",                                                                         \
          &rmsnorm2d,                                                                              \
          py::arg("input"),                                                                        \
          py::arg("weight"),                                                                       \
          py::arg("epsilon"),                                                                      \
          py::arg("use_model_sensitive_rmsnorm") = 0);                                             \
    m.def("rmsnorm2d_fwd_with_add",                                                                \
          &rmsnorm2d_with_add,                                                                     \
          py::arg("out"),                                                                          \
          py::arg("input"),                                                                        \
          py::arg("residual_in"),                                                                  \
          py::arg("residual_out"),                                                                 \
          py::arg("weight"),                                                                       \
          py::arg("epsilon"),                                                                      \
          py::arg("use_model_sensitive_rmsnorm") = 0);                                             \
    m.def("rmsnorm2d_fwd_with_smoothquant", &rmsnorm2d_with_smoothquant);                          \
    m.def("rmsnorm2d_fwd_with_add_smoothquant",                                                    \
          &rmsnorm2d_with_add_smoothquant,                                                         \
          py::arg("out"),                                                                          \
          py::arg("input"),                                                                        \
          py::arg("residual_in"),                                                                  \
          py::arg("residual_out"),                                                                 \
          py::arg("xscale"),                                                                       \
          py::arg("yscale"),                                                                       \
          py::arg("weight"),                                                                       \
          py::arg("epsilon"),                                                                      \
          py::arg("out_before_quant")            = std::nullopt,                                   \
          py::arg("use_model_sensitive_rmsnorm") = 0);                                             \
    m.def("rmsnorm2d_fwd_with_dynamicquant", &rmsnorm2d_with_dynamicquant);                        \
    m.def("rmsnorm2d_fwd_with_add_dynamicquant", &rmsnorm2d_with_add_dynamicquant);

#define ROPE_GENERAL_FWD_PYBIND                                 \
    m.def("rope_fwd_impl", &rope_fwd_impl);                     \
    m.def("rope_2c_fwd_impl", &rope_2c_fwd_impl);               \
    m.def("rope_cached_fwd_impl", &rope_cached_fwd_impl);       \
    m.def("rope_cached_2c_fwd_impl", &rope_cached_2c_fwd_impl); \
    m.def("rope_thd_fwd_impl", &rope_thd_fwd_impl);             \
    m.def("rope_2d_fwd_impl", &rope_2d_fwd_impl);

#define ROPE_GENERAL_BWD_PYBIND                                 \
    m.def("rope_bwd_impl", &rope_bwd_impl);                     \
    m.def("rope_2c_bwd_impl", &rope_2c_bwd_impl);               \
    m.def("rope_cached_bwd_impl", &rope_cached_bwd_impl);       \
    m.def("rope_cached_2c_bwd_impl", &rope_cached_2c_bwd_impl); \
    m.def("rope_thd_bwd_impl", &rope_thd_bwd_impl);             \
    m.def("rope_2d_bwd_impl", &rope_2d_bwd_impl);

#define ROPE_POS_FWD_PYBIND                                                                   \
    m.def("rope_cached_positions_fwd_impl", &rope_cached_positions_fwd_impl);                 \
    m.def("rope_cached_positions_2c_fwd_impl", &rope_cached_positions_2c_fwd_impl);           \
    m.def("rope_cached_positions_offsets_fwd_impl", &rope_cached_positions_offsets_fwd_impl); \
    m.def("rope_cached_positions_offsets_2c_fwd_impl", &rope_cached_positions_offsets_2c_fwd_impl);

#define SMOOTHQUANT_PYBIND                      \
    m.def("smoothquant_fwd", &smoothquant_fwd); \
    m.def("moe_smoothquant_fwd", &moe_smoothquant_fwd);

#define HIPBSOLGEMM_PYBIND                                                         \
    m.def("hipb_create_extension", &hipb_create_extension, "create_extension");    \
    m.def("hipb_destroy_extension", &hipb_destroy_extension, "destroy_extension"); \
    m.def("hipb_mm",                                                               \
          &hipb_mm,                                                                \
          "hipb_mm",                                                               \
          py::arg("mat1"),                                                         \
          py::arg("mat2"),                                                         \
          py::arg("solution_index"),                                               \
          py::arg("bias")      = std::nullopt,                                     \
          py::arg("out_dtype") = std::nullopt,                                     \
          py::arg("scaleA")    = std::nullopt,                                     \
          py::arg("scaleB")    = std::nullopt,                                     \
          py::arg("scaleOut")  = std::nullopt);                                     \
    m.def("hipb_findallsols",                                                      \
          &hipb_findallsols,                                                       \
          "hipb_findallsols",                                                      \
          py::arg("mat1"),                                                         \
          py::arg("mat2"),                                                         \
          py::arg("bias")      = std::nullopt,                                     \
          py::arg("out_dtype") = std::nullopt,                                     \
          py::arg("scaleA")    = std::nullopt,                                     \
          py::arg("scaleB")    = std::nullopt,                                     \
          py::arg("scaleC")    = std::nullopt);                                       \
    m.def("getHipblasltKernelName", &getHipblasltKernelName);

#define ROCSOLGEMM_PYBIND                                                          \
    m.def("rocb_create_extension", &rocb_create_extension, "create_extension");    \
    m.def("rocb_destroy_extension", &rocb_destroy_extension, "destroy_extension"); \
    m.def("rocb_mm", &RocSolIdxBlas, "mm");                                        \
    m.def("rocb_findallsols", &RocFindAllSolIdxBlas, "rocblas_find_all_sols");

#define AITER_ENUM_PYBIND                                \
    pybind11::enum_<QuantType>(m, "QuantType")           \
        .value("No", QuantType::No)                      \
        .value("per_Tensor", QuantType::per_Tensor)      \
        .value("per_Token", QuantType::per_Token)        \
        .value("per_1x32", QuantType::per_1x32)          \
        .value("per_1x128", QuantType::per_1x128)        \
        .value("per_128x128", QuantType::per_128x128)    \
        .export_values();                                \
    pybind11::enum_<ActivationType>(m, "ActivationType") \
        .value("No", ActivationType::No)                 \
        .value("Silu", ActivationType::Silu)             \
        .value("Gelu", ActivationType::Gelu)             \
        .export_values();

 #define GEMM_COMMON_PYBIND          \
 m.def("get_padded_m", &getPaddedM,  \
      py::arg("M"),                  \
      py::arg("N"),                  \
      py::arg("K"),                  \
      py::arg("gl"));    