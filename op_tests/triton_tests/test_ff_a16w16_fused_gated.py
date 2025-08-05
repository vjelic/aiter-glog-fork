import torch
import torch.nn.functional as F
import triton
import pytest
from aiter.ops.triton.ff_a16w16_fused_gated import ff_a16w16_fused_gated
from op_tests.triton_tests.test_gemm_a16w16 import minimal_x_vals
from op_tests.triton_tests.utils.types import str_to_torch_dtype
from op_tests.triton_tests.ff_test_utils import ff_gated_test

def generate_ff_a16w16_inputs(
    batch, hidden_dim, intermediate_dim, dtype, layout="TN", gating=True, output=True
):
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((batch, hidden_dim), dtype=dtype).cuda()  # (M, K)
    else:
        x = torch.randn((hidden_dim, batch), dtype=dtype).cuda().T

    if layout[1] == "T":
        if gating:
            w1 = torch.randn((hidden_dim, intermediate_dim * 2), dtype=dtype).cuda().T
        else:
            w1 = torch.randn((hidden_dim, intermediate_dim), dtype=dtype).cuda().T
        w2 = torch.randn((intermediate_dim, hidden_dim), dtype=dtype).cuda()
    else:
        if gating:
            w1 = torch.randn(
                (intermediate_dim * 2, hidden_dim), dtype=dtype
            ).cuda()  # (N*2, K)
        else:
            w1 = torch.randn((intermediate_dim, hidden_dim), dtype=dtype).cuda()
        w2 = torch.randn((hidden_dim, intermediate_dim), dtype=dtype).cuda().T

    w1 = w1 / (intermediate_dim**0.5)  # scale down output variance
    w2 = w2 / (hidden_dim**0.5)

    y = None
    if output:
        y = torch.zeros((batch, hidden_dim), dtype=dtype).cuda()
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return x, w1, w2, out_dtype, y


@pytest.mark.parametrize("activation", ["silu_exp2", "gelu_tanh", "relu", None])
@pytest.mark.parametrize("batch, hidden_dim, intermediate_dim", minimal_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_ff_a16w16_gated(
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
    ff_gated_test(ff_a16w16_fused_gated,
                  batch=batch,
                  hidden_dim=hidden_dim,
                  intermediate_dim=intermediate_dim,
                  dtype=dtype,
                  output=output,
                  activation=activation)