# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.


aiter_lib = None


def torch_compile_guard(mutates_args: list[str] = [], device: str = "cpu"):
    def decorator(func):
        try:
            import torch
            from torch.library import Library
            import inspect
        except ImportError:

            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        global aiter_lib
        aiter_lib = Library("aiter", "FRAGMENT") if aiter_lib is None else aiter_lib
        op_name = func.__name__
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation
        return_int = False
        # Only return int will cause graph breaks
        if return_annotation is int:
            return_int = True

        def outer_wrapper(*args, **kwargs):
            dummy = torch.empty(1, device=device)
            if return_int:
                result = getattr(torch.ops.aiter, op_name)(dummy, *args, **kwargs)
                _, int_value = result
                return int_value
            return getattr(torch.ops.aiter, op_name)(dummy, *args, **kwargs)

        if hasattr(torch.ops.aiter, func.__name__):
            return outer_wrapper
        if hasattr(torch.library, "infer_schema"):
            schema_str = torch.library.infer_schema(func, mutates_args=mutates_args)
        else:
            # for pytorch 2.4
            import torch._custom_op.impl

            schema_str = torch._custom_op.impl.infer_schema(
                func, mutates_args=mutates_args
            )

        input_part, output_part = schema_str.split("->", 1)
        if not sig.parameters:
            new_input = "(Tensor dummy)"
        else:
            new_input = "(Tensor dummy, " + input_part[1:]

        output_part = output_part.strip()
        if not return_int:
            new_output = output_part
        else:
            # return only int will cause graph breaks and we add dummy_out
            new_output = "(Tensor, " + output_part + ")"
        schema_str = f"{new_input} -> {new_output}".strip()

        def custom_impl(dummy_tensor, *args, **kwargs):
            out = torch.empty(1, device=device)
            if not return_int:
                return func(*args, **kwargs)
            return out, func(*args, **kwargs)

        my_lib = aiter_lib
        my_lib.define(op_name + schema_str, tags=())
        my_lib.impl(op_name, custom_impl, dispatch_key="CUDA")
        my_lib.impl(op_name, custom_impl, dispatch_key="CPU")
        my_lib._register_fake(op_name, custom_impl)

        return outer_wrapper

    return decorator
