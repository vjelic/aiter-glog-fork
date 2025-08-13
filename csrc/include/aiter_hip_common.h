// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <hip/hip_runtime.h>
#include <iostream>
#include "ck_tile/core.hpp"

enum class GPUArch {
    gfx942,
    gfx950
};

#define HIP_CALL(call)                                                                                                           \
    do                                                                                                                           \
    {                                                                                                                            \
        hipError_t err = call;                                                                                                   \
        if (err != hipSuccess)                                                                                                   \
        {                                                                                                                        \
            printf("\n[AITER] %s:%d fail to call %s ---> [HIP error](%s)\n", __FILE__, __LINE__, #call, hipGetErrorString(err)); \
            exit(0);                                                                                                             \
        }                                                                                                                        \
    } while (0)

struct p3
{
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct p2
{
    unsigned int _p0;
    unsigned int _p1;
};
struct p1
{
    unsigned int _p0;
};

struct AiterAsmKernelArgs
{
    void *args_ptr;
    void *arg_size_ptr;
    int gdx;
    int gdy;
    int gdz;
    int bdx;
    int bdy;
    int bdz;
    const hipStream_t stream;
};

class AiterAsmKernel
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    AiterAsmKernel(const char *name, const char *hsaco)
    {
        const char *AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
        std::cout << "[aiter] hipModuleLoad: " << (std::string(AITER_ASM_DIR) + hsaco).c_str() << " GetFunction: " << name;
        HIP_CALL(hipModuleLoad(&module, (std::string(AITER_ASM_DIR) + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
        std::cout << " Success" << std::endl;
    };

    ~AiterAsmKernel()
    {
        HIP_CALL(hipModuleUnload(module));
    }

    void launch_kernel(const AiterAsmKernelArgs &kargs)
    {
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs.args_ptr,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, kargs.arg_size_ptr,
                          HIP_LAUNCH_PARAM_END};

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       kargs.gdx, kargs.gdy, kargs.gdz,
                                       kargs.bdx, kargs.bdy, kargs.bdz,
                                       0, kargs.stream, nullptr, (void **)&config));
    };
};

class AiterAsmKernelFast
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    AiterAsmKernelFast(const char *name, void *hsaco)
    {
        HIP_CALL(hipModuleLoadData(&module, hsaco));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
        std::cout << " Success" << std::endl;
    };

    ~AiterAsmKernelFast()
    {
        HIP_CALL(hipModuleUnload(module));
    }

    void launch_kernel(const AiterAsmKernelArgs &kargs)
    {
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs.args_ptr,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, kargs.arg_size_ptr,
                          HIP_LAUNCH_PARAM_END};

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       kargs.gdx, kargs.gdy, kargs.gdz,
                                       kargs.bdx, kargs.bdy, kargs.bdz,
                                       0, kargs.stream, nullptr, (void **)&config));
    };
};

static const std::string get_gpu_arch() {
    int device_count;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        return "No GPU Found";
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    std::string arch_full = prop.gcnArchName;
    size_t colon_pos = arch_full.find(':');
    if (colon_pos != std::string::npos) {
        return arch_full.substr(0, colon_pos);
    } else {
        return arch_full;
    }
}

static const uint32_t get_num_cu_func()
{
    auto get_num_cu_local = [](){
        hipDevice_t dev;
        hipDeviceProp_t dev_prop;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        return dev_prop.multiProcessorCount;
    };
    static const uint32_t num_cu = get_num_cu_local();
    return num_cu;
}
