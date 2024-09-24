import ctypes, errno, os, sys

import numpy as np
from codegen.core import builder
from codegen.gemm import GemmDescription, GemmShape
from codegen.kernel.tma_wgmma_kernel_builder import TmaWgmmaKernelBuilder
from codegen.nvgpu_compiler import NvgpuCompiler
from codegen.utils import get_numpy_dtype, MainloopVariant, SmArchTag, TensorDescription
from mlir import ir, runtime as rt
from mlir.dialects import nvgpu


def test_tma_wgmma(gemm_desc: GemmDescription):
  # Create a kernel module
  module_builder = TmaWgmmaKernelBuilder(gemm_desc)
  module = module_builder()
  #print(module)

  # Compile the GEMM module
  support_lib = os.getenv("SUPPORT_LIB")
  if not os.path.exists(support_lib):
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), support_lib)

  options = f"cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"
  compiler = NvgpuCompiler(options, opt_level=3, shared_libs=[support_lib])

  engine = compiler.compile_and_jit(module)

  # Allocate input/output tensors
  dtype_a = get_numpy_dtype(gemm_desc.a.datatype)
  dtype_b = get_numpy_dtype(gemm_desc.b.datatype)
  dtype_c = get_numpy_dtype(gemm_desc.c.datatype)

  host_c = np.zeros((gemm_desc.problem_shape.m, gemm_desc.problem_shape.n), dtype_c)
  host_a = np.random.randn(gemm_desc.problem_shape.m, gemm_desc.problem_shape.k).astype(dtype_a)
  host_b = np.random.randn(gemm_desc.problem_shape.k, gemm_desc.problem_shape.n).astype(dtype_b)
  a_ptr = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(host_a)))
  b_ptr = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(host_b)))
  c_ptr = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(host_c)))

  print(f"Running GEMM kernel {gemm_desc.name()}")
  # Launch the GEMM kernel
  engine.invoke(gemm_desc.name(), a_ptr, b_ptr, c_ptr)

  # Run the reference
  ref_c = host_a.astype(dtype_a) @ host_b.astype(dtype_b)

  float_formatter = "{:.2f}".format
  np.set_printoptions(
      formatter={"float_kind": float_formatter},
      threshold=sys.maxsize,
      linewidth=sys.maxsize,
  )

  # Check the result
  np.testing.assert_allclose(host_c, ref_c, rtol=5e-03, atol=1e-01)

  # Reach here only when the test passes
  print("Test passed!")


with ir.Context() as ctx, ir.Location.unknown():
  b = builder.Builder()

  # Create testcases
  gemm_descs = [
    GemmDescription(
      TensorDescription(b.f16_ty, nvgpu.MatrixLayoutKind.ROW),  # a or lhs tensor
      TensorDescription(b.f16_ty, nvgpu.MatrixLayoutKind.ROW),  # b or rhs tensor
      TensorDescription(
        b.f32_ty, nvgpu.MatrixLayoutKind.ROW
      ),  # c or result tensor
      b.f32_ty,  # accumulation element datatype
      GemmShape(64, 128, 64),  # problem shape (in number of elements)
      GemmShape(1, 1, 1),  # cga shape (in number of cta)
      GemmShape(64, 128, 64),  # cta shape (in number of elements)
      GemmShape(4, 1, 1),  # warp shape (in number of warps)
      GemmShape(64, 128, 16),  # wgmma shape (int number of elements)
      1,  # number of smem stages
      MainloopVariant.OneStage,
      SmArchTag.Sm90,
    ),
    GemmDescription(
      TensorDescription(b.f16_ty, nvgpu.MatrixLayoutKind.ROW),  # a or lhs tensor
      TensorDescription(b.f16_ty, nvgpu.MatrixLayoutKind.ROW),  # b or rhs tensor
      TensorDescription(
        b.f32_ty, nvgpu.MatrixLayoutKind.ROW
      ),  # c or result tensor
      b.f32_ty,  # accumulation element datatype
      GemmShape(128, 128, 64),  # problem shape (in number of elements)
      GemmShape(1, 1, 1),  # cga shape (in number of cta)
      GemmShape(128, 128, 64),  # cta shape (in number of elements)
      GemmShape(4, 1, 1),  # warp shape (in number of warps)
      GemmShape(64, 128, 16),  # wgmma shape (int number of elements)
      1,  # number of smem stages
      MainloopVariant.OneStage,
      SmArchTag.Sm90,
    ),
  ]

  # Run the tests
  for gemm_desc in gemm_descs:
      test_tma_wgmma(gemm_desc)
