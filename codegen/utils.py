import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import enum
from enum import auto
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import gpu
from mlir.dialects import scf
from mlir.dialects import nvgpu
from mlir.extras import types as T
from nvcute.helpers import DataType

DEBUG = False

######################################################################
# Helper functions
######################################################################
def debug_print(fmt, *args, predicate=None, threadNumber=-1, forcePrint=False):
    if not DEBUG and not forcePrint:
        return
    type_formats = []
    for arg in args:
        ty_format = None
        if ir.IndexType.isinstance(arg.type):
            ty_format = "%llu"
        if ir.IntegerType.isinstance(arg.type):
            width = ir.IntegerType(arg.type).width
            if width == 64:
                ty_format = "%llu"
            elif width == 32:
                ty_format = "%d"
            elif width == 1:
                ty_format = "%i"
        if ir.F32Type.isinstance(arg.type):
            ty_format = "%f"
        if ty_format is None:
            raise NotImplementedError(arg.type)
        type_formats.append(ty_format)
    if threadNumber != -1:
        tidx = gpu.thread_id(gpu.Dimension.x)
        predicate = arith.cmpi(arith.CmpIPredicate.eq, tidx, c(threadNumber))
        scf.yield_([])
    if_op = scf.IfOp(predicate)
    with ir.InsertionPoint(if_op.then_block):
        gpu.printf(fmt.format(*type_formats) + "\n", args)
        scf.yield_([])

def get_size_in_bits(ty):
  """Returns the size of the data type in bits."""
  if ir.FloatType.isinstance(ty):
      return ir.FloatType(ty).width
  if ir.IntegerType.isinstance(ty):
      return ir.IntegerType(ty).width
  raise NotImplementedError(ty)

def get_type_size(ty):
  """Returns the size of the data type in bytes."""
  return get_size_in_bits(ty) // 8

######################################################################
# Helper functions to convert between numpy <=> MLIR.
######################################################################
def get_mlir_ty(dtype) -> ir.Type:
  """Returns the MLIR type for the given numpy dtype."""
  if dtype == np.float16:
      return T.f16()
  if dtype == np.float32:
      return T.f32()
  if dtype == np.float64:
      return T.f64()
  if dtype == np.int32:
      return T.i32()
  if dtype == np.int64:
      return T.i64()
  raise NotImplementedError(dtype)

def get_numpy_dtype(mlir_ty) -> np.dtype:
  """Returns the numpy dtype for the given MLIR type."""
  if T.F16Type.isinstance(mlir_ty):
      return np.float16
  if T.F32Type.isinstance(mlir_ty):
      return np.float32
  if T.F64Type.isinstance(mlir_ty):
      return np.float64
  if T.I32Type.isinstance(mlir_ty):
      return np.int32
  if T.I64Type.isinstance(mlir_ty):
      return np.int64
  raise NotImplementedError(mlir_ty)

######################################################################
# Helper functions to convert between pycute.Datatype <=> MLIR.
######################################################################
def to_mlir_ty(dtype) -> ir.Type:
  """Returns the MLIR type for the given pycute.DataType."""
  if dtype == DataType.f16:
    return T.f16()
  if dtype == DataType.f32:
    return T.f32()
  if dtype == DataType.f64:
    return T.f64()
  if dtype == DataType.i32:
    return T.i32()
  if dtype == DataType.i64:
    return T.i64()
  raise NotImplementedError(dtype)

def to_pycute_dtype(mlir_ty) -> DataType:
  if T.F16Type.isinstance(mlir_ty):
      return DataType.f16
  if T.F32Type.isinstance(mlir_ty):
      return DataType.f32
  if T.F64Type.isinstance(mlir_ty):
      return DataType.f64
  if T.I32Type.isinstance(mlir_ty):
      return DataType.i32
  if T.I64Type.isinstance(mlir_ty):
      return DataType.i64
  raise NotImplementedError(mlir_ty)

######################################################################
# Enums
######################################################################

# cuBLAS/cuDNN layout type names convention is followed for the layout names.
# https://docs.nvidia.com/cuda/cublas/index.html#cublasoperation-t
ShortLayoutTypeName = {
  nvgpu.MatrixLayoutKind.ROW: "t",
  nvgpu.MatrixLayoutKind.COL: "n",
}

# Layout names used for nvvm attributes.
LayoutTypeName = {
  nvgpu.MatrixLayoutKind.ROW: "row",
  nvgpu.MatrixLayoutKind.COL: "col",
}
######################################################################


# Note that WGMMA swizzle enum values are different from the ones in the
# used for TMA swizzle. Thus, we need to define a separate enum for WGMMA
# swizzle.
class GmmaSwizzle(enum.Enum):
  Interleave = 0
  Swizzle128B = auto()
  Swizzle64B = auto()
  Swizzle32B = auto()

GmmaSwizzleName = {
   GmmaSwizzle.Interleave: "interleave",
   GmmaSwizzle.Swizzle128B: "swizzle128b",
   GmmaSwizzle.Swizzle64B: "swizzle64b",
   GmmaSwizzle.Swizzle32B: "swizzle32b",
}

# Gmma operandA can come from shared memory or registers
class GmmaType(enum.Enum):
  SS = 0         # Operand A and B are in shared memory
  RS = auto()    # Operand A is in registers and Operand B is in shared memory

GmmaTypeName = {
  GmmaType.SS: "ss",
  GmmaType.RS: "rs",
}

######################################################################
TmaSwizzleByteDict = {nvgpu.TensorMapSwizzleKind.SWIZZLE_32B : 32,
                      nvgpu.TensorMapSwizzleKind.SWIZZLE_64B : 64,
                      nvgpu.TensorMapSwizzleKind.SWIZZLE_128B : 128}

# Enum values within the Sm90 architecture for TMA and GMMA swizzle
# are the different. We need to match the TMA swizzle to GMMA swizzle
TmaSwizzleToGmmaSwizzle = {
    nvgpu.TensorMapSwizzleKind.SWIZZLE_32B: GmmaSwizzle.Swizzle32B,
    nvgpu.TensorMapSwizzleKind.SWIZZLE_64B: GmmaSwizzle.Swizzle64B,
    nvgpu.TensorMapSwizzleKind.SWIZZLE_128B: GmmaSwizzle.Swizzle128B,
}
######################################################################

# C++ uses this value to understand whether it's dynamic or not.
MLIR_DYNAMIC = -9223372036854775808

######################################################################
class SmArchTag(enum.Enum):
  Sm80 = 80
  Sm90 = 90

SmArchTagName = {
    SmArchTag.Sm80: "ampere",
    SmArchTag.Sm90: "hopper",
}

SharedMemCapacityPerSM = {
    80: 163,  # 163KB of SMEM - 1KB reserved for the driver
    90: 227,  # 227KB of SMEM - 1KB reserved for the driver
}
######################################################################

class MainloopVariant(enum.Enum):
  """
  Mainloop variants for the mainloop that performs the GEMM computation.
  """
  OneStage = auto()                  
  Multistage = auto()
  WarpSpecializedCooperative = auto()
  WarpSpecializedPingPong = auto()

ShortMainloopVariantName = {
  MainloopVariant.OneStage: "one_stage",
  MainloopVariant.Multistage: "multistage",
  MainloopVariant.WarpSpecializedCooperative: "warp_specialized_cooperative",
  MainloopVariant.WarpSpecializedPingPong: "warp_specialized_pingpong",
}
######################################################################


class TensorDescription:
  """
  A class for tensor description capturing tensor datatype and layout.
  """

  def __init__(self, datatype, layout):
    self.datatype = datatype  # mlir built-in type
    self.layout = layout      # LayoutType enum

  def name(self):
    return "%s%s" % (self.datatype, ShortLayoutTypeName[self.layout])
######################################################################