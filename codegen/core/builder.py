from mlir import ir
from mlir.dialects import arith


class Builder:
  """
  Top-level builder class contains kernel-agnositc peices of the kernel builder.

  * Guidlines for writing IR builders *
   -  Create `mlir` types and attributes under the singular 
      ir.Context() object. The `mlir` types and attributes suffixed 
      with `_ty` with `_attr`, respectively. We separate the creation
      of types and attributes from the operations for the following:
      (a.) GEMM Kernel-agnositc: created in the `Builder`.
      (b.) GEMM Kernel-specific: created in the `KernelModuleBuilder`.
   - Create all `mlir` operations suffixed with `_op` under the
      module object and are inserted under the insertion point 
      of the `module` in the `KernelModuleBuilder` __call__.
  """

  def __init__(self):

    # create builtin datatypes
    self.f16_ty = ir.F16Type.get()
    self.f32_ty = ir.F32Type.get()
    self.f64_ty = ir.F64Type.get()
    self.index_ty = ir.IndexType.get()
    self.i1_ty = ir.IntegerType.get_signless(1)
    self.i8_ty = ir.IntegerType.get_signless(8)
    self.i16_ty = ir.IntegerType.get_signless(16)
    self.i32_ty = ir.IntegerType.get_signless(32)
    self.i64_ty = ir.IntegerType.get_signless(64)
    self.u64_ty = ir.IntegerType.get_unsigned(64)
    # create llvm pointer types
    self.llvm_smem_ptr_ty = ir.Type.parse("!llvm.ptr<3>")

    # create attributes
    self.smem_addr_space_attr = ir.Attribute.parse("#gpu.address_space<workgroup>")

    # addtional types
    self.gpu_token_ty = ir.Type.parse("!gpu.async.token")

  def underlying_llvm_struct_ty(self, memref):       
    """Returns underlying llvm struct type backing memrefs"""
    memref_ty = ir.MemRefType(memref.type)
    memory_space = memref_ty.memory_space
    llvm_ptr_ty = self.llvm_smem_ptr_ty if memory_space == self.smem_addr_space_attr else None
    if llvm_ptr_ty is None:
      raise ValueError("Only Shared Memory memrefs are supported.")
    return ir.Type.parse(f"!llvm.struct<({llvm_ptr_ty}, {llvm_ptr_ty}, "\
                         f"i64, array<{memref_ty.rank} x i64>, array<{memref_ty.rank} x i64>)>")
    
  def const_index_op(self, value):
    """Returns an arithmetic constant op (index type)."""
    return arith.constant(self.index_ty, ir.IntegerAttr.get(self.index_ty, value))

  def const_int_op(self, value, int_ty):
    """Returns an arithmetic constant op (integer type)."""
    return arith.constant(int_ty, ir.IntegerAttr.get(int_ty, value))
  
  def const_float_op(self, value, float_ty):
    """Returns an arithmetic constant op (float type)."""
    return arith.constant(float_ty, value)
######################################################################