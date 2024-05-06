from mlir.dialects import arith
from mlir.dialects import nvgpu
from mlir.dialects import nvvm
from nvcute.tiled_mma import TiledMma
from codegen.utils import *
from codegen.core.builder import Builder

class TiledMmaBuilder:
  """Builds mlir mma operations tiled on a CTA tile"""
  def __init__(self, tiled_mma : TiledMma) -> None:
    self.tiled_mma = tiled_mma
    self.atom = tiled_mma.mma_atom

    # pycute.DataType -> mlir type
    self.dtype_a = to_mlir_ty(self.atom.element_a)
    self.dtype_b = to_mlir_ty(self.atom.element_b)
    self.dtype_c = to_mlir_ty(self.atom.element_c)

    # Create types for the tiled mma
    self.accum_shape = (self.atom.layout_c.shape[1][2],
                        self.atom.layout_c.shape[1][0],
                        self.atom.layout_c.shape[1][1])
    
    # Accumulator shape for one wgmma atom 
    # (e.g. 16x2x2xf32 for 64x128x16 wgmma f32 accum mma_atom)
    self.accum_shape = (self.atom.layout_c.shape[1][2],
                        self.atom.layout_c.shape[1][0],
                        self.atom.layout_c.shape[1][1])
    
    # Accumulator type for one wgmma atom
    self.accum_ty = ir.VectorType.get(self.accum_shape, self.dtype_c)

    b = Builder()
    self.mma_shape_attr = [ir.IntegerAttr.get(b.i64_ty, x) for x in [self.atom.mma_shape.m,
                                                                     self.atom.mma_shape.n,
                                                                     self.atom.mma_shape.k]]
  
  def init_accum_tiles(self):
    """Returns initialized accumulator array along m- and n-dimension."""
    accum_op_arr = []
    for m in range(self.tiled_mma.mma_count.m):
      for n in range(self.tiled_mma.mma_count.n):
        init_accum_op = arith.ConstantOp(self.accum_ty,
                                         ir.DenseElementsAttr.get_splat(
                                            self.accum_ty, 
                                            ir.FloatAttr.get_f32(0.0)),)
        accum_op_arr.append(init_accum_op)
    return accum_op_arr
  
  def __call__(self, desc_builder_a, desc_builder_b, accum_op_arr):
    """Builds the mma operations tiled on a CTA tile"""
    for m in range(self.tiled_mma.mma_count.m):
      for n in range(self.tiled_mma.mma_count.n):

        idx = m * self.tiled_mma.mma_count.n + n
        for k in range(self.tiled_mma.mma_count.k):
          accum_op_arr[idx] = nvgpu.WarpgroupMmaOp(self.accum_ty,
                                                   desc_builder_a.desc_op.result,
                                                   desc_builder_b.desc_op.result,
                                                   accum_op_arr[idx].result,
                                                   wgmmaShape=self.mma_shape_attr,
                                                   typeA=ir.TypeAttr.get(self.dtype_a),
                                                   layoutA=nvgpu.MatrixLayoutKind.ROW,
                                                   typeB=ir.TypeAttr.get(self.dtype_b),
                                                   layoutB=nvgpu.MatrixLayoutKind.ROW)
          
          # Next k-group.
          desc_builder_a.advance_op(self.tiled_mma.advance_k_byte_a)
          desc_builder_b.advance_op(self.tiled_mma.advance_k_byte_b)

        # wgmma.commit.group.sync.aligned : commits/creates a group of gmma operations
        nvvm.WgmmaGroupSyncAlignedOp()

        # Move back to the start of the k-dimension.
        desc_builder_a.advance_op(-self.tiled_mma.advance_k_byte_a * self.tiled_mma.mma_count.k)
        desc_builder_b.advance_op(-self.tiled_mma.advance_k_byte_b * self.tiled_mma.mma_count.k)
        
        # Next accum tile in n-dimension.
        desc_builder_b.advance_op(self.tiled_mma.advance_n_byte_b)

      # Move back to the start of the n-dimension
      desc_builder_b.advance_op(-self.tiled_mma.advance_n_byte_b * self.tiled_mma.mma_count.n)

      # Next accum tile in m-dimension.
      desc_builder_a.advance_op(self.tiled_mma.advance_m_byte_a)

    # Move back to the start of the n-dimension
    desc_builder_a.advance_op(-self.tiled_mma.advance_m_byte_a * self.tiled_mma.mma_count.m)

    return accum_op_arr
