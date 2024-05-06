from mlir.dialects import arith
from mlir.dialects import nvgpu
from mlir.dialects import memref
from codegen.utils import *
from codegen.core.builder import Builder

class TmaCopyBuilder:
  """TmaCopyBuilder creates a tiled copy using TMA operations."""
  def __init__(self, tma_copy, mbarrier_group_op, smem_op) -> None:
    self.tma_copy = tma_copy
    self.mbarrier_group_op = mbarrier_group_op
    self.element_ty = to_mlir_ty(self.tma_copy.dtype)
    # Note that the shape of the box shape is (strided, contiguous) 
    # to match mlir memref restrictions.
    self.box_shape = (self.tma_copy.box_shape.strided,
                      self.tma_copy.box_shape.contiguous)
    self.box_smemref_ty = ir.MemRefType.get(self.box_shape,
                                            self.element_ty,
                                            memory_space = Builder().smem_addr_space_attr)
    
    # Shared memory starting op for the SM. 
    self.smem_op = smem_op
  
  def load(self, 
           tma_desc_op, 
           dest_smem_byte_op, # dest smem byte offset op from smem start
           src_gmem_coords,   # src gmem coords (contiguous, strided)
           mbarrier_id_op,
           pred_op):
    """Issues Tma copy operations in units of Tma boxes covering the CTA-tile."""
    b = Builder()
    for box_id in range(self.tma_copy.num_boxes):
      box_smem_byte_op = arith.addi(dest_smem_byte_op, 
                                    b.const_index_op(box_id * 
                                                     self.tma_copy.box_bytes))
      box_view_op = memref.view(self.box_smemref_ty, self.smem_op, box_smem_byte_op, [])
      coord_contiguous_op = arith.addi(src_gmem_coords[0], 
                                       b.const_index_op(box_id * 
                                                        self.tma_copy.box_shape.contiguous))
      nvgpu.TmaAsyncLoadOp(box_view_op, self.mbarrier_group_op, tma_desc_op, 
                           coordinates = [coord_contiguous_op, src_gmem_coords[1]],
                           mbarId = mbarrier_id_op,
                           predicate = pred_op)
      
      debug_print(
          "TiledCopyBuilder.load dest_smem_byte_op={} @ coodinates=({},{}), mbarrier_id_op={}",
          dest_smem_byte_op,
          coord_contiguous_op,
          src_gmem_coords[1],
          mbarrier_id_op,
          predicate=pred_op,
      )
      

'''
Notes:
1. Decouple mbarraier group from mlir.nvgpu.TmaAsyncLoadOp.
   The mlir.nvgpu.TmaAsyncLoadOp should take a smemref object 
   for the barriers just like `mlir.nvgpu.AsyncCopyOp` takes
   src and dst memref objects.
   (a). Init of this TmaCopyBuilder should NOT take and 
   depend on a mbarrier_group_op. This needs mlir.NVGPU,
   mlir.NVVM TableGen and C++ changes.
'''
    
