import numpy as np
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import func
from mlir.dialects import gpu
from mlir.dialects import memref
from mlir.dialects import nvgpu
from mlir.dialects import nvvm
from mlir.dialects import vector
from mlir.extras import types as T
from pycute.layout import Layout
from nvcute.tiled_copy import TmaCopy
from nvcute.tiled_mma import MmaAtom, TiledMma

from codegen.utils import *
from codegen.core.builder import Builder
from codegen.core.tiled_mma_builder import TiledMmaBuilder
from codegen.core.tiled_copy_builder import TmaCopyBuilder
from codegen.core.descriptor_builder import TmaDescriptorBuilder, GmmaDescriptorBuilder

################################################################################
#    Building simpliest TMA WGMMA H100 kernel
################################################################################ 
class TmaWgmmaKernelBuilder():
  def __init__(self, gemm_description) -> None:
    """Create kernel-specific types and attribute required for the GEMM kernel."""
    self.gemm_description = gemm_description
  
    b = Builder()

    # Global memory.
    self.a_memref_ty = ir.MemRefType.get([self.gemm_description.problem_shape.m,
                                          self.gemm_description.problem_shape.k],
                                          self.gemm_description.a.datatype)
    self.b_memref_ty = ir.MemRefType.get([self.gemm_description.problem_shape.k,
                                          self.gemm_description.problem_shape.n],
                                          self.gemm_description.b.datatype)
    self.c_memref_ty = ir.MemRefType.get([self.gemm_description.problem_shape.m,
                                          self.gemm_description.problem_shape.n],
                                          self.gemm_description.c.datatype)
    # Shared memory.
    self.mbarrier_group_ty = ir.Type.parse(f"!nvgpu.mbarrier.group<\
                                        memorySpace={b.smem_addr_space_attr},\
                                        num_barriers={gemm_description.num_stages}\
                                      >")

    self.dynamic_smemref_ty =  ir.MemRefType.get((MLIR_DYNAMIC,), 
                                                 b.i8_ty, 
                                                 memory_space = b.smem_addr_space_attr)
    
    # TMA shapes for A and B operands.
    # Note as of right now, the tuple format (strided, strided, ..., contiguous) 
    # a restriction at MLIR-side (check this).
    self.a_smem_stage_shape = (self.gemm_description.cta_shape.m, 
                               self.gemm_description.cta_shape.k)
    self.b_smem_stage_shape = (self.gemm_description.cta_shape.k, 
                               self.gemm_description.cta_shape.n)
    self.a_smem_stage_smemref_ty = ir.MemRefType.get(self.a_smem_stage_shape,
                                                     self.gemm_description.a.datatype,
                                                     memory_space = b.smem_addr_space_attr)
    self.b_smem_stage_smemref_ty = ir.MemRefType.get(self.b_smem_stage_shape,
                                                     self.gemm_description.b.datatype,
                                                     memory_space = b.smem_addr_space_attr)

    # Tiled copy for operands A always shape of (M, K) and layout is encoded in strides.
    self.a_tma_copy = TmaCopy(Layout((self.gemm_description.cta_shape.m, self.gemm_description.cta_shape.k),
                                              (self.gemm_description.cta_shape.k, 1)),
                                       to_pycute_dtype(self.gemm_description.a.datatype))

    # Tiled copy for operands B always shape of (N, K) and layout is encoded in strides.
    self.b_tma_copy = TmaCopy(Layout((self.gemm_description.cta_shape.n, self.gemm_description.cta_shape.k),
                                              (1, self.gemm_description.cta_shape.n)), 
                                        to_pycute_dtype(self.gemm_description.b.datatype))

    # Memref types for A, B, and C in the shared memory.
    # Note as of right now, the tuple format (strided, contiguous)
    self.a_tma_box_shape = (self.a_tma_copy.box_shape.strided, self.a_tma_copy.box_shape.contiguous) 
    self.b_tma_box_shape = (self.b_tma_copy.box_shape.strided, self.b_tma_copy.box_shape.contiguous) 
    
    # Create MmaAtom for TiledMma instances.                           
    self.mma_atom = MmaAtom(self.gemm_description.instruction_shape,
                            to_pycute_dtype(self.gemm_description.a.datatype), 
                            to_pycute_dtype(self.gemm_description.b.datatype), 
                            to_pycute_dtype(self.gemm_description.c.datatype),
                            nvgpu.MatrixLayoutKind.ROW, nvgpu.MatrixLayoutKind.ROW,
                            GmmaType.SS)
    self.tiled_mma = TiledMma(self.mma_atom, self.a_tma_copy, self.b_tma_copy)
    
  
  @property
  def dim3_grid(self):
    """Returns the grid shape as dim3 tuple in number of cta launched in ms-, n-, k-dim."""
    grid_m = ((self.gemm_description.problem_shape.m + \
               self.gemm_description.cta_shape.m - 1) // self.gemm_description.cta_shape.m)
    grid_n = ((self.gemm_description.problem_shape.n + \
               self.gemm_description.cta_shape.n - 1) // self.gemm_description.cta_shape.n)
    dim3_grid = (grid_m, grid_n, 1)
    return dim3_grid

  @property
  def dim3_block(self):
    """Returns the block shape as dim3 tuple in number of threads in m-, n-, k-dim."""
    warp_thread_count = 32
    return (self.gemm_description.warp_shape.m * warp_thread_count,
            self.gemm_description.warp_shape.n,
            self.gemm_description.warp_shape.k)
  
  @property
  def block_size(self):
    """Returns the block size in number of threads."""
    return np.product(self.dim3_block)
  
  def __call__(self):
    """Builds and returns a module in NVGPU dialect for single TMA + WGMMA kernel module."""
    b = Builder()
    module = ir.Module.create()
    
    with ir.InsertionPoint(module.body):
      func_op = func.FuncOp(self.gemm_description.name(),
                            ([self.a_memref_ty, self.b_memref_ty, self.c_memref_ty],
                             []))
      with ir.InsertionPoint(func_op.add_entry_block()):

        a_host = func_op.arguments[0]
        b_host = func_op.arguments[1]
        c_host = func_op.arguments[2]

        # Allocate device memory and memcpy.
        t1 = gpu.wait(b.gpu_token_ty, [])
        a_device, t2 = gpu.alloc(self.a_memref_ty, b.gpu_token_ty, [t1], [], [])
        b_device, t3 = gpu.alloc(self.b_memref_ty, b.gpu_token_ty, [t2], [], [])
        c_device, t4 = gpu.alloc(self.c_memref_ty, b.gpu_token_ty, [t3], [], [])
        t5 = gpu.memcpy(b.gpu_token_ty, [t4], a_device, a_host)
        t6 = gpu.memcpy(b.gpu_token_ty, [t5], b_device, b_host)
        t7 = gpu.wait(b.gpu_token_ty, [t6])

        # Create a TMA descriptor for A and B operands.
        a_tma_desc_builder = TmaDescriptorBuilder(self.a_tma_copy.swizzle,
                                          nvgpu.TensorMapL2PromoKind.L2PROMO_NONE,
                                          nvgpu.TensorMapOOBKind.OOB_ZERO,
                                          nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
                                          self.a_tma_box_shape,
                                          self.a_memref_ty)
        a_tma_desc_op = a_tma_desc_builder.tma_descriptor_op(a_device)

        b_tma_desc_builder = TmaDescriptorBuilder(self.b_tma_copy.swizzle,
                                          nvgpu.TensorMapL2PromoKind.L2PROMO_NONE,
                                          nvgpu.TensorMapOOBKind.OOB_ZERO,
                                          nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
                                          self.b_tma_box_shape,
                                          self.b_memref_ty)
        b_tma_desc_op = b_tma_desc_builder.tma_descriptor_op(b_device)
      
        # Create kernel launch.
        launch_op = gpu.LaunchOp(b.gpu_token_ty, [t7], 
                              *map(b.const_index_op, self.dim3_grid),
                              *map(b.const_index_op, self.dim3_block),
                              dynamicSharedMemorySize = b.const_int_op(self.gemm_description.smem_bytes, b.i32_ty))
        
        # Kernel body.
        launch_op.body.blocks.append(*([T.index()] * 12))
        with ir.InsertionPoint(launch_op.body.blocks[0]):

          # Create commonly used constants.
          c0_op = b.const_index_op(0)
          c1_op = b.const_index_op(1)
          ticks_op = b.const_index_op(10000000)

          memref.assume_alignment(c_device, 16)
          tidx_op = gpu.thread_id(gpu.Dimension.x)
          
          # Create shared memory constants and views.
          dynamic_smem_op = gpu.dynamic_shared_memory(self.dynamic_smemref_ty)
          a_smem_start_byte_op = b.const_index_op(0)
          a_smem_view_op = memref.view(self.a_smem_stage_smemref_ty, dynamic_smem_op, a_smem_start_byte_op, [])
          b_smem_start_byte_op = b.const_index_op(self.gemm_description.a_smem_bytes)
          b_smem_view_op = memref.view(self.b_smem_stage_smemref_ty, dynamic_smem_op, b_smem_start_byte_op, [])
          smem_txcount_bytes_op = b.const_index_op(self.gemm_description.smem_bytes_per_stage)

          # Initialize barriers.
          is_leader_op = arith.cmpi(arith.CmpIPredicate.eq, tidx_op, c0_op)
          mbarrier_group_op = nvgpu.mbarrier_create(self.mbarrier_group_ty)

          for i in range(self.gemm_description.num_stages):
            nvgpu.mbarrier_init(mbarrier_group_op, c1_op, b.const_index_op(i), predicate=is_leader_op)
          gpu.barrier()

          # Create Builders objects for Tiled Copy, GMMA descriptor, and Tiled Mma.
          a_tiled_copy = TmaCopyBuilder(self.a_tma_copy, mbarrier_group_op, dynamic_smem_op)
          b_tiled_copy = TmaCopyBuilder(self.b_tma_copy, mbarrier_group_op, dynamic_smem_op)
          a_gmma_desc_builder = GmmaDescriptorBuilder.from_bitfield(self.tiled_mma.desc_a).set_smem_addr(a_smem_view_op)
          b_gmma_desc_builder = GmmaDescriptorBuilder.from_bitfield(self.tiled_mma.desc_b).set_smem_addr(b_smem_view_op)
          tiled_mma = TiledMmaBuilder(self.tiled_mma)
          
          # Prefetch A and B TMA descriptor.
          nvgpu.tma_prefetch_descriptor(a_tma_desc_op, predicate=is_leader_op)
          nvgpu.tma_prefetch_descriptor(b_tma_desc_op, predicate=is_leader_op)          

          # Intialize barrier with TMA transaction smem A and B bytes
          nvgpu.mbarrier_arrive_expect_tx(mbarrier_group_op,
                                          smem_txcount_bytes_op, 
                                          c0_op, 
                                          predicate = is_leader_op)

          # Issue Tiled Copy for Tile A and B (issuing TMA Copy in units of box shape to cover cta shape)
          a_tiled_copy.load(a_tma_desc_op, a_smem_start_byte_op, [c0_op, c0_op], c0_op, is_leader_op)
          b_tiled_copy.load(b_tma_desc_op, b_smem_start_byte_op, [c0_op, c0_op], c0_op, is_leader_op)

          # Wait on the transaction to complete gmem->smem TMA transaction.
          # Wait inside mbarrier try wait parity until the phase is flipped from 0 to 1
          phase_bit_op = arith.constant(T.bool(), 0)
          nvgpu.MBarrierTryWaitParityOp(mbarrier_group_op, phase_bit_op, ticks_op, mbarId=c0_op)
    
          accums = tiled_mma.init_accum_tiles()

          nvvm.WgmmaFenceAlignedOp()
          
          # Issue Tiled Mma (issuing MmaAtom to cover one GEMM-K stage of CTA tile).
          accums = tiled_mma.dot(a_gmma_desc_builder, b_gmma_desc_builder, accums)

          # wgmma.wait.group.sync.aligned : waits until 0 gmma operations are pending
          nvvm.WgmmaWaitGroupSyncOp(0) 

          ## Epilogue Direct store epilogue (For clarity and simplicity in the basic TMA WGMMA kernel)
          warp_id_op = arith.DivUIOp(tidx_op, b.const_index_op(32))
          lane_id_op = arith.RemUIOp(tidx_op, b.const_index_op(32))
          quad_id_op = arith.DivUIOp(lane_id_op, b.const_index_op(4))
          lane_in_quad_id_op = arith.RemUIOp(lane_id_op, b.const_index_op(4))
          
          for m in range(self.tiled_mma.mma_count.m):
            
            accum_vector_op = accums[m]
            
            # Compute (row, col) indices in the accumulator matrix.
            row_idx_op = arith.MulIOp(warp_id_op, b.const_index_op(16))
            row_idx_op = arith.AddIOp(row_idx_op, 
                                      arith.MulIOp(b.const_index_op(m), 
                                                   b.const_index_op(64)))
            row_idx_1_op = arith.AddIOp(row_idx_op, quad_id_op)
            row_idx_2_op = arith.AddIOp(row_idx_1_op, b.const_index_op(8))
      
            for n in range(tiled_mma.accum_shape[0]):
              core_accum_vector_op = vector.ExtractOp(accum_vector_op,
                                                           [],
                                                           [n])

              col_idx_op = arith.AddIOp(arith.MulIOp(lane_in_quad_id_op,
                                                     b.const_index_op(2)), 
                                        arith.MulIOp(b.const_index_op(n),
                                                     b.const_index_op(8)))
              contiguous_accum_vector_op = vector.ExtractOp(core_accum_vector_op, [], [0])
              v1 = vector.extract(contiguous_accum_vector_op, [], [0])
              v2 = vector.extract(contiguous_accum_vector_op, [], [1])
              vector.store(contiguous_accum_vector_op, c_device, [row_idx_1_op, col_idx_op])

              contiguous_accum_vector_op = vector.ExtractOp(core_accum_vector_op, [], [1])
              v1 = vector.extract(contiguous_accum_vector_op, [], [0])
              v2 = vector.extract(contiguous_accum_vector_op, [], [1])
              vector.store(contiguous_accum_vector_op, c_device, [row_idx_2_op, col_idx_op])

          # End of kernel
          gpu.terminator()
            
        # Copy back to host
        t8 = gpu.wait(b.gpu_token_ty, [launch_op])
        t9 = gpu.memcpy(b.gpu_token_ty, [t8], c_host, c_device)
        gpu.dealloc(b.gpu_token_ty, [t8], a_device)
        gpu.dealloc(b.gpu_token_ty, [t8], b_device)
        gpu.wait(b.gpu_token_ty, [t9])
        gpu.dealloc(b.gpu_token_ty, [t9], c_device)
        func.ReturnOp([])
          
    func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

    module.operation.verify()
    return module
