from mlir.dialects import nvgpu
from pycute.layout import Layout, zipped_divide, logical_divide
from codegen.nvgpu_builder import GmmaDescriptorBitfield
from codegen.utils import TmaSwizzleToGmmaSwizzle, GmmaType
from codegen.gemm import GemmShape
from nvcute.helpers import DataType, DataTypeName, sizeof, stride
from nvcute.tiled_copy import TmaCopy


###################################################################################
#                Functions to make GMMA descriptor
###################################################################################
def make_gmma_desc(copy_traits: TmaCopy) -> GmmaDescriptorBitfield:
  """Make GMMA descriptor compile-time bits given a shared memory copy traits."""
  
  # TODO: handle datatype (f8, f16, f32)
  smem_core_matrix_contiguous_elem = None
  if copy_traits.swizzle == nvgpu.TensorMapSwizzleKind.SWIZZLE_128B:
    smem_core_matrix_contiguous_elem = 8
  elif copy_traits.swizzle == nvgpu.TensorMapSwizzleKind.SWIZZLE_64B:
    smem_core_matrix_contiguous_elem = 4
  elif copy_traits.swizzle == nvgpu.TensorMapSwizzleKind.SWIZZLE_32B:
    smem_core_matrix_contiguous_elem = 2
  else:
    raise ValueError(f"Unsupported swizzle kind {copy_traits.swizzle}")
  
  # TODO: (8, 1) in strided is fixed, but (8, 1) in contiguous is a function of 
  # dtype and swizzle. Swizzle is handled above, we need to handle dtype.
  core_matrix_tile = (Layout(smem_core_matrix_contiguous_elem,1), Layout(8,1))

  # Divide the smem layout by the fundamental tensor core tile
  core_matrix_smem_layout = logical_divide(copy_traits.pitchlinear_layout, core_matrix_tile)

  leading_offset = stride(core_matrix_smem_layout, (0,1))
  if isinstance(leading_offset, tuple):
    leading_offset = int(leading_offset[-1])
  stride_offset = stride(core_matrix_smem_layout, (1,1))

  leading_byte_offset = int(leading_offset * sizeof(copy_traits.dtype))
  stride_byte_offset = int(stride_offset * sizeof(copy_traits.dtype))

  desc = GmmaDescriptorBitfield()
  desc.leading_byte_offset = leading_byte_offset
  desc.stride_byte_offset = stride_byte_offset  
  desc.base_offset = 0
  # Match the gmma swizzle to tma swizzle
  desc.swizzle = TmaSwizzleToGmmaSwizzle[copy_traits.swizzle].value
  
  # Debug prints
  '''  
  print(f"copy_traits.pitchlinear_layout : {copy_traits.pitchlinear_layout}")
  print(f"copy_traits.smem_layout : {copy_traits.smem_layout}")
  print(f"core_matrix_tile : {core_matrix_tile}")
  print(f"core_matrix_smem_layout : {core_matrix_smem_layout}")
  print(f"leading_offset : {leading_offset}")
  print(f"stride_offset : {stride_offset}")
  print(f"leading_byte_offset : {leading_byte_offset}")
  print(f"stride_byte_offset : {stride_byte_offset}")
  print(f"gmma_desc_bitfield = {desc.data:#0{16}x}")
  '''
  return desc
###################################################################################

# WGMMA accumulator layout dictionary 
# Accum-Tile (M, N) -> Accum-Layout (pycute.layout.Layout)
WgmmaAccumulatorLayoutDict = {
  (64, 64)  : Layout(((4,8,4), (2,2, 8)), ((128,1,16), (64,8,512))),
  (64, 128) : Layout(((4,8,4), (2,2,16)), ((128,1,16), (64,8,512))),
  (64, 256) : Layout(((4,8,4), (2,2,32)), ((128,1,16), (64,8,512))),
}
###################################################################################
#                Classes to represent MMA Atom and Tiled MMA
###################################################################################
class MmaAtom:
  def __init__(self, mma_shape : GemmShape,
               element_a : DataType, 
               element_b : DataType, 
               element_c : DataType,
               layout_a : nvgpu.MatrixLayoutKind = nvgpu.MatrixLayoutKind.ROW,
               layout_b : nvgpu.MatrixLayoutKind = nvgpu.MatrixLayoutKind.COL,
               type : GmmaType = GmmaType.SS) -> None:
    self.mma_shape = mma_shape
    self.element_a = element_a
    self.element_b = element_b
    self.element_c = element_c
    self.layout_a = layout_a
    self.layout_b = layout_b
    self.layout_c = WgmmaAccumulatorLayoutDict.get((self.mma_shape.m, 
                                                    self.mma_shape.n))
    self.type = type
  
  @property
  def trans_a(self) -> bool:
    """Transpose mma operand A if it is column major."""
    return self.layout_a == nvgpu.MatrixLayoutKind.COL
  
  @property
  def trans_b(self) -> bool:
    """Transpose mma operand B if it is row major."""
    return self.layout_b == nvgpu.MatrixLayoutKind.ROW

  @property
  def tiler_a(self) -> Layout:
    """Tiler for operand A (M-mode, K-mode)."""
    return (Layout(self.mma_shape.m, 1), Layout(self.mma_shape.k, 1))
  
  @property
  def tiler_b(self) -> Layout:
    """Tiler for operand B (N-mode, K-mode)."""
    return (Layout(self.mma_shape.n, 1), Layout(self.mma_shape.k, 1))
  
  def __str__(self) -> str:
    """Unique name for the MMA atom."""
    return f"Gmma_{self.mma_shape}_{DataTypeName[self.element_a]}_{DataTypeName[self.element_b]}_"\
           f"{DataTypeName[self.element_c]}_{self.layout_a}_{self.layout_b}_{self.type}"
###################################################################################

class TiledMma:
  """Tiles the MmaAtom on operand A/B and accumulator C tiles of shared memory tile."""
  def __init__(self, mma_atom : MmaAtom, 
                     a_smem_traits : TmaCopy,
                     b_smem_traits : TmaCopy):
    self.mma_atom = mma_atom
    self.element_a = mma_atom.element_a
    self.element_b = mma_atom.element_b
    self.element_c = mma_atom.element_c
    self.a_smem_traits = a_smem_traits
    self.b_smem_traits = b_smem_traits

    ## Derived values
    self.atom_on_tile_a = zipped_divide(self.a_smem_traits.smem_layout, self.mma_atom.tiler_a)
    self.atom_on_tile_b = zipped_divide(self.b_smem_traits.smem_layout, self.mma_atom.tiler_b)
  
  @property
  def desc_a(self) -> GmmaDescriptorBitfield:
    return make_gmma_desc(self.a_smem_traits)
  
  @property
  def desc_b(self) -> GmmaDescriptorBitfield:
    return make_gmma_desc(self.b_smem_traits)
  
  @property
  def mma_count(self) -> int:
    """Number of MMA atoms in m-, n-, k-dimension to cover CTA-tile."""
    a_mma_count = self.atom_on_tile_a.shape[1] # (M, K)
    b_mma_count = self.atom_on_tile_b.shape[1] # (N, K)
    if a_mma_count[1] != b_mma_count[1]:
      raise ValueError(f"MMA atom count mismatch in K-dimension: {a_mma_count[1]} != {b_mma_count[1]}")
    return GemmShape(a_mma_count[0], b_mma_count[0], a_mma_count[1]) # (M, N, K)

  @property
  def advance_k_byte_a(self) -> int:
    """Advance value in number of bytes per K-group in smem A tile."""
    offset = self.atom_on_tile_a.stride[1][1]
    byte_offset = offset * sizeof(self.a_smem_traits.dtype)
    return byte_offset
  
  @property
  def advance_k_byte_b(self) -> int:
    """Advance value in number of bytes per K-group in smem B tile."""
    offset = self.atom_on_tile_b.stride[1][1]
    byte_offset = offset * sizeof(self.b_smem_traits.dtype)
    return byte_offset
  
  @property
  def advance_m_byte_a(self) -> int:
    """Advance value in number of bytes per M-group in smem A tile."""
    offset = self.atom_on_tile_a.stride[1][0]
    byte_offset = offset * sizeof(self.a_smem_traits.dtype)
    return byte_offset
  
  @property
  def advance_n_byte_b(self) -> int:
    """Advance value in number of bytes per N-group in smem B tile."""
    offset = self.atom_on_tile_b.stride[1][0]
    byte_offset = offset * sizeof(self.b_smem_traits.dtype)
    return byte_offset