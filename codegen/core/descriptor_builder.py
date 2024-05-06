from mlir.dialects import arith
from mlir.dialects import func
from mlir.dialects import gpu
from mlir.dialects import memref
from mlir.dialects import nvgpu
from mlir.dialects import nvvm
from mlir.dialects import llvm
from mlir.dialects import builtin
from mlir.dialects import scf
from mlir.dialects import vector
from mlir.extras import types as T
from codegen.utils import *
from codegen.core.builder import Builder

######################################################################
# A class that builds a TMA descriptor.
######################################################################
class TmaDescriptorBuilder:
  """A class that builds a TMA descriptor."""

  def __init__(self, swizzle, l2promo, oob, interleave, 
               tma_box_shape, memref_ty):
    self.swizzle = swizzle       # mlir.nvgpu.TensorMapSwizzleKind
    self.l2promo = l2promo       # mlir.nvgpu.TensorMapL2PromoKind
    self.oob = oob               # mlir.nvgpu.TensorMapOOBKind
    self.interleave = interleave # mlir.nvgpu.TensorMapInterleaveKind
    self.tma_box_shape = tma_box_shape
    self.memref_ty = memref_ty   # MemRefType

  @property
  def tensormap_descriptor_ty(self):
    """Returns a tensormap descriptor type."""
    memref_str = f"memref<{self.tma_box_shape[0]}x{self.tma_box_shape[1]}x{self.memref_ty.element_type}, 3>"
    parse_str = f"!nvgpu.tensormap.descriptor<tensor = {memref_str},\
                                              swizzle = {self.swizzle},\
                                              l2promo = {self.l2promo},\
                                              oob = {self.oob},\
                                              interleave = {self.interleave}>"
    return ir.Type.parse(parse_str)

  def tma_descriptor_op(self, device_ptr):
    """Returns a tensormap descriptor op."""
    b = Builder()
    tma_descriptor_ty = self.tensormap_descriptor_ty
    device_unranked_memref = memref.CastOp(
       ir.UnrankedMemRefType.get(self.memref_ty.element_type, self.memref_ty.memory_space), device_ptr)
    tma_descriptor_op = nvgpu.TmaCreateDescriptorOp(
        tma_descriptor_ty, device_unranked_memref, map(b.const_index_op,
                                                      self.tma_box_shape))
    return tma_descriptor_op.result


class Bitfield:
  """A class that encodes a bitfield."""
  def __init__(self):
    self.data = 0

  def _get_bits(self, offset, length):
    mask = (1 << length) - 1
    return (self.data >> offset) & mask

  def _set_bits(self, offset, length, value):
    mask = (1 << length) - 1
    self.data &= ~(mask << offset)
    self.data |= (value & mask) << offset

######################################################################
#       A classes for buidling and advancing WGMMA descriptor.
######################################################################    
class GmmaDescriptorBitfield:
  """A class that encodes the bitfield a Warp Group MMA (WGMMA) descriptor.
     The WGMMA descriptor is a 64-bit value that is encoded as follows:
     | 63 ... 62 |  51 ... 49  | 45 ... 32 | 29 ... 16 | 13 ... 0 |
     |  swizzle  | base_offset |   stride  |  leading  |  start   |
     |           |             |    byte   |   byte    | address  |
     |           |             |   offset  |   offset  |          |
     Note that this class has nothing to do with mlir and codegen.   
  """

  def __init__(self):
    """Underlying bitfield is initialized to zero."""
    self.bitfield = Bitfield()

  def __str__(self):
    """Returns a string representation of the descriptor in hexa-decimal format."""
    return f"{self.bitfield.data:#0{16}x}"
  
  @property
  def start_address(self):
    return self.bitfield._get_bits(0, 14) << 4

  @start_address.setter
  def start_address(self, value):
    self.bitfield._set_bits(0, 14, (value >> 4))

  @property
  def leading_byte_offset(self):
    return self.bitfield._get_bits(16, 14) << 4

  @leading_byte_offset.setter
  def leading_byte_offset(self, value):
    self.bitfield._set_bits(16, 14, (value >> 4))

  @property
  def stride_byte_offset(self):
    return self.bitfield._get_bits(32, 14) << 4

  @stride_byte_offset.setter
  def stride_byte_offset(self, value):
    self.bitfield._set_bits(32, 14, (value >> 4)) 

  @property
  def base_offset(self):
    return self.bitfield._get_bits(49, 3)

  @base_offset.setter
  def base_offset(self, value):
    self.bitfield._set_bits(49, 3, value)

  @property
  def swizzle(self):
    return self.bitfield._get_bits(62, 2)

  @swizzle.setter
  def swizzle(self, value):
    self.bitfield._set_bits(62, 2, value)

  @property
  def data(self):
    return self.bitfield.data
  
  @data.setter
  def data(self, value):
    self.bitfield.data = value
######################################################################


class GmmaDescriptorBuilder:
  """A class that builds a WGMMA descriptor."""

  def __init__(self):
    self.builder = Builder()
    # create descriptor bitfield
    self.desc_bitfield = GmmaDescriptorBitfield()
    self._desc_op = None

  @classmethod
  def from_bitfield(cls, desc_bitfield):
    """Create a GMMA descriptor from an instance of GmmaDescriptorBitfield."""
    obj = cls()
    # Compile-time bits are set.
    obj.desc_bitfield = desc_bitfield
    return obj
  
  @classmethod
  def from_field_values(cls, leading_byte_offset, stride_byte_offset, 
                            base_offset, swizzle):
    """Create a GMMA descriptor from individual desc fields."""
    obj = cls()
    # Compile-time fields are set.
    obj.desc_bitfield.leading_byte_offset = leading_byte_offset
    obj.desc_bitfield.stride_byte_offset = stride_byte_offset
    obj.desc_bitfield.base_offset = base_offset
    obj.desc_bitfield.swizzle = swizzle
    return obj

  def init_compile_time_bits(self):
    """Set the compile-time bits of the GMMA descriptor."""
    desc_bitfield_const_op = self.builder.const_int_op(self.desc_bitfield.data, 
                                                       self.builder.i64_ty)
    self._desc_op = llvm.OrOp(self._desc_op, desc_bitfield_const_op)

  def set_smem_addr(self, smemref):
    """Initialize GMMA descriptor takes the smemref for smem addr
    and additionaly set the compile-time bits using bitfield."""
    # Runtime smem address bits are set in desc_op.
    smemref_ty = ir.MemRefType(smemref.type)
    underlying_llvm_struct_ty = \
      self.builder.underlying_llvm_struct_ty(smemref)
    underlying_llvm_struct_op = builtin.UnrealizedConversionCastOp(
      [underlying_llvm_struct_ty], [smemref])
    smem_aligned_ptr_op = llvm.extractvalue(self.builder.llvm_smem_ptr_ty, 
                                            underlying_llvm_struct_op, [1])
    offset_elements_op = llvm.extractvalue(self.builder.i64_ty, 
                                         underlying_llvm_struct_op, [2])
                                         
    size_in_bytes = get_size_in_bits(smemref_ty.element_type) // 8
    size_in_bytes_op = self.builder.const_int_op(size_in_bytes,
                                                 self.builder.i64_ty)
    offset_bytes_op = llvm.mul(offset_elements_op, size_in_bytes_op,
                               overflow_flags=llvm.IntegerOverflowFlags.none)
    smem_ptr_value_op = llvm.add(llvm.ptrtoint(self.builder.i64_ty, 
                                         smem_aligned_ptr_op),
                                  offset_bytes_op,
                                  overflow_flags=llvm.IntegerOverflowFlags.none)
    # Create 18 bit mask.
    mask_op = self.builder.const_int_op(0x3FFFF, self.builder.i64_ty)
    c4_op = self.builder.const_int_op(4, self.builder.i64_ty)
    self._desc_op = llvm.LShrOp(llvm.AndOp(smem_ptr_value_op, mask_op), c4_op)
    
    # Set the non smem fields which are zero-ed above with smem addr setting.
    self.init_compile_time_bits()
    return self
  
  def advance_op(self, byte_offset):
    """Advance the smem address bits of the GMMA descriptor by `byte_offset`."""
    byte_offset = byte_offset >> 4 # ignore the lower 4 bits
    byte_offset_op = self.builder.const_int_op(byte_offset, self.builder.i64_ty)
    self._desc_op = llvm.AddOp(self._desc_op, byte_offset_op,
                               overflowFlags=llvm.IntegerOverflowFlags.none)
    return self._desc_op

  @property
  def desc_op(self):
    """Returns the GMMA descriptor op."""
    if self._desc_op is None:
      raise ValueError("`set_smem_addr` first and pass smemref for runtime addr.")
    return self._desc_op
  
  @property
  def desc_bits(self):
    """Returns the compile-time bits of the GMMA descriptor for inspection."""
    return self.desc_bitfield.data
