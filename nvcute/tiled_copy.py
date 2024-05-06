from mlir.dialects import nvgpu
from pycute.layout import Layout, coalesce

from codegen.utils import TmaSwizzleByteDict
from codegen.gemm import PitchLinearShape
from nvcute.helpers import DataType, is_MN_major_layout, sizeof

class TmaCopy:
  """Details for TMA copy."""
  # TODO: In the the final API TmaCopy should serve as an instance of TiledCopy.
  # We can implement Tiled Copy using TmaCopy (Sm90) or AsyncCopy (Sm80). 
  def __init__(self, canonical_layout: Layout, dtype: DataType):
    """Initialize TmaCopy with (M, K) or (N, K) Layout and element dtype."""
    # Canonical layout with shape of (M, K) or (N, K)
    self.canonical_layout = canonical_layout 
    self.dtype = dtype

    ## Properties
    self._swizzle = None
    self._num_boxes = None
    self._box_shape = None
    
    # pitch-linear shape and layout (contiguous-mode, strided-mode)
    self._pitchlinear_shape = None
    self._pitchlinear_layout = None
  
  @property
  def swizzle(self):
    """Returns the swizzle kind for TMA copy (We match it for GmmaSwizzle)."""
    # We use pitchlinear shape/layout for internal calculations of this function 
    # as it is easier to read the restrictions placed on the swizzle kind on
    # the contiguous dimension. Note that everywere else we use smem_layout.
    # smem_layout is always (MN-mode, K-mode). 
    if self._swizzle is None:
      contiguous_bytes = self.pitchlinear_shape.contiguous * sizeof(self.dtype)
      if contiguous_bytes % 128 == 0:
        self._swizzle = nvgpu.TensorMapSwizzleKind.SWIZZLE_128B
      elif contiguous_bytes % 64 == 0:
        self._swizzle = nvgpu.TensorMapSwizzleKind.SWIZZLE_64B
      elif contiguous_bytes % 32 == 0:
        self._swizzle = nvgpu.TensorMapSwizzleKind.SWIZZLE_32B
      else:
        raise ValueError(f"Unsupported smem shape {self.pitchlinear_shape} for TMA copy")
    return self._swizzle
  
  @property
  def num_boxes(self) -> int:
    if self._num_boxes is None:
      elem_contiguous = TmaSwizzleByteDict[self.swizzle] // sizeof(self.dtype)
      self._num_boxes = self.pitchlinear_shape.contiguous // elem_contiguous
    return self._num_boxes
  
  @property
  def box_shape(self) -> PitchLinearShape:
    """Returns the shape of a box in TMA copy as PitchLinearShape."""
    if self._box_shape is None:
      elem_contiguous = TmaSwizzleByteDict[self.swizzle] // sizeof(self.dtype)
      self._box_shape = PitchLinearShape(elem_contiguous, self.pitchlinear_shape.strided)
    return self._box_shape

  @property
  def box_bytes(self) -> int:
    return self.box_shape.size() * sizeof(self.dtype)
  
  @property
  def pitchlinear_shape(self) -> PitchLinearShape:
    if self._pitchlinear_shape is None:
      self._pitchlinear_shape = \
        PitchLinearShape(self.canonical_layout.shape[0], self.canonical_layout.shape[1]) \
          if is_MN_major_layout(self.canonical_layout) \
          else PitchLinearShape(self.canonical_layout.shape[1], self.canonical_layout.shape[0])
    return self._pitchlinear_shape
  
  @property
  def pitchlinear_layout(self) -> Layout:
    """Returns the pitchlinear layout for shared memory (contiguous mode, strided mode)."""
    if self._pitchlinear_layout is None:
      shape = ((self.box_shape.contiguous, self.num_boxes), self.box_shape.strided)
      stride = ((1, self.box_shape.size()), self.box_shape.contiguous)
      self._pitchlinear_layout = coalesce(Layout(shape, stride), (1,1))
    return self._pitchlinear_layout

  @property
  def is_MN_major(self) -> bool:
    return is_MN_major_layout(self.canonical_layout)
  
  @property
  def smem_layout(self) -> Layout:
    """Returns the final layout for shared memory (MN-mode, K-mode) after TMA copy."""
    if self.is_MN_major:
      return self.pitchlinear_layout
    # Swap the modes for K-major layout to always return (MN-mode, K-mode).
    shape = (self.pitchlinear_layout.shape[1], self.pitchlinear_layout.shape[0])
    stride = (self.pitchlinear_layout.stride[1], self.pitchlinear_layout.stride[0])
    return Layout(shape, stride)
