from codegen.utils import get_size_in_bits, ShortMainloopVariantName

class PitchLinearShape:
  def __init__(self, c, s) -> None:
    self.contiguous = c
    self.strided = s
  def size(self) -> int:
    return self.contiguous * self.strided
  def __str__(self) -> str:
    return f"({self.contiguous}, {self.strided})"

class GemmShape:
  """
  A class for commonly occuring tuple of 3 elements (m,n,k) in GEMM operations 
  describing problem shape, cga shape, cta shape, warp shape, instruction shape.
  """

  def __init__(self, m, n, k):
    self.m = m
    self.n = n
    self.k = k
  
  def size(self) -> int:
    return self.m * self.n * self.k
  
  def __eq__(self, other):
    return self.m == other.m and self.n == other.n and self.k == other.k
  
  def __str__(self):
    return f"m{self.m}n{self.n}k{self.k}"
######################################################################


class GemmDescription:
  """
  A class that describes a compile-time GEMM operation.
  """

  def __init__(self, a, b, c, accum_ty,
               problem_shape, cga_shape, cta_shape, warp_shape, mma_shape, 
               num_stages, mainloop_variant, sm_arch):
    # Initialize the properties that describes a GEMM kernel at compile-time.
    self.a = a                               # TensorDescription
    self.b = b                               # TensorDescription
    self.c = c                               # TensorDescription
    self.accum_ty = accum_ty                 # mlir built-in type
    self.problem_shape = problem_shape       # GemmShape (in number of elements)
    self.cga_shape = cga_shape               # GemmShape (in number of cta)
    self.cta_shape = cta_shape               # GemmShape (in number of elements)
    self.warp_shape = warp_shape             # GemmShape (in number of warps)
    self.instruction_shape = mma_shape       # GemmShape (in number of elements)
    self.num_stages = num_stages             # int
    self.mainloop_variant = mainloop_variant # MainloopVariant enum
    self.sm_arch = sm_arch                   # SmArchTag enum

    # Derive additional properties of the GEMM from the properties above.
    self.a_bitwidth = get_size_in_bits(self.a.datatype)
    self.b_bitwidth = get_size_in_bits(self.b.datatype)
    self.c_bitwidth = get_size_in_bits(self.c.datatype)

    # Compute shared memory size for A and B in bytes.
    self.a_smem_bytes_per_stage = (self.cta_shape.m * self.cta_shape.k * self.a_bitwidth) // 8
    self.b_smem_bytes_per_stage = (self.cta_shape.k * self.cta_shape.n * self.b_bitwidth) // 8
    self.smem_bytes_per_stage = self.a_smem_bytes_per_stage + self.b_smem_bytes_per_stage
    self.a_smem_bytes = self.a_smem_bytes_per_stage * self.num_stages
    self.b_smem_bytes = self.b_smem_bytes_per_stage * self.num_stages

  @property
  def smem_bytes(self):
    """Returns the total shared memory size in bytes."""
    smem_bytes_per_sm = self.smem_bytes_per_stage * self.num_stages
    sm90_smem_capacity_per_sm = 227  # in KB
    if smem_bytes_per_sm > sm90_smem_capacity_per_sm * 1024:
      raise ValueError("Shared memory size exceeds the capacity of the SM.")
    return self.smem_bytes_per_stage * self.num_stages

  def name(self):
    """Returns a name for the GEMM description."""
    return "sm90_gemm_%s_a_%s_b_%s_c_%s_cga%dx%dx%d_cta%dx%dx%d_"\
             "warp%dx%dx%d_stages%d_wgmma%dx%dx%d_accum%s" % (
               ShortMainloopVariantName[self.mainloop_variant], 
               self.a.name(), self.b.name(), self.c.name(),
               self.cga_shape.m, self.cga_shape.n, self.cga_shape.k,
               self.cta_shape.m, self.cta_shape.n, self.cta_shape.k,
               self.warp_shape.m, self.warp_shape.n, self.warp_shape.k,
               self.num_stages,
               self.instruction_shape.m, self.instruction_shape.n, self.instruction_shape.k,
               self.accum_ty)

######################################################################