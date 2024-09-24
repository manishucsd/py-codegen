import unittest
from absl.testing import parameterized
from mlir.dialects import nvgpu
from pycute.layout import Layout
from codegen.gemm import GemmShape
from nvcute.tiled_copy import TmaCopy
from nvcute.tiled_mma import MmaAtom, TiledMma
from nvcute.helpers import DataType

class TiledMmaTestCase(parameterized.TestCase):
  @parameterized.named_parameters(
    (
     # Test arguments
     'Cta_64x128x64_Wgmma_64x128x16_A_f16t_B_f16t_C_f32', 
     GemmShape(64, 128, 64), GemmShape(64, 128, 16), 
     DataType.f16, DataType.f16, DataType.f32, 
     nvgpu.MatrixLayoutKind.ROW, nvgpu.MatrixLayoutKind.ROW,
     # Expected values
     {'mma_count' : GemmShape(1, 1, 4), 
      'advance_k_byte_a' : 32, 'advance_k_byte_b' : 2048, 
      'advance_m_byte_a' : 0, 'advance_n_byte_b' : 0}
    ),
    (
    # Test arguments
    'Cta_128x128x64_Wgmma_64x128x16_A_f16t_B_f16t_C_f32', 
     GemmShape(128, 128, 64), GemmShape(64, 128, 16), 
     DataType.f16, DataType.f16, DataType.f32, 
     nvgpu.MatrixLayoutKind.ROW, nvgpu.MatrixLayoutKind.ROW,
     # Expected values
     {'mma_count' : GemmShape(2, 1, 4), 
      'advance_k_byte_a' : 32, 'advance_k_byte_b' : 2048, 
      'advance_m_byte_a' : 8192, 'advance_n_byte_b' : 0}),
    )
  def test_tiled_mma(self, cta_shape, wgmma_shape, element_a, element_b, element_c, layout_a, layout_b, expected):
    a_smem_shape = (cta_shape.m, cta_shape.k) 
    a_smem_stride = (cta_shape.k, 1) if layout_a == nvgpu.MatrixLayoutKind.ROW else (1, cta_shape.m)
    b_smem_shape = (cta_shape.n, cta_shape.k)
    b_smem_stride = (cta_shape.k, 1) if layout_b == nvgpu.MatrixLayoutKind.COL else (1, cta_shape.n)
    a_smem_traits = TmaCopy(Layout(a_smem_shape, a_smem_stride), element_a)
    b_smem_traits = TmaCopy(Layout(b_smem_shape, b_smem_stride), element_b)
    mma_atom = MmaAtom(wgmma_shape, element_a, element_b, element_c, layout_a, layout_b)
    tiled_mma = TiledMma(mma_atom, a_smem_traits, b_smem_traits)
    
    '''
    print("\n")
    print(f"tiled_mma.mma_atom.layout_c: {tiled_mma.mma_atom.layout_c}")
    print(f"tiled_mma.atom_on_tile_a: {tiled_mma.atom_on_tile_a}")
    print(f"tiled_mma.atom_on_tile_b: {tiled_mma.atom_on_tile_b}")
    print(f"tiled_mma.advance_kgroup_byte_a: {tiled_mma.advance_k_byte_a}")
    print(f"tiled_mma.advance_kgroup_byte_b: {tiled_mma.advance_k_byte_b}")
    print(f"tiled_mma.advance_mgroup_byte_a: {tiled_mma.advance_m_byte_a}")
    print(f"tiled_mma.advance_ngroup_byte_b: {tiled_mma.advance_n_byte_b}")
    print(f"tiled_mma.mma_count: {tiled_mma.mma_count}")
    '''
    
    self.assertEqual(tiled_mma.mma_count, expected['mma_count'])
    self.assertEqual(tiled_mma.advance_k_byte_a, expected['advance_k_byte_a'])
    self.assertEqual(tiled_mma.advance_k_byte_b, expected['advance_k_byte_b'])
    self.assertEqual(tiled_mma.advance_m_byte_a, expected['advance_m_byte_a'])
    self.assertEqual(tiled_mma.advance_n_byte_b, expected['advance_n_byte_b'])

if __name__ == "__main__":
  unittest.TestLoader.sortTestMethodsUsing = None
  unittest.main(verbosity=2)
