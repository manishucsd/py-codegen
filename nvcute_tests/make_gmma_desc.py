import unittest
import numpy as np
from absl.testing import parameterized
from pycute.layout import Layout
from nvcute.helpers import DataType
from nvcute.tiled_copy import TmaCopy
from nvcute.tiled_mma import make_gmma_desc


class GmmaDescTestCase(parameterized.TestCase):
  @parameterized.named_parameters(
    ('GmmaDesc_Kmajor_smem_shape_64x64xf16', Layout((64, 64), (64, 1)), DataType.f16, 0x4000004000010000),
    ('GmmaDesc_Kmajor_smem_shape_128x64xf16', Layout((128, 64), (64, 1)), DataType.f16, 0x4000004000010000),
    ('GmmaDesc_MNmajor_smem_shape_128x64xf16', Layout((128, 64), (1, 128)), DataType.f16, 0x4000004002000000),
    #('GmmaDesc_Kmajor_smem_shape_64x32xf16', Layout((64, 32), (32, 1)), DataType.f16, 0x8000002000010000), # See note (1) below
    ('GmmaDesc_MNmajor_smem_shape_128x32xf16', Layout((128, 32), (1, 128)), DataType.f16, 0x4000004001000000),
  ) 
  def test_gmma_desc(self, smem_shape, dtype, expected_desc):
    tma_copy_traits = TmaCopy(smem_shape, dtype)
    desc = make_gmma_desc(tma_copy_traits)
    # Debug prints
    '''
    print(f"\ntma_copy_traits.swizzle = {tma_copy_traits.swizzle}")
    print(f"tma_copy_traits.smem_layout (MN-mode, K-mode) = {tma_copy_traits.smem_layout}")
    print(f"desc.data (in hex): {desc.data:#x}")
    print(f"expected_desc: {expected_desc:#x}")
    '''
    self.assertEqual(np.uint64(desc.data), np.uint64(expected_desc))

if __name__ == "__main__":
  unittest.main(verbosity=2)


'''
# GMMA descriptor notes
1. The case GmmaDesc_Kmajor_smem_shape_64x32xf16 generate the correct desc bits 
   0x8000002000010000 but arith.constantop doesn't handle in the unsigned 64-bit integer.

'''