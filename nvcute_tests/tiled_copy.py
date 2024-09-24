import unittest
from absl.testing import parameterized
from pycute.layout import Layout
from nvcute.tiled_copy import TmaCopy
from nvcute.helpers import DataType

class TmaCopyTestCase(parameterized.TestCase):
  @parameterized.named_parameters(
    (# Test arguments
     'K_Major_64x64xf16', Layout((64, 64), (64, 1)), DataType.f16,
     # Expected layouts 
     {'pitchlinear_layout': Layout((64, 64), (1, 64)), 
      'smem_layout'       : Layout((64, 64), (64, 1))}
    ),

    ( # Test arguments
      'MN_Major_64x64xf16', Layout((64, 64), (1, 64)), DataType.f16,
      # Expected layouts
     {'pitchlinear_layout': Layout((64, 64), (1, 64)), 
      'smem_layout'       : Layout((64, 64), (1, 64))}
    ),  
   
    (# Test arguments
    'K_Major_128x64xf16', Layout((128, 64), (64, 1)), DataType.f16,
     # Expected layouts
     {'pitchlinear_layout': Layout((64, 128), (1, 64)), 
      'smem_layout'       : Layout((128, 64), (64, 1))}
    ),
    
    (# Test arguments
      'MN_Major_128x64xf16', Layout((128, 64), (1, 128)), DataType.f16,
     # Expected layouts
     {'pitchlinear_layout': Layout(((64, 2), 64), ((1, 4096), 64)),
      'smem_layout': Layout(((64, 2), 64), ((1, 4096), 64))}
    ),
  ) 
  def test_gmma_desc(self, smem_shape, dtype, expected):
    tma_copy_traits = TmaCopy(smem_shape, dtype)

    # Debug prints
    '''
    print(f"\ntma_copy_traits.box_shape = {tma_copy_traits.box_shape}")
    print(f"tma_copy_traits.pitchlinear_layout = {tma_copy_traits.pitchlinear_layout}")
    print(f"tma_copy_traits.smem_layout = {tma_copy_traits.smem_layout}")
    '''
    
    self.assertEqual(tma_copy_traits.pitchlinear_layout, expected['pitchlinear_layout'])
    self.assertEqual(tma_copy_traits.smem_layout, expected['smem_layout'])

if __name__ == "__main__":
  unittest.TestLoader.sortTestMethodsUsing = None
  unittest.main(verbosity=2)
