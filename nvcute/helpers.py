import enum
from enum import auto as enum_auto
from pycute.layout import Layout

class DataType(enum.Enum):
  void = enum_auto()
  u8   = enum_auto()
  s8   = enum_auto()
  e4m3 = enum_auto()
  e5m2 = enum_auto()
  f16  = enum_auto()
  bf16 = enum_auto()
  f32  = enum_auto()
  tf32 = enum_auto()
  f64  = enum_auto()
  u128 = enum_auto()
  invalid = enum_auto()

DataTypeSizeofBits = {
  DataType.void : 0,
  DataType.u8   : 8,
  DataType.s8   : 8,
  DataType.e4m3: 8,
  DataType.e5m2: 8,
  DataType.f16: 16,
  DataType.bf16: 16,
  DataType.f32: 32,
  DataType.tf32: 32,
  DataType.f64: 64,
  DataType.u128: 128,
}

DataTypeName = {
  DataType.void : "void",
  DataType.u8   : "u8",
  DataType.s8   : "s8",
  DataType.e4m3: "e4m3",
  DataType.e5m2: "e5m2",
  DataType.f16: "f16",
  DataType.bf16: "bf16",
  DataType.f32: "f32",
  DataType.tf32: "tf32",
  DataType.f64: "f64",
  DataType.u128: "u128",
  DataType.invalid: "invalid",
}

# Returns value in bytes
def sizeof(dtype : DataType):
  return DataTypeSizeofBits[dtype] // 8

def depth(tup : tuple):
  if isinstance(tup, tuple):  # Check if the input is a tuple
    max_depth = 0
    for item in tup:
      d = depth(item)  # Recursively find depth for each item
      max_depth = max(max_depth, d)
    return max_depth + 1  # Add 1 for the current tuple level
  else:
      return 0

def stride(layout : Layout, indices : tuple) -> int:
  search_ = layout.stride
  iter_count = 0
  for i in indices:
    search_ = search_[i]
  return search_

def rank(var : tuple):
  if isinstance(var, tuple):
    return len(var)
  elif isinstance(var, Layout):
    return len(var.shape)
  else:
    assert False, "Object not a tuple"

def is_MN_major_layout(layout : Layout) -> bool:
  assert rank(layout.stride) == 2 and depth(layout.stride) == 1
  if layout.stride[0] == 1:
    return True