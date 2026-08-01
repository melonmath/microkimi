"""MXFP4 (e2m1 elements, e8m0 per-32 shared scale) dequantization, numpy.

On-disk layout (compressed-tensors 'mxfp4-pack-quantized', as used by Kimi-K3):
  <name>.weight_packed : U8 [rows, cols/2]  - two e2m1 nibbles per byte, low nibble first
  <name>.weight_scale  : U8 [rows, cols/32] - e8m0 exponent byte per 32-element group

Format reference: compressed-tensors MXFP4 (group_size=32, symmetric, float type).
"""
import numpy as np

# e2m1 value table (sign x 8 magnitudes): 0, 0.5, 1, 1.5, 2, 3, 4, 6
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                  -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

def dequant_mxfp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """packed U8 [R, C/2], scale U8 [R, C/32] -> float32 [R, C]"""
    r, half = packed.shape
    cols = half * 2
    idx = np.empty((r, cols), dtype=np.uint8)
    idx[:, 0::2] = packed & 0x0F
    idx[:, 1::2] = packed >> 4
    vals = _E2M1[idx]
    # e8m0: value = 2**(byte - 127); byte 255 = NaN (unused in practice)
    exp = scale.astype(np.int32) - 127
    sc = np.exp2(exp.astype(np.float32))
    vals *= np.repeat(sc, 32, axis=1)
    return vals
