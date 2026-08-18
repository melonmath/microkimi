//! IEEE half-precision conversions (round to nearest even), scalar and
//! dependency-free; used where a backend stores f16 scales.

pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp8 = (b >> 23) & 0xff;
    let man = b & 0x007f_ffff;
    if exp8 == 0xff {
        return sign | if man != 0 { 0x7e00 } else { 0x7c00 }; // qNaN / inf
    }
    let e = exp8 as i32 - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow -> signed zero
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let mid = 1u32 << (shift - 1);
        let mut h = half as u16;
        if rem > mid || (rem == mid && (half & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let half = (man >> 13) as u16;
    let rem = man & 0x1fff;
    let mut h = sign | ((e as u16) << 10) | half;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        h = h.wrapping_add(1); // a carry into the exponent rounds up correctly
    }
    h
}

#[allow(dead_code)]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 113u32; // 127 - 15 + 1
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trips_and_rounds_to_nearest_even() {
        for &x in &[0.0f32, 1.0, -2.5, 65504.0, 6.1e-5, 1e-6, 0.333_251_95, 1e-3] {
            let h = f32_to_f16(x);
            let back = f16_to_f32(h);
            assert!((back - x).abs() <= x.abs() * 1e-3 + 1e-7, "{x} -> {h:#x} -> {back}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        // ties to even: 1 + 2^-11 lies exactly between two halves -> even
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
    }
}
