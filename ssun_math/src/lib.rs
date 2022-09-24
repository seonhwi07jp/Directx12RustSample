#![feature(portable_simd)]
#![allow(dead_code)]

mod float3;
mod float4;
mod matrix;

pub use float3::*;
pub use float4::*;
pub use matrix::*;

#[macro_export]
macro_rules! check_sse2 {
    () => {
        cfg!(target_feature = "sse2") && !cfg!(feature = "no_simd")
    };
}

#[macro_export]
macro_rules! swizzle {
    ($fp0:expr, $fp1:expr, $fp2:expr, $fp3:expr) => {
        (($fp3 << 6) | ($fp2 << 4) | ($fp1 << 2) | ($fp0))
    };
}

const PI: f32 = 3.1415926535;
const TWO_PI: f32 = 2.0 * PI;
const FOUR_PI: f32 = 4.0 * PI;
const PI_OVER_180: f32 = PI / 180.0;
const INV_PI: f32 = 1.0 / PI;
const INV_2PI: f32 = 1.0 / TWO_PI;
const INV_4PI: f32 = 1.0 / FOUR_PI;
const INV_PI_OVER_180: f32 = 180.0 / PI;

#[inline(always)]
pub fn radian(deg: f32) -> f32 { deg * PI_OVER_180 }

#[inline(always)]
pub fn degree(rad: f32) -> f32 { rad * INV_PI_OVER_180 }
