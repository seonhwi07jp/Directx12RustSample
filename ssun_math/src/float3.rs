use std::{
    arch::x86_64::{__m128, _mm_mul_ps, _mm_set_ps, _mm_shuffle_ps, _mm_sub_ps},
    mem::transmute_copy,
    ops::{Div, DivAssign},
};

use crate::{check_sse2, swizzle, Float4, Normal4, Point4, Raw4, Vector4};

/* -------------------------------------------------------------------------- */
/*                                   macros                                   */
/* -------------------------------------------------------------------------- */
#[macro_export]
macro_rules! Float3 {
    ($x:expr, $y:expr, $z:expr) => {
        Float3 { x: $x, y: $y, z: $z }
    };

    ($val:expr) => {
        Float3 {
            x: $val,
            y: $val,
            z: $val,
        }
    };
}

/* -------------------------------------------------------------------------- */
/*                                struct Float3                               */
/* -------------------------------------------------------------------------- */
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Float3 {
    pub const ZERO: Float3 = Float3!(0.0, 0.0, 0.0);
    pub const ONE: Float3 = Float3!(1.0, 1.0, 1.0);
}

impl From<Raw4> for Float3 {
    fn from(raw: Raw4) -> Self { unsafe { transmute_copy::<Raw4, Float3>(&raw) } }
}

impl From<Float4> for Float3 {
    fn from(float: Float4) -> Self { unsafe { transmute_copy::<Float4, Float3>(&float) } }
}

impl From<Vector4> for Float3 {
    fn from(vector: Vector4) -> Self { unsafe { transmute_copy::<Vector4, Float3>(&vector) } }
}

impl From<Point4> for Float3 {
    fn from(point: Point4) -> Self { unsafe { transmute_copy::<Point4, Float3>(&point) } }
}

impl From<Normal4> for Float3 {
    fn from(normal: Normal4) -> Self { unsafe { transmute_copy::<Normal4, Float3>(&normal) } }
}

/* ------------------------------ Float3Algebra ----------------------------- */
impl Float3 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(4.0, 5.0, 6.0);
    /// let result = flo1.dot(flo2);
    ///
    /// assert_eq!(result, flo1.x * flo2.x + flo1.y * flo2.y + flo1.z * flo2.z)
    /// ```
    #[inline]
    pub fn dot(&self, rhs: Float3) -> f32 { self.x * rhs.x + self.y * rhs.y + self.z * rhs.z }

    #[inline]
    pub fn abs_dot(&self, rhs: Float3) -> f32 { self.dot(rhs).abs() }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(4.0, 5.0, 6.0);
    /// let result = flo1.cross(flo2);
    /// let expect = Float3!(flo1.y * flo2.z - flo1.z * flo2.y,
    ///                      flo1.z * flo2.x - flo1.x * flo2.z,
    ///                      flo1.x * flo2.y - flo1.y * flo2.x);
    ///
    /// assert_eq!(result, expect);
    /// ```
    #[inline]
    pub fn cross(&self, rhs: Float3) -> Float3 {
        if check_sse2!() {
            const YZXW: i32 = swizzle!(1, 2, 0, 3);
            const ZXYW: i32 = swizzle!(2, 0, 1, 3);

            unsafe {
                let lhs: __m128 = _mm_set_ps(0.0, self.z, self.y, self.x);
                let rhs: __m128 = _mm_set_ps(0.0, rhs.z, rhs.y, rhs.x);

                let v1_yzxw = _mm_shuffle_ps::<YZXW>(lhs, lhs);
                let v2_zxyw = _mm_shuffle_ps::<ZXYW>(rhs, rhs);

                let r1 = _mm_mul_ps(v1_yzxw, v2_zxyw);

                let v1_zxyw = _mm_shuffle_ps::<ZXYW>(lhs, lhs);
                let v2_yzxw = _mm_shuffle_ps::<YZXW>(rhs, rhs);

                let r2 = _mm_mul_ps(v1_zxyw, v2_yzxw);

                transmute_copy::<__m128, Float3>(&_mm_sub_ps(r1, r2))
            }
        } else {
            Float3!(
                self.y * rhs.z - self.z * rhs.y,
                self.z * rhs.x - self.x * rhs.z,
                self.x * rhs.y - self.y * rhs.x
            )
        }
    }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let result = flo.length();
    ///
    /// assert_eq!(result, (flo.x * flo.x + flo.y * flo.y + flo.z * flo.z).sqrt())
    /// ```
    #[inline]
    pub fn length(&self) -> f32 { (self.x * self.x + self.y + self.y + self.z * self.z).sqrt() }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let result = flo.length_squared();
    ///
    /// assert_eq!(result, (flo.x * flo.x + flo.y * flo.y + flo.z * flo.z))
    /// ```
    #[inline]
    pub fn length_squared(&self) -> f32 { self.x * self.x + self.y + self.y + self.z * self.z }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let length = flo.length();
    /// let inv_length = 1.0 / length;
    /// let result = flo.norm();
    ///
    /// assert_eq!(result, Float3!(flo.x * inv_length, flo.y * inv_length, flo.z * inv_length))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let flo = Float3!(0.0, 0.0, 0.0);
    /// let panic = flo.norm();
    /// ```
    #[inline]
    pub fn norm(&self) -> Float3 {
        let length = self.length();
        debug_assert!(length != 0.0, "Float3 with 0 length cannot be normalized");

        self.div(length)
    }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let length = flo.length();
    /// let inv_length = 1.0 / length;
    /// let mut result = flo;
    /// result.normalize();
    ///
    /// assert_eq!(result, Float3!(flo.x * inv_length, flo.y * inv_length, flo.z * inv_length))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut flo = Float3!(0.0, 0.0, 0.0);
    /// let panic = flo.normalize();
    /// ```
    #[inline]
    pub fn normalize(&mut self) -> &Float3 {
        let length = self.length();
        debug_assert!(length != 0.0, "Float3 with 0 length cannot be normalized");

        self.div_assign(length);

        self
    }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(5.0, 6.0, 7.0);
    /// let result = flo1.distance(flo2);
    /// let x_diff = flo1.x - flo2.x;
    /// let y_diff = flo1.y - flo2.y;
    /// let z_diff = flo1.z - flo2.z;
    /// let expect = (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff).sqrt();
    ///
    /// assert_eq!(result, expect)
    /// ```
    #[inline]
    pub fn distance(&self, rhs: Float3) -> f32 {
        let x_diff = self.x - rhs.x;
        let y_diff = self.y - rhs.y;
        let z_diff = self.z - rhs.z;

        (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff).sqrt()
    }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(5.0, 6.0, 7.0);
    /// let result = flo1.distance_squared(flo2);
    /// let x_diff = flo1.x - flo2.x;
    /// let y_diff = flo1.y - flo2.y;
    /// let z_diff = flo1.z - flo2.z;
    /// let expect = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
    ///
    /// assert_eq!(result, expect)
    /// ```
    #[inline]
    pub fn distance_squared(&self, rhs: Float3) -> f32 {
        let x_diff = self.x - rhs.x;
        let y_diff = self.y - rhs.y;
        let z_diff = self.z - rhs.z;

        x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
    }
}

/* ---------------------------------- Unary --------------------------------- */
impl std::ops::Neg for Float3 {
    type Output = Float3;

    #[inline]
    fn neg(self) -> Self::Output { Float3!(-self.x, -self.y, -self.z) }
}

impl std::ops::Index<usize> for Float3 {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("out of range of Float3's index"),
        }
    }
}

impl std::ops::IndexMut<usize> for Float3 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("out of range of Float3's index"),
        }
    }
}

/* ----------------------------------- Add ---------------------------------- */
impl std::ops::Add<Float3> for Float3 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(4.0, 5.0, 6.0);
    /// let result = flo1 + flo2;
    ///
    /// assert_eq!(result, Float3!(flo1.x + flo2.x, flo1.y + flo2.y, flo1.z + flo2.z))
    /// ```
    #[inline]
    fn add(self, rhs: Float3) -> Self::Output { Float3!(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z) }
}

impl std::ops::Add<f32> for Float3 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let f: f32 = 1.0;
    /// let result = flo + f;
    ///
    /// assert_eq!(result, Float3!(flo.x + f, flo.y + f, flo.z + f))
    /// ```
    #[inline]
    fn add(self, rhs: f32) -> Self::Output { Float3!(self.x + rhs, self.y + rhs, self.z + rhs) }
}

impl std::ops::Add<Float3> for f32 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 1.0;
    /// let flo = Float3!(2.0, 3.0, 4.0);
    /// let result = f + flo;
    ///
    /// assert_eq!(result, Float3!(f + flo.x, f + flo.y, f + flo.z))
    /// ```
    #[inline]
    fn add(self, rhs: Float3) -> Self::Output { Float3!(self + rhs.x, self + rhs.y, self + rhs.z) }
}

/* -------------------------------- AddAssign ------------------------------- */
impl std::ops::AddAssign<Float3> for Float3 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(4.0, 5.0, 6.0);
    /// let mut result = flo1;
    /// result += flo2;
    ///
    /// assert_eq!(result, Float3!(flo1.x + flo2.x, flo1.y + flo2.y, flo1.z + flo2.z))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Float3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::AddAssign<f32> for Float3 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let f: f32 = 1.0;
    /// let mut result = flo;
    /// result += f;
    ///
    /// assert_eq!(result, Float3!(flo.x + f, flo.y + f, flo.z + f))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
    }
}

/* ----------------------------------- Sub ---------------------------------- */
impl std::ops::Sub<Float3> for Float3 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(4.0, 5.0, 6.0);
    /// let result = flo1 - flo2;
    ///
    /// assert_eq!(result, Float3!(flo1.x - flo2.x, flo1.y - flo2.y, flo1.z - flo2.z))
    /// ```
    #[inline]
    fn sub(self, rhs: Float3) -> Self::Output { Float3!(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z) }
}

impl std::ops::Sub<f32> for Float3 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let f: f32 = 1.0;
    /// let result = flo - f;
    ///
    /// assert_eq!(result, Float3!(flo.x - f, flo.y - f, flo.z - f))
    /// ```
    #[inline]
    fn sub(self, rhs: f32) -> Self::Output { Float3!(self.x - rhs, self.y - rhs, self.z - rhs) }
}

impl std::ops::Sub<Float3> for f32 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 1.0;
    /// let flo = Float3!(2.0, 3.0, 4.0);
    /// let result = f - flo;
    ///
    /// assert_eq!(result, Float3!(f - flo.x, f - flo.y, f - flo.z))
    /// ```
    #[inline]
    fn sub(self, rhs: Float3) -> Self::Output { Float3!(self - rhs.x, self - rhs.y, self - rhs.z) }
}

/* -------------------------------- SubAssign ------------------------------- */
impl std::ops::SubAssign<Float3> for Float3 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(4.0, 5.0, 6.0);
    /// let mut result = flo1;
    /// result -= flo2;
    ///
    /// assert_eq!(result, Float3!(flo1.x - flo2.x, flo1.y - flo2.y, flo1.z - flo2.z))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Float3) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl std::ops::SubAssign<f32> for Float3 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let f: f32 = 1.0;
    /// let mut result = flo;
    /// result -= f;
    ///
    /// assert_eq!(result, Float3!(flo.x - f, flo.y - f, flo.z - f))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
    }
}

/* ----------------------------------- Mul ---------------------------------- */
impl std::ops::Mul<Float3> for Float3 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(4.0, 5.0, 6.0);
    /// let result = flo1 * flo2;
    ///
    /// assert_eq!(result, Float3!(flo1.x * flo2.x, flo1.y * flo2.y, flo1.z * flo2.z))
    /// ```
    #[inline]
    fn mul(self, rhs: Float3) -> Self::Output { Float3!(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z) }
}

impl std::ops::Mul<f32> for Float3 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let result = flo * f;
    ///
    /// assert_eq!(result, Float3!(flo.x * f, flo.y * f, flo.z * f))
    /// ```
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output { Float3!(self.x * rhs, self.y * rhs, self.z * rhs) }
}

impl std::ops::Mul<Float3> for f32 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 2.0;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let result = f * flo;
    ///
    /// assert_eq!(result, Float3!(f * flo.x, f * flo.y, f * flo.z))
    /// ```
    #[inline]
    fn mul(self, rhs: Float3) -> Self::Output { Float3!(self * rhs.x, self * rhs.y, self * rhs.z) }
}

impl std::ops::Mul<(f32, f32, f32)> for Float3 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let tuple = (4.0, 5.0, 6.0);
    /// let result = flo * tuple;
    ///
    /// assert_eq!(result, Float3!(flo.x * tuple.0, flo.y * tuple.1, flo.z * tuple.2))
    /// ```
    #[inline]
    fn mul(self, rhs: (f32, f32, f32)) -> Self::Output { Float3!(self.x * rhs.0, self.y * rhs.1, self.z * rhs.2) }
}

/* -------------------------------- MulAssign ------------------------------- */
impl std::ops::MulAssign for Float3 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(4.0, 5.0, 6.0);
    /// let mut result = flo1;
    /// result *= flo2;
    ///
    /// assert_eq!(result, Float3!(flo1.x * flo2.x, flo1.y * flo2.y, flo1.z * flo2.z))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl std::ops::MulAssign<f32> for Float3 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let mut result = flo;
    /// result *= f;
    ///
    /// assert_eq!(result, Float3!(flo.x * f, flo.y * f, flo.z * f))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

/* ----------------------------------- Div ---------------------------------- */
impl std::ops::Div<Float3> for Float3 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(4.0, 5.0, 6.0);
    /// let result = flo1 / flo2;
    ///
    /// assert_eq!(result, Float3!(flo1.x / flo2.x, flo1.y / flo2.y, flo1.z / flo2.z))
    /// ```
    #[inline]
    fn div(self, rhs: Float3) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Float3 cannot be divided by 0"
        );

        Float3!(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl std::ops::Div<f32> for Float3 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let f: f32 = 2.0;
    /// let inv = 1.0 / f;
    /// let result = flo / f;
    ///
    /// assert_eq!(result, Float3!(flo.x * inv, flo.y * inv, flo.z * inv))
    /// ```
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        debug_assert!(rhs != 0.0, "Float3 cannot be divided by 0");
        let inv = 1.0 / rhs;

        Float3!(self.x * inv, self.y * inv, self.z * inv)
    }
}

impl std::ops::Div<(f32, f32, f32)> for Float3 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let tuple = (4.0, 5.0, 6.0);
    /// let result = flo / tuple;
    ///
    /// assert_eq!(result, Float3!(flo.x / tuple.0, flo.y / tuple.1, flo.z / tuple.2))
    /// ```
    #[inline]
    fn div(self, rhs: (f32, f32, f32)) -> Self::Output {
        debug_assert!(
            rhs.0 != 0.0 && rhs.1 != 0.0 && rhs.2 != 0.0,
            "Float3 cannot be divided by 0"
        );

        Float3!(self.x / rhs.0, self.y / rhs.1, self.z / rhs.2)
    }
}

impl std::ops::Div<Float3> for f32 {
    type Output = Float3;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 1.0;
    /// let flo = Float3!(2.0, 3.0, 4.0);
    /// let result = f / flo;
    ///
    /// assert_eq!(result, Float3!(f / flo.x, f / flo.y, f / flo.z))
    /// ```
    #[inline]
    fn div(self, rhs: Float3) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Float3 cannot be divided by 0"
        );

        Float3!(self / rhs.x, self / rhs.y, self / rhs.z)
    }
}

/* -------------------------------- DivAssign ------------------------------- */
impl std::ops::DivAssign<Float3> for Float3 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float3!(1.0, 2.0, 3.0);
    /// let flo2 = Float3!(4.0, 5.0, 6.0);
    /// let mut result = flo1;
    /// result /= flo2;
    ///
    /// assert_eq!(result, Float3!(flo1.x / flo2.x, flo1.y / flo2.y, flo1.z / flo2.z))
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Float3) {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Float3 cannot be divided by 0"
        );

        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl std::ops::DivAssign<f32> for Float3 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let f: f32 = 2.0;
    /// let inv = 1.0 / f;
    /// let mut result = flo;
    /// result /= f;
    ///
    /// assert_eq!(result, Float3!(flo.x * inv, flo.y * inv, flo.z * inv))
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        debug_assert!(rhs != 0.0, "Float3 cannot be divided by 0");
        let inv = 1.0 / rhs;

        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

impl std::ops::DivAssign<(f32, f32, f32)> for Float3 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float3!(1.0, 2.0, 3.0);
    /// let tuple = (4.0, 5.0, 6.0);
    /// let mut result = flo;
    /// result /= tuple;
    ///
    /// assert_eq!(result, Float3!(flo.x / tuple.0, flo.y / tuple.1, flo.z / tuple.2))
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: (f32, f32, f32)) {
        debug_assert!(
            rhs.0 != 0.0 && rhs.1 != 0.0 && rhs.2 != 0.0,
            "Float3 cannot be divided by 0"
        );

        self.x /= rhs.0;
        self.y /= rhs.1;
        self.z /= rhs.2;
    }
}
