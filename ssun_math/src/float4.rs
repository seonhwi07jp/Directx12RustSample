use std::{
    arch::x86_64::{__m128, _mm_add_ps, _mm_mul_ps, _mm_set1_ps, _mm_set_ps, _mm_shuffle_ps, _mm_sub_ps},
    mem::transmute,
    ops::{Div, DivAssign},
};

use crate::{check_sse2, swizzle};

/* -------------------------------------------------------------------------- */
/*                                   macros                                   */
/* -------------------------------------------------------------------------- */
#[macro_export]
macro_rules! Vector4 {
    ($x:expr, $y:expr, $z:expr) => {
        Vector4 {
            x: $x,
            y: $y,
            z: $z,
            w: 0.0,
        }
    };
    ($val:expr) => {
        Vector4 {
            x: $val,
            y: $val,
            z: $val,
            w: 0.0,
        }
    };
}

#[macro_export]
macro_rules! Point4 {
    ($x:expr, $y:expr, $z:expr) => {
        Point4 {
            x: $x,
            y: $y,
            z: $z,
            w: 1.0,
        }
    };
    ($val:expr) => {
        Point4 {
            x: $val,
            y: $val,
            z: $val,
            w: 0.0,
        }
    };
}

#[macro_export]
macro_rules! Normal4 {
    ($x:expr, $y:expr, $z:expr) => {
        Normal4 {
            x: $x,
            y: $y,
            z: $z,
            w: 0.0,
        }
    };
    ($val:expr) => {
        Normal4 {
            x: $val,
            y: $val,
            z: $val,
            w: 0.0,
        }
    };
}

#[macro_export]
macro_rules! Float4 {
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        Float4 {
            x: $x,
            y: $y,
            z: $z,
            w: $w,
        }
    };
    ($val:expr) => {
        Float4 {
            x: $val,
            y: $val,
            z: $val,
            w: $val,
        }
    };
}

#[macro_export]
macro_rules! Raw4 {
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        Raw4 {
            x: $x,
            y: $y,
            z: $z,
            w: $w,
        }
    };
    ($val:expr) => {
        Raw4 {
            x: $val,
            y: $val,
            z: $val,
            w: $val,
        }
    };
}

#[macro_export]
macro_rules! dot {
    ($x:expr, $y:expr) => {
        $x.dot($y)
    };
}

#[macro_export]
macro_rules! abs_dot {
    ($x:expr, $y:expr) => {
        $x.abs_dot($y)
    };
}

#[macro_export]
macro_rules! cross {
    ($x:expr, $y:expr) => {
        $x.cross($y)
    };
}

#[macro_export]
macro_rules! length {
    ($x:expr) => {
        $x.length()
    };
}

#[macro_export]
macro_rules! length_squared {
    ($x:expr) => {
        $x.length_squared()
    };
}

#[macro_export]
macro_rules! norm {
    ($x:expr) => {
        $x.norm()
    };
}

#[macro_export]
macro_rules! normalize {
    ($x:expr) => {
        $x.normalize()
    };
}

#[macro_export]
macro_rules! distance {
    ($x:expr, $y:expr) => {
        $x.distance($y)
    };
}

#[macro_export]
macro_rules! distance_squared {
    ($x:expr, $y:expr) => {
        $x.distance_squared($y)
    };
}

/* -------------------------------------------------------------------------- */
/*                           Vector4ComponentAlgebra                          */
/* -------------------------------------------------------------------------- */
pub trait Vector4ComponentAlgebra<T = Self> {
    type Output;

    fn dot(&self, rhs: T) -> f32;
    #[inline]
    fn abs_dot(&self, rhs: T) -> f32 { self.dot(rhs) }
    fn cross(&self, rhs: T) -> Self::Output;
}

/* -------------------------------------------------------------------------- */
/*                           Point4ComponentAlgebra                           */
/* -------------------------------------------------------------------------- */
pub trait Point4ComponentAlgebra<T = Self> {
    fn distance(&self, rhs: T) -> f32;
    fn distance_squared(&self, rhs: T) -> f32;
}

/* -------------------------------------------------------------------------- */
/*                                 struct Raw4                                */
/* -------------------------------------------------------------------------- */
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Raw4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl From<Float4> for Raw4 {
    #[inline]
    fn from(float: Float4) -> Self { unsafe { transmute(float) } }
}

impl From<Vector4> for Raw4 {
    #[inline]
    fn from(vector: Vector4) -> Self { unsafe { transmute(vector) } }
}

impl From<Point4> for Raw4 {
    #[inline]
    fn from(point: Point4) -> Self { unsafe { transmute(point) } }
}

impl From<Normal4> for Raw4 {
    #[inline]
    fn from(normal: Normal4) -> Self { unsafe { transmute(normal) } }
}

/* -------------------------------------------------------------------------- */
/*                                struct Float4                               */
/* -------------------------------------------------------------------------- */
#[repr(C, align(16))]
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Float4 {
    pub const ZERO: Float4 = Float4!(0.0, 0.0, 0.0, 0.0);
    pub const ONE: Float4 = Float4!(1.0, 1.0, 1.0, 1.0);
}

impl From<Raw4> for Float4 {
    #[inline]
    fn from(raw: Raw4) -> Self { unsafe { transmute(raw) } }
}

impl From<Vector4> for Float4 {
    #[inline]
    fn from(vector: Vector4) -> Self { unsafe { transmute(vector) } }
}

impl From<Point4> for Float4 {
    #[inline]
    fn from(point: Point4) -> Self { unsafe { transmute(point) } }
}

impl From<Normal4> for Float4 {
    #[inline]
    fn from(normal: Normal4) -> Self { unsafe { transmute(normal) } }
}

/* ------------------------------ Float4Algebra ----------------------------- */
impl Float4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 0.0);
    /// let result = flo.length();
    ///
    /// assert_eq!(result, (flo.x * flo.x + flo.y * flo.y + flo.z * flo.z + flo.w * flo.w).sqrt())
    /// ```
    #[inline]
    pub fn length(&self) -> f32 { (self.x * self.x + self.y + self.y + self.z * self.z + self.w * self.w).sqrt() }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 0.0);
    /// let result = flo.length_squared();
    ///
    /// assert_eq!(result, (flo.x * flo.x + flo.y * flo.y + flo.z * flo.z + flo.w * flo.w))
    /// ```
    #[inline]
    pub fn length_squared(&self) -> f32 { self.x * self.x + self.y + self.y + self.z * self.z + self.w * self.w }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let length = flo.length();
    /// let inv_length = 1.0 / length;
    /// let result = flo.norm();
    ///
    /// assert_eq!(result, Float4!(flo.x * inv_length, flo.y * inv_length, flo.z * inv_length, flo.w * inv_length))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let flo = Float4!(0.0, 0.0, 0.0, 0.0);
    /// let panic = flo.norm();
    /// ```
    #[inline]
    pub fn norm(&self) -> Float4 {
        let length = self.length();
        debug_assert!(length != 0.0, "Float4 with 0 length cannot be normalized");

        self.div(length)
    }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let length = flo.length();
    /// let inv_length = 1.0 / length;
    /// let mut result = flo;
    /// result.normalize();
    ///
    /// assert_eq!(result, Float4!(flo.x * inv_length, flo.y * inv_length, flo.z * inv_length, flo.w * inv_length))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut flo = Float4!(0.0, 0.0, 0.0, 0.0);
    /// let panic = flo.normalize();
    /// ```
    #[inline]
    pub fn normalize(&mut self) -> &Float4 {
        let length = self.length();
        debug_assert!(length != 0.0, "Float4 with 0 length cannot be normalized");

        self.div_assign(length);

        self
    }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let flo2 = Float4!(4.0, 5.0, 6.0, 7.0);
    /// let result = flo1.dot(flo2);
    ///
    /// assert_eq!(result, flo1.x * flo2.x + flo1.y * flo2.y + flo1.z * flo2.z + flo1.w * flo2.w)
    /// ```
    #[inline]
    pub fn dot(&self, rhs: Float4) -> f32 { self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(1.0, 2.0, 3.0, 0.0);
    /// let flo2 = Float4!(4.0, 5.0, 6.0, 0.0);
    /// let result = flo1.cross(flo2);
    /// let expect = Float4!(flo1.y * flo2.z - flo1.z * flo2.y,
    ///                      flo1.z * flo2.x - flo1.x * flo2.z,
    ///                      flo1.x * flo2.y - flo1.y * flo2.x,
    ///                      0.0);
    ///
    /// assert_eq!(result, expect);
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let flo1 = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let flo2 = Float4!(4.0, 5.0, 6.0, 7.0);
    /// let panic = flo1.cross(flo2);
    /// ```
    #[inline]
    pub fn cross(&self, rhs: Float4) -> Float4 {
        debug_assert!(
            self.w == 0.0 && rhs.w == 0.0,
            "Float4 with w value not zero cannot operate cross product"
        );

        if check_sse2!() {
            const YZXW: i32 = swizzle!(1, 2, 0, 3);
            const ZXYW: i32 = swizzle!(2, 0, 1, 3);

            unsafe {
                let lhs: __m128 = transmute(*self);
                let rhs: __m128 = transmute(rhs);

                let v1_yzxw = _mm_shuffle_ps::<YZXW>(lhs, lhs);
                let v2_zxyw = _mm_shuffle_ps::<ZXYW>(rhs, rhs);

                let r1 = _mm_mul_ps(v1_yzxw, v2_zxyw);

                let v1_zxyw = _mm_shuffle_ps::<ZXYW>(lhs, lhs);
                let v2_yzxw = _mm_shuffle_ps::<YZXW>(rhs, rhs);

                let r2 = _mm_mul_ps(v1_zxyw, v2_yzxw);

                transmute(_mm_sub_ps(r1, r2))
            }
        } else {
            Float4!(
                self.y * rhs.z - self.z * rhs.y,
                self.z * rhs.x - self.x * rhs.z,
                self.x * rhs.y - self.y * rhs.x,
                0.0
            )
        }
    }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let flo2 = Float4!(5.0, 6.0, 7.0, 8.0);
    /// let result = flo1.distance(flo2);
    /// let x_diff = flo1.x - flo2.x;
    /// let y_diff = flo1.y - flo2.y;
    /// let z_diff = flo1.z - flo2.z;
    /// let w_diff = flo1.w - flo2.w;
    /// let expect = (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff + w_diff * w_diff).sqrt();
    ///
    /// assert_eq!(result, expect)
    /// ```
    #[inline]
    pub fn distance(&self, rhs: Float4) -> f32 {
        let x_diff = self.x - rhs.x;
        let y_diff = self.y - rhs.y;
        let z_diff = self.z - rhs.z;
        let w_diff = self.w - rhs.w;

        (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff + w_diff * w_diff).sqrt()
    }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let flo2 = Float4!(5.0, 6.0, 7.0, 8.0);
    /// let result = flo1.distance_squared(flo2);
    /// let x_diff = flo1.x - flo2.x;
    /// let y_diff = flo1.y - flo2.y;
    /// let z_diff = flo1.z - flo2.z;
    /// let w_diff = flo1.w - flo2.w;
    /// let expect = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff + w_diff * w_diff;
    ///
    /// assert_eq!(result, expect)
    /// ```
    #[inline]
    pub fn distance_squared(&self, rhs: Float4) -> f32 {
        let x_diff = self.x - rhs.x;
        let y_diff = self.y - rhs.y;
        let z_diff = self.z - rhs.z;
        let w_diff = self.w - rhs.w;

        x_diff * x_diff + y_diff * y_diff + z_diff * z_diff + w_diff * w_diff
    }
}

/* ---------------------------------- Unary --------------------------------- */
impl std::ops::Neg for Float4 {
    type Output = Float4;

    #[inline]
    fn neg(self) -> Self::Output { Float4!(-self.x, -self.y, -self.z, -self.w) }
}

impl std::ops::Index<usize> for Float4 {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("out of range of Float4's index"),
        }
    }
}

impl std::ops::IndexMut<usize> for Float4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("out of range of Float4's index"),
        }
    }
}

/* ----------------------------------- Add ---------------------------------- */
impl std::ops::Add for Float4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let flo2 = Float4!(4.0, 5.0, 6.0, 7.0);
    /// let result = flo1 + flo2;
    ///
    /// assert_eq!(result, Float4!(flo1.x + flo2.x, flo1.y + flo2.y, flo1.z + flo2.z, flo1.w + flo2.w))
    /// ```
    #[inline]
    fn add(self, rhs: Float4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_add_ps(transmute(self), transmute(rhs))) }
        } else {
            Float4!(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w)
        }
    }
}

impl std::ops::Add<f32> for Float4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let f: f32 = 4.0;
    /// let result = flo + f;
    ///
    /// assert_eq!(result, Float4!(flo.x + f, flo.y + f, flo.z + f, flo.w + f))
    /// ```
    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set1_ps(rhs);
                transmute(_mm_add_ps(transmute(self), xmm))
            }
        } else {
            Float4!(self.x + rhs, self.y + rhs, self.z + rhs, self.w + rhs)
        }
    }
}

impl std::ops::Add<Float4> for f32 {
    type Output = Float4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 1.0;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let result = f + flo;
    ///
    /// assert_eq!(result, Float4!(f + flo.x, f + flo.y, f + flo.z, f + flo.w))
    /// ```
    #[inline]
    fn add(self, rhs: Float4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set1_ps(self);
                transmute(_mm_add_ps(xmm, transmute(rhs)))
            }
        } else {
            Float4!(self + rhs.x, self + rhs.y, self + rhs.z, self + rhs.w)
        }
    }
}

/* -------------------------------- AddAssign ------------------------------- */
impl std::ops::AddAssign for Float4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let flo2 = Float4!(4.0, 5.0, 6.0, 7.0);
    /// let mut result = flo1;
    ///
    /// result += flo2;
    ///
    /// assert_eq!(result, Float4!(flo1.x + flo2.x, flo1.y + flo2.y, flo1.z + flo2.z, flo1.w + flo2.w))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_add_ps(transmute(*self), transmute(rhs))) };
        } else {
            self.x += rhs.x;
            self.y += rhs.y;
            self.z += rhs.z;
            self.w += rhs.w;
        }
    }
}

impl std::ops::AddAssign<f32> for Float4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let f: f32 = 4.0;
    ///
    /// let mut result = flo;
    /// result += f;
    ///
    /// assert_eq!(result, Float4!(flo.x + f, flo.y + f, flo.z + f, flo.w + f))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set1_ps(rhs);
                *self = transmute(_mm_add_ps(transmute(*self), xmm));
            }
        } else {
            self.x += rhs;
            self.y += rhs;
            self.z += rhs;
            self.w += rhs;
        }
    }
}

/* ----------------------------------- Sub ---------------------------------- */
impl std::ops::Sub for Float4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(4.0, 5.0, 6.0, 7.0);
    /// let flo2 = Float4!(3.0, 2.0, 1.0, 4.0);
    /// let result = flo1 - flo2;
    ///
    /// assert_eq!(result, Float4!(flo1.x - flo2.x, flo1.y - flo2.y, flo1.z - flo2.z, flo1.w - flo2.w))
    /// ```
    ///
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_sub_ps(transmute(self), transmute(rhs))) }
        } else {
            Float4!(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)
        }
    }
}

impl std::ops::Sub<f32> for Float4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(5.0, 3.0, 1.0, -1.0);
    /// let f: f32 = 1.0;
    /// let result = flo - f;
    ///
    /// assert_eq!(result, Float4!(flo.x - f, flo.y - f, flo.z - f, flo.w - f))
    /// ```
    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set1_ps(rhs);
                transmute(_mm_sub_ps(transmute(self), xmm))
            }
        } else {
            Float4!(self.x - rhs, self.y - rhs, self.z - rhs, self.w - rhs)
        }
    }
}

impl std::ops::Sub<Float4> for f32 {
    type Output = Float4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 5.0;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let result = f - flo;
    ///
    /// assert_eq!(result, Float4!(f - flo.x, f - flo.y, f - flo.z, f - flo.w))
    /// ```
    #[inline]
    fn sub(self, rhs: Float4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set1_ps(self);
                transmute(_mm_sub_ps(xmm, transmute(rhs)))
            }
        } else {
            Float4!(self - rhs.x, self - rhs.y, self - rhs.z, self - rhs.w)
        }
    }
}

/* -------------------------------- SubAssign ------------------------------- */
impl std::ops::SubAssign for Float4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(4.0, 5.0, 6.0, 7.0);
    /// let flo2 = Float4!(3.0, 2.0, 1.0, 4.0);
    /// let mut result = flo1;
    /// result -= flo2;
    ///
    /// assert_eq!(result, Float4!(flo1.x - flo2.x, flo1.y - flo2.y, flo1.z - flo2.z, flo1.w - flo2.w))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_sub_ps(transmute(*self), transmute(rhs))) };
        } else {
            self.x -= rhs.x;
            self.y -= rhs.y;
            self.z -= rhs.z;
            self.w -= rhs.w;
        }
    }
}

impl std::ops::SubAssign<f32> for Float4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(4.0, 5.0, 6.0, 7.0);
    /// let f: f32 = 1.0;
    /// let mut result = flo;
    /// result -= f;
    ///
    /// assert_eq!(result, Float4!(flo.x - f, flo.y - f, flo.z - f, flo.w - f))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set1_ps(rhs);
                *self = transmute(_mm_sub_ps(transmute(*self), xmm));
            }
        } else {
            self.x -= rhs;
            self.y -= rhs;
            self.z -= rhs;
            self.w -= rhs;
        }
    }
}

/* ----------------------------------- Mul ---------------------------------- */
impl std::ops::Mul for Float4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let flo2 = Float4!(4.0, 5.0, 6.0, 7.0);
    /// let result = flo1 * flo2;
    ///
    /// assert_eq!(result, Float4!(flo1.x * flo2.x, flo1.y * flo2.y, flo1.z * flo2.z, flo1.w * flo2.w))
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_mul_ps(transmute(self), transmute(rhs))) }
        } else {
            Float4!(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z, self.w * rhs.w)
        }
    }
}

impl std::ops::Mul<f32> for Float4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let f: f32 = 4.0;
    /// let result = flo * f;
    ///
    /// assert_eq!(result, Float4!(flo.x * f, flo.y * f, flo.z * f, flo.w * f))
    /// ```
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            let xmm = unsafe { _mm_set1_ps(rhs) };
            unsafe { transmute(_mm_mul_ps(transmute(self), xmm)) }
        } else {
            Float4!(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
        }
    }
}

impl std::ops::Mul<(f32, f32, f32, f32)> for Float4 {
    type Output = Self;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let tuple = (5.0, 6.0, 7.0, 8.0);
    /// let result = flo * tuple;
    ///
    /// assert_eq!(result, Float4!(flo.x * tuple.0, flo.y * tuple.1, flo.z * tuple.2, flo.w * tuple.3))
    /// ```
    #[inline]
    fn mul(self, rhs: (f32, f32, f32, f32)) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_mul_ps(transmute(self), transmute(rhs))) }
        } else {
            Float4!(self.x * rhs.0, self.y * rhs.1, self.z * rhs.2, self.w * rhs.3)
        }
    }
}

impl std::ops::Mul<Float4> for f32 {
    type Output = Float4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 4.0;
    /// let flo = Float4!(4.0, 3.0, 2.0, 1.0);
    /// let result = f * flo;
    ///
    /// assert_eq!(result, Float4!(f * flo.x, f * flo.y, f * flo.z, f * flo.w))
    /// ```
    #[inline]
    fn mul(self, rhs: Float4) -> Self::Output {
        if check_sse2!() {
            let xmm = unsafe { _mm_set1_ps(self) };
            unsafe { transmute(_mm_mul_ps(xmm, transmute(rhs))) }
        } else {
            Float4!(self * rhs.x, self * rhs.y, self * rhs.z, self * rhs.w)
        }
    }
}

/* -------------------------------- MulAssign ------------------------------- */
impl std::ops::MulAssign for Float4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let flo2 = Float4!(4.0, 5.0, 6.0, 7.0);
    /// let mut result = flo1;
    /// result *= flo2;
    ///
    /// assert_eq!(result, Float4!(flo1.x * flo2.x, flo1.y * flo2.y, flo1.z * flo2.z, flo1.w * flo2.w))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_mul_ps(transmute(*self), transmute(rhs))) };
        } else {
            self.x *= rhs.x;
            self.y *= rhs.y;
            self.z *= rhs.z;
            self.w *= rhs.w;
        }
    }
}

impl std::ops::MulAssign<f32> for Float4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let f: f32 = 4.0;
    /// let mut result = flo;
    /// result *= f;
    ///
    /// assert_eq!(result, Float4!(flo.x * f, flo.y * f, flo.z * f, flo.w * f))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set1_ps(rhs);
                *self = transmute(_mm_mul_ps(transmute(*self), xmm));
            }
        } else {
            self.x *= rhs;
            self.y *= rhs;
            self.z *= rhs;
            self.w *= rhs;
        }
    }
}

impl std::ops::MulAssign<(f32, f32, f32, f32)> for Float4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let mut result = flo;
    /// let tuple = (4.0, 5.0, 6.0, 7.0);
    /// result *= tuple;
    ///
    /// assert_eq!(result, Float4!(flo.x * tuple.0, flo.y * tuple.1, flo.z * tuple.2, flo.w * tuple.3))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: (f32, f32, f32, f32)) {
        if check_sse2!() {
            unsafe {
                *self = transmute(_mm_mul_ps(transmute(*self), transmute(rhs)));
            }
        } else {
            self.x *= rhs.0;
            self.y *= rhs.1;
            self.z *= rhs.2;
            self.w *= rhs.3;
        }
    }
}

/* ----------------------------------- Div ---------------------------------- */
impl std::ops::Div for Float4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(6.0, 8.0, 10.0, 12.0);
    /// let flo2 = Float4!(3.0, 4.0, 5.0, 6.0);
    /// let result = flo1 / flo2;
    ///
    /// assert_eq!(result, Float4!(flo1.x / flo2.x, flo1.y / flo2.y, flo1.z / flo2.z, flo1.w / flo2.w))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let flo1 = Float4!(6.0, 8.0, 10.0, 12.0);
    /// let flo2 = Float4!(0.0, 0.0, 0.0, 0.0);
    /// let panic = flo1 / flo2;
    /// ```
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0 && rhs.w != 0.0,
            "Float4 cannot be divided by 0"
        );

        Float4!(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z, self.w / rhs.w)
    }
}

impl std::ops::Div<f32> for Float4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(4.0, 6.0, 8.0, 10.0);
    /// let f: f32 = 2.0;
    /// let result = flo / f;
    ///
    /// assert_eq!(result, Float4!(flo.x / f, flo.y / f, flo.z / f, flo.w / f))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let flo = Float4!(4.0, 6.0, 8.0, 10.0);
    /// let f: f32 = 0.0;
    /// let panic = flo / f;
    /// ```
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        debug_assert!(rhs != 0.0, "Float4 cannot be divided by 0");
        let inv = 1.0 / rhs;

        Float4!(self.x * inv, self.y * inv, self.z * inv, self.w * inv)
    }
}

impl std::ops::Div<(f32, f32, f32, f32)> for Float4 {
    type Output = Self;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let tuple = (5.0, 6.0, 7.0, 8.0);
    /// let result = flo / tuple;
    ///
    /// assert_eq!(result, Float4!(flo.x / tuple.0, flo.y / tuple.1, flo.z / tuple.2, flo.w / tuple.3))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let panic = flo / (0.0, 0.0, 0.0, 0.0);
    /// ```
    #[inline]
    fn div(self, rhs: (f32, f32, f32, f32)) -> Self::Output {
        debug_assert!(
            rhs.0 != 0.0 && rhs.1 != 0.0 && rhs.2 != 0.0 && rhs.3 != 0.0,
            "Float4 cannot be divided by 0"
        );

        Float4!(self.x / rhs.0, self.y / rhs.1, self.z / rhs.2, self.w / rhs.3)
    }
}

impl std::ops::Div<Float4> for f32 {
    type Output = Float4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let f = 10.0;
    /// let flo = Float4!(2.0, 5.0, 10.0, 12.0);
    /// let result = f / flo;
    ///
    /// assert_eq!(result, Float4!(f / flo.x, f / flo.y, f / flo.z, f / flo.w))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::Float4;
    /// let f = 10.0;
    /// let flo = Float4!(0.0, 0.0, 0.0, 0.0);
    /// let panic = f / flo;
    /// ```
    #[inline]
    fn div(self, rhs: Float4) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0 && rhs.w != 0.0,
            "Float4 with a value of 0 cannot be a numerator"
        );

        Float4!(self / rhs.x, self / rhs.y, self / rhs.z, self / rhs.w)
    }
}

/* -------------------------------- DivAssign ------------------------------- */
impl std::ops::DivAssign for Float4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo1 = Float4!(2.0, 4.0, 6.0, 8.0);
    /// let flo2 = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let mut result = flo1;
    /// result /= flo2;
    ///
    /// assert_eq!(result, Float4!(flo1.x / flo2.x, flo1.y / flo2.y, flo1.z / flo2.z, flo1.w / flo2.w))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut panic = Float4!(2.0, 4.0, 6.0, 8.0);
    /// let flo = Float4!(0.0, 0.0, 0.0, 0.0);
    /// panic /= flo;
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0 && rhs.w != 0.0,
            "Float4 cannot be divided by 0"
        );

        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
        self.w /= rhs.w;
    }
}

impl std::ops::DivAssign<f32> for Float4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(4.0, 6.0, 8.0, 10.0);
    /// let f: f32 = 2.0;
    /// let mut result = flo;
    /// result /= f;
    ///
    /// assert_eq!(result, Float4!(flo.x / f, flo.y / f, flo.z / f, flo.w / f))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut panic = Float4!(1.0, 2.0, 3.0, 4.0);
    /// let f: f32 = 0.0;
    /// panic /= f;
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        debug_assert!(rhs != 0.0, "Float4 cannot be divided by 0");
        let inv = 1.0 / rhs;

        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
        self.w *= inv;
    }
}

impl std::ops::DivAssign<(f32, f32, f32, f32)> for Float4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(4.0, 6.0, 8.0, 10.0);
    /// let tuple = (2.0, 3.0, 4.0, 5.0);
    /// let mut result = flo;
    /// result /= tuple;
    ///
    /// assert_eq!(result, Float4!(flo.x / tuple.0, flo.y / tuple.1, flo.z / tuple.2, flo.w / tuple.3))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut panic = Float4!(4.0, 6.0, 8.0, 10.0);
    /// panic /= (0.0, 0.0, 0.0, 0.0);
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: (f32, f32, f32, f32)) {
        debug_assert!(
            rhs.0 != 0.0 && rhs.1 != 0.0 && rhs.2 != 0.0 && rhs.3 != 0.0,
            "Float4 cannot be divided by 0"
        );

        self.x /= rhs.0;
        self.y /= rhs.1;
        self.z /= rhs.2;
        self.w /= rhs.3;
    }
}

/* -------------------------------------------------------------------------- */
/*                               struct Vector4                               */
/* -------------------------------------------------------------------------- */
#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C, align(16))]
pub struct Vector4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vector4 {
    pub const ZERO: Vector4 = Vector4!(0.0, 0.0, 0.0);
    pub const ONE: Vector4 = Vector4!(1.0, 1.0, 1.0);
}

impl From<Raw4> for Vector4 {
    #[inline]
    fn from(raw: Raw4) -> Self { Vector4!(raw.x, raw.y, raw.z) }
}

impl From<Float4> for Vector4 {
    #[inline]
    fn from(float: Float4) -> Self { Vector4!(float.x, float.y, float.z) }
}

impl From<Point4> for Vector4 {
    #[inline]
    fn from(point: Point4) -> Self { Vector4!(point.x, point.y, point.z) }
}

impl From<Normal4> for Vector4 {
    #[inline]
    fn from(normal: Normal4) -> Self { unsafe { transmute(normal) } }
}

/* --------------------------------- Algebra -------------------------------- */
impl Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let mut vec1 = Vector4!(1.0, 2.0, 3.0);
    /// let vec2 = vec1;
    /// let length = vec2.length();
    /// let result = *vec1.normalize();
    ///
    /// assert_eq!(result, Vector4!(vec2.x / length, vec2.y / length, vec2.z / length))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut panic = Vector4!(0.0, 0.0, 0.0);
    /// panic.normalize();
    /// ```
    #[inline]
    pub fn normalize(&mut self) -> &Vector4 {
        let length = self.length();

        debug_assert!(length != 0.0, "zero Vector4 cannot be normalized");

        self.div_assign(length);

        self
    }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let length = vec.length();
    /// let result = vec.norm();
    ///
    /// assert_eq!(result, Vector4!(vec.x / length, vec.y / length, vec.z / length))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let panic = Vector4!(0.0, 0.0, 0.0);
    /// panic.norm();
    /// ```
    #[inline]
    pub fn norm(&self) -> Vector4 {
        let length = self.length();

        debug_assert!(length != 0.0, "zero Vector4 cannot be normalized");

        self.div(length)
    }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    ///
    /// assert_eq!(vec.length(), (vec.x * vec.x + vec.y * vec.y + vec.z * vec.z).sqrt())
    /// ```
    #[inline]
    pub fn length(&self) -> f32 { (self.x * self.x + self.y * self.y + self.z * self.z).sqrt() }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    ///
    /// assert_eq!(vec.length_squared(), (vec.x * vec.x + vec.y * vec.y + vec.z * vec.z))
    /// ```
    #[inline]
    pub fn length_squared(&self) -> f32 { self.x * self.x + self.y * self.y + self.z * self.z }
}

impl Vector4ComponentAlgebra<Vector4> for Vector4 {
    type Output = Vector4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let vec1 = Vector4!(1.0, 2.0, 3.0);
    /// let vec2 = Vector4!(4.0, 5.0, 6.0);
    /// let result = vec1.dot(vec2);
    ///
    /// assert_eq!(result, vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z)
    /// ```
    #[inline]
    fn dot(&self, rhs: Vector4) -> f32 { self.x * rhs.x + self.y * rhs.y + self.z * rhs.z }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let vec1 = Vector4!(1.0, 2.0, 3.0);
    /// let vec2 = Vector4!(4.0, 5.0, 6.0);
    /// let result = vec1.cross(vec2);
    /// let expect = Vector4!(vec1.y * vec2.z - vec1.z * vec2.y,
    ///                       vec1.z * vec2.x - vec1.x * vec2.z,
    ///                       vec1.x * vec2.y - vec1.y * vec2.x);
    ///
    /// assert_eq!(result, expect);
    /// ```
    #[inline]
    fn cross(&self, rhs: Vector4) -> Vector4 {
        if check_sse2!() {
            const YZXW: i32 = swizzle!(1, 2, 0, 3);
            const ZXYW: i32 = swizzle!(2, 0, 1, 3);

            unsafe {
                let lhs: __m128 = transmute(*self);
                let rhs: __m128 = transmute(rhs);

                let v1_yzxw = _mm_shuffle_ps::<YZXW>(lhs, lhs);
                let v2_zxyw = _mm_shuffle_ps::<ZXYW>(rhs, rhs);

                let r1 = _mm_mul_ps(v1_yzxw, v2_zxyw);

                let v1_zxyw = _mm_shuffle_ps::<ZXYW>(lhs, lhs);
                let v2_yzxw = _mm_shuffle_ps::<YZXW>(rhs, rhs);

                let r2 = _mm_mul_ps(v1_zxyw, v2_yzxw);

                transmute(_mm_sub_ps(r1, r2))
            }
        } else {
            Vector4!(
                self.y * rhs.z - self.z * rhs.y,
                self.z * rhs.x - self.x * rhs.z,
                self.x * rhs.y - self.y * rhs.x
            )
        }
    }
}

impl Vector4ComponentAlgebra<Normal4> for Vector4 {
    type Output = Vector4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let result = vec.dot(nor);
    ///
    /// assert_eq!(result, vec.x * nor.x + vec.y * nor.y + vec.z * nor.z)
    /// ```
    #[inline]
    fn dot(&self, rhs: Normal4) -> f32 { self.x * rhs.x + self.y * rhs.y + self.z * rhs.z }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let result = vec.cross(nor);
    /// let expect = Vector4!(vec.y * nor.z - vec.z * nor.y,
    ///                       vec.z * nor.x - vec.x * nor.z,
    ///                       vec.x * nor.y - vec.y * nor.x);
    ///
    /// assert_eq!(result, expect);
    /// ```
    #[inline]
    fn cross(&self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            const YZXW: i32 = swizzle!(1, 2, 0, 3);
            const ZXYW: i32 = swizzle!(2, 0, 1, 3);

            unsafe {
                let lhs: __m128 = transmute(*self);
                let rhs: __m128 = transmute(rhs);

                let v1_yzxw = _mm_shuffle_ps::<YZXW>(lhs, lhs);
                let v2_zxyw = _mm_shuffle_ps::<ZXYW>(rhs, rhs);

                let r1 = _mm_mul_ps(v1_yzxw, v2_zxyw);

                let v1_zxyw = _mm_shuffle_ps::<ZXYW>(lhs, lhs);
                let v2_yzxw = _mm_shuffle_ps::<YZXW>(rhs, rhs);

                let r2 = _mm_mul_ps(v1_zxyw, v2_yzxw);

                transmute(_mm_sub_ps(r1, r2))
            }
        } else {
            Vector4!(
                self.y * rhs.z - self.z * rhs.y,
                self.z * rhs.x - self.x * rhs.z,
                self.x * rhs.y - self.y * rhs.x
            )
        }
    }
}

/* ---------------------------------- Unary --------------------------------- */
impl std::ops::Neg for Vector4 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_set_ps(0.0, -self.z, -self.y, -self.x)) }
        } else {
            Vector4!(-self.x, -self.y, -self.z)
        }
    }
}

impl std::ops::Index<usize> for Vector4 {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("out of range of Vector4's index"),
        }
    }
}

impl std::ops::IndexMut<usize> for Vector4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("out of range of Vector4's index"),
        }
    }
}

/* ----------------------------------- Add ---------------------------------- */
impl std::ops::Add for Vector4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec1 = Vector4!(1.0, 2.0, 3.0);
    /// let vec2 = Vector4!(4.0, 5.0, 6.0);
    /// let result = vec1 + vec2;
    ///
    /// assert_eq!(result, Vector4!(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z))
    /// ```
    #[inline]
    fn add(self, rhs: Vector4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_add_ps(transmute(self), transmute(rhs))) }
        } else {
            Vector4!(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
        }
    }
}

impl std::ops::Add<f32> for Vector4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let result = vec + f;
    ///
    /// assert_eq!(result, Vector4!(vec.x + f, vec.y + f, vec.z + f))
    /// ```
    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                transmute(_mm_add_ps(transmute(self), xmm))
            }
        } else {
            Vector4!(self.x + rhs, self.y + rhs, self.z + rhs)
        }
    }
}

impl std::ops::Add<Vector4> for f32 {
    type Output = Vector4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let f: f32 = 1.0;
    /// let vec = Vector4!(2.0, 3.0, 4.0);
    /// let result = f + vec;
    ///
    /// assert_eq!(result, Vector4!(f + vec.x, f + vec.y, f + vec.z))
    /// ```
    #[inline]
    fn add(self, rhs: Vector4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, self, self, self);
                transmute(_mm_add_ps(xmm, transmute(rhs)))
            }
        } else {
            Vector4!(self + rhs.x, self + rhs.y, self + rhs.z)
        }
    }
}

impl std::ops::Add<Normal4> for Vector4 {
    type Output = Vector4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let result = vec + nor;
    ///
    /// assert_eq!(result, Vector4!(vec.x + nor.x, vec.y + nor.y, vec.z + nor.z))
    /// ```
    #[inline]
    fn add(self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_add_ps(transmute(self), transmute(rhs))) }
        } else {
            Vector4!(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
        }
    }
}

impl std::ops::Add<Point4> for Vector4 {
    type Output = Point4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let poi = Point4!(4.0, 5.0, 6.0);
    /// let result = vec + poi;
    ///
    /// assert_eq!(result, Point4!(vec.x + poi.x, vec.y + poi.y, vec.z + poi.z))
    /// ```
    #[inline]
    fn add(self, rhs: Point4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_add_ps(transmute(self), transmute(rhs))) }
        } else {
            Point4!(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
        }
    }
}

/* -------------------------------- AddAssign ------------------------------- */
impl std::ops::AddAssign for Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec1 = Vector4!(1.0, 2.0, 3.0);
    /// let vec2 = Vector4!(4.0, 5.0, 6.0);
    /// let mut result = vec1;
    ///
    /// result += vec2;
    ///
    /// assert_eq!(result, Vector4!(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_add_ps(transmute(*self), transmute(rhs))) };
        } else {
            self.x += rhs.x;
            self.y += rhs.y;
            self.z += rhs.z;
        }
    }
}

impl std::ops::AddAssign<f32> for Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    ///
    /// let mut result = vec;
    /// result += f;
    ///
    /// assert_eq!(result, Vector4!(vec.x + f, vec.y + f, vec.z + f))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                *self = transmute(_mm_add_ps(transmute(*self), transmute(xmm)));
            }
        } else {
            self.x += rhs;
            self.y += rhs;
            self.z += rhs;
        }
    }
}

impl std::ops::AddAssign<Normal4> for Vector4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let mut result = vec;
    /// result += nor;
    ///
    /// assert_eq!(result, Vector4!(vec.x + nor.x, vec.y + nor.y, vec.z + nor.z))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Normal4) {
        if check_sse2!() {
            unsafe { *self = transmute(_mm_add_ps(transmute(*self), transmute(rhs))) }
        } else {
            self.x += rhs.x;
            self.y += rhs.y;
            self.z += rhs.z;
        }
    }
}

/* ----------------------------------- Sub ---------------------------------- */
impl std::ops::Sub for Vector4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec1 = Vector4!(4.0, 5.0, 6.0);
    /// let vec2 = Vector4!(3.0, 2.0, 1.0);
    /// let result = vec1 - vec2;
    ///
    /// assert_eq!(result, Vector4!(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z))
    /// ```
    ///
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_sub_ps(transmute(self), transmute(rhs))) }
        } else {
            Vector4!(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
}

impl std::ops::Sub<f32> for Vector4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(5.0, 3.0, 1.0);
    /// let f: f32 = 1.0;
    /// let result = vec - f;
    ///
    /// assert_eq!(result, Vector4!(vec.x - f, vec.y - f, vec.z - f))
    /// ```
    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                transmute(_mm_sub_ps(transmute(self), xmm))
            }
        } else {
            Vector4!(self.x - rhs, self.y - rhs, self.z - rhs)
        }
    }
}

impl std::ops::Sub<Vector4> for f32 {
    type Output = Vector4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let f: f32 = 5.0;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let result = f - vec;
    ///
    /// assert_eq!(result, Vector4!(f - vec.x, f - vec.y, f - vec.z))
    /// ```
    #[inline]
    fn sub(self, rhs: Vector4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, self, self, self);
                transmute(_mm_sub_ps(xmm, transmute(rhs)))
            }
        } else {
            Vector4!(self - rhs.x, self - rhs.y, self - rhs.z)
        }
    }
}

impl std::ops::Sub<Normal4> for Vector4 {
    type Output = Vector4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(4.0, 5.0, 6.0);
    /// let nor = Vector4!(1.0, 2.0, 3.0);
    /// let result = vec - nor;
    ///
    /// assert_eq!(result, Vector4!(vec.x - nor.x, vec.y - nor.y, vec.z - nor.z))
    /// ```
    #[inline]
    fn sub(self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_sub_ps(transmute(self), transmute(rhs))) }
        } else {
            Vector4!(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
}

/* -------------------------------- SubAssign ------------------------------- */
impl std::ops::SubAssign for Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec1 = Vector4!(4.0, 5.0, 6.0);
    /// let vec2 = Vector4!(3.0, 2.0, 1.0);
    /// let mut result = vec1;
    /// result -= vec2;
    ///
    /// assert_eq!(result, Vector4!(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_sub_ps(transmute(*self), transmute(rhs))) };
        } else {
            self.x -= rhs.x;
            self.y -= rhs.y;
            self.z -= rhs.z;
        }
    }
}

impl std::ops::SubAssign<f32> for Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(4.0, 5.0, 6.0);
    /// let f: f32 = 1.0;
    /// let mut result = vec;
    /// result -= f;
    ///
    /// assert_eq!(result, Vector4!(vec.x - f, vec.y - f, vec.z - f))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                *self = transmute(_mm_sub_ps(transmute(*self), xmm));
            }
        } else {
            self.x -= rhs;
            self.y -= rhs;
            self.z -= rhs;
        }
    }
}

impl std::ops::SubAssign<Normal4> for Vector4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(4.0, 5.0, 6.0);
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let mut result = vec;
    /// result -= nor;
    ///
    /// assert_eq!(result, Vector4!(vec.x - nor.x, vec.y - nor.y, vec.z - nor.z))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Normal4) {
        if check_sse2!() {
            unsafe { *self = transmute(_mm_sub_ps(transmute(*self), transmute(rhs))) }
        } else {
            self.x -= rhs.x;
            self.y -= rhs.y;
            self.z -= rhs.z;
        }
    }
}

/* ----------------------------------- Mul ---------------------------------- */
impl std::ops::Mul for Vector4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec1 = Vector4!(1.0, 2.0, 3.0);
    /// let vec2 = Vector4!(4.0, 5.0, 6.0);
    /// let result = vec1 * vec2;
    ///
    /// assert_eq!(result, Vector4!(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z))
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_mul_ps(transmute(self), transmute(rhs))) }
        } else {
            Vector4!(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
        }
    }
}

impl std::ops::Mul<Normal4> for Vector4 {
    type Output = Vector4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let result = vec * nor;
    ///
    /// assert_eq!(result, Vector4!(vec.x * nor.x, vec.y * nor.y, vec.z * nor.z))
    /// ```
    #[inline]
    fn mul(self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_mul_ps(transmute(self), transmute(rhs))) }
        } else {
            Vector4!(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
        }
    }
}

impl std::ops::Mul<f32> for Vector4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let result = vec * f;
    ///
    /// assert_eq!(result, Vector4!(vec.x * f, vec.y * f, vec.z * f))
    /// ```
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            let xmm = unsafe { _mm_set_ps(0.0, rhs, rhs, rhs) };
            unsafe { transmute(_mm_mul_ps(transmute(self), xmm)) }
        } else {
            Vector4!(self.x * rhs, self.y * rhs, self.z * rhs)
        }
    }
}

impl std::ops::Mul<(f32, f32, f32)> for Vector4 {
    type Output = Self;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let tuple = (1.0, 2.0, 3.0);
    /// let result = vec * tuple;
    ///
    /// assert_eq!(result, Vector4!(vec.x * tuple.0, vec.y * tuple.1, vec.z * tuple.2))
    /// ```
    #[inline]
    fn mul(self, rhs: (f32, f32, f32)) -> Self::Output {
        if check_sse2!() {
            let xmm = unsafe { _mm_set_ps(0.0, rhs.2, rhs.1, rhs.0) };
            unsafe { transmute(_mm_mul_ps(transmute(self), xmm)) }
        } else {
            Vector4!(self.x * rhs.0, self.y * rhs.1, self.z * rhs.2)
        }
    }
}

impl std::ops::Mul<Vector4> for f32 {
    type Output = Vector4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let f: f32 = 4.0;
    /// let vec = Vector4!(3.0, 2.0, 1.0);
    /// let result = f * vec;
    ///
    /// assert_eq!(result, Vector4!(f * vec.x, f * vec.y, f * vec.z))
    /// ```
    #[inline]
    fn mul(self, rhs: Vector4) -> Self::Output {
        if check_sse2!() {
            let xmm = unsafe { _mm_set_ps(0.0, self, self, self) };
            unsafe { transmute(_mm_mul_ps(xmm, transmute(rhs))) }
        } else {
            Vector4!(self * rhs.x, self * rhs.y, self * rhs.z)
        }
    }
}

/* -------------------------------- MulAssign ------------------------------- */
impl std::ops::MulAssign for Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec1 = Vector4!(1.0, 2.0, 3.0);
    /// let vec2 = Vector4!(4.0, 5.0, 6.0);
    /// let mut result = vec1;
    /// result *= vec2;
    ///
    /// assert_eq!(result, Vector4!(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_mul_ps(transmute(*self), transmute(rhs))) };
        } else {
            self.x *= rhs.x;
            self.y *= rhs.y;
            self.z *= rhs.z;
        }
    }
}

impl std::ops::MulAssign<Normal4> for Vector4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let mut result = vec;
    /// result *= nor;
    ///
    /// assert_eq!(result, Vector4!(vec.x * nor.x, vec.y * nor.y, vec.z * nor.z))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Normal4) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_mul_ps(transmute(*self), transmute(rhs))) }
        } else {
            self.x *= rhs.x;
            self.y *= rhs.y;
            self.z *= rhs.z;
        }
    }
}

impl std::ops::MulAssign<f32> for Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let mut result = vec;
    /// result *= f;
    ///
    /// assert_eq!(result, Vector4!(vec.x * f, vec.y * f, vec.z * f))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                *self = transmute(_mm_mul_ps(transmute(*self), xmm));
            }
        } else {
            self.x *= rhs;
            self.y *= rhs;
            self.z *= rhs;
        }
    }
}

impl std::ops::MulAssign<(f32, f32, f32)> for Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let mut result = vec;
    /// let tuple = (4.0, 5.0, 6.0);
    /// result *= tuple;
    ///
    /// assert_eq!(result, Vector4!(vec.x * tuple.0, vec.y * tuple.1, vec.z * tuple.2))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: (f32, f32, f32)) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs.2, rhs.1, rhs.0);
                *self = transmute(_mm_mul_ps(transmute(*self), xmm));
            }
        } else {
            self.x *= rhs.0;
            self.y *= rhs.1;
            self.z *= rhs.2;
        }
    }
}

/* ----------------------------------- Div ---------------------------------- */
impl std::ops::Div for Vector4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec1 = Vector4!(6.0, 8.0, 10.0);
    /// let vec2 = Vector4!(3.0, 4.0, 5.0);
    /// let result = vec1 / vec2;
    ///
    /// assert_eq!(result, Vector4!(vec1.x / vec2.x, vec1.y / vec2.y, vec1.z / vec2.z))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::Vector4;
    /// let vec1 = Vector4!(6.0, 8.0, 10.0);
    /// let vec2 = Vector4!(0.0, 0.0, 0.0);
    /// let panic = vec1 / vec2;
    /// ```
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Vector4 cannot be divided by 0"
        );

        Vector4!(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl std::ops::Div<Normal4> for Vector4 {
    type Output = Vector4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(4.0, 6.0, 8.0);
    /// let nor = Normal4!(2.0, 3.0, 4.0);
    /// let result = vec / nor;
    ///
    /// assert_eq!(result, Vector4!(vec.x / nor.x, vec.y / nor.y, vec.z / nor.z))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let vec = Vector4!(4.0, 6.0, 8.0);
    /// let nor = Normal4!(0.0, 0.0, 0.0);
    /// let panic = vec / nor;
    /// ```
    #[inline]
    fn div(self, rhs: Normal4) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Vector4 cannot be divided by 0"
        );

        Vector4!(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl std::ops::Div<f32> for Vector4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(4.0, 6.0, 8.0);
    /// let f: f32 = 2.0;
    /// let result = vec / f;
    ///
    /// assert_eq!(result, Vector4!(vec.x / f, vec.y / f, vec.z / f))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(4.0, 6.0, 8.0);
    /// let f: f32 = 0.0;
    /// let panic = vec / f;
    /// ```
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        debug_assert!(rhs != 0.0, "Vector4 cannot be divided by 0");
        let inv = 1.0 / rhs;

        Vector4!(self.x * inv, self.y * inv, self.z * inv)
    }
}

impl std::ops::Div<(f32, f32, f32)> for Vector4 {
    type Output = Self;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(4.0, 6.0, 8.0);
    /// let tuple = (1.0, 2.0, 3.0);
    /// let result = vec / tuple;
    ///
    /// assert_eq!(result, Vector4!(vec.x / tuple.0, vec.y / tuple.1, vec.z / tuple.2))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let vec = Vector4!(4.0, 6.0, 8.0);
    /// let panic = vec / (0.0, 0.0, 0.0);
    /// ```
    #[inline]
    fn div(self, rhs: (f32, f32, f32)) -> Self::Output {
        debug_assert!(
            rhs.0 != 0.0 && rhs.1 != 0.0 && rhs.2 != 0.0,
            "Vector4 cannot be divided by 0"
        );

        Vector4!(self.x / rhs.0, self.y / rhs.1, self.z / rhs.2)
    }
}

impl std::ops::Div<Vector4> for f32 {
    type Output = Vector4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let f = 10.0;
    /// let vec = Vector4!(2.0, 5.0, 10.0);
    /// let result = f / vec;
    ///
    /// assert_eq!(result, Vector4!(f / vec.x, f / vec.y, f / vec.z))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::Vector4;
    /// let f = 10.0;
    /// let vec = Vector4!(0.0, 0.0, 0.0);
    /// let panic = f / vec;
    /// ```
    #[inline]
    fn div(self, rhs: Vector4) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Vector4 with a value of 0 cannot be a numerator"
        );

        Vector4!(self / rhs.x, self / rhs.y, self / rhs.z)
    }
}

/* -------------------------------- DivAssign ------------------------------- */
impl std::ops::DivAssign for Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec1 = Vector4!(2.0, 4.0, 6.0);
    /// let vec2 = Vector4!(1.0, 2.0, 3.0);
    /// let mut result = vec1;
    /// result /= vec2;
    ///
    /// assert_eq!(result, Vector4!(vec1.x / vec2.x, vec1.y / vec2.y, vec1.z / vec2.z))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::Vector4;
    /// let mut panic = Vector4!(2.0, 4.0, 6.0);
    /// let vec = Vector4!(0.0, 0.0, 0.0);
    /// panic /= vec;
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Vector4 cannot be divided by 0"
        );

        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl std::ops::DivAssign<Normal4> for Vector4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let vec = Vector4!(4.0, 6.0, 8.0);
    /// let nor = Normal4!(2.0, 3.0, 4.0);
    /// let mut result = vec;
    /// result /= nor;
    ///
    /// assert_eq!(result, Vector4!(vec.x / nor.x, vec.y / nor.y, vec.z / nor.z))
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Normal4) {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Vector4 cannot be divided by 0"
        );

        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl std::ops::DivAssign<f32> for Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(4.0, 6.0, 8.0);
    /// let f: f32 = 2.0;
    /// let mut result = vec;
    /// result /= f;
    ///
    /// assert_eq!(result, Vector4!(vec.x / f, vec.y / f, vec.z / f))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::Vector4;
    /// let mut panic = Vector4!(1.0, 2.0, 3.0);
    /// let f: f32 = 0.0;
    /// panic /= f;
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        debug_assert!(rhs != 0.0, "Vector4 cannot be divided by 0");
        let inv = 1.0 / rhs;

        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

impl std::ops::DivAssign<(f32, f32, f32)> for Vector4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::Vector4;
    /// let vec = Vector4!(4.0, 6.0, 8.0);
    /// let tuple = (2.0, 3.0, 4.0);
    /// let mut result = vec;
    /// result /= tuple;
    ///
    /// assert_eq!(result, Vector4!(vec.x / tuple.0, vec.y / tuple.1, vec.z / tuple.2))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::Vector4;
    /// let mut panic = Vector4!(4.0, 6.0, 8.0);
    /// panic /= (0.0, 0.0, 0.0);
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: (f32, f32, f32)) {
        debug_assert!(
            rhs.0 != 0.0 && rhs.1 != 0.0 && rhs.2 != 0.0,
            "Vector4 cannot be divided by 0"
        );

        self.x /= rhs.0;
        self.y /= rhs.1;
        self.z /= rhs.2;
    }
}

/* -------------------------------------------------------------------------- */
/*                                struct Point                                */
/* -------------------------------------------------------------------------- */
#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C, align(16))]
pub struct Point4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Point4 {
    pub const ZERO: Point4 = Point4!(0.0, 0.0, 0.0);
    pub const ONE: Point4 = Point4!(1.0, 1.0, 1.0);
}

impl From<Raw4> for Point4 {
    #[inline]
    fn from(raw: Raw4) -> Self { Point4!(raw.x, raw.y, raw.z) }
}

impl From<Float4> for Point4 {
    #[inline]
    fn from(float: Float4) -> Self { Point4!(float.x, float.y, float.z) }
}

impl From<Vector4> for Point4 {
    #[inline]
    fn from(vector: Vector4) -> Self { Point4!(vector.x, vector.y, vector.z) }
}

impl From<Normal4> for Point4 {
    #[inline]
    fn from(normal: Normal4) -> Self { Point4!(normal.x, normal.y, normal.z) }
}

/* ---------------------------------- Unary --------------------------------- */
impl std::ops::Neg for Point4 {
    type Output = Point4;
    #[inline]
    fn neg(self) -> Self::Output { Point4!(-self.x, -self.y, -self.z) }
}

impl std::ops::Index<usize> for Point4 {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("out of range of Point4's index"),
        }
    }
}

impl std::ops::IndexMut<usize> for Point4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("out of range of Point4's index"),
        }
    }
}

/* --------------------------------- Algebra -------------------------------- */
impl Point4ComponentAlgebra<Point4> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi1 = Point4!(1.0, 2.0, 3.0);
    /// let poi2 = Point4!(4.0, 5.0, 6.0);
    /// let result = poi1.distance(poi2);
    ///
    /// let x_diff = poi1.x - poi2.x;
    /// let y_diff = poi1.y - poi2.y;
    /// let z_diff = poi1.z - poi2.z;
    ///
    /// assert_eq!(result, (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff).sqrt())
    /// ```
    #[inline]
    fn distance(&self, rhs: Point4) -> f32 {
        let x_diff = self.x - rhs.x;
        let y_diff = self.y - rhs.y;
        let z_diff = self.z - rhs.z;

        (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff).sqrt()
    }

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi1 = Point4!(1.0, 2.0, 3.0);
    /// let poi2 = Point4!(4.0, 5.0, 6.0);
    /// let result = poi1.distance_squared(poi2);
    ///
    /// let x_diff = poi1.x - poi2.x;
    /// let y_diff = poi1.y - poi2.y;
    /// let z_diff = poi1.z - poi2.z;
    ///
    /// assert_eq!(result, (x_diff * x_diff) + (y_diff * y_diff) + (z_diff * z_diff))
    #[inline]
    fn distance_squared(&self, rhs: Point4) -> f32 {
        let x_diff = self.x - rhs.x;
        let y_diff = self.y - rhs.y;
        let z_diff = self.z - rhs.z;

        x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
    }
}

/* ----------------------------------- Add ---------------------------------- */
impl std::ops::Add<Vector4> for Point4 {
    type Output = Point4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let vec = Vector4!(4.0, 5.0, 6.0);
    /// let result = poi + vec;
    ///
    /// assert_eq!(result, Point4!(poi.x + vec.x, poi.y + vec.y, poi.z + vec.z))
    /// ```
    #[inline]
    fn add(self, rhs: Vector4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_add_ps(transmute(self), transmute(rhs))) }
        } else {
            Point4!(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
        }
    }
}

impl std::ops::Add<f32> for Point4 {
    type Output = Point4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let result = poi + f;
    ///
    /// assert_eq!(result, Point4!(poi.x + f, poi.y + f, poi.z + f))
    /// ```
    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                transmute(_mm_add_ps(transmute(self), transmute(xmm)))
            }
        } else {
            Point4!(self.x + rhs, self.y + rhs, self.z + rhs)
        }
    }
}

impl std::ops::Add<Point4> for f32 {
    type Output = Point4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let f = 1.0;
    /// let poi = Point4!(2.0, 3.0, 4.0);
    /// let result = f + poi;
    ///
    /// assert_eq!(result, Point4!(f + poi.x, f + poi.y, f + poi.z))
    /// ```
    #[inline]
    fn add(self, rhs: Point4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, self, self, self);
                transmute(_mm_add_ps(xmm, transmute(rhs)))
            }
        } else {
            Point4!(self + rhs.x, self + rhs.y, self + rhs.z)
        }
    }
}

impl std::ops::Add<Normal4> for Point4 {
    type Output = Point4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let result = poi + nor;
    ///
    /// assert_eq!(result, Point4!(poi.x + nor.x, poi.y + nor.y, poi.z + nor.z))
    /// ```
    #[inline]
    fn add(self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_add_ps(transmute(self), transmute(rhs))) }
        } else {
            Point4!(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
        }
    }
}

/* -------------------------------- AddAssign ------------------------------- */
impl std::ops::AddAssign<Vector4> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let vec = Vector4!(4.0, 5.0, 6.0);
    /// let mut result = poi;
    /// result += vec;
    ///
    /// assert_eq!(result, Point4!(poi.x + vec.x, poi.y + vec.y, poi.z + vec.z))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Vector4) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_add_ps(transmute(*self), transmute(rhs))) };
        } else {
            self.x += rhs.x;
            self.y += rhs.y;
            self.z += rhs.z;
        }
    }
}

impl std::ops::AddAssign<f32> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let mut result = poi;
    /// result += f;
    ///
    /// assert_eq!(result, Point4!(poi.x + f, poi.y + f, poi.z + f))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                *self = transmute(_mm_add_ps(transmute(*self), xmm));
            }
        } else {
            self.x += rhs;
            self.y += rhs;
            self.z += rhs;
        }
    }
}

impl std::ops::AddAssign<Normal4> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let mut result = poi;
    /// result += nor;
    ///
    /// assert_eq!(result, Point4!(poi.x + nor.x, poi.y + nor.y, poi.z + nor.z))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Normal4) {
        if check_sse2!() {
            unsafe {
                *self = transmute(_mm_add_ps(transmute(*self), transmute(rhs)));
            }
        } else {
            self.x += rhs.x;
            self.y += rhs.y;
            self.z += rhs.z;
        }
    }
}

/* ----------------------------------- Sub ---------------------------------- */
impl std::ops::Sub<Vector4> for Point4 {
    type Output = Point4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(4.0, 5.0, 6.0);
    /// let vec = Vector4!(3.0, 2.0, 1.0);
    /// let result = poi - vec;
    ///
    /// assert_eq!(result, Point4!(poi.x - vec.x, poi.y - vec.y, poi.z - vec.z))
    /// ```
    #[inline]
    fn sub(self, rhs: Vector4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_sub_ps(transmute(self), transmute(rhs))) }
        } else {
            Point4!(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
}

impl std::ops::Sub<Normal4> for Point4 {
    type Output = Point4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(4.0, 5.0, 6.0);
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let result = poi - nor;
    ///
    /// assert_eq!(result, Point4!(poi.x - nor.x, poi.y - nor.y, poi.z - nor.z))
    /// ```
    #[inline]
    fn sub(self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_sub_ps(transmute(self), transmute(rhs))) }
        } else {
            Point4!(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
}

impl std::ops::Sub for Point4 {
    type Output = Vector4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let poi1 = Point4!(4.0, 5.0, 6.0);
    /// let poi2 = Point4!(3.0, 2.0, 1.0);
    /// let result = poi1 - poi2;
    ///
    /// assert_eq!(result, Vector4!(poi1.x - poi2.x, poi1.y - poi2.y, poi1.z - poi2.z))
    /// ```
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_sub_ps(transmute(self), transmute(rhs))) }
        } else {
            Vector4!(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
}

impl std::ops::Sub<f32> for Point4 {
    type Output = Vector4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(4.0, 5.0, 6.0);
    /// let f: f32 = 3.0;
    /// let result = poi - f;
    ///
    /// assert_eq!(result, Vector4!(poi.x - f, poi.y - f, poi.z - f))
    /// ```
    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(1.0, rhs, rhs, rhs);
                transmute(_mm_sub_ps(transmute(self), transmute(xmm)))
            }
        } else {
            Vector4!(self.x - rhs, self.y - rhs, self.z - rhs)
        }
    }
}

impl std::ops::Sub<Point4> for f32 {
    type Output = Vector4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 6.0;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let result = f - poi;
    ///
    /// assert_eq!(result, Vector4!(f - poi.x, f - poi.y, f - poi.z))
    #[inline]
    fn sub(self, rhs: Point4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(1.0, self, self, self);
                transmute(_mm_sub_ps(xmm, transmute(rhs)))
            }
        } else {
            Vector4!(self - rhs.x, self - rhs.y, self - rhs.z)
        }
    }
}

/* -------------------------------- SubAssign ------------------------------- */
impl std::ops::SubAssign<Vector4> for Point4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(4.0, 5.0, 6.0);
    /// let vec = Vector4!(3.0, 2.0, 1.0);
    /// let mut result = poi;
    /// result -= vec;
    ///
    /// assert_eq!(result, Point4!(poi.x - vec.x, poi.y - vec.y, poi.z - vec.z))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Vector4) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_sub_ps(transmute(*self), transmute(rhs))) };
        } else {
            self.x -= rhs.x;
            self.y -= rhs.y;
            self.z -= rhs.z;
        }
    }
}

impl std::ops::SubAssign<Normal4> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(4.0, 5.0, 6.0);
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let mut result = poi;
    /// result -= nor;
    ///
    /// assert_eq!(result, Point4!(poi.x - nor.x, poi.y - nor.y, poi.z - nor.z))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Normal4) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_sub_ps(transmute(*self), transmute(rhs))) }
        } else {
            self.x -= rhs.x;
            self.y -= rhs.y;
            self.z -= rhs.z;
        }
    }
}

impl std::ops::SubAssign<f32> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(4.0, 3.0, 2.0);
    /// let f: f32 = 1.0;
    /// let mut result = poi;
    /// result -= f;
    ///
    /// assert_eq!(result, Point4!(poi.x - f, poi.y - f, poi.z - f))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                *self = transmute(_mm_sub_ps(transmute(*self), xmm));
            }
        } else {
            self.x -= rhs;
            self.y -= rhs;
            self.z -= rhs;
        }
    }
}

/* ----------------------------------- Mul ---------------------------------- */
impl std::ops::Mul for Point4 {
    type Output = Point4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let poi1 = Point4!(1.0, 2.0, 3.0);
    /// let poi2 = Point4!(4.0, 5.0, 6.0);
    /// let result = poi1 * poi2;
    ///
    /// assert_eq!(result, Point4!(poi1.x * poi2.x, poi1.y * poi2.y, poi1.z * poi2.z))
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_mul_ps(transmute(self), transmute(rhs))) }
        } else {
            Point4!(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
        }
    }
}

impl std::ops::Mul<f32> for Point4 {
    type Output = Point4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let f = 4.0;
    /// let result = poi * f;
    ///
    /// assert_eq!(result, Point4!(poi.x * f, poi.y * f, poi.z * f))
    /// ```
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(1.0, rhs, rhs, rhs);
                transmute(_mm_mul_ps(transmute(self), xmm))
            }
        } else {
            Point4!(self.x * rhs, self.y * rhs, self.z * rhs)
        }
    }
}

impl std::ops::Mul<(f32, f32, f32)> for Point4 {
    type Output = Point4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let tuple = (4.0, 5.0, 6.0);
    /// let result = poi * tuple;
    ///
    /// assert_eq!(result, Point4!(poi.x * tuple.0, poi.y * tuple.1, poi.z * tuple.2))
    /// ```
    #[inline]
    fn mul(self, rhs: (f32, f32, f32)) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(1.0, rhs.2, rhs.1, rhs.0);
                transmute(_mm_mul_ps(transmute(self), xmm))
            }
        } else {
            Point4!(self.x * rhs.0, self.y * rhs.1, self.z * rhs.2)
        }
    }
}

/* -------------------------------- MulAssign ------------------------------- */
impl std::ops::MulAssign<Point4> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi1 = Point4!(1.0, 2.0, 3.0);
    /// let poi2 = Point4!(4.0, 5.0, 6.0);
    /// let mut result = poi1;
    /// result *= poi2;
    ///
    /// assert_eq!(result, Point4!(poi1.x * poi2.x, poi1.y * poi2.y, poi1.z * poi2.z))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Point4) {
        if check_sse2!() {
            unsafe {
                *self = transmute(_mm_mul_ps(transmute(*self), transmute(rhs)));
            }
        } else {
            self.x *= rhs.x;
            self.y *= rhs.y;
            self.z *= rhs.z;
        }
    }
}

impl std::ops::MulAssign<f32> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let mut result = poi;
    /// result *= f;
    ///
    /// assert_eq!(result, Point4!(poi.x * f, poi.y * f, poi.z * f))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(1.0, rhs, rhs, rhs);
                *self = transmute(_mm_mul_ps(transmute(*self), xmm));
            }
        } else {
            self.x *= rhs;
            self.y *= rhs;
            self.z *= rhs;
        }
    }
}

impl std::ops::MulAssign<(f32, f32, f32)> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(1.0, 2.0, 3.0);
    /// let tuple = (4.0, 5.0, 6.0);
    /// let mut result = poi;
    /// result *= tuple;
    ///
    /// assert_eq!(result, Point4!(poi.x * tuple.0, poi.y * tuple.1, poi.z * tuple.2))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: (f32, f32, f32)) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(1.0, rhs.2, rhs.1, rhs.0);
                *self = transmute(_mm_mul_ps(transmute(*self), xmm));
            }
        } else {
            self.x *= rhs.0;
            self.y *= rhs.1;
            self.z *= rhs.2;
        }
    }
}

/* ----------------------------------- Div ---------------------------------- */
impl std::ops::Div for Point4 {
    type Output = Point4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi1 = Point4!(4.0, 6.0, 8.0);
    /// let poi2 = Point4!(2.0, 3.0, 4.0);
    /// let result = poi1 / poi2;
    ///
    /// assert_eq!(result, Point4!(poi1.x / poi2.x, poi1.y / poi2.y, poi1.z / poi2.z))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let poi1 = Point4!(4.0, 6.0, 8.0);
    /// let poi2 = Point4!(0.0, 0.0, 0.0);
    /// let panic = poi1 / poi2;
    /// ```
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Point4 cannot be divided by 0"
        );

        Point4!(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl std::ops::Div<f32> for Point4 {
    type Output = Point4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(2.0, 4.0, 8.0);
    /// let f: f32 = 2.0;
    /// let result = poi / f;
    ///
    /// assert_eq!(result, Point4!(poi.x / f, poi.y / f, poi.z / f))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let poi = Point4!(2.0, 4.0, 8.0);
    /// let f: f32 = 0.0;
    /// let panic = poi / f;
    /// ```
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        debug_assert!(rhs != 0.0, "Point4 cannot be divided by 0");

        let inv = 1.0 / rhs;

        Point4!(self.x * inv, self.y * inv, self.z * inv)
    }
}

impl std::ops::Div<(f32, f32, f32)> for Point4 {
    type Output = Point4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(2.0, 4.0, 6.0);
    /// let tuple = (2.0, 2.0, 2.0);
    /// let result = poi / tuple;
    ///
    /// assert_eq!(result, Point4!(poi.x / tuple.0, poi.y / tuple.1, poi.z / tuple.2))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let poi = Point4!(2.0, 4.0, 6.0);
    /// let tuple = (0.0, 0.0, 0.0);
    /// let panic = poi / tuple;
    /// ```
    #[inline]
    fn div(self, rhs: (f32, f32, f32)) -> Self::Output {
        debug_assert!(
            rhs.0 != 0.0 && rhs.1 != 0.0 && rhs.2 != 0.0,
            "Point4 cannot be divided by 0"
        );

        Point4!(self.x / rhs.0, self.y / rhs.1, self.z / rhs.2)
    }
}

/* -------------------------------- DivAssign ------------------------------- */
impl std::ops::DivAssign<Point4> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi1 = Point4!(2.0, 4.0, 6.0);
    /// let poi2 = Point4!(1.0, 2.0, 3.0);
    /// let mut result = poi1;
    /// result /= poi2;
    ///
    /// assert_eq!(result, Point4!(poi1.x / poi2.x, poi1.y / poi2.y, poi1.z / poi2.z))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut panic = Point4!(2.0, 4.0, 6.0);
    /// let poi = Point4!(0.0, 0.0, 0.0);
    /// panic /= poi;
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Point4) {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Point4 cannot be divided by 0"
        );

        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl std::ops::DivAssign<f32> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(2.0, 4.0, 6.0);
    /// let f: f32 = 2.0;
    /// let mut result = poi;
    ///
    /// result /= f;
    ///
    /// assert_eq!(result, Point4!(poi.x / f, poi.y / f, poi.z / f))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut panic = Point4!(2.0, 4.0, 6.0);
    /// panic /= 0.0;
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        debug_assert!(rhs != 0.0, "Point4 cannot be divided by 0");
        let inv = 1.0 / rhs;

        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

impl std::ops::DivAssign<(f32, f32, f32)> for Point4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let poi = Point4!(2.0, 4.0, 6.0);
    /// let tuple = (1.0, 2.0, 3.0);
    /// let mut result = poi;
    /// result /= tuple;
    ///
    /// assert_eq!(result, Point4!(poi.x / tuple.0, poi.y / tuple.1, poi.z / tuple.2))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut panic = Point4!(2.0, 4.0, 6.0);
    /// panic /= (0.0, 0.0, 0.0);
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: (f32, f32, f32)) {
        debug_assert!(rhs.0 != 0.0 && rhs.1 != 0.0 && rhs.2 != 0.0);

        self.x /= rhs.0;
        self.y /= rhs.1;
        self.z /= rhs.2;
    }
}

/* -------------------------------------------------------------------------- */
/*                                   Normal4                                  */
/* -------------------------------------------------------------------------- */
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct Normal4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Normal4 {
    pub const ZERO: Normal4 = Normal4!(0.0);
    pub const ONE: Normal4 = Normal4!(1.0);
}

impl From<Raw4> for Normal4 {
    #[inline]
    fn from(raw: Raw4) -> Self { Normal4!(raw.x, raw.y, raw.z) }
}

impl From<Float4> for Normal4 {
    #[inline]
    fn from(float: Float4) -> Self { Normal4!(float.x, float.y, float.z) }
}

impl From<Vector4> for Normal4 {
    #[inline]
    fn from(vector: Vector4) -> Self { unsafe { transmute(vector) } }
}

impl From<Point4> for Normal4 {
    #[inline]
    fn from(point: Point4) -> Self { Normal4!(point.x, point.y, point.z) }
}

/* --------------------------------- Algebra -------------------------------- */
impl Normal4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    ///
    /// assert_eq!(nor.length(), (nor.x * nor.x + nor.y * nor.y + nor.z * nor.z).sqrt())
    /// ```
    #[inline]
    pub fn length(&self) -> f32 { (self.x * self.x + self.y * self.y + self.z * self.z).sqrt() }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    ///
    /// assert_eq!(nor.length_squared(), (nor.x * nor.x + nor.y * nor.y + nor.z * nor.z))
    /// ```
    #[inline]
    pub fn length_squared(&self) -> f32 { self.x * self.x + self.y * self.y + self.z * self.z }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let mut nor1 = Normal4!(1.0, 2.0, 3.0);
    /// let nor2 = nor1;
    /// let length = nor2.length();
    /// let result = *nor1.normalize();
    ///
    /// assert_eq!(result, Normal4!(nor2.x / length, nor2.y / length, nor2.z / length))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut panic = Normal4!(0.0, 0.0, 0.0);
    /// panic.normalize();
    /// ```
    #[inline]
    pub fn normalize(&mut self) -> &Normal4 {
        let length = self.length();

        debug_assert!(length != 0.0, "zero Normal4 cannot be normalized");

        self.div_assign(length);

        self
    }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let length = nor.length();
    /// let result = nor.norm();
    ///
    /// assert_eq!(result, Normal4!(nor.x / length, nor.y / length, nor.z / length))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let panic = Normal4!(0.0, 0.0, 0.0);
    /// panic.norm();
    /// ```
    #[inline]
    pub fn norm(&self) -> Normal4 {
        let length = self.length();

        debug_assert!(length != 0.0, "zero Normal4 cannot be normalized");

        self.div(length)
    }
}

impl Vector4ComponentAlgebra<Normal4> for Normal4 {
    type Output = Normal4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(1.0, 2.0, 3.0);
    /// let nor2 = Normal4!(4.0, 5.0, 6.0);
    /// let result = nor1.dot(nor2);
    ///
    /// assert_eq!(result, nor1.x * nor2.x + nor1.y * nor2.y + nor1.z * nor2.z)
    /// ```
    #[inline]
    fn dot(&self, rhs: Normal4) -> f32 { self.x * rhs.x + self.y * rhs.y + self.z * rhs.z }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(1.0, 2.0, 3.0);
    /// let nor2 = Normal4!(4.0, 5.0, 6.0);
    /// let result = nor1.cross(nor2);
    /// let expect = Normal4!(nor1.y * nor2.z - nor1.z * nor2.y,
    ///                       nor1.z * nor2.x - nor1.x * nor2.z,
    ///                       nor1.x * nor2.y - nor1.y * nor2.x);
    ///
    /// assert_eq!(result, expect);
    /// ```
    #[inline]
    fn cross(&self, rhs: Normal4) -> Normal4 {
        if check_sse2!() {
            const YZXW: i32 = swizzle!(1, 2, 0, 3);
            const ZXYW: i32 = swizzle!(2, 0, 1, 3);

            unsafe {
                let lhs: __m128 = transmute(*self);
                let rhs: __m128 = transmute(rhs);

                let v1_yzxw = _mm_shuffle_ps::<YZXW>(lhs, lhs);
                let v2_zxyw = _mm_shuffle_ps::<ZXYW>(rhs, rhs);

                let r1 = _mm_mul_ps(v1_yzxw, v2_zxyw);

                let v1_zxyw = _mm_shuffle_ps::<ZXYW>(lhs, lhs);
                let v2_yzxw = _mm_shuffle_ps::<YZXW>(rhs, rhs);

                let r2 = _mm_mul_ps(v1_zxyw, v2_yzxw);

                transmute(_mm_sub_ps(r1, r2))
            }
        } else {
            Normal4!(
                self.y * rhs.z - self.z * rhs.y,
                self.z * rhs.x - self.x * rhs.z,
                self.x * rhs.y - self.y * rhs.x
            )
        }
    }
}

impl Vector4ComponentAlgebra<Vector4> for Normal4 {
    type Output = Normal4;

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let vec = Vector4!(4.0, 5.0, 6.0);
    /// let result = nor.dot(vec);
    ///
    /// assert_eq!(result, nor.x * vec.x + nor.y * vec.y + nor.z * vec.z)
    /// ```
    #[inline]
    fn dot(&self, rhs: Vector4) -> f32 { self.x * rhs.x + self.y * rhs.y + self.z * rhs.z }

    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let vec = Vector4!(4.0, 5.0, 6.0);
    /// let result = nor.cross(vec);
    /// let expect = Normal4!(nor.y * vec.z - nor.z * vec.y,
    ///                       nor.z * vec.x - nor.x * vec.z,
    ///                       nor.x * vec.y - nor.y * vec.x);
    ///
    /// assert_eq!(result, expect);
    /// ```
    #[inline]
    fn cross(&self, rhs: Vector4) -> Normal4 {
        if check_sse2!() {
            const YZXW: i32 = swizzle!(1, 2, 0, 3);
            const ZXYW: i32 = swizzle!(2, 0, 1, 3);

            unsafe {
                let lhs: __m128 = transmute(*self);
                let rhs: __m128 = transmute(rhs);

                let v1_yzxw = _mm_shuffle_ps::<YZXW>(lhs, lhs);
                let v2_zxyw = _mm_shuffle_ps::<ZXYW>(rhs, rhs);

                let r1 = _mm_mul_ps(v1_yzxw, v2_zxyw);

                let v1_zxyw = _mm_shuffle_ps::<ZXYW>(lhs, lhs);
                let v2_yzxw = _mm_shuffle_ps::<YZXW>(rhs, rhs);

                let r2 = _mm_mul_ps(v1_zxyw, v2_yzxw);

                transmute(_mm_sub_ps(r1, r2))
            }
        } else {
            Normal4!(
                self.y * rhs.z - self.z * rhs.y,
                self.z * rhs.x - self.x * rhs.z,
                self.x * rhs.y - self.y * rhs.x
            )
        }
    }
}

/* ---------------------------------- Unary --------------------------------- */
impl std::ops::Neg for Normal4 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_set_ps(0.0, -self.z, -self.y, -self.x)) }
        } else {
            Normal4!(-self.x, -self.y, -self.z)
        }
    }
}

impl std::ops::Index<usize> for Normal4 {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("out of range of Normal4's index"),
        }
    }
}

impl std::ops::IndexMut<usize> for Normal4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("out of range of Normal4's index"),
        }
    }
}

/* ----------------------------------- Add ---------------------------------- */
impl std::ops::Add for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(1.0, 2.0, 3.0);
    /// let nor2 = Normal4!(4.0, 5.0, 6.0);
    /// let result = nor1 + nor2;
    ///
    /// assert_eq!(result, Normal4!(nor1.x + nor2.x, nor1.y + nor2.y, nor1.z + nor2.z))
    /// ```
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_add_ps(transmute(self), transmute(rhs))) }
        } else {
            Normal4!(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
        }
    }
}

impl std::ops::Add<f32> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let result = nor + f;
    ///
    /// assert_eq!(result, Normal4!(nor.x + f, nor.y + f, nor.z + f))
    /// ```
    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                transmute(_mm_add_ps(transmute(self), xmm))
            }
        } else {
            Normal4!(self.x + rhs, self.y + rhs, self.z + rhs)
        }
    }
}

impl std::ops::Add<Normal4> for f32 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 1.0;
    /// let nor = Normal4!(2.0, 3.0, 4.0);
    /// let result = f + nor;
    ///
    /// assert_eq!(result, Normal4!(f + nor.x, f + nor.y, f + nor.z))
    /// ```
    #[inline]
    fn add(self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, self, self, self);
                transmute(_mm_add_ps(xmm, transmute(rhs)))
            }
        } else {
            Normal4!(self + rhs.x, self + rhs.y, self + rhs.z)
        }
    }
}

impl std::ops::Add<Point4> for Normal4 {
    type Output = Point4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let poi = Point4!(4.0, 5.0, 6.0);
    /// let result = nor + poi;
    ///
    /// assert_eq!(result, Point4!(nor.x + poi.x, nor.y + poi.y, nor.z + poi.z))
    /// ```
    #[inline]
    fn add(self, rhs: Point4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_add_ps(transmute(self), transmute(rhs))) }
        } else {
            Point4!(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
        }
    }
}

/* -------------------------------- AddAssign ------------------------------- */
impl std::ops::AddAssign<Normal4> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(1.0, 2.0, 3.0);
    /// let nor2 = Normal4!(4.0, 5.0, 6.0);
    /// let mut result = nor1;
    /// result += nor2;
    ///
    /// assert_eq!(result, Normal4!(nor1.x + nor2.x, nor1.y + nor2.y, nor1.z + nor2.z))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Normal4) {
        if check_sse2!() {
            *self = unsafe { transmute(_mm_add_ps(transmute(*self), transmute(rhs))) };
        } else {
            self.x += rhs.x;
            self.y += rhs.y;
            self.z += rhs.z;
        }
    }
}

impl std::ops::AddAssign<f32> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let mut result = nor;
    /// result += f;
    ///
    /// assert_eq!(result, Normal4!(nor.x + f, nor.y + f, nor.z + f))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                *self = transmute(_mm_add_ps(transmute(*self), xmm));
            }
        } else {
            self.x += rhs;
            self.y += rhs;
            self.z += rhs;
        }
    }
}

/* ----------------------------------- Sub ---------------------------------- */
impl std::ops::Sub<Normal4> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(4.0, 5.0, 6.0);
    /// let nor2 = Normal4!(1.0, 2.0, 3.0);
    /// let result = nor1 - nor2;
    ///
    /// assert_eq!(result, Normal4!(nor1.x - nor2.x, nor1.y - nor2.y, nor1.z - nor2.z))
    /// ```
    #[inline]
    fn sub(self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_sub_ps(transmute(self), transmute(rhs))) }
        } else {
            Normal4!(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
}

impl std::ops::Sub<Vector4> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let result = nor - vec;
    ///
    /// assert_eq!(result, Normal4!(nor.x - vec.x, nor.y - vec.y, nor.z - vec.z))
    /// ```
    #[inline]
    fn sub(self, rhs: Vector4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_sub_ps(transmute(self), transmute(rhs))) }
        } else {
            Normal4!(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
        }
    }
}

impl std::ops::Sub<f32> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let f: f32 = 1.0;
    /// let result = nor - f;
    ///
    /// assert_eq!(result, Normal4!(nor.x - f, nor.y - f, nor.z - f))
    /// ```
    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                transmute(_mm_sub_ps(transmute(self), xmm))
            }
        } else {
            Normal4!(self.x - rhs, self.y - rhs, self.z - rhs)
        }
    }
}

impl std::ops::Sub<Normal4> for f32 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 4.0;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let result = f - nor;
    ///
    /// assert_eq!(result, Normal4!(f - nor.x, f - nor.y, f - nor.z))
    /// ```
    #[inline]
    fn sub(self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, self, self, self);
                transmute(_mm_sub_ps(xmm, transmute(rhs)))
            }
        } else {
            Normal4!(self - rhs.x, self - rhs.y, self - rhs.z)
        }
    }
}

/* -------------------------------- SubAssign ------------------------------- */
impl std::ops::SubAssign<Normal4> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(4.0, 5.0, 6.0);
    /// let nor2 = Normal4!(1.0, 2.0, 3.0);
    /// let mut result = nor1;
    /// result -= nor2;
    ///
    /// assert_eq!(result, Normal4!(nor1.x - nor2.x, nor1.y - nor2.y, nor1.z - nor2.z))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Normal4) {
        if check_sse2!() {
            unsafe {
                *self = transmute(_mm_sub_ps(transmute(*self), transmute(rhs)));
            }
        } else {
            self.x -= rhs.x;
            self.y -= rhs.y;
            self.z -= rhs.z;
        }
    }
}

impl std::ops::SubAssign<Vector4> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let vec = Vector4!(1.0, 2.0, 3.0);
    /// let mut result = nor;
    /// result -= vec;
    ///
    /// assert_eq!(result, Normal4!(nor.x - vec.x, nor.y - vec.y, nor.z - vec.z))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Vector4) {
        if check_sse2!() {
            unsafe {
                *self = transmute(_mm_sub_ps(transmute(*self), transmute(rhs)));
            }
        } else {
            self.x -= rhs.x;
            self.y -= rhs.y;
            self.z -= rhs.z;
        }
    }
}

impl std::ops::SubAssign<f32> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 5.0, 6.0);
    /// let f: f32 = 1.0;
    /// let mut result = nor;
    /// result -= f;
    ///
    /// assert_eq!(result, Normal4!(nor.x - f, nor.y - f, nor.z - f))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                *self = transmute(_mm_sub_ps(transmute(*self), xmm));
            }
        } else {
            self.x -= rhs;
            self.y -= rhs;
            self.z -= rhs;
        }
    }
}

/* ----------------------------------- Mul ---------------------------------- */
impl std::ops::Mul<Normal4> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(1.0, 2.0, 3.0);
    /// let nor2 = Normal4!(4.0, 5.0, 6.0);
    /// let result = nor1 * nor2;
    ///
    /// assert_eq!(result, Normal4!(nor1.x * nor2.x, nor1.y * nor2.y, nor1.z * nor2.z))
    /// ```
    #[inline]
    fn mul(self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_mul_ps(transmute(self), transmute(rhs))) }
        } else {
            Normal4!(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
        }
    }
}

impl std::ops::Mul<Vector4> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let vec = Vector4!(4.0, 5.0, 6.0);
    /// let result = nor * vec;
    ///
    /// assert_eq!(result, Normal4!(nor.x * vec.x, nor.y * vec.y, nor.z * vec.z))
    /// ```
    #[inline]
    fn mul(self, rhs: Vector4) -> Self::Output {
        if check_sse2!() {
            unsafe { transmute(_mm_mul_ps(transmute(self), transmute(rhs))) }
        } else {
            Normal4!(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
        }
    }
}

impl std::ops::Mul<f32> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let result = nor * f;
    ///
    /// assert_eq!(result, Normal4!(nor.x * f, nor.y * f, nor.z * f))
    /// ```
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                transmute(_mm_mul_ps(transmute(self), xmm))
            }
        } else {
            Normal4!(self.x * rhs, self.y * rhs, self.z * rhs)
        }
    }
}

impl std::ops::Mul<(f32, f32, f32)> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let tuple = (4.0, 5.0, 6.0);
    /// let result = nor * tuple;
    ///
    /// assert_eq!(result, Normal4!(nor.x * tuple.0, nor.y * tuple.1, nor.z * tuple.2))
    /// ```
    #[inline]
    fn mul(self, rhs: (f32, f32, f32)) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs.2, rhs.1, rhs.0);
                transmute(_mm_mul_ps(transmute(self), xmm))
            }
        } else {
            Normal4!(self.x * rhs.0, self.y * rhs.1, self.z * rhs.2)
        }
    }
}

impl std::ops::Mul<Normal4> for f32 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 4.0;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let result = f * nor;
    ///
    /// assert_eq!(result, Normal4!(f * nor.x, f * nor.y, f * nor.z))
    /// ```
    #[inline]
    fn mul(self, rhs: Normal4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, self, self, self);
                transmute(_mm_mul_ps(xmm, transmute(rhs)))
            }
        } else {
            Normal4!(self * rhs.x, self * rhs.y, self * rhs.z)
        }
    }
}

/* -------------------------------- MulAssign ------------------------------- */
impl std::ops::MulAssign<Normal4> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(1.0, 2.0, 3.0);
    /// let nor2 = Normal4!(4.0, 5.0, 6.0);
    /// let mut result = nor1;
    /// result *= nor2;
    ///
    /// assert_eq!(result, Normal4!(nor1.x * nor2.x, nor1.y * nor2.y, nor1.z * nor2.z))
    #[inline]
    fn mul_assign(&mut self, rhs: Normal4) {
        if check_sse2!() {
            unsafe {
                *self = transmute(_mm_mul_ps(transmute(*self), transmute(rhs)));
            }
        } else {
            self.x *= rhs.x;
            self.y *= rhs.y;
            self.z *= rhs.z;
        }
    }
}

impl std::ops::MulAssign<Vector4> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let vec = Vector4!(4.0, 5.0, 6.0);
    /// let mut result = nor;
    /// result *= vec;
    ///
    /// assert_eq!(result, Normal4!(nor.x * vec.x, nor.y * vec.y, nor.z * vec.z))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Vector4) {
        if check_sse2!() {
            unsafe {
                *self = transmute(_mm_mul_ps(transmute(*self), transmute(rhs)));
            }
        } else {
            self.x *= rhs.x;
            self.y *= rhs.y;
            self.z *= rhs.z;
        }
    }
}

impl std::ops::MulAssign<f32> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let f: f32 = 4.0;
    /// let mut result = nor;
    /// result *= f;
    ///
    /// assert_eq!(result, Normal4!(nor.x * f, nor.y * f, nor.z * f))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs, rhs, rhs);
                *self = transmute(_mm_mul_ps(transmute(*self), xmm));
            }
        } else {
            self.x *= rhs;
            self.y *= rhs;
            self.z *= rhs;
        }
    }
}

impl std::ops::MulAssign<(f32, f32, f32)> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(1.0, 2.0, 3.0);
    /// let tuple = (4.0, 5.0, 6.0);
    /// let mut result = nor;
    /// result *= tuple;
    ///
    /// assert_eq!(result, Normal4!(nor.x * tuple.0, nor.y * tuple.1, nor.z * tuple.2))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: (f32, f32, f32)) {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set_ps(0.0, rhs.2, rhs.1, rhs.0);
                *self = transmute(_mm_mul_ps(transmute(*self), xmm));
            }
        } else {
            self.x *= rhs.0;
            self.y *= rhs.1;
            self.z *= rhs.2;
        }
    }
}

/* ----------------------------------- Div ---------------------------------- */
impl std::ops::Div<Normal4> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(4.0, 6.0, 8.0);
    /// let nor2 = Normal4!(2.0, 3.0, 4.0);
    /// let result = nor1 / nor2;
    ///
    /// assert_eq!(result, Normal4!(nor1.x / nor2.x, nor1.y / nor2.y, nor1.z / nor2.z))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let nor1 = Normal4!(4.0, 6.0, 8.0);
    /// let nor2 = Normal4!(0.0, 0.0, 0.0);
    /// let panic = nor1 / nor2;
    /// ```
    #[inline]
    fn div(self, rhs: Normal4) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Normal4 cannot be divided by 0"
        );

        Normal4!(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl std::ops::Div<Vector4> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 6.0, 8.0);
    /// let vec = Vector4!(2.0, 3.0, 4.0);
    /// let result = nor / vec;
    ///
    /// assert_eq!(result, Normal4!(nor.x / vec.x, nor.y / vec.y, nor.z / vec.z))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 6.0, 8.0);
    /// let vec = Vector4!(0.0, 0.0, 0.0);
    /// let panic = nor / vec;
    /// ```
    #[inline]
    fn div(self, rhs: Vector4) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Normal4 cannot be divided by 0"
        );

        Normal4!(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl std::ops::Div<f32> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 6.0, 8.0);
    /// let f: f32 = 2.0;
    /// let result = nor / f;
    ///
    /// assert_eq!(result, Normal4!(nor.x / f, nor.y / f, nor.z / f))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 6.0, 8.0);
    /// let panic = nor / 0.0;
    /// ```
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        debug_assert!(rhs != 0.0, "Normal4 cannot be divided by 0");
        let inv = 1.0 / rhs;

        Normal4!(self.x * inv, self.y * inv, self.z * inv)
    }
}

impl std::ops::Div<(f32, f32, f32)> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 6.0, 8.0);
    /// let tuple = (2.0, 3.0, 4.0);
    /// let result = nor / tuple;
    ///
    /// assert_eq!(result, Normal4!(nor.x / tuple.0, nor.y / tuple.1, nor.z / tuple.2))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 6.0, 8.0);
    /// let tuple = (0.0, 0.0, 0.0);
    /// let panic = nor / tuple;
    /// ```
    #[inline]
    fn div(self, rhs: (f32, f32, f32)) -> Self::Output {
        debug_assert!(
            rhs.0 != 0.0 && rhs.1 != 0.0 && rhs.2 != 0.0,
            "Normal4 cannot be divided by 0"
        );

        Normal4!(self.x / rhs.0, self.y / rhs.1, self.z / rhs.2)
    }
}

impl std::ops::Div<Normal4> for f32 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 1.0;
    /// let nor = Normal4!(2.0, 3.0, 4.0);
    /// let result = f / nor;
    ///
    /// assert_eq!(result, Normal4!(f / nor.x, f / nor.y, f / nor.z))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let f: f32 = 1.0;
    /// let nor = Normal4!(0.0, 0.0, 0.0);
    /// let panic = f / nor;
    /// ```
    #[inline]
    fn div(self, rhs: Normal4) -> Self::Output {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Normal4 cannot be divided by 0"
        );

        Normal4!(self / rhs.x, self / rhs.y, self / rhs.z)
    }
}

/* -------------------------------- DivAssign ------------------------------- */
impl std::ops::DivAssign<Normal4> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(4.0, 6.0, 8.0);
    /// let nor2 = Normal4!(2.0, 3.0, 4.0);
    /// let mut result = nor1;
    /// result /= nor2;
    ///
    /// assert_eq!(result, Normal4!(nor1.x / nor2.x, nor1.y / nor2.y, nor1.z / nor2.z))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let nor1 = Normal4!(4.0, 6.0, 8.0);
    /// let nor2 = Normal4!(0.0, 0.0, 0.0);
    /// let mut panic = nor1;
    /// panic /= nor2;
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Normal4) {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Normal4 cannot be divided by 0"
        );

        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl std::ops::DivAssign<Vector4> for Normal4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let nor1 = Normal4!(4.0, 6.0, 8.0);
    /// let nor2 = Vector4!(2.0, 3.0, 4.0);
    /// let mut result = nor1;
    /// result /= nor2;
    ///
    /// assert_eq!(result, Normal4!(nor1.x / nor2.x, nor1.y / nor2.y, nor1.z / nor2.z))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 6.0, 8.0);
    /// let vec = Vector4!(0.0, 0.0, 0.0);
    /// let mut panic = nor;
    /// panic /= vec;
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Vector4) {
        debug_assert!(
            rhs.x != 0.0 && rhs.y != 0.0 && rhs.z != 0.0,
            "Normal4 cannot be divided by 0"
        );

        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl std::ops::DivAssign<f32> for Normal4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 6.0, 8.0);
    /// let f: f32 = 2.0;
    /// let mut result = nor;
    /// result /= f;
    ///
    /// assert_eq!(result, Normal4!(nor.x / f, nor.y / f, nor.z / f))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut panic = Normal4!(1.0, 2.0, 3.0);
    /// let f: f32 = 0.0;
    /// panic /= f;
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        debug_assert!(rhs != 0.0, "Normal4 cannot be divided by 0");
        let inv = 1.0 / rhs;

        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

impl std::ops::DivAssign<(f32, f32, f32)> for Normal4 {
    /// # Examples
    ///
    /// ```
    /// use ssun_math::*;
    /// let nor = Normal4!(4.0, 6.0, 8.0);
    /// let tuple = (2.0, 3.0, 4.0);
    /// let mut result = nor;
    /// result /= tuple;
    ///
    /// assert_eq!(result, Normal4!(nor.x / tuple.0, nor.y / tuple.1, nor.z / tuple.2))
    /// ```
    ///
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mut panic = Normal4!(4.0, 6.0, 8.0);
    /// panic /= (0.0, 0.0, 0.0);
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: (f32, f32, f32)) {
        debug_assert!(
            rhs.0 != 0.0 && rhs.1 != 0.0 && rhs.2 != 0.0,
            "Normal4 cannot be divided by 0"
        );

        self.x /= rhs.0;
        self.y /= rhs.1;
        self.z /= rhs.2;
    }
}
