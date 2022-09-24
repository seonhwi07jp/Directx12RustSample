use crate::{float3::*, float4::*, *};
use std::arch::x86_64::{__m128, _mm_add_ps, _mm_mul_ps, _mm_set1_ps, _mm_shuffle_ps, _mm_sub_ps};
use std::intrinsics::transmute;
use std::mem::MaybeUninit;

/* -------------------------------------------------------------------------- */
/*                                   macros                                   */
/* -------------------------------------------------------------------------- */
#[macro_export]
macro_rules! Matrix4x4 {
    ($m00:expr, $m01:expr, $m02:expr, $m03:expr,
    $m10:expr, $m11:expr, $m12:expr, $m13:expr,
    $m20:expr, $m21:expr, $m22:expr, $m23:expr,
    $m30:expr, $m31:expr, $m32:expr, $m33:expr) => {
        Matrix4x4 {
            m00: $m00,
            m01: $m01,
            m02: $m02,
            m03: $m03,
            m10: $m10,
            m11: $m11,
            m12: $m12,
            m13: $m13,
            m20: $m20,
            m21: $m21,
            m22: $m22,
            m23: $m23,
            m30: $m30,
            m31: $m31,
            m32: $m32,
            m33: $m33,
        }
    };
    ($val:expr) => {
        Matrix4x4 {
            m00: $val,
            m01: $val,
            m02: $val,
            m03: $val,
            m10: $val,
            m11: $val,
            m12: $val,
            m13: $val,
            m20: $val,
            m21: $val,
            m22: $val,
            m23: $val,
            m30: $val,
            m31: $val,
            m32: $val,
            m33: $val,
        }
    };
    () => {
        Matrix4x4 {
            m00: 1.0,
            m01: 0.0,
            m02: 0.0,
            m03: 0.0,
            m10: 0.0,
            m11: 1.0,
            m12: 0.0,
            m13: 0.0,
            m20: 0.0,
            m21: 0.0,
            m22: 1.0,
            m23: 0.0,
            m30: 0.0,
            m31: 0.0,
            m32: 0.0,
            m33: 1.0,
        }
    };
}

#[macro_export]
macro_rules! inverse {
    ($x:expr) => {
        $x.inverse()
    };
}

#[macro_export]
macro_rules! transpose {
    ($x:expr) => {
        $x.transpose()
    };
}

/* -------------------------------------------------------------------------- */
/*                                struct Matrix                               */
/* -------------------------------------------------------------------------- */
#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C, align(16))]
pub struct Matrix4x4 {
    pub m00: f32,
    pub m01: f32,
    pub m02: f32,
    pub m03: f32,
    pub m10: f32,
    pub m11: f32,
    pub m12: f32,
    pub m13: f32,
    pub m20: f32,
    pub m21: f32,
    pub m22: f32,
    pub m23: f32,
    pub m30: f32,
    pub m31: f32,
    pub m32: f32,
    pub m33: f32,
}

impl Default for Matrix4x4 {
    #[inline]
    fn default() -> Self { Matrix4x4!(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0) }
}

impl Matrix4x4 {
    pub const ZERO: Matrix4x4 = Matrix4x4!(0.0);
    pub const IDENTITY: Matrix4x4 = Matrix4x4!();

    #[cfg(target_feature = "sse2")]
    #[inline]
    pub fn from_xmm(r1: __m128, r2: __m128, r3: __m128, r4: __m128) -> Matrix4x4 {
        unsafe { transmute((r1, r2, r3, r4)) }
    }

    #[cfg(target_feature = "sse2")]
    #[inline]
    pub fn into_xmm(&self) -> (__m128, __m128, __m128, __m128) { unsafe { transmute(*self) } }
}

/* --------------------------------- Algebra -------------------------------- */
impl Matrix4x4 {
    #[inline]
    pub fn transpose(&self) -> Matrix4x4 {
        Matrix4x4!(
            self.m00, self.m10, self.m20, self.m30, self.m01, self.m11, self.m21, self.m31, self.m02, self.m12,
            self.m22, self.m32, self.m03, self.m13, self.m23, self.m33
        )
    }

    #[inline]
    pub fn det(&self) -> f32 {
        (self.m00 * self.m11 * self.m22 * self.m33)
            + (self.m00 * self.m12 * self.m23 * self.m31)
            + (self.m00 * self.m13 * self.m21 * self.m32)
            - (self.m00 * self.m13 * self.m22 * self.m31)
            - (self.m00 * self.m12 * self.m21 * self.m33)
            - (self.m00 * self.m11 * self.m23 * self.m32)
            - (self.m01 * self.m10 * self.m22 * self.m33)
            - (self.m02 * self.m10 * self.m23 * self.m31)
            - (self.m03 * self.m10 * self.m21 * self.m32)
            + (self.m03 * self.m10 * self.m22 * self.m31)
            + (self.m02 * self.m10 * self.m21 * self.m33)
            + (self.m01 * self.m10 * self.m23 * self.m32)
            + (self.m01 * self.m12 * self.m20 * self.m33)
            + (self.m02 * self.m13 * self.m20 * self.m31)
            + (self.m03 * self.m11 * self.m20 * self.m32)
            - (self.m03 * self.m12 * self.m20 * self.m31)
            - (self.m02 * self.m11 * self.m20 * self.m33)
            - (self.m01 * self.m13 * self.m20 * self.m32)
            - (self.m01 * self.m12 * self.m23 * self.m30)
            - (self.m02 * self.m13 * self.m21 * self.m30)
            - (self.m03 * self.m11 * self.m22 * self.m30)
            + (self.m03 * self.m12 * self.m21 * self.m30)
            + (self.m02 * self.m11 * self.m23 * self.m30)
            + (self.m01 * self.m13 * self.m22 * self.m30)
    }

    #[inline]
    pub fn inverse(&self) -> Matrix4x4 {
        let det = self.det();

        debug_assert!(
            det != 0.0,
            "Matrix4x4 with determinant 0 cannot convert to inverse matrix"
        );

        let temp = Matrix4x4!(
            // 00
            (self.m11 * self.m22 * self.m33) + (self.m12 * self.m23 * self.m31) + (self.m13 * self.m21 * self.m32)
                - (self.m13 * self.m22 * self.m31)
                - (self.m12 * self.m21 * self.m33)
                - (self.m11 * self.m23 * self.m32),
            // 01
            -(self.m01 * self.m22 * self.m33) - (self.m02 * self.m23 * self.m31) - (self.m03 * self.m21 * self.m32)
                + (self.m03 * self.m22 * self.m31)
                + (self.m02 * self.m21 * self.m33)
                + (self.m01 * self.m23 * self.m32),
            //02
            (self.m01 * self.m12 * self.m33) + (self.m02 * self.m13 * self.m31) + (self.m03 * self.m11 * self.m32)
                - (self.m03 * self.m12 * self.m31)
                - (self.m02 * self.m11 * self.m33)
                - (self.m01 * self.m13 * self.m32),
            // 03
            -(self.m01 * self.m12 * self.m23) - (self.m02 * self.m13 * self.m21) - (self.m03 * self.m11 * self.m22)
                + (self.m03 * self.m12 * self.m21)
                + (self.m02 * self.m11 * self.m23)
                + (self.m01 * self.m13 * self.m22),
            // 10
            -(self.m10 * self.m22 * self.m33) - (self.m12 * self.m23 * self.m30) - (self.m13 * self.m20 * self.m32)
                + (self.m13 * self.m22 * self.m30)
                + (self.m12 * self.m20 * self.m33)
                + (self.m10 * self.m23 * self.m32),
            // 11
            (self.m00 * self.m22 * self.m33) + (self.m02 * self.m23 * self.m30) + (self.m03 * self.m20 * self.m32)
                - (self.m03 * self.m22 * self.m30)
                - (self.m02 * self.m20 * self.m33)
                - (self.m00 * self.m23 * self.m32),
            // 12
            -(self.m00 * self.m12 * self.m33) - (self.m02 * self.m13 * self.m30) - (self.m03 * self.m10 * self.m32)
                + (self.m03 * self.m12 * self.m30)
                + (self.m02 * self.m10 * self.m33)
                + (self.m00 * self.m13 * self.m32),
            // 13
            (self.m00 * self.m12 * self.m23) + (self.m02 * self.m13 * self.m20) + (self.m03 * self.m10 * self.m22)
                - (self.m03 * self.m12 * self.m20)
                - (self.m02 * self.m10 * self.m23)
                - (self.m00 * self.m13 * self.m22),
            // 20
            (self.m10 * self.m21 * self.m33) + (self.m11 * self.m23 * self.m30) + (self.m13 * self.m20 * self.m31)
                - (self.m13 * self.m21 * self.m30)
                - (self.m11 * self.m20 * self.m33)
                - (self.m10 * self.m23 * self.m31),
            // 21
            -(self.m00 * self.m21 * self.m33) - (self.m01 * self.m23 * self.m30) - (self.m03 * self.m20 * self.m31)
                + (self.m03 * self.m21 * self.m30)
                + (self.m01 * self.m20 * self.m33)
                + (self.m00 * self.m23 * self.m31),
            // 22
            (self.m00 * self.m11 * self.m33) + (self.m01 * self.m13 * self.m30) + (self.m03 * self.m10 * self.m31)
                - (self.m03 * self.m11 * self.m30)
                - (self.m01 * self.m10 * self.m33)
                - (self.m00 * self.m13 * self.m31),
            // 23
            -(self.m00 * self.m11 * self.m23) - (self.m01 * self.m13 * self.m20) - (self.m03 * self.m10 * self.m21)
                + (self.m03 * self.m11 * self.m20)
                + (self.m01 * self.m10 * self.m23)
                + (self.m00 * self.m13 * self.m21),
            // 30
            -(self.m10 * self.m21 * self.m32) - (self.m11 * self.m22 * self.m30) - (self.m12 * self.m20 * self.m31)
                + (self.m12 * self.m21 * self.m30)
                + (self.m11 * self.m20 * self.m32)
                + (self.m10 * self.m22 * self.m31),
            // 31
            (self.m00 * self.m21 * self.m32) + (self.m01 * self.m22 * self.m30) + (self.m02 * self.m20 * self.m31)
                - (self.m02 * self.m21 * self.m30)
                - (self.m01 * self.m20 * self.m32)
                - (self.m00 * self.m22 * self.m31),
            // 32
            -(self.m00 * self.m11 * self.m32) - (self.m01 * self.m12 * self.m30) - (self.m02 * self.m10 * self.m31)
                + (self.m02 * self.m11 * self.m30)
                + (self.m01 * self.m10 * self.m32)
                + (self.m00 * self.m12 * self.m31),
            // 33
            (self.m00 * self.m11 * self.m22) + (self.m01 * self.m12 * self.m20) + (self.m02 * self.m10 * self.m21)
                - (self.m02 * self.m11 * self.m20)
                - (self.m01 * self.m10 * self.m22)
                - (self.m00 * self.m12 * self.m21)
        );

        temp * (1.0 / det)
    }

    #[inline]
    pub fn translate(x: f32, y: f32, z: f32) -> Matrix4x4 {
        Matrix4x4!(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0)
    }

    #[inline]
    pub fn translate_from_float3(float3: Float3) -> Matrix4x4 {
        Matrix4x4!(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, float3.x, float3.y, float3.z, 1.0)
    }

    #[inline]
    pub fn scale(x: f32, y: f32, z: f32) -> Matrix4x4 {
        Matrix4x4!(x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 0.0, z, 0.0, 0.0, 0.0, 0.0, 1.0)
    }

    #[inline]
    pub fn scale_from_float3(float3: Float3) -> Matrix4x4 {
        Matrix4x4!(float3.x, 0.0, 0.0, 0.0, 0.0, float3.y, 0.0, 0.0, 0.0, 0.0, float3.z, 0.0, 0.0, 0.0, 0.0, 1.0)
    }

    #[inline]
    pub fn rotate_x_by_deg(deg: f32) -> Matrix4x4 {
        let sin_theta = radian(deg).sin();
        let cos_theta = radian(deg).cos();

        Matrix4x4!(
            1.0, 0.0, 0.0, 0.0, 0.0, cos_theta, sin_theta, 0.0, 0.0, -sin_theta, cos_theta, 0.0, 0.0, 0.0, 0.0, 1.0
        )
    }

    #[inline]
    pub fn rotate_x_by_rad(rad: f32) -> Matrix4x4 {
        let sin_theta = rad.sin();
        let cos_theta = rad.cos();

        Matrix4x4!(
            1.0, 0.0, 0.0, 0.0, 0.0, cos_theta, sin_theta, 0.0, 0.0, -sin_theta, cos_theta, 0.0, 0.0, 0.0, 0.0, 1.0
        )
    }

    #[inline]
    pub fn rotate_y_by_deg(deg: f32) -> Matrix4x4 {
        let sin_theta = radian(deg).sin();
        let cos_theta = radian(deg).cos();

        Matrix4x4!(
            cos_theta, 0.0, -sin_theta, 0.0, 0.0, 1.0, 0.0, 0.0, sin_theta, 0.0, cos_theta, 0.0, 0.0, 0.0, 0.0, 1.0
        )
    }

    #[inline]
    pub fn rotate_y_by_rad(rad: f32) -> Matrix4x4 {
        let sin_theta = rad.sin();
        let cos_theta = rad.cos();

        Matrix4x4!(
            cos_theta, 0.0, -sin_theta, 0.0, 0.0, 1.0, 0.0, 0.0, sin_theta, 0.0, cos_theta, 0.0, 0.0, 0.0, 0.0, 1.0
        )
    }

    #[inline]
    pub fn rotate_z_by_deg(deg: f32) -> Matrix4x4 {
        let sin_theta = radian(deg).sin();
        let cos_theta = radian(deg).cos();

        Matrix4x4!(
            cos_theta, sin_theta, 0.0, 0.0, -sin_theta, cos_theta, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0
        )
    }

    #[inline]
    pub fn rotate_z_by_rad(rad: f32) -> Matrix4x4 {
        let sin_theta = rad.sin();
        let cos_theta = rad.cos();

        Matrix4x4!(
            cos_theta, sin_theta, 0.0, 0.0, -sin_theta, cos_theta, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0
        )
    }

    #[inline]
    pub fn rotate_by_deg(axis: Float3, deg: f32) -> Matrix4x4 {
        let a = axis.norm();

        let sin_theta = radian(deg).sin();
        let cos_theta = radian(deg).cos();

        Matrix4x4!(
            a.x * a.x + (1.0 - a.x * a.x) * cos_theta,
            a.x * a.y * (1.0 - cos_theta) + a.z * sin_theta,
            a.x * a.z * (1.0 - cos_theta) - a.y * sin_theta,
            0.0,
            a.x * a.y + (1.0 - cos_theta) - a.z * sin_theta,
            a.y * a.y + (1.0 - a.y * a.y) * cos_theta,
            a.y * a.z * (1.0 - cos_theta) + a.x * sin_theta,
            0.0,
            a.x * a.z + (1.0 - cos_theta) + a.y * sin_theta,
            a.y * a.z * (1.0 - cos_theta) - a.x * sin_theta,
            a.z * a.z + (1.0 - a.z * a.z) * cos_theta,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0
        )
    }

    #[inline]
    pub fn rotate_by_rad(axis: Float3, rad: f32) -> Matrix4x4 {
        let a = axis.norm();

        let sin_theta = rad.sin();
        let cos_theta = rad.cos();

        Matrix4x4!(
            a.x * a.x + (1.0 - a.x * a.x) * cos_theta,
            a.x * a.y * (1.0 - cos_theta) + a.z * sin_theta,
            a.x * a.z * (1.0 - cos_theta) - a.y * sin_theta,
            0.0,
            a.x * a.y + (1.0 - cos_theta) - a.z * sin_theta,
            a.y * a.y + (1.0 - a.y * a.y) * cos_theta,
            a.y * a.z * (1.0 - cos_theta) + a.x * sin_theta,
            0.0,
            a.x * a.z + (1.0 - cos_theta) + a.y * sin_theta,
            a.y * a.z * (1.0 - cos_theta) - a.x * sin_theta,
            a.z * a.z + (1.0 - a.z * a.z) * cos_theta,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0
        )
    }

    #[inline]
    pub fn look_at(origin: Point4, target: Point4, up: Vector4, right: Vector4) -> Matrix4x4 {
        let new_forward = norm!(target - origin);
        let mut new_right = norm!(cross!(up, new_forward));
        let mut new_up = norm!(cross!(new_forward, new_right));

        if new_right.length_squared() == 0.0 {
            new_right = right;

            new_up = norm!(cross!(new_forward, new_right));
            new_right = norm!(cross!(new_up, new_forward));
        }

        Matrix4x4!(
            new_right.x,
            new_up.x,
            new_forward.x,
            0.0,
            new_right.y,
            new_up.y,
            new_forward.y,
            0.0,
            new_right.z,
            new_up.z,
            new_forward.z,
            0.0,
            -origin.x,
            -origin.y,
            -origin.z,
            1.0
        )
    }

    #[inline]
    pub fn perspective_by_deg(fov: f32, ratio: f32, n: f32, f: f32) -> Matrix4x4 {
        let inv_tan = 1.0 / (radian(fov) * 0.5).tan();

        Matrix4x4!(
            inv_tan / ratio,
            0.0,
            0.0,
            0.0,
            0.0,
            inv_tan,
            0.0,
            0.0,
            0.0,
            0.0,
            f / (f - n),
            1.0,
            0.0,
            0.0,
            -f * n / (f - n),
            0.0
        )
    }

    #[inline]
    pub fn perspective_by_rad(fov: f32, ratio: f32, n: f32, f: f32) -> Matrix4x4 {
        let inv_tan = 1.0 / (fov * 0.5).tan();

        Matrix4x4!(
            inv_tan / ratio,
            0.0,
            0.0,
            0.0,
            0.0,
            inv_tan,
            0.0,
            0.0,
            0.0,
            0.0,
            f / (f - n),
            1.0,
            0.0,
            0.0,
            -f * n / (f - n),
            0.0
        )
    }

    #[inline]
    pub fn orthographic(width: f32, height: f32, n: f32, f: f32) -> Matrix4x4 {
        let z_range = 1.0 / (f - n);

        Matrix4x4!(
            2.0 / width,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / height,
            0.0,
            0.0,
            0.0,
            0.0,
            z_range,
            0.0,
            0.0,
            0.0,
            -z_range * n,
            1.0
        )
    }
}

/* ------------------------- Matrix-Vector Multiply ------------------------- */
impl std::ops::Mul<Matrix4x4> for Float4 {
    type Output = Float4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Float4!(1.0, 2.0, 3.0, 0.0);
    /// let mat = Matrix4x4!(
    ///                     2.0, 0.0, 0.0, 0.0,
    ///                     0.0, 3.0, 0.0, 0.0,
    ///                     0.0, 0.0, 4.0, 0.0,
    ///                     2.0, 3.0, 1.0, 1.0);
    /// let result = flo * mat;
    ///
    /// assert_eq!(result, Float4!(
    ///                               flo.x * mat.m00 + flo.y * mat.m10 + flo.z * mat.m20 + flo.w * mat.m30,
    ///                               flo.x * mat.m01 + flo.y * mat.m11 + flo.z * mat.m21 + flo.w * mat.m31,
    ///                               flo.x * mat.m02 + flo.y * mat.m12 + flo.z * mat.m22 + flo.w * mat.m32,
    ///                               flo.x * mat.m03 + flo.y * mat.m13 + flo.z * mat.m23 + flo.w * mat.m33))
    /// ```
    #[inline]
    fn mul(self, rhs: Matrix4x4) -> Self::Output {
        if check_sse2!() {
            const XXXX: i32 = swizzle!(0, 0, 0, 0);
            const YYYY: i32 = swizzle!(1, 1, 1, 1);
            const ZZZZ: i32 = swizzle!(2, 2, 2, 2);
            const WWWW: i32 = swizzle!(3, 3, 3, 3);

            unsafe {
                let xxxx = _mm_shuffle_ps::<XXXX>(transmute(self), transmute(self));
                let yyyy = _mm_shuffle_ps::<YYYY>(transmute(self), transmute(self));
                let zzzz = _mm_shuffle_ps::<ZZZZ>(transmute(self), transmute(self));
                let wwww = _mm_shuffle_ps::<WWWW>(transmute(self), transmute(self));

                let (r1, r2, r3, r4) = rhs.into_xmm();

                let l1 = _mm_mul_ps(xxxx, r1);
                let l2 = _mm_mul_ps(yyyy, r2);

                let l5 = _mm_add_ps(l1, l2);

                let l3 = _mm_mul_ps(zzzz, r3);
                let l4 = _mm_mul_ps(wwww, r4);

                let l6 = _mm_add_ps(l3, l4);

                transmute(_mm_add_ps(l5, l6))
            }
        } else {
            Float4!(
                self.x * rhs.m00 + self.y * rhs.m10 + self.z * rhs.m20 + self.w * rhs.m30,
                self.x * rhs.m01 + self.y * rhs.m11 + self.z * rhs.m21 + self.w * rhs.m31,
                self.x * rhs.m02 + self.y * rhs.m12 + self.z * rhs.m22 + self.w * rhs.m32,
                self.x * rhs.m03 + self.y * rhs.m13 + self.z * rhs.m23 + self.w * rhs.m33
            )
        }
    }
}

impl std::ops::Mul<Matrix4x4> for Vector4 {
    type Output = Vector4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Vector4!(1.0, 2.0, 3.0);
    /// let mat = Matrix4x4!(
    ///                     2.0, 0.0, 0.0, 0.0,
    ///                     0.0, 3.0, 0.0, 0.0,
    ///                     0.0, 0.0, 4.0, 0.0,
    ///                     2.0, 3.0, 1.0, 1.0);
    /// let result = flo * mat;
    ///
    /// assert_eq!(result, Vector4!(
    ///                               flo.x * mat.m00 + flo.y * mat.m10 + flo.z * mat.m20 + flo.w * mat.m30,
    ///                               flo.x * mat.m01 + flo.y * mat.m11 + flo.z * mat.m21 + flo.w * mat.m31,
    ///                               flo.x * mat.m02 + flo.y * mat.m12 + flo.z * mat.m22 + flo.w * mat.m32))
    /// ```
    #[inline]
    fn mul(self, rhs: Matrix4x4) -> Self::Output {
        if check_sse2!() {
            const XXXX: i32 = swizzle!(0, 0, 0, 0);
            const YYYY: i32 = swizzle!(1, 1, 1, 1);
            const ZZZZ: i32 = swizzle!(2, 2, 2, 2);
            const WWWW: i32 = swizzle!(3, 3, 3, 3);

            unsafe {
                let xxxx = _mm_shuffle_ps::<XXXX>(transmute(self), transmute(self));
                let yyyy = _mm_shuffle_ps::<YYYY>(transmute(self), transmute(self));
                let zzzz = _mm_shuffle_ps::<ZZZZ>(transmute(self), transmute(self));
                let wwww = _mm_shuffle_ps::<WWWW>(transmute(self), transmute(self));

                let (r1, r2, r3, r4) = rhs.into_xmm();

                let l1 = _mm_mul_ps(xxxx, r1);
                let l2 = _mm_mul_ps(yyyy, r2);

                let l5 = _mm_add_ps(l1, l2);

                let l3 = _mm_mul_ps(zzzz, r3);
                let l4 = _mm_mul_ps(wwww, r4);

                let l6 = _mm_add_ps(l3, l4);

                transmute(_mm_add_ps(l5, l6))
            }
        } else {
            Vector4!(
                self.x * rhs.m00 + self.y * rhs.m10 + self.z * rhs.m20,
                self.x * rhs.m01 + self.y * rhs.m11 + self.z * rhs.m21,
                self.x * rhs.m02 + self.y * rhs.m12 + self.z * rhs.m22
            )
        }
    }
}

impl std::ops::Mul<Matrix4x4> for Point4 {
    type Output = Point4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Point4!(1.0, 2.0, 3.0);
    /// let mat = Matrix4x4!(
    ///                     2.0, 0.0, 0.0, 0.0,
    ///                     0.0, 3.0, 0.0, 0.0,
    ///                     0.0, 0.0, 4.0, 0.0,
    ///                     2.0, 3.0, 1.0, 1.0);
    /// let result = flo * mat;
    ///
    /// assert_eq!(result, Point4!(
    ///                               flo.x * mat.m00 + flo.y * mat.m10 + flo.z * mat.m20 + flo.w * mat.m30,
    ///                               flo.x * mat.m01 + flo.y * mat.m11 + flo.z * mat.m21 + flo.w * mat.m31,
    ///                               flo.x * mat.m02 + flo.y * mat.m12 + flo.z * mat.m22 + flo.w * mat.m32))
    /// ```
    #[inline]
    fn mul(self, rhs: Matrix4x4) -> Self::Output {
        if check_sse2!() {
            const XXXX: i32 = swizzle!(0, 0, 0, 0);
            const YYYY: i32 = swizzle!(1, 1, 1, 1);
            const ZZZZ: i32 = swizzle!(2, 2, 2, 2);
            const WWWW: i32 = swizzle!(3, 3, 3, 3);

            unsafe {
                let xxxx = _mm_shuffle_ps::<XXXX>(transmute(self), transmute(self));
                let yyyy = _mm_shuffle_ps::<YYYY>(transmute(self), transmute(self));
                let zzzz = _mm_shuffle_ps::<ZZZZ>(transmute(self), transmute(self));
                let wwww = _mm_shuffle_ps::<WWWW>(transmute(self), transmute(self));

                let (r1, r2, r3, r4) = rhs.into_xmm();

                let l1 = _mm_mul_ps(xxxx, r1);
                let l2 = _mm_mul_ps(yyyy, r2);

                let l5 = _mm_add_ps(l1, l2);

                let l3 = _mm_mul_ps(zzzz, r3);
                let l4 = _mm_mul_ps(wwww, r4);

                let l6 = _mm_add_ps(l3, l4);

                transmute(_mm_add_ps(l5, l6))
            }
        } else {
            Point4!(
                self.x * rhs.m00 + self.y * rhs.m10 + self.z * rhs.m20 + self.w * rhs.m30,
                self.x * rhs.m01 + self.y * rhs.m11 + self.z * rhs.m21 + self.w * rhs.m31,
                self.x * rhs.m02 + self.y * rhs.m12 + self.z * rhs.m22 + self.w * rhs.m32
            )
        }
    }
}

impl std::ops::Mul<Matrix4x4> for Normal4 {
    type Output = Normal4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let flo = Normal4!(1.0, 2.0, 3.0);
    /// let mat = Matrix4x4!(
    ///                     2.0, 0.0, 0.0, 0.0,
    ///                     0.0, 3.0, 0.0, 0.0,
    ///                     0.0, 0.0, 4.0, 0.0,
    ///                     2.0, 3.0, 1.0, 1.0);
    /// let result = flo * mat;
    ///
    /// assert_eq!(result, Normal4!(
    ///                               flo.x * mat.m00 + flo.y * mat.m10 + flo.z * mat.m20 + flo.w * mat.m30,
    ///                               flo.x * mat.m01 + flo.y * mat.m11 + flo.z * mat.m21 + flo.w * mat.m31,
    ///                               flo.x * mat.m02 + flo.y * mat.m12 + flo.z * mat.m22 + flo.w * mat.m32))
    /// ```
    #[inline]
    fn mul(self, rhs: Matrix4x4) -> Self::Output {
        if check_sse2!() {
            const XXXX: i32 = swizzle!(0, 0, 0, 0);
            const YYYY: i32 = swizzle!(1, 1, 1, 1);
            const ZZZZ: i32 = swizzle!(2, 2, 2, 2);
            const WWWW: i32 = swizzle!(3, 3, 3, 3);

            unsafe {
                let xxxx = _mm_shuffle_ps::<XXXX>(transmute(self), transmute(self));
                let yyyy = _mm_shuffle_ps::<YYYY>(transmute(self), transmute(self));
                let zzzz = _mm_shuffle_ps::<ZZZZ>(transmute(self), transmute(self));
                let wwww = _mm_shuffle_ps::<WWWW>(transmute(self), transmute(self));

                let (r1, r2, r3, r4) = rhs.into_xmm();

                let l1 = _mm_mul_ps(xxxx, r1);
                let l2 = _mm_mul_ps(yyyy, r2);

                let l5 = _mm_add_ps(l1, l2);

                let l3 = _mm_mul_ps(zzzz, r3);
                let l4 = _mm_mul_ps(wwww, r4);

                let l6 = _mm_add_ps(l3, l4);

                transmute(_mm_add_ps(l5, l6))
            }
        } else {
            Normal4!(
                self.x * rhs.m00 + self.y * rhs.m10 + self.z * rhs.m20,
                self.x * rhs.m01 + self.y * rhs.m11 + self.z * rhs.m21,
                self.x * rhs.m02 + self.y * rhs.m12 + self.z * rhs.m22
            )
        }
    }
}

/* ---------------------------------- Unary --------------------------------- */
impl std::ops::Neg for Matrix4x4 {
    type Output = Matrix4x4;

    #[inline]
    fn neg(self) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let (l1, l2, l3, l4) = self.into_xmm();
                let xmm = _mm_set1_ps(-1.0);
                Matrix4x4::from_xmm(
                    _mm_mul_ps(xmm, l1),
                    _mm_mul_ps(xmm, l2),
                    _mm_mul_ps(xmm, l3),
                    _mm_mul_ps(xmm, l4),
                )
            }
        } else {
            Matrix4x4!(
                -self.m00, -self.m01, -self.m02, -self.m03, -self.m10, -self.m11, -self.m12, -self.m13, -self.m20,
                -self.m21, -self.m22, -self.m23, -self.m30, -self.m31, -self.m32, -self.m33
            )
        }
    }
}

impl std::ops::Index<usize> for Matrix4x4 {
    type Output = [f32; 4];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let mat_arr = (self as *const Matrix4x4) as *const [f32; 4];

        unsafe { mat_arr.offset(index as isize).as_ref().unwrap_unchecked() }
    }
}

impl std::ops::IndexMut<usize> for Matrix4x4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let mat_arr = (self as *mut Matrix4x4) as *mut [f32; 4];

        unsafe { mat_arr.offset(index as isize).as_mut().unwrap_unchecked() }
    }
}

/* ----------------------------------- Add ---------------------------------- */
impl std::ops::Add<Matrix4x4> for Matrix4x4 {
    type Output = Matrix4x4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let mat2 = Matrix4x4!(
    ///            4.0, 3.0, 2.0, 1.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            12.0, 11.0, 10.0, 9.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let result = mat1 + mat2;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 + mat2.m00, mat1.m01 + mat2.m01, mat1.m02 + mat2.m02, mat1.m03 + mat2.m03,
    ///                    mat1.m10 + mat2.m10, mat1.m11 + mat2.m11, mat1.m12 + mat2.m12, mat1.m13 + mat2.m13,
    ///                    mat1.m20 + mat2.m20, mat1.m21 + mat2.m21, mat1.m22 + mat2.m22, mat1.m23 + mat2.m23,
    ///                    mat1.m30 + mat2.m30, mat1.m31 + mat2.m31, mat1.m32 + mat2.m32, mat1.m33 + mat2.m33))
    /// ```
    #[inline]
    fn add(self, rhs: Matrix4x4) -> Self::Output {
        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();
            let (r1, r2, r3, r4) = rhs.into_xmm();

            unsafe {
                Matrix4x4::from_xmm(
                    _mm_add_ps(l1, r1),
                    _mm_add_ps(l2, r2),
                    _mm_add_ps(l3, r3),
                    _mm_add_ps(l4, r4),
                )
            }
        } else {
            Matrix4x4!(
                self.m00 + rhs.m00,
                self.m01 + rhs.m01,
                self.m02 + rhs.m02,
                self.m03 + rhs.m03,
                self.m10 + rhs.m10,
                self.m11 + rhs.m11,
                self.m12 + rhs.m12,
                self.m13 + rhs.m13,
                self.m20 + rhs.m20,
                self.m21 + rhs.m21,
                self.m22 + rhs.m22,
                self.m23 + rhs.m23,
                self.m30 + rhs.m30,
                self.m31 + rhs.m31,
                self.m32 + rhs.m32,
                self.m33 + rhs.m33
            )
        }
    }
}

impl std::ops::Add<f32> for Matrix4x4 {
    type Output = Matrix4x4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let f: f32 = 3.0;
    /// let result = mat1 + f;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 + f, mat1.m01 + f, mat1.m02 + f, mat1.m03 + f,
    ///                    mat1.m10 + f, mat1.m11 + f, mat1.m12 + f, mat1.m13 + f,
    ///                    mat1.m20 + f, mat1.m21 + f, mat1.m22 + f, mat1.m23 + f,
    ///                    mat1.m30 + f, mat1.m31 + f, mat1.m32 + f, mat1.m33 + f))
    /// ```
    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();

            unsafe {
                let xmm = _mm_set1_ps(rhs);
                Matrix4x4::from_xmm(
                    _mm_add_ps(l1, xmm),
                    _mm_add_ps(l2, xmm),
                    _mm_add_ps(l3, xmm),
                    _mm_add_ps(l4, xmm),
                )
            }
        } else {
            Matrix4x4!(
                self.m00 + rhs,
                self.m01 + rhs,
                self.m02 + rhs,
                self.m03 + rhs,
                self.m10 + rhs,
                self.m11 + rhs,
                self.m12 + rhs,
                self.m13 + rhs,
                self.m20 + rhs,
                self.m21 + rhs,
                self.m22 + rhs,
                self.m23 + rhs,
                self.m30 + rhs,
                self.m31 + rhs,
                self.m32 + rhs,
                self.m33 + rhs
            )
        }
    }
}

impl std::ops::Add<Matrix4x4> for f32 {
    type Output = Matrix4x4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 3.0;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let result = f + mat1;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    f + mat1.m00, f + mat1.m01, f + mat1.m02, f + mat1.m03,
    ///                    f + mat1.m10, f + mat1.m11, f + mat1.m12, f + mat1.m13,
    ///                    f + mat1.m20, f + mat1.m21, f + mat1.m22, f + mat1.m23,
    ///                    f + mat1.m30, f + mat1.m31, f + mat1.m32, f + mat1.m33))
    /// ```
    #[inline]
    fn add(self, rhs: Matrix4x4) -> Self::Output {
        if check_sse2!() {
            let (r1, r2, r3, r4) = rhs.into_xmm();
            unsafe {
                let xmm = _mm_set1_ps(self);
                Matrix4x4::from_xmm(
                    _mm_add_ps(xmm, r1),
                    _mm_add_ps(xmm, r2),
                    _mm_add_ps(xmm, r3),
                    _mm_add_ps(xmm, r4),
                )
            }
        } else {
            Matrix4x4!(
                self + rhs.m00,
                self + rhs.m01,
                self + rhs.m02,
                self + rhs.m03,
                self + rhs.m10,
                self + rhs.m11,
                self + rhs.m12,
                self + rhs.m13,
                self + rhs.m20,
                self + rhs.m21,
                self + rhs.m22,
                self + rhs.m23,
                self + rhs.m30,
                self + rhs.m31,
                self + rhs.m32,
                self + rhs.m33
            )
        }
    }
}

/* -------------------------------- AddAssign ------------------------------- */
impl std::ops::AddAssign<Matrix4x4> for Matrix4x4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let mat2 = Matrix4x4!(
    ///            4.0, 3.0, 2.0, 1.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            12.0, 11.0, 10.0, 9.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let mut result = mat1;
    /// result += mat2;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 + mat2.m00, mat1.m01 + mat2.m01, mat1.m02 + mat2.m02, mat1.m03 + mat2.m03,
    ///                    mat1.m10 + mat2.m10, mat1.m11 + mat2.m11, mat1.m12 + mat2.m12, mat1.m13 + mat2.m13,
    ///                    mat1.m20 + mat2.m20, mat1.m21 + mat2.m21, mat1.m22 + mat2.m22, mat1.m23 + mat2.m23,
    ///                    mat1.m30 + mat2.m30, mat1.m31 + mat2.m31, mat1.m32 + mat2.m32, mat1.m33 + mat2.m33))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Matrix4x4) {
        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();
            let (r1, r2, r3, r4) = rhs.into_xmm();
            unsafe {
                *self = Matrix4x4::from_xmm(
                    _mm_add_ps(l1, r1),
                    _mm_add_ps(l2, r2),
                    _mm_add_ps(l3, r3),
                    _mm_add_ps(l4, r4),
                );
            }
        } else {
            self.m00 += rhs.m00;
            self.m01 += rhs.m01;
            self.m02 += rhs.m02;
            self.m03 += rhs.m03;
            self.m10 += rhs.m10;
            self.m11 += rhs.m11;
            self.m12 += rhs.m12;
            self.m13 += rhs.m13;
            self.m20 += rhs.m20;
            self.m21 += rhs.m21;
            self.m22 += rhs.m22;
            self.m23 += rhs.m23;
            self.m30 += rhs.m30;
            self.m31 += rhs.m31;
            self.m32 += rhs.m32;
            self.m33 += rhs.m33;
        }
    }
}

impl std::ops::AddAssign<f32> for Matrix4x4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let f: f32 = 3.0;
    /// let mut result = mat1;
    /// result += f;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 + f, mat1.m01 + f, mat1.m02 + f, mat1.m03 + f,
    ///                    mat1.m10 + f, mat1.m11 + f, mat1.m12 + f, mat1.m13 + f,
    ///                    mat1.m20 + f, mat1.m21 + f, mat1.m22 + f, mat1.m23 + f,
    ///                    mat1.m30 + f, mat1.m31 + f, mat1.m32 + f, mat1.m33 + f))
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();
            unsafe {
                let xmm = _mm_set1_ps(rhs);
                *self = Matrix4x4::from_xmm(
                    _mm_add_ps(l1, xmm),
                    _mm_add_ps(l2, xmm),
                    _mm_add_ps(l3, xmm),
                    _mm_add_ps(l4, xmm),
                );
            }
        } else {
            self.m00 += rhs;
            self.m01 += rhs;
            self.m02 += rhs;
            self.m03 += rhs;
            self.m10 += rhs;
            self.m11 += rhs;
            self.m12 += rhs;
            self.m13 += rhs;
            self.m20 += rhs;
            self.m21 += rhs;
            self.m22 += rhs;
            self.m23 += rhs;
            self.m30 += rhs;
            self.m31 += rhs;
            self.m32 += rhs;
            self.m33 += rhs;
        }
    }
}

/* ----------------------------------- Sub ---------------------------------- */
impl std::ops::Sub<Matrix4x4> for Matrix4x4 {
    type Output = Matrix4x4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let mat2 = Matrix4x4!(
    ///            4.0, 3.0, 2.0, 1.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            12.0, 11.0, 10.0, 9.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let result = mat1 - mat2;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 - mat2.m00, mat1.m01 - mat2.m01, mat1.m02 - mat2.m02, mat1.m03 - mat2.m03,
    ///                    mat1.m10 - mat2.m10, mat1.m11 - mat2.m11, mat1.m12 - mat2.m12, mat1.m13 - mat2.m13,
    ///                    mat1.m20 - mat2.m20, mat1.m21 - mat2.m21, mat1.m22 - mat2.m22, mat1.m23 - mat2.m23,
    ///                    mat1.m30 - mat2.m30, mat1.m31 - mat2.m31, mat1.m32 - mat2.m32, mat1.m33 - mat2.m33))
    /// ```
    #[inline]
    fn sub(self, rhs: Matrix4x4) -> Self::Output {
        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();
            let (r1, r2, r3, r4) = rhs.into_xmm();

            unsafe {
                Matrix4x4::from_xmm(
                    _mm_sub_ps(l1, r1),
                    _mm_sub_ps(l2, r2),
                    _mm_sub_ps(l3, r3),
                    _mm_sub_ps(l4, r4),
                )
            }
        } else {
            Matrix4x4!(
                self.m00 - rhs.m00,
                self.m01 - rhs.m01,
                self.m02 - rhs.m02,
                self.m03 - rhs.m03,
                self.m10 - rhs.m10,
                self.m11 - rhs.m11,
                self.m12 - rhs.m12,
                self.m13 - rhs.m13,
                self.m20 - rhs.m20,
                self.m21 - rhs.m21,
                self.m22 - rhs.m22,
                self.m23 - rhs.m23,
                self.m30 - rhs.m30,
                self.m31 - rhs.m31,
                self.m32 - rhs.m32,
                self.m33 - rhs.m33
            )
        }
    }
}

impl std::ops::Sub<f32> for Matrix4x4 {
    type Output = Matrix4x4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let f: f32 = 3.0;
    /// let result = mat1 - f;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 - f, mat1.m01 - f, mat1.m02 - f, mat1.m03 - f,
    ///                    mat1.m10 - f, mat1.m11 - f, mat1.m12 - f, mat1.m13 - f,
    ///                    mat1.m20 - f, mat1.m21 - f, mat1.m22 - f, mat1.m23 - f,
    ///                    mat1.m30 - f, mat1.m31 - f, mat1.m32 - f, mat1.m33 - f))
    /// ```
    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();

            unsafe {
                let xmm = _mm_set1_ps(rhs);
                Matrix4x4::from_xmm(
                    _mm_sub_ps(l1, xmm),
                    _mm_sub_ps(l2, xmm),
                    _mm_sub_ps(l3, xmm),
                    _mm_sub_ps(l4, xmm),
                )
            }
        } else {
            Matrix4x4!(
                self.m00 - rhs,
                self.m01 - rhs,
                self.m02 - rhs,
                self.m03 - rhs,
                self.m10 - rhs,
                self.m11 - rhs,
                self.m12 - rhs,
                self.m13 - rhs,
                self.m20 - rhs,
                self.m21 - rhs,
                self.m22 - rhs,
                self.m23 - rhs,
                self.m30 - rhs,
                self.m31 - rhs,
                self.m32 - rhs,
                self.m33 - rhs
            )
        }
    }
}

impl std::ops::Sub<Matrix4x4> for f32 {
    type Output = Matrix4x4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 3.0;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let result = f - mat1;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    f - mat1.m00, f - mat1.m01, f - mat1.m02, f - mat1.m03,
    ///                    f - mat1.m10, f - mat1.m11, f - mat1.m12, f - mat1.m13,
    ///                    f - mat1.m20, f - mat1.m21, f - mat1.m22, f - mat1.m23,
    ///                    f - mat1.m30, f - mat1.m31, f - mat1.m32, f - mat1.m33))
    /// ```
    #[inline]
    fn sub(self, rhs: Matrix4x4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set1_ps(self);
                let (r1, r2, r3, r4) = rhs.into_xmm();

                Matrix4x4::from_xmm(
                    _mm_sub_ps(xmm, r1),
                    _mm_sub_ps(xmm, r2),
                    _mm_sub_ps(xmm, r3),
                    _mm_sub_ps(xmm, r4),
                )
            }
        } else {
            Matrix4x4!(
                self - rhs.m00,
                self - rhs.m01,
                self - rhs.m02,
                self - rhs.m03,
                self - rhs.m10,
                self - rhs.m11,
                self - rhs.m12,
                self - rhs.m13,
                self - rhs.m20,
                self - rhs.m21,
                self - rhs.m22,
                self - rhs.m23,
                self - rhs.m30,
                self - rhs.m31,
                self - rhs.m32,
                self - rhs.m33
            )
        }
    }
}

/* -------------------------------- SubAssign ------------------------------- */
impl std::ops::SubAssign<Matrix4x4> for Matrix4x4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let mat2 = Matrix4x4!(
    ///            4.0, 3.0, 2.0, 1.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            12.0, 11.0, 10.0, 9.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let mut result = mat1;
    /// result -= mat2;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 - mat2.m00, mat1.m01 - mat2.m01, mat1.m02 - mat2.m02, mat1.m03 - mat2.m03,
    ///                    mat1.m10 - mat2.m10, mat1.m11 - mat2.m11, mat1.m12 - mat2.m12, mat1.m13 - mat2.m13,
    ///                    mat1.m20 - mat2.m20, mat1.m21 - mat2.m21, mat1.m22 - mat2.m22, mat1.m23 - mat2.m23,
    ///                    mat1.m30 - mat2.m30, mat1.m31 - mat2.m31, mat1.m32 - mat2.m32, mat1.m33 - mat2.m33))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Matrix4x4) {
        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();
            let (r1, r2, r3, r4) = rhs.into_xmm();

            unsafe {
                *self = Matrix4x4::from_xmm(
                    _mm_sub_ps(l1, r1),
                    _mm_sub_ps(l2, r2),
                    _mm_sub_ps(l3, r3),
                    _mm_sub_ps(l4, r4),
                )
            }
        } else {
            self.m00 -= rhs.m00;
            self.m01 -= rhs.m01;
            self.m02 -= rhs.m02;
            self.m03 -= rhs.m03;
            self.m10 -= rhs.m10;
            self.m11 -= rhs.m11;
            self.m12 -= rhs.m12;
            self.m13 -= rhs.m13;
            self.m20 -= rhs.m20;
            self.m21 -= rhs.m21;
            self.m22 -= rhs.m22;
            self.m23 -= rhs.m23;
            self.m30 -= rhs.m30;
            self.m31 -= rhs.m31;
            self.m32 -= rhs.m32;
            self.m33 -= rhs.m33;
        }
    }
}

impl std::ops::SubAssign<f32> for Matrix4x4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let f: f32 = 3.0;
    /// let mut result = mat1;
    /// result -= f;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 - f, mat1.m01 - f, mat1.m02 - f, mat1.m03 - f,
    ///                    mat1.m10 - f, mat1.m11 - f, mat1.m12 - f, mat1.m13 - f,
    ///                    mat1.m20 - f, mat1.m21 - f, mat1.m22 - f, mat1.m23 - f,
    ///                    mat1.m30 - f, mat1.m31 - f, mat1.m32 - f, mat1.m33 - f))
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();

            unsafe {
                let xmm = _mm_set1_ps(rhs);
                *self = Matrix4x4::from_xmm(
                    _mm_sub_ps(l1, xmm),
                    _mm_sub_ps(l2, xmm),
                    _mm_sub_ps(l3, xmm),
                    _mm_sub_ps(l4, xmm),
                )
            }
        } else {
            self.m00 -= rhs;
            self.m01 -= rhs;
            self.m02 -= rhs;
            self.m03 -= rhs;
            self.m10 -= rhs;
            self.m11 -= rhs;
            self.m12 -= rhs;
            self.m13 -= rhs;
            self.m20 -= rhs;
            self.m21 -= rhs;
            self.m22 -= rhs;
            self.m23 -= rhs;
            self.m30 -= rhs;
            self.m31 -= rhs;
            self.m32 -= rhs;
            self.m33 -= rhs;
        }
    }
}

/* ----------------------------------- Mul ---------------------------------- */
impl std::ops::Mul<Matrix4x4> for Matrix4x4 {
    type Output = Matrix4x4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let mat2 = Matrix4x4!(
    ///            4.0, 3.0, 2.0, 1.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            12.0, 11.0, 10.0, 9.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let result = mat1 * mat2;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                               mat1.m00 * mat2.m00 + mat1.m01 * mat2.m10 + mat1.m02 * mat2.m20 + mat1.m03 * mat2.m30,
    ///                               mat1.m00 * mat2.m01 + mat1.m01 * mat2.m11 + mat1.m02 * mat2.m21 + mat1.m03 * mat2.m31,
    ///                               mat1.m00 * mat2.m02 + mat1.m01 * mat2.m12 + mat1.m02 * mat2.m22 + mat1.m03 * mat2.m32,
    ///                               mat1.m00 * mat2.m03 + mat1.m01 * mat2.m13 + mat1.m02 * mat2.m23 + mat1.m03 * mat2.m33,
    ///
    ///                               mat1.m10 * mat2.m00 + mat1.m11 * mat2.m10 + mat1.m12 * mat2.m20 + mat1.m13 * mat2.m30,
    ///                               mat1.m10 * mat2.m01 + mat1.m11 * mat2.m11 + mat1.m12 * mat2.m21 + mat1.m13 * mat2.m31,
    ///                               mat1.m10 * mat2.m02 + mat1.m11 * mat2.m12 + mat1.m12 * mat2.m22 + mat1.m13 * mat2.m32,
    ///                               mat1.m10 * mat2.m03 + mat1.m11 * mat2.m13 + mat1.m12 * mat2.m23 + mat1.m13 * mat2.m33,
    ///                               
    ///                               mat1.m20 * mat2.m00 + mat1.m21 * mat2.m10 + mat1.m22 * mat2.m20 + mat1.m23 * mat2.m30,
    ///                               mat1.m20 * mat2.m01 + mat1.m21 * mat2.m11 + mat1.m22 * mat2.m21 + mat1.m23 * mat2.m31,
    ///                               mat1.m20 * mat2.m02 + mat1.m21 * mat2.m12 + mat1.m22 * mat2.m22 + mat1.m23 * mat2.m32,
    ///                               mat1.m20 * mat2.m03 + mat1.m21 * mat2.m13 + mat1.m22 * mat2.m23 + mat1.m23 * mat2.m33,
    ///
    ///                               mat1.m30 * mat2.m00 + mat1.m31 * mat2.m10 + mat1.m32 * mat2.m20 + mat1.m33 * mat2.m30,
    ///                               mat1.m30 * mat2.m01 + mat1.m31 * mat2.m11 + mat1.m32 * mat2.m21 + mat1.m33 * mat2.m31,
    ///                               mat1.m30 * mat2.m02 + mat1.m31 * mat2.m12 + mat1.m32 * mat2.m22 + mat1.m33 * mat2.m32,
    ///                               mat1.m30 * mat2.m03 + mat1.m31 * mat2.m13 + mat1.m32 * mat2.m23 + mat1.m33 * mat2.m33))
    /// ```
    #[inline]
    fn mul(self, rhs: Matrix4x4) -> Self::Output {
        if check_sse2!() {
            let mut xmms: [__m128; 4] = unsafe { std::mem::zeroed() };

            for i in 0..4 {
                unsafe {
                    let xxxx = _mm_set1_ps(self[i][0]);
                    let yyyy = _mm_set1_ps(self[i][1]);
                    let zzzz = _mm_set1_ps(self[i][2]);
                    let wwww = _mm_set1_ps(self[i][3]);

                    let (r1, r2, r3, r4) = rhs.into_xmm();

                    let l1 = _mm_mul_ps(xxxx, r1);
                    let l2 = _mm_mul_ps(yyyy, r2);

                    let l5 = _mm_add_ps(l1, l2);

                    let l3 = _mm_mul_ps(zzzz, r3);
                    let l4 = _mm_mul_ps(wwww, r4);

                    let l6 = _mm_add_ps(l3, l4);

                    xmms[i] = _mm_add_ps(l5, l6);
                }
            }

            unsafe { transmute(xmms) }
        } else {
            Matrix4x4!(
                // r0
                self.m00 * rhs.m00 + self.m01 * rhs.m10 + self.m02 * rhs.m20 + self.m03 * rhs.m30,
                self.m00 * rhs.m01 + self.m01 * rhs.m11 + self.m02 * rhs.m21 + self.m03 * rhs.m31,
                self.m00 * rhs.m02 + self.m01 * rhs.m12 + self.m02 * rhs.m22 + self.m03 * rhs.m32,
                self.m00 * rhs.m03 + self.m01 * rhs.m13 + self.m02 * rhs.m23 + self.m03 * rhs.m33,
                // r1
                self.m10 * rhs.m00 + self.m11 * rhs.m10 + self.m12 * rhs.m20 + self.m13 * rhs.m30,
                self.m10 * rhs.m01 + self.m11 * rhs.m11 + self.m12 * rhs.m21 + self.m13 * rhs.m31,
                self.m10 * rhs.m02 + self.m11 * rhs.m12 + self.m12 * rhs.m22 + self.m13 * rhs.m32,
                self.m10 * rhs.m03 + self.m11 * rhs.m13 + self.m12 * rhs.m23 + self.m13 * rhs.m33,
                // r2
                self.m20 * rhs.m00 + self.m21 * rhs.m10 + self.m22 * rhs.m20 + self.m23 * rhs.m30,
                self.m20 * rhs.m01 + self.m21 * rhs.m11 + self.m22 * rhs.m21 + self.m23 * rhs.m31,
                self.m20 * rhs.m02 + self.m21 * rhs.m12 + self.m22 * rhs.m22 + self.m23 * rhs.m32,
                self.m20 * rhs.m03 + self.m21 * rhs.m13 + self.m22 * rhs.m23 + self.m23 * rhs.m33,
                // r3
                self.m30 * rhs.m00 + self.m31 * rhs.m10 + self.m32 * rhs.m20 + self.m33 * rhs.m30,
                self.m30 * rhs.m01 + self.m31 * rhs.m11 + self.m32 * rhs.m21 + self.m33 * rhs.m31,
                self.m30 * rhs.m02 + self.m31 * rhs.m12 + self.m32 * rhs.m22 + self.m33 * rhs.m32,
                self.m30 * rhs.m03 + self.m31 * rhs.m13 + self.m32 * rhs.m23 + self.m33 * rhs.m33
            )
        }
    }
}

impl std::ops::Mul<f32> for Matrix4x4 {
    type Output = Matrix4x4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let f: f32 = 3.0;
    /// let result = mat1 * f;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 * f, mat1.m01 * f, mat1.m02 * f, mat1.m03 * f,
    ///                    mat1.m10 * f, mat1.m11 * f, mat1.m12 * f, mat1.m13 * f,
    ///                    mat1.m20 * f, mat1.m21 * f, mat1.m22 * f, mat1.m23 * f,
    ///                    mat1.m30 * f, mat1.m31 * f, mat1.m32 * f, mat1.m33 * f))
    /// ```
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();

            unsafe {
                let xmm = _mm_set1_ps(rhs);
                Matrix4x4::from_xmm(
                    _mm_mul_ps(l1, xmm),
                    _mm_mul_ps(l2, xmm),
                    _mm_mul_ps(l3, xmm),
                    _mm_mul_ps(l4, xmm),
                )
            }
        } else {
            Matrix4x4!(
                self.m00 * rhs,
                self.m01 * rhs,
                self.m02 * rhs,
                self.m03 * rhs,
                self.m10 * rhs,
                self.m11 * rhs,
                self.m12 * rhs,
                self.m13 * rhs,
                self.m20 * rhs,
                self.m21 * rhs,
                self.m22 * rhs,
                self.m23 * rhs,
                self.m30 * rhs,
                self.m31 * rhs,
                self.m32 * rhs,
                self.m33 * rhs
            )
        }
    }
}

impl std::ops::Mul<Matrix4x4> for f32 {
    type Output = Matrix4x4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let f: f32 = 3.0;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let result = f * mat1;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    f * mat1.m00, f * mat1.m01, f * mat1.m02, f * mat1.m03,
    ///                    f * mat1.m10, f * mat1.m11, f * mat1.m12, f * mat1.m13,
    ///                    f * mat1.m20, f * mat1.m21, f * mat1.m22, f * mat1.m23,
    ///                    f * mat1.m30, f * mat1.m31, f * mat1.m32, f * mat1.m33))
    /// ```
    #[inline]
    fn mul(self, rhs: Matrix4x4) -> Self::Output {
        if check_sse2!() {
            unsafe {
                let xmm = _mm_set1_ps(self);
                let (r1, r2, r3, r4) = rhs.into_xmm();
                Matrix4x4::from_xmm(
                    _mm_mul_ps(xmm, r1),
                    _mm_mul_ps(xmm, r2),
                    _mm_mul_ps(xmm, r3),
                    _mm_mul_ps(xmm, r4),
                )
            }
        } else {
            Matrix4x4!(
                self * rhs.m00,
                self * rhs.m01,
                self * rhs.m02,
                self * rhs.m03,
                self * rhs.m10,
                self * rhs.m11,
                self * rhs.m12,
                self * rhs.m13,
                self * rhs.m20,
                self * rhs.m21,
                self * rhs.m22,
                self * rhs.m23,
                self * rhs.m30,
                self * rhs.m31,
                self * rhs.m32,
                self * rhs.m33
            )
        }
    }
}

/* -------------------------------- MulAssign ------------------------------- */
impl std::ops::MulAssign<Matrix4x4> for Matrix4x4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let mat2 = Matrix4x4!(
    ///            4.0, 3.0, 2.0, 1.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            12.0, 11.0, 10.0, 9.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let mut result = mat1;
    /// result *= mat2;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                               mat1.m00 * mat2.m00 + mat1.m01 * mat2.m10 + mat1.m02 * mat2.m20 + mat1.m03 * mat2.m30,
    ///                               mat1.m00 * mat2.m01 + mat1.m01 * mat2.m11 + mat1.m02 * mat2.m21 + mat1.m03 * mat2.m31,
    ///                               mat1.m00 * mat2.m02 + mat1.m01 * mat2.m12 + mat1.m02 * mat2.m22 + mat1.m03 * mat2.m32,
    ///                               mat1.m00 * mat2.m03 + mat1.m01 * mat2.m13 + mat1.m02 * mat2.m23 + mat1.m03 * mat2.m33,
    ///
    ///                               mat1.m10 * mat2.m00 + mat1.m11 * mat2.m10 + mat1.m12 * mat2.m20 + mat1.m13 * mat2.m30,
    ///                               mat1.m10 * mat2.m01 + mat1.m11 * mat2.m11 + mat1.m12 * mat2.m21 + mat1.m13 * mat2.m31,
    ///                               mat1.m10 * mat2.m02 + mat1.m11 * mat2.m12 + mat1.m12 * mat2.m22 + mat1.m13 * mat2.m32,
    ///                               mat1.m10 * mat2.m03 + mat1.m11 * mat2.m13 + mat1.m12 * mat2.m23 + mat1.m13 * mat2.m33,
    ///                               
    ///                               mat1.m20 * mat2.m00 + mat1.m21 * mat2.m10 + mat1.m22 * mat2.m20 + mat1.m23 * mat2.m30,
    ///                               mat1.m20 * mat2.m01 + mat1.m21 * mat2.m11 + mat1.m22 * mat2.m21 + mat1.m23 * mat2.m31,
    ///                               mat1.m20 * mat2.m02 + mat1.m21 * mat2.m12 + mat1.m22 * mat2.m22 + mat1.m23 * mat2.m32,
    ///                               mat1.m20 * mat2.m03 + mat1.m21 * mat2.m13 + mat1.m22 * mat2.m23 + mat1.m23 * mat2.m33,
    ///
    ///                               mat1.m30 * mat2.m00 + mat1.m31 * mat2.m10 + mat1.m32 * mat2.m20 + mat1.m33 * mat2.m30,
    ///                               mat1.m30 * mat2.m01 + mat1.m31 * mat2.m11 + mat1.m32 * mat2.m21 + mat1.m33 * mat2.m31,
    ///                               mat1.m30 * mat2.m02 + mat1.m31 * mat2.m12 + mat1.m32 * mat2.m22 + mat1.m33 * mat2.m32,
    ///                               mat1.m30 * mat2.m03 + mat1.m31 * mat2.m13 + mat1.m32 * mat2.m23 + mat1.m33 * mat2.m33))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Matrix4x4) {
        if check_sse2!() {
            let mut xmms: [MaybeUninit<__m128>; 4] = unsafe { MaybeUninit::uninit().assume_init() };

            for i in 0..4 {
                unsafe {
                    let xxxx = _mm_set1_ps(self[i][0]);
                    let yyyy = _mm_set1_ps(self[i][1]);
                    let zzzz = _mm_set1_ps(self[i][2]);
                    let wwww = _mm_set1_ps(self[i][3]);

                    let (r0, r1, r2, r3) = rhs.into_xmm();

                    let l0 = _mm_mul_ps(xxxx, r0);
                    let l1 = _mm_mul_ps(yyyy, r1);

                    let l4 = _mm_add_ps(l0, l1);

                    let l2 = _mm_mul_ps(zzzz, r2);
                    let l3 = _mm_mul_ps(wwww, r3);

                    let l5 = _mm_add_ps(l2, l3);

                    xmms[i].write(_mm_add_ps(l4, l5));
                }
            }

            *self = unsafe { transmute(xmms) };
        } else {
            *self = Matrix4x4!(
                // r0
                self.m00 * rhs.m00 + self.m01 * rhs.m10 + self.m02 * rhs.m20 + self.m03 * rhs.m30,
                self.m00 * rhs.m01 + self.m01 * rhs.m11 + self.m02 * rhs.m21 + self.m03 * rhs.m31,
                self.m00 * rhs.m02 + self.m01 * rhs.m12 + self.m02 * rhs.m22 + self.m03 * rhs.m32,
                self.m00 * rhs.m03 + self.m01 * rhs.m13 + self.m02 * rhs.m23 + self.m03 * rhs.m33,
                // r1
                self.m10 * rhs.m00 + self.m11 * rhs.m10 + self.m12 * rhs.m20 + self.m13 * rhs.m30,
                self.m10 * rhs.m01 + self.m11 * rhs.m11 + self.m12 * rhs.m21 + self.m13 * rhs.m31,
                self.m10 * rhs.m02 + self.m11 * rhs.m12 + self.m12 * rhs.m22 + self.m13 * rhs.m32,
                self.m10 * rhs.m03 + self.m11 * rhs.m13 + self.m12 * rhs.m23 + self.m13 * rhs.m33,
                // r2
                self.m20 * rhs.m00 + self.m21 * rhs.m10 + self.m22 * rhs.m20 + self.m23 * rhs.m30,
                self.m20 * rhs.m01 + self.m21 * rhs.m11 + self.m22 * rhs.m21 + self.m23 * rhs.m31,
                self.m20 * rhs.m02 + self.m21 * rhs.m12 + self.m22 * rhs.m22 + self.m23 * rhs.m32,
                self.m20 * rhs.m03 + self.m21 * rhs.m13 + self.m22 * rhs.m23 + self.m23 * rhs.m33,
                // r3
                self.m30 * rhs.m00 + self.m31 * rhs.m10 + self.m32 * rhs.m20 + self.m33 * rhs.m30,
                self.m30 * rhs.m01 + self.m31 * rhs.m11 + self.m32 * rhs.m21 + self.m33 * rhs.m31,
                self.m30 * rhs.m02 + self.m31 * rhs.m12 + self.m32 * rhs.m22 + self.m33 * rhs.m32,
                self.m30 * rhs.m03 + self.m31 * rhs.m13 + self.m32 * rhs.m23 + self.m33 * rhs.m33
            );
        }
    }
}

impl std::ops::MulAssign<f32> for Matrix4x4 {
    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let f: f32 = 3.0;
    /// let mut result = mat1;
    /// result *= f;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 * f, mat1.m01 * f, mat1.m02 * f, mat1.m03 * f,
    ///                    mat1.m10 * f, mat1.m11 * f, mat1.m12 * f, mat1.m13 * f,
    ///                    mat1.m20 * f, mat1.m21 * f, mat1.m22 * f, mat1.m23 * f,
    ///                    mat1.m30 * f, mat1.m31 * f, mat1.m32 * f, mat1.m33 * f))
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();

            unsafe {
                let xmm = _mm_set1_ps(rhs);
                *self = Matrix4x4::from_xmm(
                    _mm_mul_ps(l1, xmm),
                    _mm_mul_ps(l2, xmm),
                    _mm_mul_ps(l3, xmm),
                    _mm_mul_ps(l4, xmm),
                );
            }
        } else {
            self.m00 *= rhs;
            self.m01 *= rhs;
            self.m02 *= rhs;
            self.m03 *= rhs;
            self.m10 *= rhs;
            self.m11 *= rhs;
            self.m12 *= rhs;
            self.m13 *= rhs;
            self.m20 *= rhs;
            self.m21 *= rhs;
            self.m22 *= rhs;
            self.m23 *= rhs;
            self.m30 *= rhs;
            self.m31 *= rhs;
            self.m32 *= rhs;
            self.m33 *= rhs;
        }
    }
}

/* ----------------------------------- Div ---------------------------------- */
impl std::ops::Div<f32> for Matrix4x4 {
    type Output = Matrix4x4;

    /// # Examples
    /// ```
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let f: f32 = 3.0;
    /// let inv = 1.0 / f;
    /// let result = mat1 / f;
    ///
    /// assert_eq!(result, Matrix4x4!(
    ///                    mat1.m00 * inv, mat1.m01 * inv, mat1.m02 * inv, mat1.m03 * inv,
    ///                    mat1.m10 * inv, mat1.m11 * inv, mat1.m12 * inv, mat1.m13 * inv,
    ///                    mat1.m20 * inv, mat1.m21 * inv, mat1.m22 * inv, mat1.m23 * inv,
    ///                    mat1.m30 * inv, mat1.m31 * inv, mat1.m32 * inv, mat1.m33 * inv))
    /// ```
    /// ``` should_panic
    /// use ssun_math::*;
    /// let mat1 = Matrix4x4!(
    ///            1.0, 2.0, 3.0, 4.0,
    ///            5.0, 6.0, 7.0, 8.0,
    ///            9.0, 10.0, 11.0, 12.0,
    ///            13.0, 14.0, 15.0, 16.0);
    /// let f: f32 = 0.0;
    /// let panic = mat1 / f;
    /// ```
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        debug_assert!(rhs != 0.0, "Matrix4x4 cannot be divided by 0");
        let inv = 1.0 / rhs;

        if check_sse2!() {
            let (l1, l2, l3, l4) = self.into_xmm();

            unsafe {
                let xmm = _mm_set1_ps(inv);
                Matrix4x4::from_xmm(
                    _mm_mul_ps(l1, xmm),
                    _mm_mul_ps(l2, xmm),
                    _mm_mul_ps(l3, xmm),
                    _mm_mul_ps(l4, xmm),
                )
            }
        } else {
            Matrix4x4!(
                self.m00 * inv,
                self.m01 * inv,
                self.m02 * inv,
                self.m03 * inv,
                self.m10 * inv,
                self.m11 * inv,
                self.m12 * inv,
                self.m13 * inv,
                self.m20 * inv,
                self.m21 * inv,
                self.m22 * inv,
                self.m23 * inv,
                self.m30 * inv,
                self.m31 * inv,
                self.m32 * inv,
                self.m33 * inv
            )
        }
    }
}
