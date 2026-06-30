pub trait LossyFrom<Source>: Sized {
    fn lossy_from(source: Source) -> Self;
}

macro_rules! impl_lossy_from {
    ($( $src:ty $(=> $dst_dir:ty)? $(= $dst_bi:ty)? ),* $(,)?) => {
        $( impl_lossy_from!(@parse $src $(=> $dst_dir)? $(= $dst_bi)?); )*
    };

    // Identity
    (@parse $ty:ty) => {
        impl_lossy_from!(@generate $ty, $ty);
    };

    // One-way: f64 => f32
    (@parse $src:ty => $dst:ty) => {
        impl_lossy_from!(@generate $src, $dst);
    };

    // Bidirectional: f64 = f32
    (@parse $t1:ty = $t2:ty) => {
        impl_lossy_from!(@generate $t1, $t2);
        impl_lossy_from!(@generate $t2, $t1);
    };

    (@generate $src:ty, $dst:ty) => {
        impl LossyFrom<$src> for $dst {
            #[inline]
            fn lossy_from(n: $src) -> Self {
                n as Self
            }
        }
    };
}

impl_lossy_from![
    f32, f64,
    i8, i16, i32, i64, i128,
    u8, u16, u32, u64, u128,
    isize, usize,
    f32 = f64,

    // All possible combinations:
    // f32 = i8,
    // f32 = i16,
    // f32 = i32,
    // f32 = i64,
    // f32 = i128,
    // f32 = u8,
    // f32 = u16,
    // f32 = u32,
    // f32 = u64,
    // f32 = u128,
    // f32 = isize,
    // f32 = usize,
    //
    // f64 = i8,
    // f64 = i16,
    // f64 = i32,
    // f64 = i64,
    // f64 = i128,
    // f64 = u8,
    // f64 = u16,
    // f64 = u32,
    // f64 = u64,
    // f64 = u128,
    // f64 = isize,
    // f64 = usize,
    //
    // i8 = i16,
    // i8 = i32,
    // i8 = i64,
    // i8 = i128,
    // i8 = u8,
    // i8 = u16,
    // i8 = u32,
    // i8 = u64,
    // i8 = u128,
    // i8 = isize,
    // i8 = usize,
    //
    // i16 = i32,
    // i16 = i64,
    // i16 = i128,
    // i16 = u8,
    // i16 = u16,
    // i16 = u32,
    // i16 = u64,
    // i16 = u128,
    // i16 = isize,
    // i16 = usize,
    //
    // i32 = i64,
    // i32 = i128,
    // i32 = u8,
    // i32 = u16,
    // i32 = u32,
    // i32 = u64,
    // i32 = u128,
    // i32 = isize,
    // i32 = usize,
    //
    // i64 = i128,
    // i64 = u8,
    // i64 = u16,
    // i64 = u32,
    // i64 = u64,
    // i64 = u128,
    // i64 = isize,
    // i64 = usize,
    //
    // i128 = u8,
    // i128 = u16,
    // i128 = u32,
    // i128 = u64,
    // i128 = u128,
    // i128 = isize,
    // i128 = usize,
    //
    // u8 = u16,
    // u8 = u32,
    // u8 = u64,
    // u8 = u128,
    // u8 = isize,
    // u8 = usize,
    //
    // u16 = u32,
    // u16 = u64,
    // u16 = u128,
    // u16 = isize,
    // u16 = usize,
    //
    // u32 = u64,
    // u32 = u128,
    // u32 = isize,
    // u32 = usize,
    //
    // u64 = u128,
    // u64 = isize,
    // u64 = usize,
    //
    // u128 = isize,
    // u128 = usize,
    //
    // isize = usize,
];
