//! Compatibility module for alloc/std types.
//!
//! Mirrors whippyunits' own `alloc` shim: it provides a single import path for
//! the heap types that differ between `std` and `no_std + alloc` environments,
//! so `alloc`-gated code can `use crate::alloc::*` without duplicating the
//! feature-flag dance at each site.

#![allow(unused_imports)]

#[cfg(not(feature = "std"))]
pub use alloc_crate::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[cfg(feature = "std")]
pub use std::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
