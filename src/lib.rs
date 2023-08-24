//! # packing_list
//!
//! This is a kind of [free list](https://en.wikipedia.org/wiki/Free_list) implementation where new elements are
//! *guaranteed* to be placed in the smallest available index of the list. 

pub mod packing_list;

#[doc(inline)]
pub use crate::packing_list::PackingList;

