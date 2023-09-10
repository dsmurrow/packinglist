//! # packinglist
//!
//! This is a kind of [free list](https://en.wikipedia.org/wiki/Free_list) implementation where new elements are
//! *guaranteed* to be placed in the smallest available index of the list.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::vec;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[cfg(not(feature = "std"))]
use alloc::collections::{BTreeSet, btree_set, BTreeMap, BinaryHeap};
#[cfg(feature = "std")]
use std::collections::{BTreeSet, btree_set, BTreeMap, BinaryHeap};

#[cfg(not(feature = "std"))]
use core::cmp::Reverse;
#[cfg(feature = "std")]
use std::cmp::Reverse;

#[cfg(not(feature = "std"))]
use core::convert::From;
#[cfg(feature = "std")]
use std::convert::From;

#[cfg(not(feature = "std"))]
use core::fmt;
#[cfg(feature = "std")]
use std::fmt;

#[cfg(not(feature = "std"))]
use core::hash::{Hash, Hasher};
#[cfg(feature = "std")]
use std::hash::{Hash, Hasher};

#[cfg(not(feature = "std"))]
use core::iter::{Enumerate, FilterMap};
#[cfg(feature = "std")]
use std::iter::{Enumerate, FilterMap};

#[cfg(not(feature = "std"))]
use core::mem;
#[cfg(feature = "std")]
use std::mem;

#[cfg(not(feature = "std"))]
use core::ops::{Deref, DerefMut, Drop, Index};
#[cfg(feature = "std")]
use std::ops::{Deref, DerefMut, Drop, Index};

#[cfg(not(feature = "std"))]
use core::slice::SliceIndex;
#[cfg(feature = "std")]
use std::slice::SliceIndex;

#[cfg(any(feature = "serde-std", feature = "serde-nostd"))]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub type TransformTable = BTreeMap<usize, usize>;

pub type IndexIter = btree_set::IntoIter<usize>;

/// A FreeList implementation this will always put a new element at the smallest empty index of the
/// list.
#[derive(Clone, Default)]
pub struct PackingList<T> {
    list: Vec<Option<T>>,
    empty_spots: BinaryHeap<Reverse<usize>>,
}

impl<T> PackingList<T> {
    /// Creates a new, empty `PackingList<T>`.
    ///
    /// Will not allocate until elements are added.
    #[inline]
    pub fn new() -> Self {
        PackingList {
            list: Vec::new(),
            empty_spots: BinaryHeap::new(),
        }
    }

    /// Clears all entries from the list. Has no effect on the allocated capacity of the list.
    #[inline]
    pub fn clear(&mut self) {
        self.list.clear();
        self.empty_spots.clear();
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    #[inline]
    pub fn as_vec(&self) -> &'_ Vec<Option<T>> {
        &self.list
    }

    /// Returns a smart pointer to the vector containing the list, which can be modified like any
    /// mutable [`Vec`].
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let mut list = PackingList::from([Some(1), None, Some(2), Some(5)]);
    ///
    /// {
    ///     let mut ptr = list.as_vec_mut();
    ///     ptr[0] = None;
    ///     ptr[1] = Some(5);
    ///     ptr[2] = None;
    ///     ptr.push(None);
    /// }
    ///
    /// assert_eq!(list.add(0), 0);
    /// assert_eq!(list, [Some(0), Some(5), None, Some(5)]);
    /// ```
    #[inline]
    pub fn as_vec_mut(&mut self) -> VecPtr<'_, T> {
        VecPtr { list: self }
    }

    /// Returns the number of non-empty entries in the list.
    #[inline]
    pub fn count(&self) -> usize {
        self.list.len() - self.empty_spots.len()
    }

    /// Removes all empty spots in the list and shrinks it to fit.
    ///
    /// Returns a [`TransformTable`] mapping the old indeces to the new ones.
    ///
    /// # Examples
    ///
    /// ```
    /// # use packinglist::PackingList;
    /// let mut list = PackingList::from([Some(0), None, Some(2), None, Some(4), None]);
    ///
    /// let transform = list.pack();
    ///
    /// assert_eq!(list, [Some(0), Some(2), Some(4)]);
    ///
    /// assert_eq!(transform.get(&0), Some(&0));
    /// assert_eq!(transform.get(&2), Some(&1));
    /// assert_eq!(transform.get(&4), Some(&2));
    /// ```
    pub fn pack(&mut self) -> TransformTable {
        let mut old_list: Vec<Option<T>> = Vec::with_capacity(self.count());
        mem::swap(&mut old_list, &mut self.list);

        self.empty_spots.clear();

        let mut table = TransformTable::new();

        for (i, v) in old_list
            .into_iter()
            .enumerate()
            .filter(|(_, v)| v.is_some())
        {
            self.list.push(v);
            table.insert(i, self.list.len() - 1);
        }

        table
    }

    /// Empties `other` into `self`, using the earliest available spaces.
    ///
    /// Returns a [`TransformTable`] mapping the indeces of the values in `other` to their new indeces in `self`.
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let mut a = PackingList::from([Some(0), None, Some(2), None, Some(4)]);
    /// let mut b = PackingList::from([Some(1), Some(3)]);
    ///
    /// let table = a.combine(&mut b);
    ///
    /// assert!(b.is_empty());
    ///
    /// assert_eq!(*a.as_vec(), [Some(0), Some(1), Some(2), Some(3), Some(4)]);
    ///
    /// assert_eq!(table.get(&0), Some(&1));
    /// assert_eq!(table.get(&1), Some(&3));
    /// ```
    pub fn combine(&mut self, other: &mut PackingList<T>) -> TransformTable {
        let mut old_other_list: Vec<Option<T>> = Vec::new();
        mem::swap(&mut old_other_list, &mut other.list);

        other.empty_spots.clear();

        let mut table = TransformTable::new();

        let iter = old_other_list
            .into_iter()
            .enumerate()
            .filter(|(_, v)| v.is_some())
            .map(|(i, v)| (i, v.unwrap()));

        for (i, v) in iter {
            table.insert(i, self.add(v));
        }

        table
    }

    /// Returns an iterator over the indeces of the list that contain items. The items are yielded
    /// in increasing order.
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let list = PackingList::from([None, Some(1), Some(9), Some(3), None, None, Some(4)]);
    ///
    /// let indeces: Vec<usize> = list.index_iter().collect();
    ///
    /// assert_eq!(indeces, [1, 2, 3, 6]);
    /// ```
    #[inline]
    pub fn index_iter(&self) -> IndexIter {
        let all_indeces = BTreeSet::from_iter(0..self.list.len());
        let excluded_indeces = BTreeSet::from_iter(self.empty_spots.clone().into_iter().map(|i| i.0));

        all_indeces.difference(&excluded_indeces).cloned().collect::<BTreeSet<usize>>().into_iter()
    }

    /// Returns an iterator over the items in the list in order of increasing index.
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let list = PackingList::from([None, Some(1), Some(9), Some(3), None, None, Some(4)]);
    ///
    /// let items: Vec<&i32> = list.item_iter().collect();
    ///
    /// assert_eq!(items, [&1, &9, &3, &4]);
    /// ```
    #[inline]
    pub fn item_iter(&self) -> ItemIter<'_, T> {
        ItemIter {
            list: &self.list,
            index_iter: self.index_iter(),
        }
    }

    /// Returns an iterator that allows modifying each non-empty value in the list, in order of
    /// increasing index.
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let mut list = PackingList::from([Some(2), Some(5), None, Some(11)]);
    ///
    /// for v in list.iter_mut() {
    ///     *v += 1
    /// }
    ///
    /// assert_eq!(list, [Some(3), Some(6), None, Some(12)]);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            items: self.item_iter(),
        }
    }

    /// Places `data` in the first available spot in the list. Returns the index it was placed at.
    ///
    /// # Examples
    ///
    /// ```
    /// # use packinglist::PackingList;
    /// let ex_vec = [Some(0), Some(1), Some(2)];
    /// let list = PackingList::from(ex_vec.clone());
    /// assert_eq!(list, ex_vec);
    ///
    /// let mut list = PackingList::from([Some(0), None, Some(2), None]);
    /// let idx = list.add(1);
    /// assert_eq!(idx, 1); // 1 was the smallest empty index
    /// assert_eq!(list[idx], Some(1));
    /// assert_eq!(list, [Some(0), Some(1), Some(2), None]);
    /// ```
    ///
    /// # Time complexity
    ///
    /// The expected cost of `add` is *O*(1). The worst possible case for a single call is *O*(n)
    /// if more memory needs to be allocated.
    #[inline]
    pub fn add(&mut self, data: T) -> usize {
        if let Some(idx) = self.empty_spots.pop() {
            self.list[idx.0] = Some(data);
            idx.0
        } else {
            self.list.push(Some(data));
            self.list.len() - 1
        }
    }

    /// Removes the item at the index `idx` of the list if it is not `None`. Returns entry at
    /// `idx`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use packinglist::PackingList;
    /// let mut list = PackingList::from([Some(0), Some(1), Some(2), Some(3)]);
    ///
    /// list.remove(1);
    /// list.remove(2);
    ///
    /// assert_eq!(list, [Some(0), None, None, Some(3)]);
    /// ```
    ///
    /// # Time complexity
    ///
    /// If `idx` is the last index in the list then it'll take *O*(1) time. Otherwise the
    /// worst-case performance is *O*(log(*n*)).
    #[inline]
    pub fn remove(&mut self, idx: usize) -> Option<T> {
        self.list.get_mut(idx)?.take().map(|v| {
            if idx == self.list.len() - 1 {
                self.list.pop();
            } else {
                self.empty_spots.push(Reverse(idx));
            }
            v
        })
    }

    /// Get a mutable reference to the item at `idx` if it exists., for unmutable functionality,
    /// just index the list.
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let mut list = PackingList::from([Some(19), None]);
    ///
    /// let mut_ref = list.get_mut(1);
    /// assert_eq!(mut_ref, None);
    ///
    /// let mut_ref = list.get_mut(0);
    /// assert_eq!(mut_ref, Some(&mut 19));
    ///
    /// *mut_ref.unwrap() = 10;
    /// assert_eq!(list[0], Some(10));
    /// ```
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        self.list.get_mut(idx)?.as_mut()
    }

    /// Retains only the non-empty elements of the list specified by the predicate.
    ///
    /// Each element that stays is at the same index as it was before `retain` was called.
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let mut list = PackingList::from([Some(0), Some(1), None, Some(3), Some(4)]);
    /// list.retain(|&n| n > 2);
    /// assert_eq!(list, [None, None, None, Some(3), Some(4)]);
    /// ```
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|elem| f(elem));
    }


    /// Retains only the non-empty elements of the list specified by the predicate, passing a
    /// mutable reference to it.
    ///
    /// Each element that stays is at the same index as it was before `retain_mut` was called.
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let mut list = PackingList::from([Some(0), Some(1), None, Some(3), Some(4)]);
    /// list.retain_mut(|n| if *n > 2 {
    ///     *n += 1;
    ///     true
    /// } else {
    ///     false
    /// });
    /// assert_eq!(list, [None, None, None, Some(4), Some(5)]);
    /// ```
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        for i in 0..self.list.len() {
            if let Some(value) = self.get_mut(i) {
                if !f(value) {
                    self.remove(i);
                }
            }
        }
    }


    /// Removes all trailing `None`'s. All user-facing instances of `PackingList` should already be
    /// trimmed, so this is for internal purposes.
    #[inline]
    fn trim_vec(&mut self) {
        while self.list.last().is_some_and(|opt| opt.is_none()) {
            self.list.pop();
        }
    }
}

impl<T, I: SliceIndex<[Option<T>]>> Index<I> for PackingList<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        self.list.index(index)
    }
}

impl<T> fmt::Debug for PackingList<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        let len = self.list.len();

        for opt in &self.list[0..(len - 1)] {
            match opt {
                Some(v) => write!(f, "{:?}, ", v)?,
                None => write!(f, "_, ")?,
            };
        }

        match self.list.last() {
            Some(Some(v)) => write!(f, "{:?}]", v)?,
            _ => write!(f, "]")?,
        };

        Ok(())
    }
}

impl<T> FromIterator<Option<T>> for PackingList<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        Self::from(iter.into_iter().collect::<Vec<Option<T>>>())
    }
}

#[cfg(any(feature = "serde-std", feature = "serde-nostd"))]
impl<T: Serialize> Serialize for PackingList<T> {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.list.serialize(serializer)
    }
}

#[cfg(any(feature = "serde-std", feature = "serde-nostd"))]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for PackingList<T> {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::from(Vec::deserialize(deserializer)?))
    }
}

pub type ListIter<T> =
    FilterMap<Enumerate<vec::IntoIter<Option<T>>>, fn((usize, Option<T>)) -> Option<(usize, T)>>;

impl<T> IntoIterator for PackingList<T> {
    type Item = (usize, T);
    type IntoIter = ListIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.list
            .into_iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.map(|v| (i, v)))
    }
}

impl<T, const N: usize> From<[Option<T>; N]> for PackingList<T> {
    #[inline]
    fn from(arr: [Option<T>; N]) -> Self {
        Self::from(Vec::from(arr))
    }
}

impl<T: Clone> From<&[Option<T>]> for PackingList<T> {
    #[inline]
    fn from(s: &[Option<T>]) -> Self {
        Self::from(s.to_vec())
    }
}

impl<T: Clone> From<&mut [Option<T>]> for PackingList<T> {
    #[inline]
    fn from(s: &mut [Option<T>]) -> Self {
        Self::from(s.to_vec())
    }
}

impl<T> From<Box<[Option<T>]>> for PackingList<T> {
    #[inline]
    fn from(b: Box<[Option<T>]>) -> Self {
        Self::from(Vec::from(b))
    }
}

impl<T> From<Vec<Option<T>>> for PackingList<T> {
    fn from(vec: Vec<Option<T>>) -> Self {
        let empty_spots: BinaryHeap<Reverse<usize>> = vec
            .iter()
            .enumerate()
            .filter(|(_, opt)| opt.is_none())
            .map(|(i, _)| Reverse(i))
            .collect();

        PackingList {
            list: vec,
            empty_spots,
        }
    }
}

impl<T: Clone> From<&Vec<Option<T>>> for PackingList<T> {
    #[inline]
    fn from(vec: &Vec<Option<T>>) -> Self {
        Self::from(vec.clone())
    }
}

impl<T: Clone> From<&mut Vec<Option<T>>> for PackingList<T> {
    #[inline]
    fn from(vec: &mut Vec<Option<T>>) -> Self {
        Self::from(vec.clone())
    }
}

impl<T> Hash for PackingList<T>
where
    T: Hash,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.list.hash(state);
    }
}

impl<T, const N: usize, U> PartialEq<[Option<U>; N]> for PackingList<T>
where
    Option<T>: PartialEq<Option<U>>,
{
    #[inline]
    fn eq(&self, other: &[Option<U>; N]) -> bool {
        self.list.eq(other)
    }
}

impl<T, const N: usize, U> PartialEq<&[Option<U>; N]> for PackingList<T>
where
    Option<T>: PartialEq<Option<U>>,
{
    #[inline]
    fn eq(&self, other: &&[Option<U>; N]) -> bool {
        self.list.eq(other)
    }
}

impl<T, U> PartialEq<[Option<U>]> for PackingList<T>
where
    Option<T>: PartialEq<Option<U>>,
{
    #[inline]
    fn eq(&self, other: &[Option<U>]) -> bool {
        self.list.eq(other)
    }
}

impl<T, U> PartialEq<&[Option<U>]> for PackingList<T>
where
    Option<T>: PartialEq<Option<U>>,
{
    #[inline]
    fn eq(&self, other: &&[Option<U>]) -> bool {
        self.list.eq(other)
    }
}

impl<T, U> PartialEq<Vec<Option<U>>> for PackingList<T> 
where
    Option<T>: PartialEq<Option<U>>,
{
    #[inline]
    fn eq(&self, other: &Vec<Option<U>>) -> bool {
        self.list == *other
    }
}

impl<T, U> PartialEq<PackingList<U>> for PackingList<T> 
where
    Option<T>: PartialEq<Option<U>>,
{
    #[inline]
    fn eq(&self, other: &PackingList<U>) -> bool {
        self.list == other.list
    }
}

impl<T: Eq> Eq for PackingList<T> {}

pub struct VecPtr<'a, T> {
    list: &'a mut PackingList<T>,
}

impl<'a, T> Deref for VecPtr<'a, T> {
    type Target = Vec<Option<T>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.list.list
    }
}

impl<'a, T> DerefMut for VecPtr<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.list.list
    }
}

impl<'a, T> Drop for VecPtr<'a, T> {
    fn drop(&mut self) {
        self.list.trim_vec();
        self.list.empty_spots.clear();

        let empty_indeces = self
            .list
            .list
            .iter()
            .enumerate()
            .filter(|(_, opt)| opt.is_none())
            .map(|(i, _)| i);

        for i in empty_indeces {
            self.list.empty_spots.push(Reverse(i));
        }
    }
}

pub struct ItemIter<'a, T> {
    list: &'a Vec<Option<T>>,
    index_iter: IndexIter,
}

impl<'a, T> Iterator for ItemIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iter.next() {
            Some(i) => self.list[i].as_ref(),
            None => None,
        }
    }
}

pub struct IterMut<'a, T> {
    items: ItemIter<'a, T>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.items.next() {
            Some(r) => {
                // This is my first REAL time doing stuff with Rust pointers. It feels illegal but
                // satisfying to pull off!
                unsafe { (r as *const T).cast_mut().as_mut() }
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::format;

    #[test]
    fn add_does_fill() {
        let mut list = PackingList::from([Some(0), None, Some(2), None, Some(4)]);

        assert_eq!(list.add(1), 1);
        assert_eq!(list.add(3), 3);
        assert_eq!(
            list,
            [Some(0), Some(1), Some(2), Some(3), Some(4)]
        );
    }

    #[test]
    fn add_does_push() {
        let mut list = PackingList::from([Some(0), Some(1)]);

        assert_eq!(list.add(2), 2);
        assert_eq!(list, [Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn debug_format() {
        let list = PackingList::from([Some(1), None, None, Some(5)]);
        assert_eq!("[1, _, _, 5]", format!("{:?}", list));
    }

    #[test]
    fn remove_makes_empty() {
        let mut list = PackingList::from([None, Some(1)]);

        assert_eq!(list.remove(1), Some(1));
        assert!(list.is_empty());
    }

    #[test]
    fn remove_none_is_none() {
        let vec = [Some(0), None, Some(1)];
        let mut list = PackingList::from(vec.clone());

        assert_eq!(list.remove(1), None);
        assert_eq!(list, vec);
    }

    #[test]
    fn iter_mut_weirdness() {
        let mut list = PackingList::new();

        assert_eq!(0, list.add(0));
        assert_eq!(1, list.add(1));
        assert_eq!(2, list.add(2));
        assert_eq!(3, list.add(3));
        assert_eq!(list, [Some(0), Some(1), Some(2), Some(3)]);

        assert_eq!(0, list.remove(0).unwrap());
        for v in list.iter_mut() {
            if *v > 0 {
                *v -= 1;
            }
        }
        assert_eq!(list, [None, Some(0), Some(1), Some(2)]);

        assert_eq!(1, list.remove(2).unwrap());
        assert_eq!(list.index_iter().collect::<Vec<usize>>(), vec![1, 3]);
        for v in list.iter_mut() {
            if *v > 1 {
                *v -= 1;
            }
        }
        assert_eq!(list, [None, Some(0), None, Some(1)]);

        assert_eq!(0, list.remove(1).unwrap());
        assert_eq!(list.index_iter().collect::<Vec<usize>>(), vec![3]);
        for v in list.iter_mut() {
            if *v > 0 {
                *v -= 1;
            }
        }
        assert_eq!(list, [None, None, None, Some(0)]);
    }

    #[test]
    fn into_iter() {
        let mut iter = PackingList::from([
            None,
            Some(1),
            Some(2),
            None,
            None,
            None,
            Some(3),
            None,
            Some(4),
        ])
        .into_iter();

        assert_eq!(iter.next(), Some((1, 1)));
        assert_eq!(iter.next(), Some((2, 2)));
        assert_eq!(iter.next(), Some((6, 3)));
        assert_eq!(iter.next(), Some((8, 4)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn indexing() {
        let vec = [Some(1), Some(100), None, None, Some(3)];
        let list = PackingList::from(vec.clone());

        for (i, item) in vec.into_iter().enumerate() {
            assert_eq!(item, list[i]);
        }
    }

    #[test]
    #[cfg(any(feature = "serde-std", feature = "serde-nostd"))]
    fn test_serialize() {
        let list = PackingList::from([Some(1), None, None, Some(400)]);
        let serialized = serde_json::to_string(&list).unwrap();
        assert_eq!("[1,null,null,400]", serialized);
    }
}
