//! # packinglist
//!
//! This is a kind of [free list](https://en.wikipedia.org/wiki/Free_list) implementation where new elements are
//! *guaranteed* to be placed in the smallest available index of the list. 

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::collections::{ binary_heap, BinaryHeap };
#[cfg(feature = "std")]
use std::collections::{ binary_heap, BinaryHeap };

#[cfg(not(feature = "std"))]
use core::cmp::Reverse;
#[cfg(feature = "std")]
use std::cmp::Reverse;

#[cfg(not(feature = "std"))]
use core::convert;
#[cfg(feature = "std")]
use std::convert;

#[cfg(not(feature = "std"))]
use core::fmt;
#[cfg(feature = "std")]
use std::fmt;

#[cfg(not(feature = "std"))]
use core::hash::{ Hash, Hasher };
#[cfg(feature = "std")]
use std::hash::{ Hash, Hasher };

#[cfg(not(feature = "std"))]
use core::iter::{ Enumerate, FilterMap, Peekable };
#[cfg(feature = "std")]
use std::iter::{ Enumerate, FilterMap, Peekable };

#[cfg(not(feature = "std"))]
use core::ops::{ Deref, DerefMut, Drop, Index };
#[cfg(feature = "std")]
use std::ops::{ Deref, DerefMut, Drop, Index };

#[cfg(not(feature = "std"))]
use core::slice::SliceIndex;
#[cfg(feature = "std")]
use std::slice::SliceIndex;

#[cfg(feature = "serde")]
use serde::{ Deserialize, Deserializer, Serialize, Serializer };

pub type TransformTable = std::collections::BTreeMap<usize, usize>;

/// A FreeList implementation this will always put a new element at the smallest empty index of the
/// list.
#[derive(Clone, Default)]
pub struct PackingList<T> {
    list: Vec<Option<T>>,
    empty_spots: BinaryHeap<Reverse<usize>>
}

impl<T> PackingList<T> {
    /// Creates a new, empty `PackingList<T>`.
    ///
    /// Will not allocate until elements are added.
    #[inline]
    pub fn new() -> Self {
        PackingList {
            list: Vec::new(),
            empty_spots: BinaryHeap::new()
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
    /// let mut list = PackingList::from(vec![Some(1), None, Some(2), Some(5)]);
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
    /// assert_eq!(*list.as_vec(), [Some(0), Some(5), None, Some(5)]);
    /// ```
    #[inline]
    pub fn as_vec_mut<'a>(&'a mut self) -> VecPtr<'a, T> {
        VecPtr {
            list: self
        }
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
    /// let mut list = PackingList::from(vec![Some(0), None, Some(2), None, Some(4), None]);
    ///
    /// let transform = list.pack();
    ///
    /// assert_eq!(*list.as_vec(), [Some(0), Some(2), Some(4)]);
    ///
    /// assert_eq!(transform.get(&0), Some(&0));
    /// assert_eq!(transform.get(&2), Some(&1));
    /// assert_eq!(transform.get(&4), Some(&2));
    /// ```
    pub fn pack(&mut self) -> TransformTable {
        let mut old_list: Vec<Option<T>> = Vec::with_capacity(self.count());
        std::mem::swap(&mut old_list, &mut self.list);

        self.empty_spots.clear();

        let mut table = TransformTable::new();

        for (i, v) in old_list.into_iter().enumerate().filter(|(_, v)| v.is_some()) {
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
    /// let mut a = PackingList::from(vec![Some(0), None, Some(2), None, Some(4)]);
    /// let mut b = PackingList::from(vec![Some(1), Some(3)]);
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
        std::mem::swap(&mut old_other_list, &mut other.list);

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
    /// let list = PackingList::from(vec![None, Some(1), Some(9), Some(3), None, None, Some(4)]);
    ///
    /// let indeces: Vec<usize> = list.index_iter().collect();
    ///
    /// assert_eq!(indeces, [1, 2, 3, 6]);
    /// ```
    #[inline]
    pub fn index_iter(&self) -> IndexIter<'_> {
        IndexIter {
            current: 0,
            end: self.list.len(),
            heap_iter: self.empty_spots.iter().peekable()
        }
    }

    /// Returns an iterator over the items in the list in order of increasing index.
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let list = PackingList::from(vec![None, Some(1), Some(9), Some(3), None, None, Some(4)]);
    ///
    /// let items: Vec<&i32> = list.item_iter().collect();
    ///
    /// assert_eq!(items, [&1, &9, &3, &4]);
    /// ```
    #[inline]
    pub fn item_iter(&self) -> ItemIter<'_, T> {
        ItemIter {
            list: &self.list,
            index_iter: self.index_iter()
        }
    }

    /// Returns an iterator that allows modifying each non-empty value in the list, in order of
    /// increasing index.
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let mut list = PackingList::from(vec![Some(2), Some(5), None, Some(11)]);
    ///
    /// for v in list.iter_mut() {
    ///     *v += 1
    /// }
    ///
    /// assert_eq!(*list.as_vec(), [Some(3), Some(6), None, Some(12)]);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            items: self.item_iter()
        }
    }

    /// Places `data` in the first available spot in the list. Returns the index it was placed at.
    ///
    /// # Examples
    ///
    /// ```
    /// # use packinglist::PackingList;
    /// let ex_vec = vec![Some(0), Some(1), Some(2)];
    /// let list = PackingList::from(ex_vec.clone());
    /// assert_eq!(*list.as_vec(), ex_vec);
    ///
    /// let mut list = PackingList::from(vec![Some(0), None, Some(2), None]);
    /// let idx = list.add(1);
    /// assert_eq!(idx, 1); // 1 was the smallest empty index
    /// assert_eq!(list[idx], Some(1));
    /// assert_eq!(*list.as_vec(), [Some(0), Some(1), Some(2), None]);
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
    /// let mut list = PackingList::from(vec![Some(0), Some(1), Some(2), Some(3)]);
    ///
    /// list.remove(1);
    /// list.remove(2);
    ///
    /// assert_eq!(*list.as_vec(), [Some(0), None, None, Some(3)]);
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

    /// Get a mutable reference to the item at `idx` if it exists.
    ///
    /// # Examples
    /// ```
    /// # use packinglist::PackingList;
    /// let mut list = PackingList::from(vec![Some(19), None]);
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

    /// Removes all trailing `None`'s. All user-facing instances of `PackingList` should already be
    /// trimmed, so this is for internal purposes.
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

impl<T> fmt::Debug for PackingList<T> where T: fmt::Debug {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "[")?;

		let len = self.list.len();

		for opt in &self.list[0..(len - 1)] {
			match opt {
				Some(v) => write!(f, "{:?}, ", v)?,
				None => write!(f, "_, ")?
			};
		}

		match self.list.last() {
			Some(Some(v)) => write!(f, "{:?}]", v)?,
			_ => write!(f, "]")?
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

#[cfg(feature = "serde")]
impl<T: Serialize> Serialize for PackingList<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> 
    where
        S: Serializer,
    {
        self.list.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for PackingList<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>
    {
        Ok(Self::from(Vec::deserialize(deserializer)?))
    }
}

pub type ListIter<T> = FilterMap<Enumerate<std::vec::IntoIter<Option<T>>>, fn((usize, Option<T>)) -> Option<(usize, T)>>;

impl<T> IntoIterator for PackingList<T> {
    type Item = (usize, T);
    type IntoIter = ListIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.list.into_iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.map(|v| (i, v)))
    }
}

impl<T> convert::From<Vec<Option<T>>> for PackingList<T> {
    fn from(vec: Vec<Option<T>>) -> Self {
        let empty_spots: BinaryHeap<Reverse<usize>> = vec.iter()
            .enumerate()
            .filter(|(_, opt)| opt.is_none())
            .map(|(i, _)| Reverse(i)).collect();

        PackingList {
            list: vec,
            empty_spots
        }
    }
}

impl<T> Hash for PackingList<T> where T: Hash {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.list.hash(state);
    }
}

impl<T: PartialEq> PartialEq for PackingList<Option<T>> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.list == other.list
    }
}

pub struct VecPtr<'a, T> {
    list: &'a mut PackingList<T>
}

impl<'a, T> Deref for VecPtr<'a, T> {
    type Target = Vec<Option<T>>;

    fn deref(&self) -> &Self::Target {
        &self.list.list
    }
}

impl<'a, T> DerefMut for VecPtr<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.list.list
    }
}

impl<'a, T> Drop for VecPtr<'a, T> {
    fn drop(&mut self) {
        self.list.trim_vec();
        self.list.empty_spots.clear();

        let empty_indeces = self.list.list.iter()
            .enumerate()
            .filter(|(_, opt)| opt.is_none())
            .map(|(i, _)| i);

        for i in empty_indeces {
            self.list.empty_spots.push(Reverse(i));
        }
    }
}


pub struct IndexIter<'a> {
    current: usize,
    end: usize,
    heap_iter: Peekable<binary_heap::Iter<'a, Reverse<usize>>>
}

impl<'a> Iterator for IndexIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(peek) = self.heap_iter.peek() {
            if self.current < peek.0 {
                self.current += 1;
                Some(self.current - 1)
            } else {
                let mut popped = self.heap_iter.next().unwrap().0;

                while self.heap_iter.peek().is_some_and(|v| popped == v.0 - 1) {
                    popped = self.heap_iter.next().unwrap().0;
                }

                if popped + 1 < self.end {
                    self.current = popped + 2;
                    Some(self.current - 1)
                } else {
                    None
                }
            }
        } else if self.current < self.end {
            self.current += 1;
            Some(self.current - 1)
        } else {
            None
        }
    }
}

pub struct ItemIter<'a, T> {
    list: &'a Vec<Option<T>>,
    index_iter: IndexIter<'a>
}

impl<'a, T> Iterator for ItemIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iter.next() {
            Some(i) => self.list[i].as_ref(),
            None => None
        }
    }
}

pub struct IterMut<'a, T> {
    items: ItemIter<'a, T>
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.items.next() {
            Some(r) => {
                // This is my first REAL time doing stuff with Rust pointers. It feels illegal but
                // satisfying to pull off!
                unsafe {
                    (r as *const T).cast_mut().as_mut()
                }
            },
            None => None
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_does_fill() {
        let mut list = PackingList::from(vec![Some(0), None, Some(2), None, Some(4)]);

        assert_eq!(list.add(1), 1);
        assert_eq!(list.add(3), 3);
        assert_eq!(*list.as_vec(), [Some(0), Some(1), Some(2), Some(3), Some(4)]);
    }

    #[test]
    fn add_does_push() {
        let mut list = PackingList::from(vec![Some(0), Some(1)]);

        assert_eq!(list.add(2), 2);
        assert_eq!(*list.as_vec(), [Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn remove_makes_empty() {
        let mut list = PackingList::from(vec![None, Some(1)]);

        assert_eq!(list.remove(1), Some(1));
        assert!(list.is_empty());
    }

    #[test]
    fn remove_none_is_none() {
        let vec = vec![Some(0), None, Some(1)];
        let mut list = PackingList::from(vec.clone());

        assert_eq!(list.remove(1), None);
        assert_eq!(*list.as_vec(), vec);
    }

    #[test]
    fn into_iter() {
        let mut iter = PackingList::from(vec![None, Some(1), Some(2), None, None, None, Some(3), None, Some(4)])
            .into_iter();

        assert_eq!(iter.next(), Some((1, 1)));
        assert_eq!(iter.next(), Some((2, 2)));
        assert_eq!(iter.next(), Some((6, 3)));
        assert_eq!(iter.next(), Some((8, 4)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serialize() {
        let list = PackingList::from(vec![Some(1), None, None, Some(400)]);
        let serialized = serde_json::to_string(&list).unwrap();
        assert_eq!("[1,null,null,400]", serialized);
    }
}


