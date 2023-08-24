use std::collections::{ binary_heap, BinaryHeap };
use std::cmp::Reverse;
use std::convert;
use std::fmt;
use std::hash::{ Hash, Hasher };
use std::ops::Index;
use std::slice::SliceIndex;

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

    pub fn list<'a>(&'a self) -> &'a Vec<Option<T>> {
        &self.list
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
    /// # use packing_list::PackingList;
    /// let mut list = PackingList::from(vec![Some(0), None, Some(2), None, Some(4), None]);
    ///
    /// let transform = list.pack();
    ///
    /// assert_eq!(*list.list(), vec![Some(0), Some(2), Some(4)]);
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

    #[inline]
    pub fn index_iter<'a>(&'a self) -> IndexIter<'a> {
        IndexIter {
            current: 0,
            end: self.list.len(),
            heap_iter: self.empty_spots.iter().peekable()
        }
    }

    #[inline]
    pub fn item_iter<'a>(&'a self) -> ItemIter<'a, T> {
        ItemIter {
            list: &self.list,
            index_iter: self.index_iter()
        }
    }

    /// Places `data` in the first available spot in the list. Returns the index it was placed at.
    ///
    /// # Examples
    ///
    /// ```
    /// # use packing_list::PackingList;
    /// let ex_vec = vec![Some(0), Some(1), Some(2)];
    /// let list = PackingList::from(ex_vec.clone());
    ///
    /// assert_eq!(*list.list(), ex_vec);
    /// ```
    ///
    /// ```
    /// # use packing_list::PackingList;
    /// # // TODO: move this to from trait implementation doc
    /// let mut list = PackingList::from(vec![Some(0), None, Some(2), None]);
    ///
    /// let idx = list.add(1);
    ///
    /// assert_eq!(idx, 1); // 1 was the smallest empty index
    /// assert_eq!(list[idx], Some(1));
    ///
    /// assert_eq!(*list.list(), vec![Some(0), Some(1), Some(2), None]);
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
    /// # use packing_list::PackingList;
    /// let mut list = PackingList::from(vec![Some(0), Some(1), Some(2), Some(3)]);
    ///
    /// list.remove_by_idx(1);
    /// list.remove_by_idx(2);
    ///
    /// assert_eq!(*list.list(), vec![Some(0), None, None, Some(3)]);
    /// ```
    ///
    /// # Time complexity
    ///
    /// If `idx` is the last index in the list then it'll take *O*(1) time. Otherwise the
    /// worst-case performance is *O*(log(*n*)).
    #[inline]
    pub fn remove_by_idx(&mut self, idx: usize) -> Option<T> {
        self.list.get_mut(idx)?.take().and_then(|v| {
            if idx == self.list.len() - 1 {
                self.list.pop();
            } else {
                self.empty_spots.push(Reverse(idx));
            }
            Some(v)
        })
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.list.hash(state);
    }
}

impl<T: PartialEq> PartialEq for PackingList<Option<T>> {
    fn eq(&self, other: &Self) -> bool {
        self.list == other.list
    }
}



use std::iter::Peekable;

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
        if let Some(i) = self.index_iter.next() {
            Some(self.list[i].as_ref().unwrap())
        } else {
            None
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

}
