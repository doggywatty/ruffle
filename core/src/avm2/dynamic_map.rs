use crate::avm2::object::Object;
use crate::string::AvmString;
use fnv::FnvBuildHasher;
use gc_arena::Collect;
use hashbrown::HashTable;
use hashbrown::hash_table::Entry;
use std::cell::Cell;
use std::hash::{BuildHasher, Hash};

#[derive(Debug, Collect, Copy, Clone)]
#[collect(no_drop)]
pub struct DynamicProperty<V> {
    pub value: V,
    pub enumerable: bool,
}

#[derive(Eq, PartialEq, Hash, Copy, Clone, Collect)]
#[collect(no_drop)]
pub enum DynamicKey<'gc> {
    String(AvmString<'gc>),
    // When the name parses as a non-negative integer, we use that integer as the key
    // See `ScriptObject::get_property_local` and `ScriptObject::set_property_local`.
    // This is observable when iterating over the object keys, as the key
    // can be `number`
    Uint(u32),
    Object(Object<'gc>),
}

/// A HashMap designed for dynamic properties on an object.
///
/// Uses `HashTable` directly to expose stable bucket indices, which are
/// needed for correct iteration when entries are added or removed mid-iteration.

/*
	16 bytes instead of 48.
	Every AS3 object holds one, and nearly all of them, instances of sealed classes, never get a dynamic property:
	the table is allocated with the first one, and the iteration indices are 32 bits.
	Super Smash Flash 2 keeps millions of small objects (its hitbox caches) and ran out of memory with 600-800 MB of AS3 heap.
*/
#[derive(Debug, Clone, Collect)]
#[collect(no_drop)]
pub struct DynamicMap<K, V> {
    table: Option<Box<HashTable<(K, DynamicProperty<V>)>>>,
    #[collect(require_static)]
    hasher: FnvBuildHasher,
    // The last index that was given back to flash
    #[collect(require_static)]
    public_index: Cell<u32>,
    // The actual bucket index that represents where an item is in the table
    #[collect(require_static)]
    real_index: Cell<u32>,
}

impl<K: Eq + Hash, V> Default for DynamicMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash, V> DynamicMap<K, V> {
    fn hash_key(&self, key: &K) -> u64 {
        self.hasher.hash_one(key)
    }

    pub fn new() -> Self {
        Self {
            table: None,
            hasher: FnvBuildHasher::default(),
            public_index: Cell::new(0),
            real_index: Cell::new(0),
        }
    }

    pub fn get(&self, key: &K) -> Option<&DynamicProperty<V>> {
        let table = self.table.as_ref()?;
        let hash = self.hash_key(key);

        table.find(hash, |(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        let Some(table) = &self.table else {
            return false;
        };
        let hash = self.hash_key(key);

        table.find(hash, |(k, _)| k == key).is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &DynamicProperty<V>)> {
        self.table.iter().flat_map(|table| table.iter()).map(|(k, v)| (k, v))
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.table.iter().flat_map(|table| table.iter()).map(|(k, _)| k)
    }

    pub fn len(&self) -> usize {
        self.table.as_ref().map_or(0, |table| table.len())
    }

    pub fn is_empty(&self) -> bool {
        self.table.as_ref().is_none_or(|table| table.is_empty())
    }

    fn num_buckets(&self) -> usize {
        self.table.as_ref().map_or(0, |table| table.num_buckets())
    }

    fn get_bucket(&self, index: usize) -> Option<&(K, DynamicProperty<V>)> {
        self.table.as_ref()?.get_bucket(index)
    }

    /// Gets the bucket index from the current public index.
    /// Returns `None` if the index is out of bounds.
    fn public_to_real_index(&self, index: usize) -> Option<usize> {
        let mut count = 0;
        let num_buckets = self.num_buckets();

        for i in 0..num_buckets {
            if let Some((_, v)) = self.get_bucket(i)
                && v.enumerable
            {
                count += 1;

                if count >= index {
                    return Some(i);
                }
            }
        }

        None
    }

    pub fn insert(&mut self, key: K, new_value: V) {
        let hash = self.hash_key(&key);
        let hasher = &self.hasher;
        let table = self.table.get_or_insert_with(|| Box::new(HashTable::new()));

        match table.entry(hash, |(k, _)| *k == key, |(k, _)| hasher.hash_one(k)) {
            Entry::Occupied(mut occupied) => {
                // NOTE: When inserting a new value into an already-occupied entry,
                // the value of the `enumerable` field isn't reset to `true`
                occupied.get_mut().1.value = new_value;
            }
            Entry::Vacant(vacant) => {
                vacant.insert((
                    key,
                    DynamicProperty {
                        value: new_value,
                        enumerable: true,
                    },
                ));
            }
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<DynamicProperty<V>> {
        let hash = self.hash_key(key);
        let table = self.table.as_mut()?;

        match table.find_entry(hash, |(k, _)| k == key) {
            Ok(occupied) => Some(occupied.remove().0.1),
            Err(_) => None,
        }
    }

    // NOTE: Per the docs on `HashTable::find_bucket_index`, bucket indices are
    // "only meaningful as long as the table is not resized and no entries are
    // added or removed". The current implementation uses tombstones and never
    // moves elements, so indices of other entries are preserved on removal in
    // practice, but this is not a guaranteed API contract and should not be
    // relied on long-term.
    // https://docs.rs/hashbrown/0.16.1/hashbrown/struct.HashTable.html#method.find_bucket_index
    pub fn next(&self, index: usize) -> Option<usize> {
        // Start iteration from the beginning
        if index == 0 {
            if let Some(real) = self.public_to_real_index(1) {
                self.real_index.set(real as u32);
                self.public_index.set(1);
                return Some(1);
            } else {
                self.public_index.set(0);
                self.real_index.set(0);
                return None;
            }
        }
        // If the public index is zero than this is the first time this is being enumerated,
        // if index != self.public_index, then we are enumerating twice OR we mutated while enumerating.
        //
        // Regardless of the reason, we just need to sync the supplied index to the real index.
        if self.public_index.get() == 0 || index != self.public_index.get() as usize {
            if let Some(real) = self.public_to_real_index(index) {
                // Pick up where we left off in the iteration
                // See https://github.com/adobe/avmplus/blob/858d034a3bd3a54d9b70909386435cf4aec81d21/core/avmplusHashtable.cpp#L395
                self.real_index.set(real as u32);
                self.public_index.set(index as u32);
            } else {
                self.public_index.set(0);
                self.real_index.set(0);
                return None;
            }
        }

        let real = self.real_index.get() as usize + 1;
        let num_buckets = self.num_buckets();

        if !self.is_empty() && real < num_buckets {
            for i in real..num_buckets {
                if let Some((_, v)) = self.get_bucket(i)
                    && v.enumerable
                {
                    self.real_index.set(i as u32);
                    self.public_index.set(self.public_index.get() + 1);
                    return Some(self.public_index.get() as usize);
                }
            }
        }

        None
    }

    pub fn pair_at(&self, index: usize) -> Option<&(K, DynamicProperty<V>)> {
        let real_index =
            if self.public_index.get() == 0 || self.public_index.get() as usize != index {
                self.public_to_real_index(index)?
            } else {
                self.real_index.get() as usize
            };

        self.get_bucket(real_index)
    }

    pub fn key_at(&self, index: usize) -> Option<&K> {
        self.pair_at(index).map(|(k, _)| k)
    }

    pub fn value_at(&self, index: usize) -> Option<&V> {
        self.pair_at(index).map(|(_, p)| &p.value)
    }

    pub fn set_enumerable(&mut self, key: &K, enumerable: bool) {
        let hash = self.hash_key(key);

        if let Some((_, prop)) = self.table.as_mut().and_then(|table| table.find_mut(hash, |(k, _)| k == key))
        {
            prop.enumerable = enumerable;
        }
    }
}

#[cfg(test)]
mod tests {

    use super::DynamicMap;

    #[test]
    fn test_dynamic_map() {
        let mut map: DynamicMap<&'static str, i32> = DynamicMap::new();
        map.insert("a", 0);
        map.insert("b", 0);
        map.insert("c", 0);
        map.insert("d", 0);
        let mut current = 0;
        while let Some(next) = map.next(current) {
            if current == 2 {
                map.insert("e", 0);
                map.insert("f", 0);
            }
            println!("{}", map.key_at(current).unwrap());
            current = next;
        }
        println!("next");
        current = 0;
        while let Some(next) = map.next(current) {
            println!("{}", map.key_at(current).unwrap());
            current = next;
        }
    }
}
