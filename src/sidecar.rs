//! Derived per-segment sidecars with a consumer-owned format.
//!
//! Consumers of [`SegmentedStore`] build one derived index per immutable
//! segment (postings, a graph, bucket tables) and want it cached in memory,
//! persisted next to `segstore.seg.<id>`, and dropped when the segment's live
//! id set changes. This module owns that lifecycle and nothing else:
//!
//! - The **consumer** owns the format: magic/version tags
//!   ([`SidecarCodec::magic`], [`SidecarCodec::version`]), a recipe identifier
//!   ([`SidecarCodec::recipe`]), the payload bytes ([`SidecarCodec::encode`],
//!   [`SidecarCodec::decode`]), and the acceptance rule that decides whether
//!   decoded bytes still describe a given live-id set
//!   ([`SidecarCodec::accepts`]).
//! - **segstore** owns the lifecycle: where the sidecar lives and how it is
//!   named ([`SidecarHost`]), the envelope framing (magic, version, segment id,
//!   recipe, CRC via [`SidecarEnvelope`]), the `(segment_id, live ids)` cache
//!   key, rebuild-on-miss, eviction for deleted segments, and atomic writes.
//!
//! A sidecar is a rebuildable cache, not part of the manifest commit: it may be
//! deleted at any time, and a missing, corrupt, misnamed, recipe-mismatched, or
//! stale sidecar is always rebuilt from the source segment.
//!
//! # Cache key and stale entries
//!
//! [`SidecarCache`] keys an entry by [`SidecarKey`] = `(segment_id, live ids)`,
//! with the live ids in one canonical order ([`LiveIds`]: sorted and deduped).
//! Because the live set is part of the key, a stale entry is never served: a
//! shrunken or grown live set is a different key, so the lookup misses, the
//! cache asks [`SidecarCodec::accepts`] about the on-disk sidecar, and a
//! rejected sidecar causes exactly one rebuild. Inserting the rebuilt entry
//! also drops the segment's previous entry, so a cache holds at most one entry
//! per segment.
//!
//! # Thread safety
//!
//! [`SidecarCache`] is `Sync` when the codec is: entries sit behind a
//! `std::sync::Mutex`, so `get_or_build` may be called concurrently and a
//! reader-side cache can be shared by an `Arc`. The lock is held for the whole
//! get/load/build step on purpose, which makes concurrent `get_or_build` calls
//! on the same key build exactly once; the cost is that builds (including
//! distinct keys) are serialized, so keep the build closure cheap or give each
//! costly build its own cache. A poisoned lock is recovered, not propagated: a
//! panic inside a build closure does not wedge the cache. The build closure must
//! not call back into the same cache, which would deadlock on the non-reentrant
//! mutex.
//!
//! # Example
//!
//! A toy consumer stores a sorted id list as its sidecar payload, cached per
//! segment and reloaded from disk instead of rebuilt.
//!
//! ```
//! use durability::MemoryDirectory;
//! use segstore::sidecar::{LiveIds, SidecarCache, SidecarCodec, SidecarHost, SidecarKey};
//! use segstore::{DefaultStore, SegmentedStore};
//!
//! struct IdListCodec;
//!
//! impl SidecarCodec for IdListCodec {
//!     type Id = u32;
//!     type Index = Vec<u32>;
//!
//!     fn magic(&self) -> &[u8; 8] { b"TOYIDX01" }
//!     fn version(&self) -> u32 { 1 }
//!     fn kind(&self) -> &str { "toy" }
//!     fn recipe(&self) -> &str { "u32-id-list-v1" }
//!
//!     fn encode(&self, index: &Self::Index) -> Option<Vec<u8>> {
//!         Some(index.iter().flat_map(|id| id.to_le_bytes()).collect())
//!     }
//!
//!     fn decode(&self, payload: &[u8]) -> Option<Self::Index> {
//!         if payload.len() % 4 != 0 {
//!             return None;
//!         }
//!         Some(
//!             payload
//!                 .chunks_exact(4)
//!                 .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
//!                 .collect(),
//!         )
//!     }
//!
//!     fn accepts(&self, index: &Self::Index, live: &LiveIds<u32>) -> bool {
//!         index.as_slice() == live.as_slice()
//!     }
//! }
//!
//! let dir = MemoryDirectory::arc();
//! let mut store =
//!     SegmentedStore::open(dir, DefaultStore::<u32, String>::new(), 2).unwrap();
//! store.add(1, "a".into()).unwrap();
//! store.add(2, "b".into()).unwrap();
//! store.checkpoint().unwrap();
//!
//! let seg = store.segment_ids()[0];
//! let live = store.live_ids(seg).unwrap();
//! assert_eq!(live.as_slice(), &[1, 2]);
//!
//! let cache = SidecarCache::new(IdListCodec);
//! let key = SidecarKey::new(seg, live.clone());
//! let built = cache
//!     .get_or_build(&store, key.clone(), || Some(live.as_slice().to_vec()))
//!     .unwrap();
//! assert_eq!(built.as_slice(), &[1, 2]);
//! assert_eq!(cache.persist_missing(&store, &[seg]).unwrap(), 1);
//!
//! // A fresh cache loads the persisted sidecar instead of rebuilding.
//! let reopened = SidecarCache::new(IdListCodec);
//! let loaded = reopened.get_or_build(&store, key, || None).unwrap();
//! assert_eq!(loaded.as_slice(), &[1, 2]);
//! ```

use std::collections::HashMap;
use std::hash::Hash;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use durability::{Directory, PersistenceError, PersistenceResult};
use serde::de::DeserializeOwned;

use crate::{try_index_name, SegmentCatalog, SegmentedStore, SidecarEnvelope, Store, View};

/// One segment's live ids in segstore's canonical order: sorted and deduped.
///
/// This is the live-set half of a [`SidecarKey`] and the fingerprint a
/// [`SidecarCodec::accepts`] check compares against. Build it from the ids the
/// consumer enumerated from its own decoded segment, filtered by the host's
/// liveness predicate ([`SegmentedStore::is_live_in`],
/// [`View::is_live_in`], or [`SegmentCatalog::is_live_in`]) — or, for a writer,
/// from [`SegmentedStore::live_ids`] directly.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LiveIds<Id> {
    ids: Vec<Id>,
}

impl<Id: Ord> LiveIds<Id> {
    /// Canonicalize `ids` into a sorted, deduped live-id set.
    pub fn new(ids: impl IntoIterator<Item = Id>) -> Self {
        let mut ids: Vec<Id> = ids.into_iter().collect();
        ids.sort_unstable();
        ids.dedup();
        Self { ids }
    }

    /// The live ids, sorted ascending with duplicates removed.
    pub fn as_slice(&self) -> &[Id] {
        &self.ids
    }

    /// Number of live ids.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the segment has no live ids left.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Whether `id` is in the live set.
    pub fn contains(&self, id: &Id) -> bool {
        self.ids.binary_search(id).is_ok()
    }

    /// Iterate the live ids in ascending order.
    pub fn iter(&self) -> std::slice::Iter<'_, Id> {
        self.ids.iter()
    }
}

impl<Id: Ord> FromIterator<Id> for LiveIds<Id> {
    fn from_iter<I: IntoIterator<Item = Id>>(iter: I) -> Self {
        Self::new(iter)
    }
}

/// The cache key of one sidecar: a segment id plus that segment's live-id set.
///
/// The live ids are part of the key, so a changed live set can never be served
/// an entry built for the previous one.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SidecarKey<Id> {
    segment_id: u64,
    live_ids: LiveIds<Id>,
}

impl<Id> SidecarKey<Id> {
    /// Key a sidecar by the stable segment id and that segment's live ids.
    pub fn new(segment_id: u64, live_ids: LiveIds<Id>) -> Self {
        Self {
            segment_id,
            live_ids,
        }
    }

    /// The stable segment id this sidecar belongs to.
    pub fn segment_id(&self) -> u64 {
        self.segment_id
    }

    /// The live-id set this sidecar was built for.
    pub fn live_ids(&self) -> &LiveIds<Id> {
        &self.live_ids
    }
}

/// A consumer's sidecar format and acceptance rule.
///
/// segstore frames the payload with [`SidecarEnvelope`] using these tags, so a
/// copied, corrupt, or recipe-mismatched file is rejected before
/// [`Self::decode`] runs. Everything below the envelope — what the bytes mean,
/// whether an index is usable for a given live set — is consumer policy.
pub trait SidecarCodec {
    /// The consumer's id type; ids must be orderable so [`LiveIds`] has one
    /// canonical form.
    type Id: Clone + Ord + Hash;
    /// The built per-segment index this codec encodes and decodes.
    type Index;

    /// Envelope magic tag, e.g. `b"MYIDX001"`.
    fn magic(&self) -> &[u8; 8];

    /// Envelope format version for the payload bytes.
    fn version(&self) -> u32;

    /// Sidecar namespace tag used in the file name (`segstore.idx.<id>.<kind>`).
    ///
    /// Must satisfy the documented sidecar-kind grammar (non-empty ASCII
    /// alphanumeric plus `_`/`-`); an invalid tag surfaces as an error from the
    /// cache instead of producing a name that garbage collection cannot match.
    fn kind(&self) -> &str;

    /// Recipe identifier: every build choice that changes the bytes or their
    /// meaning. A mismatch rebuilds instead of decoding.
    fn recipe(&self) -> &str;

    /// Encode the payload (not the envelope). `None` means "not encodable".
    fn encode(&self, index: &Self::Index) -> Option<Vec<u8>>;

    /// Decode a payload that already passed envelope validation. `None` means
    /// the bytes are unusable and the sidecar must be rebuilt.
    fn decode(&self, payload: &[u8]) -> Option<Self::Index>;

    /// Whether a decoded index may serve `live` for its segment.
    ///
    /// This is the acceptance check that closes the stale-sidecar hole: return
    /// `false` for an index built for a different live-id set. `false` triggers
    /// exactly one rebuild.
    fn accepts(&self, index: &Self::Index, live: &LiveIds<Self::Id>) -> bool;
}

/// Where a sidecar lives, how it is named, and how it is written.
///
/// Implemented for the writer ([`SegmentedStore`]), the manifest-only reader
/// ([`SegmentCatalog`]) and a snapshot ([`View`]), so the same cache works on all
/// three read modes without reopening the store.
pub trait SidecarHost {
    /// The directory sidecars are written to.
    fn sidecar_dir(&self) -> &Arc<dyn Directory>;

    /// The reserved sidecar name for `segment_id`, validated against this host's
    /// known segment set where the type can check it.
    fn try_sidecar_name(&self, segment_id: u64, kind: &str) -> PersistenceResult<String>;

    /// Infallible convenience wrapper around [`Self::try_sidecar_name`].
    ///
    /// Panics if the segment id is not visible to this host or `kind` violates
    /// the documented sidecar-kind grammar.
    fn sidecar_name(&self, segment_id: u64, kind: &str) -> String {
        self.try_sidecar_name(segment_id, kind)
            .unwrap_or_else(|e| panic!("invalid segstore sidecar name: {e}"))
    }

    /// Filesystem path for a sidecar, when the backend exposes paths.
    fn sidecar_file_path(&self, segment_id: u64, kind: &str) -> PersistenceResult<Option<PathBuf>> {
        let name = self.try_sidecar_name(segment_id, kind)?;
        Ok(self.sidecar_dir().file_path(&name))
    }

    /// Whether a sidecar file exists.
    fn sidecar_exists(&self, name: &str) -> bool {
        self.sidecar_dir().exists(name)
    }

    /// Read a whole sidecar, or `None` if it is absent.
    fn read_sidecar(&self, name: &str) -> PersistenceResult<Option<Vec<u8>>> {
        let dir = self.sidecar_dir();
        if !dir.exists(name) {
            return Ok(None);
        }
        let mut file = dir.open_file(name)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(Some(bytes))
    }

    /// Atomically publish a sidecar; a torn write can never be observed.
    fn write_sidecar(&self, name: &str, bytes: &[u8]) -> PersistenceResult<()> {
        self.sidecar_dir().atomic_write(name, bytes)
    }

    /// Delete a sidecar. Absent files are a no-op, like [`Directory::delete`].
    fn delete_sidecar(&self, name: &str) -> PersistenceResult<()> {
        let dir = self.sidecar_dir();
        if dir.exists(name) {
            dir.delete(name)?;
        }
        Ok(())
    }
}

impl<S: Store> SidecarHost for SegmentedStore<S> {
    fn sidecar_dir(&self) -> &Arc<dyn Directory> {
        self.dir()
    }

    fn try_sidecar_name(&self, segment_id: u64, kind: &str) -> PersistenceResult<String> {
        self.try_index_name(segment_id, kind)
    }
}

impl<Id> SidecarHost for SegmentCatalog<Id>
where
    Id: Eq + Hash + DeserializeOwned,
{
    fn sidecar_dir(&self) -> &Arc<dyn Directory> {
        self.dir()
    }

    fn try_sidecar_name(&self, segment_id: u64, kind: &str) -> PersistenceResult<String> {
        self.try_index_name(segment_id, kind)
    }
}

impl<S: Store> SidecarHost for View<S> {
    fn sidecar_dir(&self) -> &Arc<dyn Directory> {
        self.dir()
    }

    fn try_sidecar_name(&self, segment_id: u64, kind: &str) -> PersistenceResult<String> {
        if self.segment_ids().binary_search(&segment_id).is_err() {
            return Err(PersistenceError::NotFound(format!(
                "segment id {segment_id} is not in this view"
            )));
        }
        try_index_name(segment_id, kind)
    }
}

/// In-memory sidecar cache with write-through helpers, keyed by
/// `(segment_id, live ids)`.
///
/// See the module docs for the ownership split, thread safety, and stale-entry
/// behavior.
pub struct SidecarCache<C: SidecarCodec> {
    codec: C,
    entries: Mutex<EntryMap<C::Id, C::Index>>,
}

/// Cached entry table: one built index per `(segment id, live ids)`.
type EntryMap<Id, Index> = HashMap<SidecarKey<Id>, Arc<Index>>;

/// One `(key, index)` pair lifted out of the cache for persistence.
type Entry<Id, Index> = (SidecarKey<Id>, Arc<Index>);

impl<C: SidecarCodec> SidecarCache<C> {
    /// An empty cache using `codec` for framing, encoding, decoding, and
    /// acceptance.
    pub fn new(codec: C) -> Self {
        Self {
            codec,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// The codec this cache was built with.
    pub fn codec(&self) -> &C {
        &self.codec
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries().len()
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries().is_empty()
    }

    /// Whether `key` is cached in memory.
    pub fn contains(&self, key: &SidecarKey<C::Id>) -> bool {
        self.entries().contains_key(key)
    }

    /// Return the cached index for `key`, else load it from `host`, else build
    /// it.
    ///
    /// Order: memory hit, then the persisted sidecar (decoded only after the
    /// envelope's magic/version/segment-id/recipe/CRC check and only when
    /// [`SidecarCodec::accepts`] approves the decoded index for `key`'s live
    /// ids), then `build`. A missing sidecar, an envelope or decode failure, an
    /// I/O error while loading, and a rejected live-id set all count as a miss
    /// and trigger the build; the sidecar is a cache, so none of them is fatal.
    ///
    /// `build` also runs when the on-disk sidecar was rejected, so a stale file
    /// is rebuilt from source. That build is not persisted here: call
    /// [`Self::persist`] or [`Self::persist_missing`] to write it, or
    /// [`Self::invalidate`] before a mutation.
    ///
    /// Returns `None` if the sidecar was absent and `build` returned `None`;
    /// a failed build leaves no entry behind.
    pub fn get_or_build<H, F>(
        &self,
        host: &H,
        key: SidecarKey<C::Id>,
        build: F,
    ) -> Option<Arc<C::Index>>
    where
        H: SidecarHost,
        F: FnOnce() -> Option<C::Index>,
    {
        let mut entries = self.entries();
        if let Some(hit) = entries.get(&key) {
            return Some(hit.clone());
        }
        if let Some(index) = self.load(host, &key) {
            let index = Arc::new(index);
            Self::insert(&mut entries, key, index.clone());
            return Some(index);
        }
        let index = Arc::new(build()?);
        Self::insert(&mut entries, key, index.clone());
        Some(index)
    }

    /// Drop the cached entry for `segment_id` and delete its persisted sidecar.
    ///
    /// Call this before mutating a segment's live set (a re-add or delete of an
    /// id in that segment), so neither the memory entry nor the file can be
    /// served for the old set.
    pub fn invalidate<H: SidecarHost>(&self, host: &H, segment_id: u64) -> PersistenceResult<()> {
        self.entries()
            .retain(|key, _| key.segment_id() != segment_id);
        let name = host.try_sidecar_name(segment_id, self.codec.kind())?;
        host.delete_sidecar(&name)
    }

    /// Evict cached entries whose segment id is not in `segment_ids`.
    ///
    /// This is the post-compaction step: `segment_ids` is
    /// [`SegmentedStore::segment_ids`] (or a view's), so entries for merged-away
    /// segments are dropped. Persisted files for those segments are
    /// garbage-collected by segstore on the next open, not here.
    pub fn retain(&self, segment_ids: &[u64]) {
        self.entries()
            .retain(|key, _| segment_ids.contains(&key.segment_id()));
    }

    /// Atomically write the cached entry for `key`, returning whether it existed.
    ///
    /// This replaces an existing file unconditionally; use it to persist a
    /// rebuild that replaced a rejected stale sidecar.
    pub fn persist<H: SidecarHost>(
        &self,
        host: &H,
        key: &SidecarKey<C::Id>,
    ) -> PersistenceResult<bool> {
        let Some(index) = self.entries().get(key).cloned() else {
            return Ok(false);
        };
        let bytes = self.encode_envelope(key, &index)?;
        let name = host.try_sidecar_name(key.segment_id(), self.codec.kind())?;
        host.write_sidecar(&name, &bytes)?;
        Ok(true)
    }

    /// Persist every cached entry for `segment_ids` whose sidecar file is
    /// absent, returning how many were written.
    ///
    /// Existing files are left untouched by contract, even if their content is
    /// stale: a present sidecar is either accepted at load time or rebuilt and
    /// written through [`Self::persist`].
    pub fn persist_missing<H: SidecarHost>(
        &self,
        host: &H,
        segment_ids: &[u64],
    ) -> PersistenceResult<usize> {
        let pending: Vec<Entry<C::Id, C::Index>> = {
            let entries = self.entries();
            entries
                .iter()
                .filter(|(key, _)| segment_ids.contains(&key.segment_id()))
                .map(|(key, index)| (key.clone(), index.clone()))
                .collect()
        };
        let mut written = 0;
        for (key, index) in pending {
            let name = host.try_sidecar_name(key.segment_id(), self.codec.kind())?;
            if host.sidecar_exists(&name) {
                continue;
            }
            let bytes = self.encode_envelope(&key, &index)?;
            host.write_sidecar(&name, &bytes)?;
            written += 1;
        }
        Ok(written)
    }

    /// Load, validate, and accept the persisted sidecar for `key`, if any.
    fn load<H: SidecarHost>(&self, host: &H, key: &SidecarKey<C::Id>) -> Option<C::Index> {
        let name = host
            .try_sidecar_name(key.segment_id(), self.codec.kind())
            .ok()?;
        let bytes = host.read_sidecar(&name).ok()??;
        let payload = SidecarEnvelope::decode(
            self.codec.magic(),
            self.codec.version(),
            key.segment_id(),
            self.codec.recipe().as_bytes(),
            &bytes,
        )
        .ok()?;
        let index = self.codec.decode(payload)?;
        self.codec.accepts(&index, key.live_ids()).then_some(index)
    }

    /// Frame `index` for `key` in the consumer's envelope.
    fn encode_envelope(
        &self,
        key: &SidecarKey<C::Id>,
        index: &C::Index,
    ) -> PersistenceResult<Vec<u8>> {
        let payload = self.codec.encode(index).ok_or_else(|| {
            PersistenceError::InvalidConfig(format!(
                "sidecar codec {:?} could not encode segment {}",
                self.codec.kind(),
                key.segment_id()
            ))
        })?;
        SidecarEnvelope::encode(
            self.codec.magic(),
            self.codec.version(),
            key.segment_id(),
            self.codec.recipe().as_bytes(),
            &payload,
        )
    }

    /// Insert, dropping any previous entry for the same segment so a changed
    /// live-id set replaces rather than accumulates.
    fn insert(
        entries: &mut EntryMap<C::Id, C::Index>,
        key: SidecarKey<C::Id>,
        index: Arc<C::Index>,
    ) {
        let segment_id = key.segment_id();
        entries.retain(|existing, _| existing.segment_id() != segment_id);
        entries.insert(key, index);
    }

    fn entries(&self) -> MutexGuard<'_, EntryMap<C::Id, C::Index>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    use durability::MemoryDirectory;

    use crate::DefaultStore;

    /// A toy consumer: the index is just the segment's live id list.
    struct IdListCodec;

    impl SidecarCodec for IdListCodec {
        type Id = u32;
        type Index = Vec<u32>;

        fn magic(&self) -> &[u8; 8] {
            b"TOYIDX01"
        }

        fn version(&self) -> u32 {
            1
        }

        fn kind(&self) -> &str {
            "toy"
        }

        fn recipe(&self) -> &str {
            "u32-id-list-v1"
        }

        fn encode(&self, index: &Self::Index) -> Option<Vec<u8>> {
            Some(index.iter().flat_map(|id| id.to_le_bytes()).collect())
        }

        fn decode(&self, payload: &[u8]) -> Option<Self::Index> {
            if payload.len() % 4 != 0 {
                return None;
            }
            Some(
                payload
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        }

        fn accepts(&self, index: &Self::Index, live: &LiveIds<u32>) -> bool {
            index.as_slice() == live.as_slice()
        }
    }

    fn sealed_store() -> (
        Arc<dyn Directory>,
        SegmentedStore<DefaultStore<u32, String>>,
    ) {
        let dir = MemoryDirectory::arc();
        let mut store =
            SegmentedStore::open(dir.clone(), DefaultStore::<u32, String>::new(), 2).unwrap();
        store.add(1, "a".into()).unwrap();
        store.add(2, "b".into()).unwrap();
        store.checkpoint().unwrap();
        (dir, store)
    }

    fn key(segment_id: u64, ids: &[u32]) -> SidecarKey<u32> {
        SidecarKey::new(segment_id, LiveIds::new(ids.iter().copied()))
    }

    /// A directory-only host, so the cache can be exercised across threads: the
    /// writer handle is `!Sync`, but [`SidecarHost`] is not tied to it.
    struct DirHost {
        dir: Arc<dyn Directory>,
    }

    impl SidecarHost for DirHost {
        fn sidecar_dir(&self) -> &Arc<dyn Directory> {
            &self.dir
        }

        fn try_sidecar_name(&self, segment_id: u64, kind: &str) -> PersistenceResult<String> {
            try_index_name(segment_id, kind)
        }
    }

    #[test]
    fn toy_codec_round_trips_through_a_store_host() {
        let (_dir, store) = sealed_store();
        let seg = store.segment_ids()[0];
        let live = store.live_ids(seg).expect("default store lists ids");
        assert_eq!(live.as_slice(), &[1, 2]);

        let cache = SidecarCache::new(IdListCodec);
        let mut builds = 0;
        let built = cache
            .get_or_build(&store, key(seg, &[1, 2]), || {
                builds += 1;
                Some(live.as_slice().to_vec())
            })
            .expect("built");
        assert_eq!(built.as_slice(), &[1, 2]);
        assert_eq!(cache.persist_missing(&store, &[seg]).unwrap(), 1);

        let reopened = SidecarCache::new(IdListCodec);
        let loaded = reopened
            .get_or_build(&store, key(seg, &[1, 2]), || {
                builds += 1;
                None
            })
            .expect("loaded from the persisted sidecar");
        assert_eq!(loaded.as_slice(), &[1, 2]);
        assert_eq!(builds, 1, "the persisted sidecar was loaded, not rebuilt");
    }

    #[test]
    fn stale_live_ids_invalidate_and_rebuild_once() {
        let (_dir, store) = sealed_store();
        let seg = store.segment_ids()[0];
        let cache = SidecarCache::new(IdListCodec);
        let mut builds = 0;
        cache
            .get_or_build(&store, key(seg, &[1, 2]), || {
                builds += 1;
                Some(vec![1, 2])
            })
            .unwrap();
        assert_eq!(cache.persist_missing(&store, &[seg]).unwrap(), 1);

        // The segment's live set shrank, but the persisted sidecar still
        // carries ids 1 and 2.
        let rebuilt = cache
            .get_or_build(&store, key(seg, &[1]), || {
                builds += 1;
                Some(vec![1])
            })
            .unwrap();
        assert_eq!(builds, 2, "the stale sidecar was rejected and rebuilt once");
        assert_eq!(rebuilt.as_slice(), &[1]);
        assert_eq!(cache.len(), 1, "the stale entry is replaced, not kept");
        assert!(!cache.contains(&key(seg, &[1, 2])));

        let hit = cache
            .get_or_build(&store, key(seg, &[1]), || {
                builds += 1;
                None
            })
            .unwrap();
        assert_eq!(hit.as_slice(), &[1]);
        assert_eq!(builds, 2, "the rebuilt entry is served from memory");
    }

    #[test]
    fn retain_evicts_entries_for_deleted_segments() {
        let (_dir, store) = sealed_store();
        let cache = SidecarCache::new(IdListCodec);
        for seg in [0, 1, 2] {
            cache
                .get_or_build(&store, key(seg, &[1]), || Some(vec![1]))
                .unwrap();
        }
        assert_eq!(cache.len(), 3);

        cache.retain(&[0, 2]);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&key(0, &[1])));
        assert!(!cache.contains(&key(1, &[1])));
        assert!(cache.contains(&key(2, &[1])));
    }

    #[test]
    fn persist_missing_writes_only_absent_entries() {
        let (_dir, store) = sealed_store();
        let seg = store.segment_ids()[0];
        let cache = SidecarCache::new(IdListCodec);
        cache
            .get_or_build(&store, key(seg, &[1, 2]), || Some(vec![1, 2]))
            .unwrap();

        let name = store.sidecar_name(seg, "toy");
        store.write_sidecar(&name, b"present").unwrap();
        assert_eq!(
            cache.persist_missing(&store, &[seg, seg + 1]).unwrap(),
            0,
            "a present sidecar is never overwritten"
        );
        assert_eq!(store.read_sidecar(&name).unwrap().unwrap(), b"present");

        store.delete_sidecar(&name).unwrap();
        assert_eq!(
            cache.persist_missing(&store, &[seg, seg + 1]).unwrap(),
            1,
            "only the cached entry for a listed segment is written"
        );
        let written = store.read_sidecar(&name).unwrap().expect("sidecar");
        let payload = SidecarEnvelope::decode(b"TOYIDX01", 1, seg, b"u32-id-list-v1", &written)
            .expect("valid envelope");
        assert_eq!(payload, &[1, 0, 0, 0, 2, 0, 0, 0]);
    }

    #[test]
    fn invalidate_drops_the_entry_and_the_persisted_file() {
        let (_dir, store) = sealed_store();
        let seg = store.segment_ids()[0];
        let cache = SidecarCache::new(IdListCodec);
        cache
            .get_or_build(&store, key(seg, &[1, 2]), || Some(vec![1, 2]))
            .unwrap();
        assert_eq!(cache.persist_missing(&store, &[seg]).unwrap(), 1);

        cache.invalidate(&store, seg).unwrap();
        assert!(cache.is_empty());
        assert!(!store.sidecar_exists(&store.sidecar_name(seg, "toy")));

        let mut rebuilt = 0;
        cache
            .get_or_build(&store, key(seg, &[1, 2]), || {
                rebuilt += 1;
                Some(vec![1, 2])
            })
            .unwrap();
        assert_eq!(rebuilt, 1, "an invalidated key rebuilds exactly once");
    }

    #[test]
    fn concurrent_get_or_build_on_one_key_builds_once() {
        let host = DirHost {
            dir: MemoryDirectory::arc(),
        };
        let cache = SidecarCache::new(IdListCodec);
        let builds = AtomicUsize::new(0);
        let barrier = Barrier::new(8);

        let handles: Vec<Arc<Vec<u32>>> = std::thread::scope(|scope| {
            let spawned: Vec<_> = (0..8)
                .map(|_| {
                    let cache = &cache;
                    let host = &host;
                    let builds = &builds;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        cache
                            .get_or_build(host, key(0, &[1, 2]), || {
                                builds.fetch_add(1, Ordering::SeqCst);
                                Some(vec![1, 2])
                            })
                            .expect("built once")
                    })
                })
                .collect();
            spawned
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });

        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert!(
            handles
                .iter()
                .all(|handle| Arc::ptr_eq(handle, &handles[0])),
            "every caller receives the same cached index"
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn catalog_and_view_hosts_name_the_same_sidecar() {
        let (dir, store) = sealed_store();
        let seg = store.segment_ids()[0];
        let expected = store.sidecar_name(seg, "toy");

        let catalog = SegmentCatalog::<u32>::open(dir).unwrap();
        assert_eq!(catalog.sidecar_name(seg, "toy"), expected);

        let view = store.reader().view();
        assert_eq!(view.sidecar_name(seg, "toy"), expected);
        assert_eq!(view.live_ids(seg, [2, 1, 2]).as_slice(), &[1, 2]);

        let missing = view.try_sidecar_name(seg + 99, "toy").unwrap_err();
        assert!(missing.to_string().contains("not in this view"));
    }
}
