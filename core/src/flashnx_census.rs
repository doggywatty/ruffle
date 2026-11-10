/*
    Census of live objects, for the "census:" heartbeat line.

    Super Smash Flash 2's GC heap grows by ~100 MB per Classic level
    (130 MB at the first fight, 450 MB at the 4th, where the heap ran out),
    and the orphan list and the frame time grow with it.
    These counters say WHAT accumulates: movies, display objects, and AVM2 objects per class.

    The per-object guards only exist with the "flashnx_census" feature:
    they are a field of every AVM2 object and every display object,
    and the class table takes a lock on every AVM2 allocation and free.
*/

#[cfg(feature = "flashnx_census")]
mod imp {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};
    use std::sync::Mutex;

    static LIVE_DISPLAY_OBJECTS: AtomicI64 = AtomicI64::new(0);
    static ORPHANS: AtomicU64 = AtomicU64::new(0);
    static BROADCAST: AtomicU64 = AtomicU64::new(0);

    struct Entry {
        name: String,
        live: i64,
        last: i64,
    }

    /// Live AVM2 objects, keyed by their class pointer.
    static CLASSES: Mutex<Option<HashMap<usize, Entry>>> = Mutex::new(None);

    fn with_classes<R>(f: impl FnOnce(&mut HashMap<usize, Entry>) -> R) -> R {
        let mut guard = CLASSES.lock().unwrap_or_else(|e| e.into_inner());
        f(guard.get_or_insert_with(HashMap::new))
    }

    /// Field of every "ScriptObjectData": counts it under its class.
    #[derive(Debug)]
    pub struct ObjectGuard(usize);

    impl ObjectGuard {
        pub fn new(class_key: usize, name: impl FnOnce() -> String) -> Self {
            with_classes(|map| {
                map.entry(class_key).or_insert_with(|| Entry {
                        name: name(),
                        live: 0,
                        last: 0,
                    })
                    .live += 1;
            });
            Self(class_key)
        }
    }

    impl Clone for ObjectGuard {
        fn clone(&self) -> Self {
            with_classes(|map| {
                if let Some(e) = map.get_mut(&self.0) {
                    e.live += 1;
                }
            });
            Self(self.0)
        }
    }

    impl Drop for ObjectGuard {
        fn drop(&mut self) {
            with_classes(|map| {
                if let Some(e) = map.get_mut(&self.0) {
                    e.live -= 1;
                }
            });
        }
    }

    // Field of every "DisplayObjectBase".
    #[derive(Debug)]
    pub struct DisplayGuard;

    impl Default for DisplayGuard {
        fn default() -> Self {
            LIVE_DISPLAY_OBJECTS.fetch_add(1, Relaxed);
            Self
        }
    }

    impl Clone for DisplayGuard {
        fn clone(&self) -> Self {
            Self::default()
        }
    }

    impl Drop for DisplayGuard {
        fn drop(&mut self) {
            LIVE_DISPLAY_OBJECTS.fetch_sub(1, Relaxed);
        }
    }

    pub fn note_frame(orphans: usize, broadcast: usize) {
        ORPHANS.store(orphans as u64, Relaxed);
        BROADCAST.store(broadcast as u64, Relaxed);
    }

    pub fn report() -> String {
        let (movies, movie_bytes) = ruffle_common::tag_utils::live_movie_stats();
        let (shared, copied) = crate::character::bitmap_bytes_stats();
        let (snd_shared, snd_copied) = crate::backend::audio::sound_bytes_stats();
        let (released, dropped, kept) = crate::library::movie_free_stats();
        let mb = |b: u64| b / (1024 * 1024);
        with_classes(|map| {
            let total: i64 = map.values().map(|e| e.live).sum();
            let mut entries: Vec<&Entry> = map.values().filter(|e| e.live != 0).collect();
            entries.sort_by_key(|e| -e.live);
            let top: Vec<String> = entries.iter().take(12).map(|e| format!("{} {}({:+})", e.name, e.live, e.live - e.last)).collect();
            entries.sort_by_key(|e| -(e.live - e.last));
            let grew: Vec<String> = entries.iter().take(8).filter(|e| e.live > e.last).map(|e| format!("{} {:+}", e.name, e.live - e.last)).collect();
            for e in map.values_mut() {
                e.last = e.live;
            }
            format!(
                "movies={} ({}MB) released={} dropped={} kept={} bitmapBytes shared={}MB copied={}MB soundBytes shared={}MB copied={}MB displayObjects={} orphans={} broadcast={} avm2={} | top: {} | grew: {}",
                movies,
                mb(movie_bytes),
                released,
                dropped,
                kept,
                mb(shared),
                mb(copied),
                mb(snd_shared),
                mb(snd_copied),
                LIVE_DISPLAY_OBJECTS.load(Relaxed),
                ORPHANS.load(Relaxed),
                BROADCAST.load(Relaxed),
                total,
                top.join(", "),
                grew.join(", "),
            )
        })
    }
}

#[cfg(feature = "flashnx_census")]
pub use imp::{note_frame, report, DisplayGuard, ObjectGuard};

// Without "flashnx_census", only the counters that cost nothing per object.
#[cfg(not(feature = "flashnx_census"))]
pub fn report() -> String {
    let (movies, movie_bytes) = ruffle_common::tag_utils::live_movie_stats();
    let (shared, copied) = crate::character::bitmap_bytes_stats();
    let (snd_shared, snd_copied) = crate::backend::audio::sound_bytes_stats();
    let (released, dropped, kept) = crate::library::movie_free_stats();
    let mb = |b: u64| b / (1024 * 1024);
    let line = format!(
        "movies={} ({}MB) released={} dropped={} kept={} bitmapBytes shared={}MB copied={}MB soundBytes shared={}MB copied={}MB",
        movies,
        mb(movie_bytes),
        released,
        dropped,
        kept,
        mb(shared),
        mb(copied),
        mb(snd_shared),
        mb(snd_copied),
    );
    // Measuring builds: which constructor made the movies (since boot), and which clips were emptied (since the last line).
    #[cfg(feature = "flashnx_instr")]
    let line = {
        let m = ruffle_common::tag_utils::movies_made();
        format!(
            "{line} | made: empty={} fakeLen={} fakeData={} error={} data={} image={} copy={} | emptied {}",
            m[0],
            m[1],
            m[2],
            m[3],
            m[4],
            m[5],
            m[6],
            crate::display_object::flashnx_emptied_take(),
        )
    };
    line
}
