use std::cell::OnceCell;

use crate::backend::audio::SoundHandle;
use crate::binary_data::BinaryData;
use crate::display_object::{
    Avm1Button, Avm2Button, BitmapClass, EditText, Graphic, MorphShape, MovieClip, Text, Video,
};
use crate::font::Font;
use gc_arena::barrier::unlock;
use gc_arena::lock::Lock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_render::backend::RenderBackend;
use ruffle_render::bitmap::{Bitmap as RenderBitmap, BitmapHandle, BitmapSize};
use ruffle_render::error::Error as RenderError;
use ruffle_common::tag_utils::SwfSlice;
use std::borrow::Cow;
use std::sync::Arc;
use swf::DefineBitsLossless;

impl<'gc> Character<'gc> {
    /*
        whether an instance of this symbol is alive, in a collection about to sweep (see "MovieLibrary::in_use" in library.rs).
        Instances share the "shared" data of the library's copy.
        A character whose instances keep nothing of it answers false.
    */
    pub fn has_live_instances(&self, fc: &gc_arena::Finalization<'gc>) -> bool {
        match self {
            Character::MovieClip(c) => c.shared_alive(fc),
            Character::Graphic(c) => c.shared_alive(fc),
            Character::EditText(c) => c.shared_alive(fc),
            Character::Text(c) => c.shared_alive(fc),
            Character::MorphShape(c) => c.shared_alive(fc),
            Character::Avm1Button(c) => c.shared_alive(fc),
            Character::Avm2Button(c) => c.shared_alive(fc),
            _ => false,
        }
    }
}

#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub enum Character<'gc> {
    EditText(EditText<'gc>),
    Graphic(Graphic<'gc>),
    MovieClip(MovieClip<'gc>),
    Bitmap(Gc<'gc, BitmapCharacter<'gc>>),
    Avm1Button(Avm1Button<'gc>),
    Avm2Button(Avm2Button<'gc>),
    Font(Font<'gc>),
    MorphShape(MorphShape<'gc>),
    Text(Text<'gc>),
    Sound(#[collect(require_static)] SoundHandle),
    Video(Video<'gc>),
    BinaryData(Gc<'gc, BinaryData>),
}

#[derive(Collect, Debug)]
#[collect(no_drop)]
pub struct BitmapCharacter<'gc> {
    #[collect(require_static)]
    compressed: CompressedBitmap,
    /// A lazily constructed GPU handle, used when performing fills with this bitmap
    #[collect(require_static)]
    handle: OnceCell<BitmapHandle>,
    /// The bitmap class set by `SymbolClass` - this is used when we instantaite
    /// a `Bitmap` displayobject.
    avm2_class: Lock<BitmapClass<'gc>>,

    // the decoded pixels that BitmapData made from this symbol share, alive while one of them still does (see "shared_pixels").
    #[collect(require_static)]
    shared_pixels: std::cell::RefCell<std::sync::Weak<crate::bitmap::bitmap_data::SharedPixels>>,
}

impl<'gc> BitmapCharacter<'gc> {
    pub fn new(compressed: CompressedBitmap) -> Self {
        Self {
            compressed,
            handle: OnceCell::default(),
            avm2_class: Lock::new(BitmapClass::NoSubclass),
            shared_pixels: Default::default(),
        }
    }

    // the symbol's decoded pixels, for a new BitmapData to share ("SharedPixels"). Decoded again only once no BitmapData holds them.
    pub fn shared_pixels(&self) -> Arc<crate::bitmap::bitmap_data::SharedPixels> {
        if let Some(shared) = self.shared_pixels.borrow().upgrade() {
            return shared;
        }
        // Never expect this: see `decode_or_stand_in`.
        let bitmap = self.compressed.decode_or_stand_in();
        let shared = crate::bitmap::bitmap_data::SharedPixels::new(
            bitmap.width(),
            bitmap.height(),
            bitmap
                .as_colors()
                .map(crate::bitmap::bitmap_data::Color::from)
                .collect(),
        );
        *self.shared_pixels.borrow_mut() = Arc::downgrade(&shared);
        shared
    }

    pub fn compressed(&self) -> &CompressedBitmap {
        &self.compressed
    }

    pub fn avm2_class(&self) -> BitmapClass<'gc> {
        self.avm2_class.get()
    }

    pub fn set_avm2_class(this: Gc<'gc, Self>, bitmap_class: BitmapClass<'gc>, mc: &Mutation<'gc>) {
        unlock!(Gc::write(mc, this), Self, avm2_class).set(bitmap_class);
    }

    pub fn bitmap_handle(
        &self,
        backend: &mut dyn RenderBackend,
    ) -> Result<BitmapHandle, RenderError> {
        // FIXME - use `OnceCell::get_or_try_init` when stabilized.
        if let Some(handle) = self.handle.get() {
            return Ok(handle.clone());
        }

        /*
            Refuse rather than OOM-abort the process.

            Decoded handles are cached here FOREVER (see the FIXME below, which is upstream's own),
            so a movie that streams through hundreds of unique full-size frames walks the heap to death.
            
            Sonic RPG Ep.10 does exactly that: ~2.77 MB per 1363x509 frame,
            dead around 210-250 MB accumulated (its own SWF is already 284 MB).
            
            Returning Err degrades gracefully by design,
            MovieLibrarySource::bitmap_handle logs it and yields None,
            so he shape draws without its fill instead of killing the app,
            which could cost the user unsaved progress in whatever they play next.
        */
        let sz = self.compressed.size();
        let want = sz.width as u64 * sz.height as u64 * 4;
        let large = want >= LARGE_BITMAP_BYTES;
        if bitmap_cache_would_exceed(want, large) {
            tracing::warn!(
                "bitmap decode refused ({}x{}, {}): cached-bitmap budget exhausted \
                 (total {} MB / {} MB, large {} MB / {} MB)",
                sz.width,
                sz.height,
                if large { "large" } else { "small" },
                CACHED_BITMAP_BYTES.load(std::sync::atomic::Ordering::Relaxed) / (1024 * 1024),
                CACHED_BITMAP_BUDGET_BYTES.load(std::sync::atomic::Ordering::Relaxed) / (1024 * 1024),
                CACHED_LARGE_BITMAP_BYTES.load(std::sync::atomic::Ordering::Relaxed) / (1024 * 1024),
                LARGE_BITMAP_BUDGET_BYTES.load(std::sync::atomic::Ordering::Relaxed) / (1024 * 1024),
            );
            return Err(RenderError::TooLarge);
        }
        let decoded = self.compressed.decode()?;
        CACHED_BITMAP_BYTES.fetch_add(want, std::sync::atomic::Ordering::Relaxed);
        if large {
            CACHED_LARGE_BITMAP_BYTES.fetch_add(want, std::sync::atomic::Ordering::Relaxed);
        }
        /*
            The image never changes and its bytes stay in the movie,
            so the renderer may drop its texture when nothing has drawn it for a while and decode it again on the next draw (see "flashnx_bitmap_residency").
        */
        let compressed = self.compressed.clone();
        let reload: std::rc::Rc<dyn Fn() -> Option<RenderBitmap<'static>>> =
            std::rc::Rc::new(move || compressed.decode().ok());
        let new_handle = backend.register_bitmap_reloadable(decoded, reload)?;
        // FIXME - do we ever want to release this handle, to avoid taking up GPU memory?
        self.handle.set(new_handle.clone()).unwrap();
        Ok(new_handle)
    }
}

/// Holds a bitmap from an SWF tag, plus the decoded width/height.
/// We avoid decompressing the image until it's actually needed - some pathological SWFS
/// like 'House' have thousands of highly-compressed (mostly empty) bitmaps, which can
/// take over 10GB of ram if we decompress them all during preloading.
#[derive(Clone, Debug)]
pub enum CompressedBitmap {
    Jpeg {
        data: BitmapBytes,
        /*
            the movie's JPEGTables, for a DefineBits (JPEG1) tag.
            They are glued in front of `data` at decode time: upstream glues them at preload and keeps the glued copy.
        */
        tables: Option<Arc<[u8]>>,
        alpha: Option<BitmapBytes>,
        width: u32,
        height: u32,
    },
    // The fields of "DefineBitsLossless", with the pixel data kept as "BitmapBytes" instead of an owned copy.
    Lossless {
        version: u8,
        id: swf::CharacterId,
        format: swf::BitmapFormat,
        width: u16,
        height: u16,
        data: BitmapBytes,
    },
}

/*
    The compressed bytes of a bitmap tag.
    
    Upstream copies every DefineBits* payload out of the movie at preload, while the movie itself (`SwfMovie`) stays alive as long as its clips do,
    so every image of a loaded SWF is held twice.
    Super Smash Flash 2 loads its stages and characters as ~50 SWFs whose bytes are 80-90 % images
    (DAT220: 5.0 of 6.2 MB; DAT252: 3.0 of 3.8 MB), about 110 MB of duplicates by the 4th Classic level,
    where the heap ran out. A tag that lies inside he movie data now just references it.
*/
#[derive(Clone, Debug)]
pub enum BitmapBytes {
    Owned(Vec<u8>),
    Movie(SwfSlice),
}

static BITMAP_BYTES_SHARED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BITMAP_BYTES_COPIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Bytes of bitmap tags that referenced their movie / that were copied, since boot ("census:" line).
pub fn bitmap_bytes_stats() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        BITMAP_BYTES_SHARED.load(Relaxed),
        BITMAP_BYTES_COPIED.load(Relaxed),
    )
}

impl BitmapBytes {
    /// References `data` inside `swf`'s movie when it lies there, else copies it.
    pub fn from_movie(swf: &SwfSlice, data: &[u8]) -> Self {
        use std::sync::atomic::Ordering::Relaxed;
        let slice = swf.to_unbounded_subslice(data);
        if !data.is_empty() && slice.end - slice.start == data.len() && slice.end <= slice.movie.data().len()
        {
            BITMAP_BYTES_SHARED.fetch_add(data.len() as u64, Relaxed);
            Self::Movie(slice)
        } else {
            BITMAP_BYTES_COPIED.fetch_add(data.len() as u64, Relaxed);
            Self::Owned(data.to_vec())
        }
    }
}

impl std::ops::Deref for BitmapBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Owned(data) => data,
            Self::Movie(slice) => slice.data(),
        }
    }
}

/*
    Bytes of DECODED bitmap held by CompressedBitmap::bitmap_handle's cache.

    That cache is never evicted (upstream FIXME in "bitmap_handle"),
    so this only ever grows within a session.
    It is NOT the same pool as "bitmap::bitmap_data::BITMAPDATA_LIVE_BYTES",
    which tracks AVM-created BitmapData surfaces and is properly freed on GC.
*/
static CACHED_BITMAP_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/*
    Ceiling for that cache, in bytes.
    Set per movie by ["reset_bitmap_cache"], a FIXED ceiling cannot serve both ends of the range:

    Sonic RPG Ep.10: SWF is 284 MB resident -> died around 213-253 MB of bitmaps
    Super Smash Flash 2: SWF is 2 MB (assets stream from SD) -> wants far more

    so the movie's own resident size is subtracted from a total allowance.
    A fixed 160 MB starved SSF2 (13121 refusals, sprites drawn white) while still being generous for Sonic.
*/
static CACHED_BITMAP_BUDGET_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(DEFAULT_BITMAP_ALLOWANCE);

/*
    Same thing,
    counting only the bitmaps big enough to need a large contiguous buffer to decode into,
    and the ceiling for those.

    One number could not serve both shapes of movie, because the two do not fail the same way.
    
    Sonic RPG Ep.10 dies on ONE allocation:
    1363x509 frames, 2.77 MB of contiguous heap each,
    and around 210-250 MB accumulated the next one cannot be satisfied even though plenty of total memory remains.
    
    The King of Fighters Wing (issue #102) never asks for anything contiguous:
    384x224 sprites, 340 KB each, 4k of them, and it wants some 800 MB in total.
    
    A single budget tuned for the first starves the second, 423 MB here,
    exhausted at frame 346, then 5137 refusals and a white screen with a few sprites on it.

    So the strict ceiling now applies to the large ones only, where it was measured, and the total gets the room the small ones need.
*/
static CACHED_LARGE_BITMAP_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LARGE_BITMAP_BUDGET_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(LARGE_BITMAP_ALLOWANCE);

// A decoded bitmap at least this big counts as "large". 1 MB sits an order of magnitude clear of both examples above (340 KB vs 2.77 MB).
const LARGE_BITMAP_BYTES: u64 = 1024 * 1024;

/*
    Total we are willing to see resident as movie + decoded bitmaps, all sizes.
    The Switch title heap is ~3.2 GB and the vector arenas already hold 576 MB of it, so this leaves room for the atlases those bitmaps also fill.
*/
const TOTAL_BITMAP_ALLOWANCE: u64 = 1100 * 1024 * 1024;

/*
    The same, for large bitmaps alone.
    Unchanged from the single budget this splits: it is the number Sonic RPG Ep.10 was measured against.
*/
const LARGE_BITMAP_ALLOWANCE: u64 = 450 * 1024 * 1024;

// Floor, so a monstrous movie still gets room to show something.
const MIN_BITMAP_BUDGET: u64 = 96 * 1024 * 1024;
const DEFAULT_BITMAP_ALLOWANCE: u64 = TOTAL_BITMAP_ALLOWANCE;

/*
    Past its budget,
    a movie at least this big may keep decoding while the heap can still hand out one ["HEAP_ROOM_PROBE_BYTES"] block.

    Subtracting the movie's size is a proxy for heap pressure,
    and for a huge movie it is the proxy, not the heap, that says no.
    
    New Super Smash Flash is a 397 MB movie:
    its large-bitmap budget sits on the 96 MB floor, 219 images (545 MB decoded) were refused,
    among them the stage backgrounds (5120x432, 1236x4000), whose shapes then draw white.
    
    Decoding them all peaked the heap at 2.5 GB of 3.1 in a test session.
    Everything smaller keeps the fixed budget untouched:
    Super Smash Flash 2 is a 1 MB movie that also fills its large budget,
    and enters a fight with about 98 MB of margin,
    so letting it decode more would cost exactly that margin.
*/
const HEAP_ROOM_MIN_MOVIE: u64 = 100 * 1024 * 1024;

/*
    Size of the block that must still be free.
    Sonic RPG Ep.10 (284 MB, also eligible) died on a 2.77 MB contiguous block: this gives up long before.
*/
const HEAP_ROOM_PROBE_BYTES: usize = 512 * 1024 * 1024;
static HEAP_ROOM_ELIGIBLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static HEAP_ROOM_GRANTED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/*
    Can newlib still hand out one block this big? Asked of "malloc" directly:
    FlashNX's Rust allocator answers a failure by emptying the GPU buffer cache and retrying, which a probe must not trigger.
*/
fn heap_has_room(bytes: usize) -> bool {
    use std::ffi::c_void;
    unsafe extern "C" {
        #[cfg(not(target_os = "horizon"))]
        fn malloc(size: usize) -> *mut c_void;
        #[cfg(target_os = "horizon")]
        fn memalign(align: usize, size: usize) -> *mut c_void;
        fn free(ptr: *mut c_void);
    }
    unsafe {
        /*
            On Switch, big blocks (decoded images among them) come from the top of the heap (FlashNX's cpp/src/flashnx_heap.c):
            ask there, by asking the way they are asked for.
        */
        #[cfg(target_os = "horizon")]
        let p = memalign(4096, bytes);
        #[cfg(not(target_os = "horizon"))]
        let p = malloc(bytes);
        if p.is_null() {
            return false;
        }
        free(p);
    }
    true
}

/*
    The renderer dropped ("resident == false") or rebuilt the texture of a decoded SWF image registered through "register_bitmap_reloadable".
    The budget above counts what is RESIDENT:
    without this,
    a game that cycles through more images than the budget over a session would end up refused (invisible sprites)
    although the evicted ones hold no memory any more.
*/
pub fn flashnx_bitmap_residency(bytes: u64, resident: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let large = bytes >= LARGE_BITMAP_BYTES;
    if resident {
        CACHED_BITMAP_BYTES.fetch_add(bytes, Relaxed);
        if large {
            CACHED_LARGE_BITMAP_BYTES.fetch_add(bytes, Relaxed);
        }
    } else {
        // Load then store: only the player thread touches these counters.
        let v = CACHED_BITMAP_BYTES.load(Relaxed);
        CACHED_BITMAP_BYTES.store(v.saturating_sub(bytes), Relaxed);
        if large {
            let v = CACHED_LARGE_BITMAP_BYTES.load(Relaxed);
            CACHED_LARGE_BITMAP_BYTES.store(v.saturating_sub(bytes), Relaxed);
        }
    }
}

/*
    Start a new movie:
    clear the accounting and size the budget from that movie's resident bytes.
    
    MUST be called per game, the counter is process-global,
    and leaving it set made a second launch of the same game refuse almost every bitmap from the first frame (observed: refusals 54 log lines in).
*/
pub fn reset_bitmap_cache(movie_bytes: u64) {
    CACHED_BITMAP_BYTES.store(0, std::sync::atomic::Ordering::Relaxed);
    CACHED_LARGE_BITMAP_BYTES.store(0, std::sync::atomic::Ordering::Relaxed);

    // The movie's own resident bytes come off both ceilings: it is in the same heap either way, and a 284 MB movie has already spent that much of it.
    let budget = TOTAL_BITMAP_ALLOWANCE.saturating_sub(movie_bytes).max(MIN_BITMAP_BUDGET);
    let large_budget = LARGE_BITMAP_ALLOWANCE.saturating_sub(movie_bytes).max(MIN_BITMAP_BUDGET).min(budget);
    CACHED_BITMAP_BUDGET_BYTES.store(budget, std::sync::atomic::Ordering::Relaxed);
    LARGE_BITMAP_BUDGET_BYTES.store(large_budget, std::sync::atomic::Ordering::Relaxed);
    let eligible = movie_bytes >= HEAP_ROOM_MIN_MOVIE;
    HEAP_ROOM_ELIGIBLE.store(eligible, std::sync::atomic::Ordering::Relaxed);
    HEAP_ROOM_GRANTED.store(0, std::sync::atomic::Ordering::Relaxed);
    tracing::info!("bitmap cache: budget {} MB total, {} MB for large bitmaps (movie {} MB){}", budget / (1024 * 1024), large_budget / (1024 * 1024), movie_bytes / (1024 * 1024), if eligible { ", past it while the heap has a 512 MB block" } else { "" }, );
}

fn bitmap_cache_would_exceed(want: u64, large: bool) -> bool {
    let total = CACHED_BITMAP_BYTES.load(std::sync::atomic::Ordering::Relaxed).saturating_add(want) > CACHED_BITMAP_BUDGET_BYTES.load(std::sync::atomic::Ordering::Relaxed);
    let over = total || (large && CACHED_LARGE_BITMAP_BYTES.load(std::sync::atomic::Ordering::Relaxed).saturating_add(want) > LARGE_BITMAP_BUDGET_BYTES.load(std::sync::atomic::Ordering::Relaxed));
    if !over || !HEAP_ROOM_ELIGIBLE.load(std::sync::atomic::Ordering::Relaxed) || !heap_has_room(HEAP_ROOM_PROBE_BYTES)
    {
        return over;
    }
    let n = HEAP_ROOM_GRANTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n == 1 || n % 50 == 0 {
        tracing::info!("bitmap cache: over budget but the heap still has a 512 MB block, decoding anyway \
             ({} so far, total {} MB, large {} MB)", n, CACHED_BITMAP_BYTES.load(std::sync::atomic::Ordering::Relaxed) / (1024 * 1024), CACHED_LARGE_BITMAP_BYTES.load(std::sync::atomic::Ordering::Relaxed) / (1024 * 1024), );
    }
    false
}

impl CompressedBitmap {
    pub fn size(&self) -> BitmapSize {
        match self {
            CompressedBitmap::Jpeg { width, height, .. } => BitmapSize {
                width: *width,
                height: *height,
            },
            CompressedBitmap::Lossless { width, height, .. } => BitmapSize {
                width: (*width).into(),
                height: (*height).into(),
            },
        }
    }

    /*
        "decode" on a heap that has nothing left.

        Decoding allocates width × height × 4 bytes,
        so it is one of the first things to fail once memory runs out,
        and three upstream call sites "unwrap" it,
        which turns "one sprite could not be decoded" into a dead process.
        
        Measured on Super Smash Flash 2:
        entering a fight panics at "library.rs:258" with "InvalidZlibCompression(Kind(OutOfMemory))"
        while the heap sits around 2.8 GB.
        A missing sprite is a far better outcome than losing the session.

        The stand-in is 1×1 transparent RGBA on purpose.
        Returning "None" from "Library::instantiate_display_object" is NOT an option:
        "clone_sprite" in "avm1/globals/movie_clip.rs" unwraps that "Option",
        so the panic would simply move to "duplicateMovieClip".

        Logging is capped: the caller is out of memory, and formatting a message allocates.
    */
    pub fn decode_or_stand_in(&self) -> RenderBitmap<'static> {
        match self.decode() {
            Ok(b) => b,
            Err(_) => {
                use std::sync::atomic::{AtomicU32, Ordering};
                static LOGGED: AtomicU32 = AtomicU32::new(0);
                let n = LOGGED.fetch_add(1, Ordering::Relaxed);
                if n < 8 {
                    let size = self.size();
                    tracing::warn!(
                        "bitmap decode failed ({}x{}), substituting a blank — out of memory",
                        size.width,
                        size.height
                    );
                }
                RenderBitmap::new(1, 1, ruffle_render::bitmap::BitmapFormat::Rgba, std::vec![0u8; 4], )
            }
        }
    }

    pub fn decode(&self) -> Result<RenderBitmap<'static>, RenderError> {
        match self {
            CompressedBitmap::Jpeg {
                data,
                tables,
                alpha,
                width: _,
                height: _,
            } => {
                let data = ruffle_render::utils::glue_tables_to_jpeg(data, tables.as_deref());
                ruffle_render::utils::decode_define_bits_jpeg(&data, alpha.as_deref())
            }
            CompressedBitmap::Lossless {
                version,
                id,
                format,
                width,
                height,
                data,
            } => ruffle_render::utils::decode_define_bits_lossless(&DefineBitsLossless {
                version: *version,
                id: *id,
                format: *format,
                width: *width,
                height: *height,
                data: Cow::Borrowed(data),
            }),
        }
    }
}
