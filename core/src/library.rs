use crate::avm1::{PropertyMap as Avm1PropertyMap, PropertyMap};
use crate::avm2::{Class as Avm2Class, Domain as Avm2Domain};
use crate::backend::audio::SoundHandle;
use crate::character::Character;

use crate::display_object::{Bitmap, Graphic, MorphShape, MovieClip, MovieClipWeak, Text};
use crate::font::{Font, FontDescriptor, FontLike, FontQuery, FontType};
use crate::prelude::*;
use crate::string::AvmString;
use crate::tag_utils::SwfMovie;
use gc_arena::collect::Trace;
use gc_arena::{Collect, Finalization, Gc, GcWeak, Mutation};
use ruffle_render::backend::RenderBackend;
use ruffle_render::bitmap::BitmapHandle;
use ruffle_render::utils::remove_invalid_jpeg_data;

use crate::backend::ui::{FontDefinition, UiBackend};
use crate::font::DefaultFont;
use fnv::{FnvHashMap, FnvHashSet};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::{Arc, Weak};
use weak_table::{PtrWeakKeyHashMap, WeakValueHashMap, traits::WeakElement};

#[derive(Clone)]
struct MovieSymbol(Arc<SwfMovie>, CharacterId);

#[derive(Clone)]
struct WeakMovieSymbol(Weak<SwfMovie>, CharacterId);

impl WeakElement for WeakMovieSymbol {
    type Strong = MovieSymbol;

    fn new(view: &Self::Strong) -> Self {
        Self(Arc::downgrade(&view.0), view.1)
    }

    fn view(&self) -> Option<Self::Strong> {
        if let Some(strong) = self.0.upgrade() {
            Some(MovieSymbol(strong, self.1))
        } else {
            None
        }
    }
}

/// The mappings between class objects and library characters defined by
/// `SymbolClass`.
pub struct Avm2ClassRegistry<'gc> {
    /// A list of AVM2 class objects and the character IDs they are expected to
    /// instantiate.
    class_map: WeakValueHashMap<Avm2Class<'gc>, WeakMovieSymbol>,

    /*
        The movies whose library was released ("Library::release_movie"), by address.
        
        Their classes are not traced from here:
        a class holds its movie through its code,
        so tracing them would keep the movie in use for good.

        "Library::finalize_released" keeps or drops them.
    */
    released: FnvHashSet<usize>,
}

unsafe impl<'gc> Collect<'gc> for Avm2ClassRegistry<'gc> {
    fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
        for (k, symbol) in self.class_map.iter() {
            if self.released.is_empty() || !self.released.contains(&(Arc::as_ptr(&symbol.0) as usize))
            {
                cc.trace(k);
            }
        }
    }
}

impl Default for Avm2ClassRegistry<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'gc> Avm2ClassRegistry<'gc> {
    pub fn new() -> Self {
        Self {
            class_map: WeakValueHashMap::new(),
            released: FnvHashSet::default(),
        }
    }

    // Keeps the classes of "movie", released but still in use, alive for the current collection (see "Library::finalize_released").
    fn resurrect_movie(&self, fc: &Finalization<'gc>, movie: &Arc<SwfMovie>) {
        for (class, symbol) in self.class_map.iter() {
            if Arc::ptr_eq(&symbol.0, movie) {
                class.resurrect(fc);
            }
        }
    }

    /*
        Forgets the classes of a released movie whose library is dropped.
        
        They are about to be freed: none may stay in the map.
        
        By key: weak_table's "retain" can pass over the entry that its backward-shift deletion moves into a slot it has already visited
        (found by the same work upstream, see "Library::release_movie").
    */
    fn remove_movie(&mut self, movie: &Arc<SwfMovie>) {
        let classes: Vec<Avm2Class<'gc>> = self.class_map.iter().filter(|(_, symbol)| Arc::ptr_eq(&symbol.0, movie)).map(|(class, _)| *class).collect();
        for class in classes {
            self.class_map.remove(&class);
        }
        self.released.remove(&(Arc::as_ptr(movie) as usize));
    }

    /// Retrieve the library symbol for a given AVM2 class object.
    ///
    /// A value of `None` indicates that this AVM2 class is not associated with
    /// a library symbol.
    pub fn class_symbol(&self, class_def: Avm2Class<'gc>) -> Option<(Arc<SwfMovie>, CharacterId)> {
        match self.class_map.get(&class_def) {
            Some(MovieSymbol(movie, symbol)) => Some((movie, symbol)),
            None => None,
        }
    }

    /// Associate an AVM2 class definition with a given library symbol.
    pub fn set_class_symbol(
        &mut self,
        class_def: Avm2Class<'gc>,
        movie: Arc<SwfMovie>,
        symbol: CharacterId,
    ) {
        if let Some(old) = self.class_map.get(&class_def) {
            if Arc::ptr_eq(&movie, &old.0) && symbol != old.1 {
                // Flash player actually allows using the same class in multiple SymbolClass
                // entries in the same swf, with *different* symbol ids. Whichever one
                // is processed first will *win*, and the second one will be ignored.
                // We still log a warning, since we wouldn't expect this to happen outside
                // of deliberately crafted SWFs.
                tracing::warn!(
                    "Tried to overwrite class {:?} id={:?} with symbol id={:?} from same movie",
                    class_def,
                    old.1,
                    symbol,
                );
            }
            // If we're trying to overwrite the class with a symbol from a *different* SwfMovie,
            // then just ignore it. This handles the case where a Loader has a class that shadows
            // a class in the main swf (possibly with a different ApplicationDomain). This will
            // result in the original class from the parent being used, even when the child swf
            // instantiates the clip on the timeline.
            return;
        }
        self.class_map.insert(class_def, MovieSymbol(movie, symbol));
    }
}

/// Symbol library for a single given SWF.
#[derive(Collect)]
#[collect(no_drop)]
pub struct MovieLibrary<'gc> {
    swf: Arc<SwfMovie>,
    characters: HashMap<CharacterId, Character<'gc>>,
    export_characters: Avm1PropertyMap<'gc, CharacterId>,
    imported_assets: HashMap<AvmString<'gc>, CharacterId>,

    /*
        shared, so a DefineBits bitmap can keep a reference to the tables and glue them in when it is decoded,
        instead of storing its own glued copy of the image for the life of the movie.
    */
    jpeg_tables: Option<Arc<[u8]>>,

    fonts: FontMap<'gc>,
    avm2_domain: Option<Avm2Domain<'gc>>,
    // set once the Loader that loaded this movie unloaded it ("Library::release_movie"`), with the root clip it had then.
    released: bool,
    released_root: Option<MovieClipWeak<'gc>>,
    // resurrected in the current collection ("finalize_released").
    #[collect(require_static)]
    kept: std::cell::Cell<bool>,
}

impl<'gc> MovieLibrary<'gc> {
    pub fn new(swf: Arc<SwfMovie>) -> Self {
        Self {
            swf,
            characters: HashMap::new(),
            imported_assets: HashMap::new(),
            export_characters: Avm1PropertyMap::new(),
            jpeg_tables: None,
            fonts: Default::default(),
            avm2_domain: None,
            released: false,
            released_root: None,
            kept: std::cell::Cell::new(false),
        }
    }

    /*
        Whether anything outside this released (so untraced) library still uses its movie,
        in a collection about to sweep:
        its AVM2 domain (every live object, class or function of the movie's code reaches it),
        its root clip, or an instance of one of its symbols.
    */
    fn in_use(&self, fc: &Finalization<'gc>) -> bool {
        self.avm2_domain.is_some_and(|domain| !domain.is_dead(fc)) || self.released_root.is_some_and(|root| !root.is_dead(fc)) || self.characters.values().any(|character| character.has_live_instances(fc))
    }

    /// Registers a character; returns `true` if successful, or `false` if a character with
    /// the given ID already exists.
    pub fn register_character(&mut self, id: CharacterId, character: Character<'gc>) -> bool {
        use std::collections::hash_map::Entry;
        match self.characters.entry(id) {
            Entry::Vacant(e) => {
                if let Character::Font(font) = character {
                    self.fonts.register(font);
                }
                e.insert(character);
                true
            }
            Entry::Occupied(_) => {
                tracing::error!("Character ID collision: Tried to register ID {} twice", id);
                false
            }
        }
    }

    /// Registers an export name for a given character ID.
    /// This character will then be instantiable from AVM1.
    pub fn register_export(&mut self, id: CharacterId, export_name: AvmString<'gc>) {
        let character_exists = self.contains_character(id);
        debug_assert!(character_exists);
        if !character_exists {
            tracing::error!(
                "Tried to register export '{export_name}' for a non-existent character {id}"
            );
            return;
        }

        self.export_characters.insert(export_name, id, false);
    }

    pub fn characters(&self) -> &HashMap<CharacterId, Character<'gc>> {
        &self.characters
    }

    pub fn export_characters(&self) -> &PropertyMap<'gc, CharacterId> {
        &self.export_characters
    }

    pub fn contains_character(&self, id: CharacterId) -> bool {
        self.characters.contains_key(&id)
    }

    pub fn character_by_id(&self, id: CharacterId) -> Option<Character<'gc>> {
        self.characters.get(&id).copied()
    }

    pub fn character_by_export_name(
        &self,
        name: AvmString<'gc>,
    ) -> Option<(CharacterId, Character<'gc>)> {
        if let Some(id) = self.export_characters.get(name, false)
            && let Some(character) = self.characters.get(id)
        {
            return Some((*id, *character));
        }
        None
    }

    pub fn character_id_by_import_name(&self, name: AvmString<'gc>) -> Option<CharacterId> {
        self.imported_assets.get(&name).copied()
    }

    pub fn register_import(&mut self, name: AvmString<'gc>, id: CharacterId) {
        self.imported_assets.insert(name, id);
    }

    /// Instantiates the library item with the given character ID into a display object.
    /// The object must then be post-instantiated before being used.
    pub fn instantiate_by_id(
        &self,
        id: CharacterId,
        mc: &Mutation<'gc>,
    ) -> Option<DisplayObject<'gc>> {
        if let Some(&character) = self.characters.get(&id) {
            self.instantiate_display_object(id, character, mc)
        } else {
            tracing::error!("Tried to instantiate non-registered character ID {}", id);
            None
        }
    }

    /// Instantiates the library item with the given export name into a display object.
    /// The object must then be post-instantiated before being used.
    pub fn instantiate_by_export_name(
        &self,
        export_name: AvmString<'gc>,
        mc: &Mutation<'gc>,
    ) -> Option<DisplayObject<'gc>> {
        if let Some((id, character)) = self.character_by_export_name(export_name) {
            self.instantiate_display_object(id, character, mc)
        } else {
            tracing::error!(
                "Tried to instantiate non-registered character {}",
                export_name
            );
            None
        }
    }

    /// Instantiates the given character into a display object.
    /// The object must then be post-instantiated before being used.
    fn instantiate_display_object(
        &self,
        id: CharacterId,
        character: Character<'gc>,
        mc: &Mutation<'gc>,
    ) -> Option<DisplayObject<'gc>> {
        match character {
            Character::Bitmap(bitmap) => {
                let avm2_class = bitmap.avm2_class();
                // Never unwrap this (see "decode_or_stand_in").
                let bitmap = bitmap.compressed().decode_or_stand_in();
                let bitmap = Bitmap::new(mc, id, bitmap, self.swf.clone());
                bitmap.set_avm2_bitmapdata_class(mc, avm2_class);
                Some(bitmap.instantiate(mc))
            }
            Character::EditText(edit_text) => Some(edit_text.instantiate(mc)),
            Character::Graphic(graphic) => Some(graphic.instantiate(mc)),
            Character::MorphShape(morph_shape) => Some(morph_shape.instantiate(mc)),
            Character::MovieClip(movie_clip) => Some(movie_clip.instantiate(mc)),
            Character::Avm1Button(button) => Some(button.instantiate(mc)),
            Character::Avm2Button(button) => Some(button.instantiate(mc)),
            Character::Text(text) => Some(text.instantiate(mc)),
            Character::Video(video) => Some(video.instantiate(mc)),
            _ => {
                // Cannot instantiate non-display object
                None
            }
        }
    }

    pub fn get_font(&self, id: CharacterId) -> Option<Font<'gc>> {
        if let Some(&Character::Font(font)) = self.characters.get(&id) {
            Some(font)
        } else {
            None
        }
    }

    pub fn embedded_fonts(&self) -> Vec<Font<'gc>> {
        self.fonts.all()
    }

    /// Returns the `Graphic` with the given character ID.
    /// Returns `None` if the ID does not exist or is not a `Graphic`.
    pub fn get_graphic(&self, id: CharacterId) -> Option<Graphic<'gc>> {
        if let Some(&Character::Graphic(graphic)) = self.characters.get(&id) {
            Some(graphic)
        } else {
            None
        }
    }

    /// Returns the `MorphShape` with the given character ID.
    /// Returns `None` if the ID does not exist or is not a `MorphShape`.
    pub fn get_morph_shape(&self, id: CharacterId) -> Option<MorphShape<'gc>> {
        if let Some(&Character::MorphShape(morph_shape)) = self.characters.get(&id) {
            Some(morph_shape)
        } else {
            None
        }
    }

    pub fn get_sound(&self, id: CharacterId) -> Option<SoundHandle> {
        if let Some(Character::Sound(sound)) = self.characters.get(&id) {
            Some(*sound)
        } else {
            None
        }
    }

    /// Returns the `Text` with the given character ID.
    /// Returns `None` if the ID does not exist or is not a `Text`.
    pub fn get_text(&self, id: CharacterId) -> Option<Text<'gc>> {
        if let Some(&Character::Text(text)) = self.characters.get(&id) {
            Some(text)
        } else {
            None
        }
    }

    pub fn set_jpeg_tables(&mut self, data: &[u8]) {
        if self.jpeg_tables.is_some() {
            // SWF spec says there should only be one JPEGTables tag.
            // TODO: What is the behavior when there are multiples?
            tracing::warn!("SWF contains multiple JPEGTables tags");
            return;
        }
        // Some SWFs have a JPEGTables tag with 0 length; ignore these.
        // (Does this happen when there is only a single DefineBits tag?)
        self.jpeg_tables = if data.is_empty() {
            None
        } else {
            Some(Arc::from(remove_invalid_jpeg_data(data).as_ref()))
        }
    }

    pub fn jpeg_tables(&self) -> Option<&[u8]> {
        self.jpeg_tables.as_deref()
    }

    // The same tables, as a handle a bitmap can keep.
    pub fn jpeg_tables_shared(&self) -> Option<Arc<[u8]>> {
        self.jpeg_tables.clone()
    }

    pub fn set_avm2_domain(&mut self, avm2_domain: Avm2Domain<'gc>) {
        self.avm2_domain = Some(avm2_domain);
    }

    /// Get the AVM2 domain this movie runs under.
    ///
    /// Note that the presence of an AVM2 domain does *not* indicate that this
    /// movie provides AVM2 code. For example, a movie may have been loaded by
    /// AVM2 code into a particular domain, even though it turned out to be
    /// an AVM1 movie, and thus this domain is unused.
    pub fn avm2_domain(&self) -> Avm2Domain<'gc> {
        self.avm2_domain.unwrap()
    }

    pub fn try_avm2_domain(&self) -> Option<Avm2Domain<'gc>> {
        self.avm2_domain
    }
}

pub struct MovieLibrarySource<'a, 'gc> {
    pub library: &'a MovieLibrary<'gc>,
}

impl ruffle_render::bitmap::BitmapSource for MovieLibrarySource<'_, '_> {
    fn bitmap_size(&self, id: u16) -> Option<ruffle_render::bitmap::BitmapSize> {
        if let Some(Character::Bitmap(bitmap)) = self.library.characters.get(&id) {
            Some(bitmap.compressed().size())
        } else {
            None
        }
    }

    fn bitmap_handle(&self, id: u16, backend: &mut dyn RenderBackend) -> Option<BitmapHandle> {
        let Some(Character::Bitmap(bitmap)) = self.library.characters.get(&id) else {
            return None;
        };

        match bitmap.bitmap_handle(backend) {
            Ok(handle) => Some(handle),
            Err(e) => {
                tracing::error!("Failed to register bitmap character {id}: {e}");
                None
            }
        }
    }
}

struct MovieLibraries<'gc>(PtrWeakKeyHashMap<Weak<SwfMovie>, MovieLibrary<'gc>>);

unsafe impl<'gc> Collect<'gc> for MovieLibraries<'gc> {
    #[inline]
    fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
        for (_, val) in self.0.iter() {
            if val.released {
                /*
                    A released library holds its contents only while something else uses its movie:
                    "Library::finalize_released" resurrects or drops it before every sweep.
                    Its weak pointers are traced as such, so they stay valid.
                */
                val.trace(&mut WeakOnly(&mut *cc));
            } else {
                cc.trace(val);
            }
        }
    }
}

// Passes on only the weak pointers (see "MovieLibraries::trace").
struct WeakOnly<'a, C>(&'a mut C);

impl<'gc, C: Trace<'gc>> Trace<'gc> for WeakOnly<'_, C> {
    fn trace_gc(&mut self, _gc: Gc<'gc, ()>) {}

    fn trace_gc_weak(&mut self, gc: GcWeak<'gc, ()>) {
        self.0.trace_gc_weak(gc)
    }
}

/*
    Keeps every pointer it is shown alive for the current collection (see "Library::finalize_released"), and through them all they hold.
    Weak pointers need nothing: they were traced as such.
*/
struct Resurrect<'a, 'gc>(&'a Finalization<'gc>);

impl<'gc> Trace<'gc> for Resurrect<'_, 'gc> {
    fn trace_gc(&mut self, gc: Gc<'gc, ()>) {
        Gc::resurrect(self.0, gc)
    }

    fn trace_gc_weak(&mut self, _gc: GcWeak<'gc, ()>) {}
}

// "moviefree.off" marker (see "Library::release_movie").
static MOVIE_FREE: AtomicBool = AtomicBool::new(true);
static MOVIES_RELEASED: AtomicU32 = AtomicU32::new(0);
static MOVIES_DROPPED: AtomicU32 = AtomicU32::new(0);
static MOVIES_KEPT: AtomicU32 = AtomicU32::new(0);

// whether the libraries of unloaded movies may be dropped.
pub fn set_movie_free(enabled: bool) {
    MOVIE_FREE.store(enabled, Relaxed);
}

/*
    movies released since boot, libraries dropped since boot,
    and released ones still in use at the last collection ("census:" line).
*/
pub fn movie_free_stats() -> (u32, u32, u32) {
    (
        MOVIES_RELEASED.load(Relaxed),
        MOVIES_DROPPED.load(Relaxed),
        MOVIES_KEPT.load(Relaxed),
    )
}

impl<'gc> MovieLibraries<'gc> {
    fn new() -> Self {
        Self(PtrWeakKeyHashMap::new())
    }

    fn get(&self, key: &Arc<SwfMovie>) -> Option<&MovieLibrary<'gc>> {
        self.0.get(key)
    }

    fn get_or_insert_mut(&mut self, movie: Arc<SwfMovie>) -> &mut MovieLibrary<'gc> {
        self.0
            .entry(movie.clone())
            .or_insert_with(|| MovieLibrary::new(movie))
    }

    fn known_movies(&self) -> impl Iterator<Item = Arc<SwfMovie>> {
        self.0.keys()
    }
}

/// Symbol library for multiple movies.
#[derive(Collect)]
#[collect(no_drop)]
pub struct Library<'gc> {
    /// All the movie libraries.
    movie_libraries: MovieLibraries<'gc>,

    /// A cache of seen device fonts.
    // TODO: Descriptors shouldn't be stored in fonts. Fonts should be a list that we iterate and ask "do you match". A font can have zero or many names.
    device_fonts: FontMap<'gc>,

    /// "Global" embedded fonts, registered through AVM2 `Font.registerFont`.
    /// These should be checked before any Movie-specific library's own fonts.
    global_fonts: FontMap<'gc>,

    /// A set of which fonts we've asked from the backend already, to help with negative caching.
    /// If we've asked for a specific font, record it here and don't ask again.
    font_lookup_cache: FnvHashSet<FontQuery>,

    /// Cached font sort queries.
    font_sort_cache: FnvHashMap<FontQuery, Vec<Font<'gc>>>,

    /// The implementation names of each default font.
    default_font_names: FnvHashMap<DefaultFont, Vec<String>>,

    /// The cached list of implementations per default font.
    default_font_cache: FnvHashMap<(DefaultFont, bool, bool), Vec<Font<'gc>>>,

    /// A list of the symbols associated with specific AVM2 constructor
    /// prototypes.
    avm2_class_registry: Avm2ClassRegistry<'gc>,
}

impl<'gc> Library<'gc> {
    pub fn empty() -> Self {
        Self {
            movie_libraries: MovieLibraries::new(),
            device_fonts: Default::default(),
            global_fonts: Default::default(),
            font_lookup_cache: Default::default(),
            font_sort_cache: Default::default(),
            default_font_names: Default::default(),
            default_font_cache: Default::default(),
            avm2_class_registry: Default::default(),
        }
    }

    pub fn library_for_movie(&self, movie: Arc<SwfMovie>) -> Option<&MovieLibrary<'gc>> {
        self.movie_libraries.get(&movie)
    }

    pub fn library_for_movie_mut(&mut self, movie: Arc<SwfMovie>) -> &mut MovieLibrary<'gc> {
        self.movie_libraries.get_or_insert_mut(movie)
    }

    pub fn known_movies(&self) -> impl Iterator<Item = Arc<SwfMovie>> {
        self.movie_libraries.known_movies()
    }

    /*
        Called when a Loader unloads "movie" ("LoaderInfoObject::unload").

        Upstream never frees a loaded movie:
        its library holds it, and the libraries are held for good.
        Super Smash Flash 2 unloads its oldest characters and stages past 12 and 10 ("Resource.unload", "unloadAndStop")
        and loads them again when needed:
        
        each load was a new movie kept forever, until the heap ran out in a second Classic run.
        
        A released library is no longer traced:
        each collection keeps it whole while anything else uses the movie,
        and drops it otherwise ("finalize_released"),
        after which the collector frees all that only it held.
        
        Upstream's work on the same leak: ruffle PRs #22908 and #23071, open.
    */
    pub fn release_movie(&mut self, movie: &Arc<SwfMovie>, root: Option<MovieClip<'gc>>) {
        if !MOVIE_FREE.load(Relaxed) {
            return;
        }
        let Some(library) = self.movie_libraries.0.get_mut(movie) else {
            return;
        };
        if library.released {
            return;
        }
        library.released = true;
        library.released_root = root.map(|root| root.downgrade());
        self.avm2_class_registry.released.insert(Arc::as_ptr(movie) as usize);
        MOVIES_RELEASED.fetch_add(1, Relaxed);
    }

    // Whether a collection has released libraries to finalize.
    pub fn has_released(&self) -> bool {
        !self.avm2_class_registry.released.is_empty()
    }

    /*
        1 finalization pass, run with the arena fully marked:
        the passes run to their end before EVERY sweep ("finalize_released_libraries" in player.rs):
        a released library is not traced, so one left in place without being resurrected would point at freed objects.

        Resurrects, whole, each released library not yet kept in this collection whose movie is in use ("MovieLibrary::in_use"),
        and returns true if it did:
        
        what it holds may reach another released library's anchors (a kept movie holding a class of another),
        which "is_dead" only shows once marking has run again, so another pass must follow.
        
        A pass that resurrects nothing is the last:
        it drops the released libraries not kept, with their symbol classes,
        and returns their sounds for the audio backend to forget.
    */
    pub fn finalize_released(&mut self, fc: &Finalization<'gc>) -> (bool, Vec<SoundHandle>) {
        let mut resurrected = false;
        for (movie, library) in self.movie_libraries.0.iter() {
            if library.released && !library.kept.get() && library.in_use(fc) {
                library.trace(&mut Resurrect(fc));
                self.avm2_class_registry.resurrect_movie(fc, &movie);
                library.kept.set(true);
                resurrected = true;
            }
        }
        if resurrected {
            return (true, Vec::new());
        }
        let mut dead = Vec::new();
        let mut kept = 0;
        for (movie, library) in self.movie_libraries.0.iter() {
            if !library.released {
                continue;
            }
            if library.kept.replace(false) {
                kept += 1;
            } else {
                dead.push(movie);
            }
        }
        MOVIES_KEPT.store(kept, Relaxed);
        let mut sounds = Vec::new();
        for movie in dead {
            self.avm2_class_registry.remove_movie(&movie);
            if let Some(library) = self.movie_libraries.0.remove(&movie) {
                sounds.extend(library.characters.values().filter_map(|c| match c {
                    Character::Sound(sound) => Some(*sound),
                    _ => None,
                }));
                tracing::info!(
                    "moviefree: dropped the library of an unloaded movie ({} KB, {} characters, {} still released)",
                    movie.data().len() / 1024,
                    library.characters.len(),
                    kept,
                );
            }
            MOVIES_DROPPED.fetch_add(1, Relaxed);
        }
        (false, sounds)
    }

    /// Returns the default Font implementations behind the built in names (ie `_sans`)
    pub fn default_font(
        &mut self,
        name: DefaultFont,
        is_bold: bool,
        is_italic: bool,
        ui: &dyn UiBackend,
        renderer: &mut dyn RenderBackend,
        gc_context: &Mutation<'gc>,
    ) -> Vec<Font<'gc>> {
        // Can't use entry api here as we want to use self for `load_device_font`.
        // Cache the value as this will be looked up a lot, and font lookup by name can be expensive if lots of fonts exist.
        if let Some(cache) = self.default_font_cache.get(&(name, is_bold, is_italic)) {
            return cache.clone();
        }

        let mut result = vec![];
        // First try to find any exactly matching fonts.
        for name in self.default_font_names.entry(name).or_default().clone() {
            let query = FontQuery::new(FontType::Device, name, is_bold, is_italic);
            if let Some(font) = self.get_or_load_exact_device_font(&query, ui, renderer, gc_context)
            {
                result.push(font);
                break; // TODO: Return multiple fonts when it's needed.
            }
        }

        // Nothing found, try a compatible font.
        if result.is_empty() {
            for name in self.default_font_names.entry(name).or_default().clone() {
                let query = FontQuery::new(FontType::Device, name, is_bold, is_italic);
                if let Some(font) = self.device_fonts.find(&query) {
                    result.push(font);
                    break; // TODO: Return multiple fonts when it's needed.
                }
            }
        }

        self.default_font_cache
            .insert((name, is_bold, is_italic), result.clone());
        result
    }

    /// Returns the device font exactly matching the requested options.
    fn get_or_load_exact_device_font(
        &mut self,
        query: &FontQuery,
        ui: &dyn UiBackend,
        renderer: &mut dyn RenderBackend,
        gc_context: &Mutation<'gc>,
    ) -> Option<Font<'gc>> {
        // If we have the exact matching font already, use that
        // TODO: We should instead ask each font if it matches a given name. Partial matches are allowed, and fonts may have any amount of names.
        if let Some(font) = self.device_fonts.get(query) {
            return Some(*font);
        }

        // We don't have this font already. Did we ask for it before?
        let new_request = self.font_lookup_cache.insert(query.clone());
        if new_request {
            // First time asking for this font, see if our backend can provide anything relevant
            ui.load_device_font(query, &mut |definition| {
                self.register_device_font(gc_context, renderer, definition)
            });

            // Check again. A backend may or may not have provided some new fonts,
            // and they may or may not be relevant to the one we're asking for.
            if let Some(font) = self.device_fonts.get(query) {
                return Some(*font);
            }

            let name = &query.name;
            let is_bold = query.is_bold;
            let is_italic = query.is_italic;
            warn!("Unknown device font \"{name}\" (bold: {is_bold}, italic: {is_italic})");
        }

        None
    }

    /// Returns the device font compatible with the requested options.
    pub fn get_or_load_device_font(
        &mut self,
        name: &str,
        is_bold: bool,
        is_italic: bool,
        ui: &dyn UiBackend,
        renderer: &mut dyn RenderBackend,
        gc_context: &Mutation<'gc>,
    ) -> Option<Font<'gc>> {
        let query = FontQuery::new(FontType::Device, name.to_owned(), is_bold, is_italic);

        // Try to find an exactly matching font.
        if let Some(font) = self.get_or_load_exact_device_font(&query, ui, renderer, gc_context) {
            return Some(font);
        }

        // Fallback: Try to find an existing font to re-use instead of giving up.
        self.device_fonts.find(&query)
    }

    fn sort_device_fonts(
        &mut self,
        query: &FontQuery,
        ui: &dyn UiBackend,
        renderer: &mut dyn RenderBackend,
        gc_context: &Mutation<'gc>,
    ) -> Vec<Font<'gc>> {
        // First, ask the backend to sort the fonts for us.
        let fonts = ui.sort_device_fonts(query, &mut |definition| {
            self.register_device_font(gc_context, renderer, definition)
        });

        let fonts: Vec<Font<'gc>> = fonts
            .iter()
            .filter_map(|font_query| self.device_fonts.get(font_query))
            .copied()
            .collect();

        if !fonts.is_empty() {
            return fonts;
        }

        // When the backend failed (or doesn't support sorting fonts), fall back
        // to loading one font only without sorting.
        let font = self.get_or_load_device_font(
            &query.name,
            query.is_bold,
            query.is_italic,
            ui,
            renderer,
            gc_context,
        );
        font.map(|font| vec![font]).unwrap_or_default()
    }

    pub fn get_or_sort_device_fonts(
        &mut self,
        name: &str,
        is_bold: bool,
        is_italic: bool,
        ui: &dyn UiBackend,
        renderer: &mut dyn RenderBackend,
        gc_context: &Mutation<'gc>,
    ) -> Vec<Font<'gc>> {
        // TODO We should be able to return a &Vec here, but (1) the borrow
        //   checker is too strict and doesn't allow if branching, and
        //   (2) there's no way to insert a value and get a reference to
        //   it at the same time.
        let query = FontQuery::new(FontType::Device, name.to_owned(), is_bold, is_italic);
        if let Some(fonts) = self.font_sort_cache.get(&query) {
            return fonts.clone();
        }

        let fonts = self.sort_device_fonts(&query, ui, renderer, gc_context);
        self.font_sort_cache.insert(query, fonts.clone());
        fonts
    }

    pub fn set_default_font(&mut self, font: DefaultFont, names: Vec<String>) {
        self.default_font_names.insert(font, names);
        self.default_font_cache.clear();
    }

    pub fn register_device_font(
        &mut self,
        gc_context: &Mutation<'gc>,
        renderer: &mut dyn RenderBackend,
        definition: FontDefinition<'_>,
    ) {
        match definition {
            FontDefinition::SwfTag(tag, encoding) => {
                let font =
                    Font::from_swf_tag(gc_context, renderer, tag, encoding, FontType::Device);
                let name = font.descriptor().name().to_owned();
                let is_bold = font.descriptor().bold();
                let is_italic = font.descriptor().italic();
                tracing::debug!(
                    "Loaded new device font \"{name}\" (bold: {is_bold}, italic: {is_italic}) from swf tag"
                );
                self.device_fonts.register(font);
            }
            FontDefinition::FontFile {
                name,
                is_bold,
                is_italic,
                data,
                index,
            } => {
                let descriptor = FontDescriptor::from_parts(&name, is_bold, is_italic);
                if let Ok(font) =
                    Font::from_font_file(gc_context, descriptor, data, index, FontType::Device)
                {
                    let name = font.descriptor().name().to_owned();
                    tracing::debug!(
                        "Loaded new device font \"{name}\" (bold: {is_bold}, italic: {is_italic}) from file"
                    );
                    self.device_fonts.register(font);
                } else {
                    warn!("Failed to load device font from file");
                }
            }
            FontDefinition::ExternalRenderer {
                name,
                is_bold,
                is_italic,
                font_renderer,
            } => {
                let descriptor = FontDescriptor::from_parts(&name, is_bold, is_italic);
                let font = Font::from_renderer(gc_context, descriptor, font_renderer);
                tracing::debug!(
                    "Loaded new externally rendered font \"{name}\" (bold: {is_bold}, italic: {is_italic})"
                );
                self.device_fonts.register(font);
            }
        }
        self.default_font_cache.clear();
    }

    /// Find a font by it's name and parameters.
    pub fn get_embedded_font_by_name(
        &self,
        name: &str,
        font_type: FontType,
        is_bold: bool,
        is_italic: bool,
        movie: Option<Arc<SwfMovie>>,
    ) -> Option<Font<'gc>> {
        let query = FontQuery::new(font_type, name.to_owned(), is_bold, is_italic);
        if let Some(font) = self.global_fonts.find(&query) {
            return Some(font);
        }
        if let Some(movie) = movie
            && let Some(library) = self.library_for_movie(movie)
            && let Some(font) = library.fonts.find(&query)
        {
            return Some(font);
        }
        None
    }

    pub fn global_fonts(&self) -> Vec<Font<'gc>> {
        self.global_fonts.all()
    }

    pub fn register_global_font(&mut self, font: Font<'gc>) {
        self.global_fonts.register(font);
    }

    /// Get the AVM2 class registry.
    pub fn avm2_class_registry(&self) -> &Avm2ClassRegistry<'gc> {
        &self.avm2_class_registry
    }

    /// Mutate the AVM2 class registry.
    pub fn avm2_class_registry_mut(&mut self) -> &mut Avm2ClassRegistry<'gc> {
        &mut self.avm2_class_registry
    }
}

#[derive(Collect, Default)]
#[collect(no_drop)]
struct FontMap<'gc>(FnvHashMap<FontQuery, Font<'gc>>);

impl<'gc> FontMap<'gc> {
    pub fn register(&mut self, font: Font<'gc>) {
        let descriptor = font.descriptor();
        self.0
            .entry(FontQuery::from_descriptor(font.font_type(), descriptor))
            .or_insert(font);
    }

    pub fn get(&self, font_query: &FontQuery) -> Option<&Font<'gc>> {
        self.0.get(font_query)
    }

    pub fn find(&self, font_query: &FontQuery) -> Option<Font<'gc>> {
        // The order here is specific, and tested in `tests/swfs/fonts/embed_matching/fallback_preferences`

        // Exact match
        if let Some(font) = self.get(font_query) {
            return Some(*font);
        }

        let is_italic = font_query.is_italic;
        let is_bold = font_query.is_bold;

        let mut fallback_query = font_query.clone();
        if is_italic ^ is_bold {
            // If one is set (but not both), then try upgrading to bold italic...
            fallback_query.is_bold = true;
            fallback_query.is_italic = true;
            if let Some(font) = self.get(&fallback_query) {
                return Some(*font);
            }

            // and then downgrading to regular
            fallback_query.is_bold = false;
            fallback_query.is_italic = false;
            if let Some(font) = self.get(&fallback_query) {
                return Some(*font);
            }

            // and then finally whichever one we don't have set
            fallback_query.is_bold = !is_bold;
            fallback_query.is_italic = !is_italic;
            if let Some(font) = self.get(&fallback_query) {
                return Some(*font);
            }
        } else {
            // We don't have an exact match and we were either looking for regular or bold-italic

            if is_italic && is_bold {
                // Do we have regular? (unless we already looked for it)
                fallback_query.is_bold = false;
                fallback_query.is_italic = false;
                if let Some(font) = self.get(&fallback_query) {
                    return Some(*font);
                }
            }

            // Do we have bold?
            fallback_query.is_bold = true;
            fallback_query.is_italic = false;
            if let Some(font) = self.get(&fallback_query) {
                return Some(*font);
            }

            // Do we have italic?
            fallback_query.is_bold = false;
            fallback_query.is_italic = true;
            if let Some(font) = self.get(&fallback_query) {
                return Some(*font);
            }

            if !is_bold && !is_italic {
                // Do we have bold italic? (unless we already looked for it)
                fallback_query.is_bold = true;
                fallback_query.is_italic = true;
                if let Some(font) = self.get(&fallback_query) {
                    return Some(*font);
                }
            }
        }

        None
    }

    pub fn all(&self) -> Vec<Font<'gc>> {
        self.0.values().copied().collect()
    }
}
