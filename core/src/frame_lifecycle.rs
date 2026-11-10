/*
    Frame events management

    This module aids in keeping track of which frame execution phase we are in.

    For AVM2 code, display objects execute a series of discrete phases,
    and each object is notified about the current frame phase in rendering order.
    When objects are created,
    they are "caught up" to the current frame phase to ensure correct order of operations.

    AVM1 code (presumably, either on an AVM1 stage or within an "AVM1Movie")
    runs in one phase, with timeline operations executing with all phases inline in the order that clips were originally created.
*/

use crate::avm2::{Avm2, EventObject};
use crate::context::UpdateContext;
use crate::display_object::{DisplayObject, MovieClip, TDisplayObject};
use crate::loader::LoadManager;
use crate::orphan_manager::OrphanManager;
use tracing::instrument;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/*
    whether an inner goto may skip orphans whose tree has not changed since they were last processed (see "run_inner_goto_frame").
    On by default: off to compare, from FlashNX's `gotoskip.off` marker.
*/
static ORPHAN_SKIP: AtomicBool = AtomicBool::new(true);
static INNER_GOTOS: AtomicU64 = AtomicU64::new(0);
static ORPHANS_RUN: AtomicU64 = AtomicU64::new(0);
static ORPHANS_SKIPPED: AtomicU64 = AtomicU64::new(0);

/*
    The orphan mark epoch at the start of the last frame-script pass that processed every dirty orphan with no new mark arriving meanwhile:
    while the epoch still equals it, every orphan is clean. "u64::MAX" = unknown.
*/
static ORPHANS_CLEAN_AT: AtomicU64 = AtomicU64::new(u64::MAX);
static GOTOS_LIST_SKIPPED: AtomicU64 = AtomicU64::new(0);

pub fn set_orphan_skip(enabled: bool) {
    ORPHAN_SKIP.store(enabled, Ordering::Relaxed);
}

/*
    Whether an explicit goto on an AVM1 clip leaves out the inner frame.
    
    Everything "run_inner_goto_frame" does belongs to AVM2:
    orphans (AVM2 objects off the display list),
    frame construction (AVM1 clips are exempt from it),
    
    AS3 frame scripts, the "frameConstructed" and "exitFrame" broadcasts, LoaderInfo events.
    
    Upstream already leaves it out of AVM1 no-op gotos ("no_op_goto"),
    not of the others, so every "gotoAndStop" or "nextFrame" of an AS2 game walked the whole stage twice: ~117 us a goto in
    Madness: Project Nexus (issue #109),
    which redresses every character with dozens of them a frame, 82 % of its goto time.
    
    On by default: off from FlashNX's "avm1goto.off" marker, to compare.
*/
static AVM1_INNER_SKIP: AtomicBool = AtomicBool::new(true);
static AVM1_INNER_SKIPPED: AtomicU64 = AtomicU64::new(0);

pub fn set_avm1_inner_goto_skip(enabled: bool) {
    AVM1_INNER_SKIP.store(enabled, Ordering::Relaxed);
}

// AVM1 gotos that left out their inner frame since the last call.
pub fn take_avm1_inner_skipped() -> u64 {
    AVM1_INNER_SKIPPED.swap(0, Ordering::Relaxed)
}

// Whether an explicit goto on "clip" runs the inner frame (see "AVM1_INNER_SKIP").
pub fn explicit_goto_runs_inner_frame(clip: MovieClip<'_>) -> bool {
    if clip.movie().is_action_script_3() || !AVM1_INNER_SKIP.load(Ordering::Relaxed) {
        return true;
    }
    AVM1_INNER_SKIPPED.fetch_add(1, Ordering::Relaxed);
    false
}

/*
    ruffle#19434:
    a clip emptied by "unloadMovie()" no longer gets its "onClipEvent(enterFrame)" and "onEnterFrame":
    
    Flash stops them, Ruffle kept running them on the empty clip.
    
    In Combat Tournament Legends (#123) every fighter not picked unloads itself in its load handler,
    then its frame 1 script gives the emptied clip an "onEnterFrame" with the whole fighter logic:
    7 ghost fighters thinking every frame, 85 % of the fight's ActionScript. 
    
    On by default: off from FlashNX's "unloadedframe.off" marker, to compare.
*/
static AVM1_UNLOADED_QUIET: AtomicBool = AtomicBool::new(true);
static AVM1_UNLOADED_SKIPPED: AtomicU64 = AtomicU64::new(0);

pub fn set_avm1_unloaded_quiet(enabled: bool) {
    AVM1_UNLOADED_QUIET.store(enabled, Ordering::Relaxed);
}

/// enterFrame events not sent to unloaded clips since the last call.
pub fn take_avm1_unloaded_skipped() -> u64 {
    AVM1_UNLOADED_SKIPPED.swap(0, Ordering::Relaxed)
}

// Whether an unloaded clip still gets its enterFrame event (see "AVM1_UNLOADED_QUIET").
pub fn unloaded_clip_gets_enter_frame() -> bool {
    if AVM1_UNLOADED_QUIET.load(Ordering::Relaxed) {
        AVM1_UNLOADED_SKIPPED.fetch_add(1, Ordering::Relaxed);
        return false;
    }
    true
}

/// (inner gotos, orphans processed by them, orphans skipped, gotos that did not
/// even iterate the orphan list) since the last call.
pub fn take_inner_goto_stats() -> (u64, u64, u64, u64) {
    (
        INNER_GOTOS.swap(0, Ordering::Relaxed),
        ORPHANS_RUN.swap(0, Ordering::Relaxed),
        ORPHANS_SKIPPED.swap(0, Ordering::Relaxed),
        GOTOS_LIST_SKIPPED.swap(0, Ordering::Relaxed),
    )
}

/// Which phase of the frame we're currently in.
///
/// AVM2 frames exist in one of four phases: `Enter`, `Construct`,
/// `FrameScripts`, or `Exit`. An additional `Idle` phase covers rendering and
/// event processing.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum FramePhase {
    /// We're entering the next frame.
    ///
    /// When movie clips enter a new frame, they must do two things:
    ///
    ///  - Remove all children that should not exist on the next frame.
    ///  - Increment their current frame number.
    ///
    /// Once this phase ends, we fire `enterFrame` on the broadcast list.
    Enter,

    /// We're constructing children of existing display objects.
    ///
    /// All `PlaceObject` tags should execute at this time.
    ///
    /// Once we construct the frame, we fire `frameConstructed` on the
    /// broadcast list.
    Construct,

    /// We're running all queued frame scripts.
    ///
    /// Frame scripts are the AS3 equivalent of old-style `DoAction` tags. They
    /// are queued in the `Update` phase if the current timeline frame number
    /// differs from the prior frame's one.
    FrameScripts,

    /// We're finishing frame processing.
    ///
    /// When we exit a completed frame, we fire `exitFrame` on the broadcast
    /// list.
    Exit,

    /// We're not currently executing any frame code.
    ///
    /// At this point in time, event handlers are expected to run. No frame
    /// catch-up work should execute.
    #[default]
    Idle,
}

/// Run one frame according to AVM2 frame order.
/// NOTE: The `each_orphan_movie` calls are in really odd places,
/// but this is needed to match Flash Player's output. There may
/// still be lurking bugs, but the current code matches Flash's
/// output exactly for two complex test cases (see `avm2/orphan_movie*`)
#[instrument(level = "debug", skip_all)]
pub fn run_all_phases_avm2(context: &mut UpdateContext<'_>) {
    let stage = context.stage;

    if !stage.movie().is_action_script_3() {
        return;
    }

    #[cfg(feature = "flashnx_census")]
    crate::flashnx_census::note_frame(context.orphan_manager.len(), context.avm2.broadcast_len());

    /*
        Full frames pass over unchanged subtrees too (see "display_object::set_full_frame_skip"):
        
        quiet ones while entering the frame, clean ones while constructing it and running its scripts,
        by the same rules as an inner goto.
        
        Only around the walks: the broadcasts in between run as before.
    */
    let full_skip = crate::display_object::full_frame_skip();

    *context.frame_phase = FramePhase::Enter;
    OrphanManager::each_orphan_obj(context, |orphan, context| {
        if crate::display_object::skip_quiet_child(orphan) {
            return;
        }
        orphan.enter_frame(context);
    });
    stage.enter_frame(context);

    *context.frame_phase = FramePhase::Construct;
    let was_skipping = crate::display_object::set_skipping_clean_subtrees(full_skip);
    OrphanManager::each_orphan_obj(context, |orphan, context| {
        if crate::display_object::skip_clean_child(orphan) {
            return;
        }
        orphan.construct_frame(context);
    });
    stage.construct_frame(context);
    crate::display_object::set_skipping_clean_subtrees(was_skipping);
    broadcast_frame_constructed(context);

    *context.frame_phase = FramePhase::FrameScripts;
    let epoch = crate::display_object::orphan_mark_epoch();
    let mut left_dirty = false;
    let was_skipping = crate::display_object::set_skipping_clean_subtrees(full_skip);
    OrphanManager::each_orphan_obj(context, |orphan, context| {
        if crate::display_object::skip_clean_child(orphan) {
            return;
        }
        orphan.base().set_lifecycle_dirty(false);
        orphan.run_frame_scripts(context);
        left_dirty |= orphan.base().lifecycle_dirty();
    });
    if !left_dirty {
        note_orphans_clean_if_unmarked(epoch);
    }
    stage.run_frame_scripts(context);
    crate::display_object::set_skipping_clean_subtrees(was_skipping);
    run_frame_script_cleanup(context);

    *context.frame_phase = FramePhase::Exit;
    broadcast_frame_exited(context);

    // The correct time to run context3DCreated events seems to be here
    stage.check_requested_context3ds(context);

    // We cannot easily remove dead `GcWeak` instances from the orphan list
    // inside `each_orphan_movie`, since the callback may modify the orphan list.
    // Instead, we do one cleanup at the end of the frame.
    // This performs special handling of clips which became orphaned as
    // a result of a RemoveObject tag - see `cleanup_dead_orphans` for details.
    context.orphan_manager.cleanup_dead_orphans(context.gc());

    *context.frame_phase = FramePhase::Idle;
}

/// Like `run_all_phases_avm2`, but specialized for the "nested frame" triggered
/// by a goto. This is different enough to not be worth combining into a single
/// method with extra parameters.
///
/// During a goto, we run frame construction, framescripts, and frame exits for the *entire stage*.
/// This even extends to orphans - for example, calling `gotoAndStop` on an orphan will
/// cause frame construction to get run for the *current frame* of other objects on the timeline
/// (even if the goto was called from an enterFrame event handler).
pub fn run_inner_goto_frame<'gc>(
    context: &mut UpdateContext<'gc>,
    removed_frame_scripts: &[DisplayObject<'gc>],
    initial_clip: MovieClip<'gc>,
) {
    if initial_clip.swf_version() <= 9 && initial_clip.movie().is_action_script_3() {
        // We skip the next `enter_frame` call, so that we will still run the framescripts
        // queued for our target frame.
        initial_clip.base().set_skip_next_enter_frame(true);

        // The next enter_frame must visit it to consume the flag.
        crate::display_object::mark_enter_needed(initial_clip.into());

        /*
            The objects placed by this goto are not constructed here,
            but they must be before the frame script of the target frame runs (see "MovieClip::run_local_frame_scripts").
        */
        initial_clip.set_goto_construct_pending();

        return;
    }

    let stage = context.stage;
    let old_phase = *context.frame_phase;

    // When performing goto, frame scripts behave the same as when entering a new frame
    // so no separate cleanup is performed on ones registered during frame script phase
    context.frame_script_cleanup_queue.clear();

    // Note - we do *not* call `enter_frame` or dispatch an `enterFrame` event

    /*
        Every inner goto used to walk the whole tree of every orphan,
        twice (construct, then frame scripts), whatever the goto touched.
        
        On an orphan whose tree has not changed since it was last processed,
        both walks are no-ops:
        every object is constructed, every frame script of its current frame has run.
        
        Super Smash Flash 2 keeps a growing crowd of orphans and fires gotos all frame long:
        that walk was 54 % of a fight frame, and fights slowed down one after the other
        
        (ruffle#24509 is the same wall on desktop).
        
        A tree is dirty from the moment something in it changes frame, gains a child,
        or queues a frame script or a goto ("mark_lifecycle_dirty"), until its frame scripts run.
        
        The stage is still walked every time.
    */
    let skip = ORPHAN_SKIP.load(Ordering::Relaxed);
    INNER_GOTOS.fetch_add(1, Ordering::Relaxed);

    /*
        Only the dirty orphans, found through the queue their marks fill ("queue_dirty_orphan"),
        in orphan-list order:
        
        the same objects in the same order as iterating the list and passing over the clean ones,
        without iterating it (~800 weak references in an SSF2 fight, twice per goto).
        
        And inside the trees it does walk, the stage's included, pass over every unchanged subtree ("skipping_clean_subtrees").
        
        Only around the walks: the broadcasts in between run as before.
    */
    *context.frame_phase = FramePhase::Construct;
    let was_skipping = crate::display_object::set_skipping_clean_subtrees(skip);
    if skip {
        let done = for_each_dirty_orphan(context, |orphan, context| {
            orphan.construct_frame(context);
        });
        if done.is_empty() {
            GOTOS_LIST_SKIPPED.fetch_add(1, Ordering::Relaxed);
        }
        // Constructing does not clean them: the frame-script pass below must find them again.
        for orphan in done {
            crate::display_object::queue_dirty_orphan(orphan);
        }
    } else {
        OrphanManager::each_orphan_obj(context, |orphan, context| {
            orphan.construct_frame(context);
        });
    }
    stage.construct_frame(context);
    crate::display_object::set_skipping_clean_subtrees(was_skipping);
    broadcast_frame_constructed(context);

    *context.frame_phase = FramePhase::FrameScripts;
    let was_skipping = crate::display_object::set_skipping_clean_subtrees(skip);
    stage.run_frame_scripts(context);
    let run_scripts = |orphan: DisplayObject<'gc>, context: &mut UpdateContext<'gc>| {
        orphan.base().set_lifecycle_dirty(false);
        ORPHANS_RUN.fetch_add(1, Ordering::Relaxed);
        orphan.run_frame_scripts(context);
    };
    if skip {
        for_each_dirty_orphan(context, run_scripts);
    } else {
        OrphanManager::each_orphan_obj(context, run_scripts);
    }
    crate::display_object::set_skipping_clean_subtrees(was_skipping);

    for child in removed_frame_scripts {
        child.run_frame_scripts(context);
    }

    *context.frame_phase = FramePhase::Exit;
    broadcast_frame_exited(context);

    // We cannot easily remove dead `GcWeak` instances from the orphan list
    // inside `each_orphan_movie`, since the callback may modify the orphan list.
    // Instead, we do one cleanup at the end of the frame.
    // This performs special handling of clips which became orphaned as
    // a result of a RemoveObject tag - see `cleanup_dead_orphans` for details.

    // Only when such an orphan was added: dead and re-parented entries are passed over by every walk and go at the frame's end.
    if !skip || context.orphan_manager.needs_cleanup() {
        context.orphan_manager.cleanup_dead_orphans(context.gc());
    }

    *context.frame_phase = old_phase;
}

/*
    Run "f" on every dirty orphan, in orphan-list order,
    without iterating the list.
    
    An orphan that becomes dirty while this runs is taken in if it comes later in the list
    (iterating the list would have reached it),

    left queued for the next pass otherwise (iterating would have passed it already).
    
    Returns the orphans "f" ran on.
*/
fn for_each_dirty_orphan<'gc>(context: &mut UpdateContext<'gc>, mut f: impl FnMut(DisplayObject<'gc>, &mut UpdateContext<'gc>), ) -> Vec<DisplayObject<'gc>> {
    use crate::display_object::{queue_dirty_orphan, take_dirty_orphans};
    let mut pending = context.orphan_manager.resolve_dirty(context.gc(), &take_dirty_orphans());
    let mut done = Vec::with_capacity(pending.len());
    let mut late = Vec::new();
    let mut i = 0;
    while i < pending.len() {
        let (seq, orphan) = pending[i];
        i += 1;
        f(orphan, context);
        done.push(orphan);
        let fresh = take_dirty_orphans();
        if fresh.is_empty() {
            continue;
        }
        for (s, o) in context.orphan_manager.resolve_dirty(context.gc(), &fresh) {
            if s > seq {
                let ahead = &pending[i..];
                if !ahead.iter().any(|e| e.0 == s) {
                    let at = i + ahead.partition_point(|e| e.0 < s);
                    pending.insert(at, (s, o));
                }
            } else {
                late.push(o);
            }
        }
    }
    for o in late {
        queue_dirty_orphan(o);
    }
    done
}

// A frame-script pass over the orphans just cleaned every dirty one. If nothing got marked while it ran, every orphan is clean as of "epoch".
fn note_orphans_clean_if_unmarked(epoch: u64) {
    if crate::display_object::orphan_mark_epoch() == epoch {
        ORPHANS_CLEAN_AT.store(epoch, Ordering::Relaxed);
    }
}

/// Broadcast a `enterFrame` event to all `DisplayObject`s.
pub fn broadcast_frame_entered<'gc>(context: &mut UpdateContext<'gc>) {
    let enter_frame_evt = EventObject::bare_default_event(context, "enterFrame");
    let dobject_constr = context.avm2.classes().display_object;
    Avm2::broadcast_event(context, enter_frame_evt, dobject_constr);
}

/// Broadcast a `frameConstructed` event to all `DisplayObject`s.
pub fn broadcast_frame_constructed<'gc>(context: &mut UpdateContext<'gc>) {
    let frame_constructed_evt = EventObject::bare_default_event(context, "frameConstructed");
    let dobject_constr = context.avm2.classes().display_object;
    Avm2::broadcast_event(context, frame_constructed_evt, dobject_constr);
}

/// Broadcast a `exitFrame` event to all `DisplayObject`s.
pub fn broadcast_frame_exited<'gc>(context: &mut UpdateContext<'gc>) {
    let exit_frame_evt = EventObject::bare_default_event(context, "exitFrame");
    let dobject_constr = context.avm2.classes().display_object;
    Avm2::broadcast_event(context, exit_frame_evt, dobject_constr);

    LoadManager::run_exit_frame(context);
}

/// Empty the `context.frame_script_cleanup_queue` by running frame scripts for
/// each clip in the queue.
fn run_frame_script_cleanup<'gc>(context: &mut UpdateContext<'gc>) {
    while let Some(clip) = context.frame_script_cleanup_queue.pop_front() {
        clip.set_has_pending_script(true);
        clip.set_last_queued_script_frame(None);
        clip.run_local_frame_scripts(context);
    }
}

/// Run all previously-executed frame phases on a newly-constructed display
/// object.
///
/// This is a no-op on AVM1, which has it's own catch-up logic.
pub fn catchup_display_object_to_frame<'gc>(
    context: &mut UpdateContext<'gc>,
    dobj: DisplayObject<'gc>,
) {
    if !dobj.movie().is_action_script_3() {
        return;
    }

    match *context.frame_phase {
        FramePhase::Enter => {
            dobj.enter_frame(context);
        }
        FramePhase::Construct | FramePhase::FrameScripts | FramePhase::Exit | FramePhase::Idle => {
            dobj.enter_frame(context);
            dobj.construct_frame(context);
        }
    }
}
