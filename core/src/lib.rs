// This lint is helpful, but right now we have too many instances of it.
// TODO: Remove this once all instances are fixed.
#![allow(clippy::needless_pass_by_ref_mut)]
// This lint is good in theory, but in AVMs we often need to do `let x = args.get(0); let y = args.get(1);` etc.
// It'd make those much less readable and consistent.
#![allow(clippy::get_first)]

// The measurement build reads an op's discriminant to count it.
#![cfg_attr(any(feature = "flashnx_opcount", feature = "flashnx_jit"), feature(core_intrinsics))]
#![cfg_attr(any(feature = "flashnx_opcount", feature = "flashnx_jit"), allow(internal_features))]

#[macro_use]
mod display_object;
pub use display_object::{StageAlign, StageDisplayState, StageScaleMode};

#[macro_use]
extern crate num_derive;

#[macro_use]
mod avm1;
mod avm2;

// Per-game "Capabilities.playerType" override (Disney/Yamago shells need opposite answers, see "avm2::globals::flash::system::capabilities").
pub use avm2::globals::flash::system::capabilities::set_plugin_player_type;
mod avm_rng;
mod binary_data;
pub mod bitmap;
mod character;

// Per-movie reset + sizing of the decoded-bitmap cache budget.
pub use character::{flashnx_bitmap_residency, reset_bitmap_cache};
pub use flashnx_census::report as census_report;
pub use library::set_movie_free;
pub use flashnx_as3prof::{report as as3prof_report, set_as3_profiler};
pub use avm2::activation::{
    jit_map_text, jit_report, jit_reset, set_fast_calls, set_jit, set_jit_ab, set_jit_direct_calls, set_jit_map,
    set_jit_tos_cache,
};
pub use avm1::set_avm1_fast_push;

// AVM2 opcode census over the last "frames" frames, then reset ("avm2ops:" lines). Empty unless built with "flashnx_opcount".
pub fn avm2ops_report(frames: u64) -> String {
    #[cfg(feature = "flashnx_opcount")]
    {
        flashnx_avm2ops::report(frames)
    }
    #[cfg(not(feature = "flashnx_opcount"))]
    {
        let _ = frames;
        String::new()
    }
}
pub use bitmap::operations::{take_threshold_memo_hits, take_threshold_stats};
pub use frame_lifecycle::{
    set_avm1_inner_goto_skip, set_avm1_unloaded_quiet, set_orphan_skip, take_avm1_inner_skipped,
    take_avm1_unloaded_skipped, take_inner_goto_stats,
};
pub use display_object::set_lazy_shapes;
pub use display_object::{set_full_frame_skip, set_stage_skip, take_enter_skipped, take_subtrees_skipped};
pub mod context;
pub mod context_menu;
mod drawing;
mod ecma_conversions;
pub mod events;
pub mod focus_tracker;
pub mod font;
mod flashnx_as3prof;
mod flashnx_census;
#[cfg(feature = "flashnx_opcount")]
mod flashnx_avm2ops;
mod frame_lifecycle;
mod html;
mod input;
mod library;
pub mod limits;
pub mod loader;
mod local_connection;
mod locale;
mod net_connection;
mod orphan_manager;
pub mod pixel_bender;
mod player;
mod prelude;
pub mod socket;
mod streams;
pub mod string;
mod system_properties;
pub mod tag_utils;
mod tessellation_cache;
pub mod timer;
mod types;
mod vminterface;

pub mod backend;
pub mod compatibility_rules;
pub mod config;
#[cfg(feature = "egui")]
pub mod debug_ui;
pub mod external;
pub mod i18n;
pub mod stub;

pub use context_menu::ContextMenuItem;
pub use events::PlayerEvent;
pub use indexmap;
pub use loader::LoadBehavior;
pub use player::{Player, PlayerBuilder, PlayerMode, PlayerRuntime, StaticCallstack};

// Last tick's SWF-frame count, collector phase and arena allocation, for the periodic-spike hunt (see "player::flashnx_gc_probe").
pub use player::{flashnx_frame_accumulator_us, flashnx_gc_probe, set_render_cost, set_script_hook};
pub use player::set_bitmap_cache;
pub use player::set_gc_small_pacing;
pub use display_object::set_flashnx_mask_fixes;

// The AVM1 clip-reference cache switch, and the AVM1 probes the Switch measurement build reads (zeros without "flashnx_instr").
pub use player::{
    flashnx_avm1_named_take, flashnx_avm1_ops, flashnx_avm1_probe, flashnx_getname_probe,
    flashnx_goto_probe, flashnx_mcref_probe, flashnx_set_avm1_named, flashnx_set_mcref_cache,
};
pub use ruffle_common::duration::FloatDuration;
pub use ruffle_render::backend::ViewportDimensions;
pub use swf;
pub use swf::Color;
pub use ttf_parser;

/// The newest Flash Player version known to Ruffle.
pub const NEWEST_PLAYER_VERSION: u8 = 51;

/// The default Flash Player version that Ruffle will emulate.
pub const DEFAULT_PLAYER_VERSION: u8 = 32;
