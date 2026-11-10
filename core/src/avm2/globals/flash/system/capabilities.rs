//! `flash.display.Capabilities` native methods

use crate::avm2::{Activation, AvmString, Error, Value};
use crate::player::PlayerRuntime;

/// Implements `flash.system.Capabilities.os`
pub fn get_os<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let os = match activation.avm2().player_runtime {
        // For most normal Flash Player usage, the OS should not matter,
        // so let's pretend it's Windows for the broadest possible compatibility.
        PlayerRuntime::FlashPlayer => "Windows 8",
        PlayerRuntime::AIR => {
            if cfg!(windows) {
                "Windows 10"
            } else if cfg!(target_vendor = "apple") {
                "Mac OS 10.5.2"
            } else {
                "Linux 5.10.49"
            }
        }
    };
    Ok(AvmString::new_utf8(activation.gc(), os).into())
}

/// Implements `flash.system.Capabilities.version`
pub fn get_version<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let os = match activation.avm2().player_runtime {
        PlayerRuntime::FlashPlayer => "WIN",
        PlayerRuntime::AIR => {
            if cfg!(windows) {
                "WIN"
            } else if cfg!(target_vendor = "apple") {
                "MAC"
            } else {
                "LNX"
            }
        }
    };
    Ok(AvmString::new_utf8(
        activation.gc(),
        format!("{os} {},0,0,0", activation.avm2().player_version),
    )
    .into())
}

/*
    Report "Capabilities.playerType" as "PlugIn" instead of Ruffle's default.
    Off unless the host turns it on for a specific game, see the note in "get_player_type".
*/
static PLUGIN_PLAYER_TYPE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

// Set by the host before building the player, per game.
pub fn set_plugin_player_type(on: bool) {
    PLUGIN_PLAYER_TYPE.store(on, core::sync::atomic::Ordering::Relaxed);
}

fn plugin_player_type() -> bool {
    PLUGIN_PLAYER_TYPE.load(core::sync::atomic::Ordering::Relaxed)
}

/// Implements `flash.system.Capabilities.playerType`
pub fn get_player_type<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    // TODO: When should "External" be returned?

    /*
        Disney/Yamago minigames branch on "UtilsSystem.isStandalone()" (playerType == "StandAlone" or "External"), and the two branches want OPPOSITE answers depending on the title:

        - Agent P Strikes Back: its standalone branch loads a DEV harness ("../../deploy/MinigameAPI_*_DEV_ONLY_DO_NOT_PUBLISH.swf")
        Disney never shipped to blank stage because it needs "PlugIn".
        - Tron Uprising: Escape from Argon City: its standalone branch is fully OFFLINE ("onApiInit(null)", saves through a SharedObject),
        while its plugin branch loads Disney's live API and then dereferences "api.dataStorageService" to null offline (TypeError #1009).
        It needs "StandAlone" (i.e. Ruffle's own default).

        So this is a per-GAME choice, not a global one.
        Default = stock Ruffle behaviour:
        the host opts a game in via "set_plugin_player_type" (see "ruffle_init").
        A blanket "PlugIn" also caused the Pursuit of Hat 2 black screen (gaforflash), so the narrower the better.
    */
    let player_type = if cfg!(target_family = "wasm") || plugin_player_type() {
        "PlugIn"
    } else {
        match activation.context.avm2.player_runtime {
            PlayerRuntime::FlashPlayer => "StandAlone",
            PlayerRuntime::AIR => "Desktop",
        }
    };

    Ok(AvmString::new_utf8(activation.gc(), player_type).into())
}

/// Implements `flash.system.Capabilities.screenResolutionX`
pub fn get_screen_resolution_x<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let viewport_dimensions = activation.context.renderer.viewport_dimensions();
    // Viewport size is adjusted for HiDPI.
    let adjusted_width = f64::from(viewport_dimensions.width) / viewport_dimensions.scale_factor;
    Ok(adjusted_width.round().into())
}

/// Implements `flash.system.Capabilities.screenResolutionY`
pub fn get_screen_resolution_y<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let viewport_dimensions = activation.context.renderer.viewport_dimensions();
    // Viewport size is adjusted for HiDPI.
    let adjusted_height = f64::from(viewport_dimensions.height) / viewport_dimensions.scale_factor;
    Ok(adjusted_height.round().into())
}

/// Implements `flash.system.Capabilities.pixelAspectRatio`
pub fn get_pixel_aspect_ratio<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    // source: https://web.archive.org/web/20230611050355/https://flylib.com/books/en/4.13.1.272/1/
    Ok(1.into())
}

/// Implements `flash.system.Capabilities.screenDPI`
pub fn get_screen_dpi<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    // source: https://tracker.adobe.com/#/view/FP-3949775
    Ok(72.into())
}

/// Implements `flash.system.Capabilities.language`
pub fn get_language<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    let language = activation
        .context
        .system
        .language
        .get_language_code(activation.context.player_version);
    Ok(AvmString::new_utf8(activation.gc(), language).into())
}
