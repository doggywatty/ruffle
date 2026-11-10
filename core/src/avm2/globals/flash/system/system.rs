//! `flash.system.System` native methods

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::parameters::ParametersExt;
use crate::avm2::value::Value;

/*
    "flash.system.System.gc", a full collection after this update.
    
    Upstream leaves it empty (release Flash Players ignore it).
    Super Smash Flash 2 calls it when it tears a match down, right before loading the next stage and fighters, which is the moment a 3 GB console runs out:
    the incremental collector is paced per frame and still holds the old match's garbage then.
    
    The same request as the open upstream PR ruffle#23897, rate limited by "request_full_gc" since some games call it every frame.
*/
pub fn gc<'gc>(
    _activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    crate::player::request_full_gc();
    Ok(Value::Undefined)
}

/// Implements `flash.system.System.setClipboard` method
pub fn set_clipboard<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    // The following restrictions only apply to the plugin.
    // TODO: Check the type of event that triggered the function call.
    #[cfg(target_family = "wasm")]
    if false {
        return Err(crate::avm2::error::make_error_2176(activation));
    }

    let new_content = args.get_string_non_null(activation, 0, "text")?;
    activation
        .context
        .ui
        .set_clipboard_content(new_content.to_string());

    Ok(Value::Undefined)
}
