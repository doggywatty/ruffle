use quick_xml::escape::EscapeError as XmlEscapeError;
use std::str::Utf8Error;

/// AVM1 XML unescaping. Decodes entities individually, even when
/// preceded by a bare '&' (e.g. "&&amp;" becomes "&&").
pub fn avm1_unescape(input: &[u8]) -> Result<String, Utf8Error> {
    custom_unescape(input, false)
}

/// AVM2 E4X XML unescaping. Does not decode entities preceded by a
/// bare '&' (e.g. "&&amp;" is preserved as "&&amp;").
pub fn avm2_unescape(input: &[u8]) -> Result<String, Utf8Error> {
    custom_unescape(input, true)
}

/*
	Handles Flash-specific XML unescaping behavior.
	We accept all XML entities, and also accept standalone '&' without a corresponding ';'.

	The two AVMs pick their candidates differently (this used to be a pair of regexes, "&[^;&]*;" for AVM1 and `&[^;]*;` for AVM2):
	- AVM2 matches from an '&' to the first ';' after it, whatever is in between, so "&&amp;" is one candidate and is preserved as-is,
	- AVM1 only matches when no other '&' comes before that ';', so "&&amp;" decodes as a literal '&' followed by "&amp;".

	Both are scanned linearly, the regex form was catastrophic on strings with bare '&' and no ';' at all
	Super Mario Bros. Crossover's level data is MBs of that (lakituStart&&HideOnDifficulties=...),
	and each '&' made the backtracker walk to the end of the string again
	(34 s for one 4 MB text node inside "XML/init")
		- upstream ruffle#21623, FlashNX -
*/
fn custom_unescape(input: &[u8], avm2: bool) -> Result<String, Utf8Error> {
    let input = std::str::from_utf8(input)?;

    let mut result = String::new();
    let mut last_end = 0;
    let mut i = 0;

    while i < input.len() {
        let Some(rel) = input[i..].find('&') else {
            break;
        };
        let amp = i + rel;

        let rest = &input[amp + 1..];
        let Some(semi_rel) = rest.find(';') else {
            /*
				No ';' after this '&', so there is none after any later '&' either.
			*/
            break;
        };
        if !avm2 && rest[..semi_rel].contains('&') {
            /*
				Another '&' comes first: this one is bare.
			*/
            i = amp + 1 + rest[..semi_rel].find('&').unwrap();
            continue;
        }

        let end = amp + 1 + semi_rel + 1;

        result.push_str(&input[last_end..amp]);

        let entity = &input[amp..end];
        /*
			Unfortunately, we need to call this on each entity individually,
			since it bails out if *any* entities in the string lack a terminating ';'
		*/
        match quick_xml::escape::unescape(entity) {
            Ok(decoded) => result.push_str(&decoded),
            /*
				Unknown or malformed entities are preserved as-is.
				The AVM2 rule can match strings like "&&amp;" which contain
				multiple '&' characters, causing unescape to report UnterminatedEntity
				for the first '&' before it reaches the ';'.
			*/
            Err(
                XmlEscapeError::InvalidCharRef(_)
                | XmlEscapeError::UnrecognizedEntity(_, _)
                | XmlEscapeError::UnterminatedEntity(_),
            ) => result.push_str(entity),
        }

        last_end = end;
        i = end;
    }

    result.push_str(&input[last_end..]);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
		Mirrors tests/tests/swfs/{avm1,avm2}/xml_unescaping/Test.as, so the regex-free scan can be checked without running a SWF.
	*/
    #[test]
    fn unescape_entities() {
        for (avm2, input, expected) in [
            (true, "A &amp; &#39; B", "A & ' B"),
            (false, "A &amp; &#39; B", "A & ' B"),
			
            /*
				A bare '&' before a real entity.
			*/
            (true, "A & &thing; B", "A & &thing; B"),
            (false, "A & &thing; B", "A & &thing; B"),
            (true, "A &&thing; B", "A &&thing; B"),
            (false, "A &&thing; B", "A &&thing; B"),
            (true, "A &&&thing; B", "A &&&thing; B"),
            (false, "A &&&thing; B", "A &&&thing; B"),
            (true, "A &&amp; B", "A &&amp; B"),
            (false, "A &&amp; B", "A && B"),
            (true, "A &amp;&amp; B", "A && B"),
            (false, "A &amp;&amp; B", "A && B"),

            /*
				No terminator at all, preserved verbatim.
			*/
            (true, "a && b && c", "a && b && c"),
            (false, "a && b && c", "a && b && c"),

            /*
				Numeric and named entities next to bare ampersands.
			*/
            (true, "&bogus; &lt;", "&bogus; <"),
            (false, "&bogus; &lt;", "&bogus; <"),
        ] {
            let got = if avm2 {
                avm2_unescape(input.as_bytes()).unwrap()
            } else {
                avm1_unescape(input.as_bytes()).unwrap()
            };
            assert_eq!(got, expected, "avm2={avm2} input={input:?}");
        }
    }

    /*
		a 4 MB text of bare '&' and no ';' used to take tens of seconds (38-ish) through the entity regex.
		It's now 1 linear pass.
	*/
    #[test]
    fn unescape_bare_ampersands_is_linear() {
        let input = "0,0,lakituStart&&HideOnDifficulties=easynormal,0,0,".repeat(90_000);
        for avm2 in [true, false] {
            let out = if avm2 {
                avm2_unescape(input.as_bytes()).unwrap()
            } else {
                avm1_unescape(input.as_bytes()).unwrap()
            };
            assert_eq!(out, input);
        }
        let input = "&amp;".repeat(1_000_000);
        for avm2 in [true, false] {
            let out = if avm2 {
                avm2_unescape(input.as_bytes()).unwrap()
            } else {
                avm1_unescape(input.as_bytes()).unwrap()
            };
            assert_eq!(out, "&".repeat(1_000_000));
        }
    }
}
