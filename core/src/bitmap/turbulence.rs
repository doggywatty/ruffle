//! This file is a Rust port of the C reference implementation of the
//! feTurbulence element in the SVG specification. It's the usual Perlin noise.
//!
//! See: <https://www.w3.org/TR/SVG11/filters.html#feTurbulenceElement>.
//! The `octave_offsets` parameter of `turbulence` was added after porting.

// Copyright © 2015 W3C® (MIT, ERCIM, Keio, Beihang).
// This software or document includes material copied from or derived
// from https://www.w3.org/TR/SVG11/filters.html#feTurbulenceElement.

/* Produces results in the range [1, 2**31 - 2].
Algorithm is: r = (a * r) mod m
where a = 16807 and m = 2**31 - 1 = 2147483647
See [Park & Miller], CACM vol. 31 no. 10 p. 1195, Oct. 1988
To test: the algorithm should produce the result 1043618065
as the 10,000th generated number if the original seed is 1.
*/
const RAND_M: i64 = 2147483647; // 2**31 - 1
const RAND_A: i64 = 16807; // 7**5; primitive root of m
const RAND_Q: i64 = 127773; // m / a
const RAND_R: i64 = 2836; // m % a
fn setup_seed(mut seed: i64) -> i64 {
    if seed <= 0 {
        seed = -(seed % (RAND_M - 1)) + 1
    };
    if seed > RAND_M - 1 {
        seed = RAND_M - 1
    };
    seed
}

fn random(seed: i64) -> i64 {
    let mut result = RAND_A * (seed % RAND_Q) - RAND_R * (seed / RAND_Q);
    if result <= 0 {
        result += RAND_M
    };
    result
}

#[derive(Copy, Clone)]
struct StitchInfo {
    /// How much width to subtract to wrap for stitching.
    width: i32,
    /// How much height to subtract to wrap for stitching.
    height: i32,
    /// Minimum value of x to wrap.
    wrap_x: i32,
    /// Minimum value of y to wrap.
    wrap_y: i32,
}

fn s_curve(t: f64) -> f64 {
    t * t * (3. - 2. * t)
}

fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

const B_SIZE: usize = 0x100;
const BM: i32 = 0xff;
const PERLIN_N: i32 = 0x1000;
// const NP: i32 = 12; // 2^PerlinN
// const NM: i32 = 0xfff;

pub struct Turbulence {
    lattice_selector: [i32; B_SIZE + B_SIZE + 2],
    gradient: [[[f64; 2]; B_SIZE + B_SIZE + 2]; 4],
}

#[expect(clippy::many_single_char_names, clippy::needless_range_loop)] // for the sake of similarity with the original
impl Turbulence {
    pub fn from_seed(mut seed: i64) -> Self {
        let mut s: f64;
        let mut lattice_selector = [0_i32; B_SIZE + B_SIZE + 2];
        let mut gradient = [[[0.0_f64; 2]; B_SIZE + B_SIZE + 2]; 4];

        seed = setup_seed(seed);
        for k in 0..4 {
            for i in 0..B_SIZE {
                lattice_selector[i] = i as i32;
                for j in 0..2 {
                    seed = random(seed);
                    gradient[k][i][j] =
                        ((seed % (B_SIZE + B_SIZE) as i64) - B_SIZE as i64) as f64 / B_SIZE as f64;
                }
                s = f64::sqrt(
                    gradient[k][i][0] * gradient[k][i][0] + gradient[k][i][1] * gradient[k][i][1],
                );
                gradient[k][i][0] /= s;
                gradient[k][i][1] /= s;
            }
        }
        for i in (1..B_SIZE).rev() {
            let k = lattice_selector[i];
            seed = random(seed);
            let j = (seed % B_SIZE as i64) as usize;
            lattice_selector[i] = lattice_selector[j];
            lattice_selector[j] = k;
        }
        for i in 0..B_SIZE + 2 {
            lattice_selector[B_SIZE + i] = lattice_selector[i];
            for k in 0..4 {
                for j in 0..2 {
                    gradient[k][B_SIZE + i][j] = gradient[k][i][j];
                }
            }
        }

        Turbulence {
            gradient,
            lattice_selector,
        }
    }

    /*
        Kept as the reference "turbulence_channels" is tested against (bit for bit):
        "perlin_noise" no longer calls it.
    */
    #[cfg(test)]
    fn noise2(
        &self,
        color_channel: usize,
        vec: (f64, f64),
        stitch_info: Option<StitchInfo>,
    ) -> f64 {
        let t = vec.0 + PERLIN_N as f64;
        let mut bx0 = t as i32;
        let mut bx1 = bx0 + 1;
        let rx0 = t - (t as i32) as f64;
        let rx1 = rx0 - 1.0;

        let t = vec.1 + PERLIN_N as f64;
        let mut by0 = t as i32;
        let mut by1 = by0 + 1;
        let ry0 = t - (t as i32) as f64;
        let ry1 = ry0 - 1.0;

        // If stitching, adjust lattice points accordingly.
        if let Some(stitch_info) = stitch_info {
            if bx0 >= stitch_info.wrap_x {
                bx0 -= stitch_info.width;
            }
            if bx1 >= stitch_info.wrap_x {
                bx1 -= stitch_info.width;
            }
            if by0 >= stitch_info.wrap_y {
                by0 -= stitch_info.height;
            }
            if by1 >= stitch_info.wrap_y {
                by1 -= stitch_info.height;
            }
        }

        bx0 &= BM;
        bx1 &= BM;
        by0 &= BM;
        by1 &= BM;

        let i = self.lattice_selector[bx0 as usize];
        let j = self.lattice_selector[bx1 as usize];
        let b00 = self.lattice_selector[(i + by0) as usize];
        let b10 = self.lattice_selector[(j + by0) as usize];
        let b01 = self.lattice_selector[(i + by1) as usize];
        let b11 = self.lattice_selector[(j + by1) as usize];

        let sx = s_curve(rx0);
        let sy = s_curve(ry0);

        let q = self.gradient[color_channel][b00 as usize];
        let u = rx0 * q[0] + ry0 * q[1];
        let q = self.gradient[color_channel][b10 as usize];
        let v = rx1 * q[0] + ry0 * q[1];
        let a = lerp(sx, u, v);

        let q = self.gradient[color_channel][b01 as usize];
        let u = rx0 * q[0] + ry1 * q[1];
        let q = self.gradient[color_channel][b11 as usize];
        let v = rx1 * q[0] + ry1 * q[1];
        let b = lerp(sx, u, v);

        lerp(sy, a, b)
    }

    // the reference, see "noise2".
    #[cfg(test)]
    #[expect(clippy::too_many_arguments)]
    pub fn turbulence(
        &self,
        color_channel: usize,
        point: (f64, f64),
        mut base_freq: (f64, f64),
        num_octaves: usize,
        fractal_sum: bool,
        do_stitching: bool,
        tile_pos: (f64, f64),
        tile_size: (f64, f64),
        octave_offsets: &[(f64, f64)],
    ) -> f64 {
        let mut stitch_info: Option<StitchInfo> = None; // Not stitching when None.
        // Adjust the base frequencies if necessary for stitching.
        if do_stitching {
            // When stitching tiled turbulence, the frequencies must be adjusted
            // so that the tile borders will be continuous.
            if base_freq.0 != 0.0 {
                let lo_freq = (tile_size.0 * base_freq.0).floor() / tile_size.0;
                let hi_freq = (tile_size.0 * base_freq.0).ceil() / tile_size.0;
                base_freq.0 = if base_freq.0 / lo_freq < hi_freq / base_freq.0 {
                    lo_freq
                } else {
                    hi_freq
                };
            }
            if base_freq.1 != 0.0 {
                let lo_freq = (tile_size.1 * base_freq.0).floor() / tile_size.1;
                let hi_freq = (tile_size.1 * base_freq.1).ceil() / tile_size.1;
                base_freq.1 = if base_freq.1 / lo_freq < hi_freq / base_freq.1 {
                    lo_freq
                } else {
                    hi_freq
                };
            }
            // Set up initial stitch values.
            let w = (tile_size.0 * base_freq.0 + 0.5) as i32;
            let h = (tile_size.1 * base_freq.1 + 0.5) as i32;
            stitch_info = Some(StitchInfo {
                width: w,
                height: h,
                wrap_x: (tile_pos.0 * base_freq.0) as i32 + PERLIN_N + w,
                wrap_y: (tile_pos.1 * base_freq.1) as i32 + PERLIN_N + h,
            });
        }
        let mut sum = 0.0;
        let mut ratio = 1.0;
        for octave in 0..num_octaves {
            let offset = octave_offsets.get(octave).unwrap();
            let vec = (
                (point.0 + offset.0) * base_freq.0 * ratio,
                (point.1 + offset.1) * base_freq.1 * ratio,
            );
            let noise = self.noise2(color_channel, vec, stitch_info);
            sum += if fractal_sum { noise } else { noise.abs() } / ratio;
            ratio *= 2.0;
            if let Some(ref mut stitch_info) = stitch_info {
                // Update stitch values. Subtracting PerlinN before the multiplication and
                // adding it afterward simplifies to subtracting it once.
                stitch_info.width *= 2;
                stitch_info.wrap_x = 2 * stitch_info.wrap_x - PERLIN_N;
                stitch_info.height *= 2;
                stitch_info.wrap_y = 2 * stitch_info.wrap_y - PERLIN_N;
            }
        }
        sum
    }

    // The half of "noise2" that does not depend on the color channel, verbatim.
    fn lattice(&self, vec: (f64, f64), stitch_info: Option<StitchInfo>) -> Lattice {
        let t = vec.0 + PERLIN_N as f64;
        let mut bx0 = t as i32;
        let mut bx1 = bx0 + 1;
        let rx0 = t - (t as i32) as f64;
        let rx1 = rx0 - 1.0;

        let t = vec.1 + PERLIN_N as f64;
        let mut by0 = t as i32;
        let mut by1 = by0 + 1;
        let ry0 = t - (t as i32) as f64;
        let ry1 = ry0 - 1.0;

        // If stitching, adjust lattice points accordingly.
        if let Some(stitch_info) = stitch_info {
            if bx0 >= stitch_info.wrap_x {
                bx0 -= stitch_info.width;
            }
            if bx1 >= stitch_info.wrap_x {
                bx1 -= stitch_info.width;
            }
            if by0 >= stitch_info.wrap_y {
                by0 -= stitch_info.height;
            }
            if by1 >= stitch_info.wrap_y {
                by1 -= stitch_info.height;
            }
        }

        bx0 &= BM;
        bx1 &= BM;
        by0 &= BM;
        by1 &= BM;

        let i = self.lattice_selector[bx0 as usize];
        let j = self.lattice_selector[bx1 as usize];
        Lattice {
            b00: self.lattice_selector[(i + by0) as usize] as usize,
            b10: self.lattice_selector[(j + by0) as usize] as usize,
            b01: self.lattice_selector[(i + by1) as usize] as usize,
            b11: self.lattice_selector[(j + by1) as usize] as usize,
            rx0,
            rx1,
            ry0,
            ry1,
            sx: s_curve(rx0),
            sy: s_curve(ry0),
        }
    }

    // The other half of "noise2", for one color channel, verbatim.
    fn noise_at(&self, color_channel: usize, l: &Lattice) -> f64 {
        let q = self.gradient[color_channel][l.b00];
        let u = l.rx0 * q[0] + l.ry0 * q[1];
        let q = self.gradient[color_channel][l.b10];
        let v = l.rx1 * q[0] + l.ry0 * q[1];
        let a = lerp(l.sx, u, v);

        let q = self.gradient[color_channel][l.b01];
        let u = l.rx0 * q[0] + l.ry1 * q[1];
        let q = self.gradient[color_channel][l.b11];
        let v = l.rx1 * q[0] + l.ry1 * q[1];
        let b = lerp(l.sx, u, v);

        lerp(l.sy, a, b)
    }

    /*
        "turbulence" for up to four color channels of one point, into "out[..channels.len()]", with the same numbers bit for bit (tested below).
        "turbulence" redid the stitching setup for every pixel and every channel, and the lattice for every channel, though neither depends on the channel:
        Happy Wheels' two "perlinNoise" calls (stage size, 10 octaves, gray and alpha) took 1.1 s each on the Switch.
    */
    #[allow(clippy::too_many_arguments)]
    pub fn turbulence_channels(&self, channels: &[usize], point: (f64, f64), setup: &TurbulenceSetup, num_octaves: usize, fractal_sum: bool, octave_offsets: &[(f64, f64)], out: &mut [f64; 4], ) {
        let base_freq = setup.base_freq;
        let mut stitch_info = setup.stitch_info;
        let mut sums = [0.0; 4];
        let mut ratio = 1.0;
        for octave in 0..num_octaves {
            let offset = octave_offsets.get(octave).unwrap();
            let vec = ((point.0 + offset.0) * base_freq.0 * ratio, (point.1 + offset.1) * base_freq.1 * ratio, );
            let lattice = self.lattice(vec, stitch_info);
            for (sum, &channel) in sums.iter_mut().zip(channels) {
                let noise = self.noise_at(channel, &lattice);
                *sum += if fractal_sum { noise } else { noise.abs() } / ratio;
            }
            ratio *= 2.0;
            if let Some(ref mut stitch_info) = stitch_info {
                stitch_info.width *= 2;
                stitch_info.wrap_x = 2 * stitch_info.wrap_x - PERLIN_N;
                stitch_info.height *= 2;
                stitch_info.wrap_y = 2 * stitch_info.wrap_y - PERLIN_N;
            }
        }
        out[..channels.len()].copy_from_slice(&sums[..channels.len()]);
    }
}

// see "Turbulence::lattice".
struct Lattice {
    b00: usize,
    b10: usize,
    b01: usize,
    b11: usize,
    rx0: f64,
    rx1: f64,
    ry0: f64,
    ry1: f64,
    sx: f64,
    sy: f64,
}

/*
    The stitching setup of "turbulence", done once per call to "perlinNoise" instead of per pixel: it only depends on the call's parameters.
    The body is turbulence's, verbatim.
*/
#[derive(Copy, Clone)]
pub struct TurbulenceSetup {
    base_freq: (f64, f64),
    stitch_info: Option<StitchInfo>,
}

impl TurbulenceSetup {
    pub fn new(mut base_freq: (f64, f64), do_stitching: bool, tile_pos: (f64, f64), tile_size: (f64, f64), ) -> Self {
        let mut stitch_info: Option<StitchInfo> = None; // Not stitching when None.
        // Adjust the base frequencies if necessary for stitching.
        if do_stitching {
            // When stitching tiled turbulence, the frequencies must be adjusted so that the tile borders will be continuous.
            if base_freq.0 != 0.0 {
                let lo_freq = (tile_size.0 * base_freq.0).floor() / tile_size.0;
                let hi_freq = (tile_size.0 * base_freq.0).ceil() / tile_size.0;
                base_freq.0 = if base_freq.0 / lo_freq < hi_freq / base_freq.0 {
                    lo_freq
                } else {
                    hi_freq
                };
            }
            if base_freq.1 != 0.0 {
                let lo_freq = (tile_size.1 * base_freq.0).floor() / tile_size.1;
                let hi_freq = (tile_size.1 * base_freq.1).ceil() / tile_size.1;
                base_freq.1 = if base_freq.1 / lo_freq < hi_freq / base_freq.1 {
                    lo_freq
                } else {
                    hi_freq
                };
            }
            // Set up initial stitch values.
            let w = (tile_size.0 * base_freq.0 + 0.5) as i32;
            let h = (tile_size.1 * base_freq.1 + 0.5) as i32;
            stitch_info = Some(StitchInfo {
                width: w,
                height: h,
                wrap_x: (tile_pos.0 * base_freq.0) as i32 + PERLIN_N + w,
                wrap_y: (tile_pos.1 * base_freq.1) as i32 + PERLIN_N + h,
            });
        }
        TurbulenceSetup {
            base_freq,
            stitch_info,
        }
    }
}

#[cfg(test)]
mod flashnx_tests {
    use super::*;

    // "turbulence_channels" against "turbulence", bit for bit, over seeds, sizes, frequencies (zero, below and above one), octave counts, offsets, stitching and fractal sums, and every channel subset "perlin_noise" uses.
    #[test]
    fn turbulence_channels_matches_turbulence() {
        let mut r: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            r
        };
        let mut checked = 0u64;
        for case in 0..600 {
            let seed = (next() % 200_000) as i64 - 100_000;
            let turb = Turbulence::from_seed(seed);
            let w = (1 + next() % 400) as f64;
            let h = (1 + next() % 400) as f64;
            let mut base = || match next() % 6 {
                0 => 0.0,
                1 => 1.0,
                2 => w,
                3 => (next() % 100_000) as f64 / 1000.0,
                4 => 0.37,
                _ => (next() % 900) as f64 + 0.5,
            };
            let (bx, by) = (base(), base());
            let base_freq = (
                if bx == 0.0 { 0.0 } else { 1.0 / bx },
                if by == 0.0 { 0.0 } else { 1.0 / by },
            );
            let octaves = (next() % 11) as usize;
            let stitch = next() % 2 == 0;
            let fractal = next() % 2 == 0;
            let offsets: Vec<(f64, f64)> = (0..octaves).map(|_| {((next() % 2_000_000) as f64 / 1000.0 - 1000.0, (next() % 2_000_000) as f64 / 1000.0 - 1000.0, )}).collect();
            let n = 1 + (case % 4);
            let channels: Vec<usize> = (0..n).collect();
            let setup = TurbulenceSetup::new(base_freq, stitch, (0.0, 0.0), (w, h));
            for _ in 0..40 {
                let p = ((next() % w as u64) as f64, (next() % h as u64) as f64);
                let mut out = [0.0; 4];
                turb.turbulence_channels(&channels, p, &setup, octaves, fractal, &offsets, &mut out);
                for (k, &c) in channels.iter().enumerate() {
                    let want = turb.turbulence(c, p, base_freq, octaves, fractal, stitch, (0.0, 0.0), (w, h), &offsets, );
                    assert_eq!(want.to_bits(), out[k].to_bits(), "case {case} channel {c} point {p:?}: {want} vs {}", out[k]);
                    checked += 1;
                }
            }
        }
        assert!(checked > 50_000);
    }
}
