/*
    AVM2 opcode census for the measurement build ("flashnx_instr").

    Fireboy & Watergirl 2 spends 84 ms of a 90 ms frame in ActionScript,
    and the sampling profiler puts the time in the interpreter's machinery
    (dispatch, stack traffic, calls) rather than in any one opcode.
    
    What it cannot say is how MANY opcodes run,
    which pairs come back most often (the candidates for fused opcodes, like avmplus' "peephole.tbl"),
    and how small the called methods are (the candidates for inlining). This counts all three.

    Single-threaded by construction: AVM2 only runs on the Ruffle worker.
*/

const N: usize = 256;

static mut COUNTS: [u64; N] = [0; N];
static mut PAIRS: [u32; N * N] = [0; N * N];
static mut PREV: usize = 0;
static mut NAMES: [Option<&'static str>; N] = [None; N];

/// Bytecode calls by size of the callee's optimized body, in ops: <=4, <=8, <=16, <=32, <=64, more.
static mut BODY: [u64; 6] = [0; 6];
static mut NATIVE_CALLS: u64 = 0;

// "i" is the op's discriminant: "op" is only formatted the first time.
#[inline(always)]
pub fn count(i: usize, op: &dyn core::fmt::Debug) {
    let i = i & (N - 1);
    unsafe {
        let counts = &raw mut COUNTS;
        if (*counts)[i] == 0 {
            name(i, op);
        }
        (*counts)[i] += 1;
        let pairs = &raw mut PAIRS;
        let p = (&raw mut PREV).read() * N + i;
        (*pairs)[p] = (*pairs)[p].wrapping_add(1);
        (&raw mut PREV).write(i);
    }
}

#[cold]
#[inline(never)]
fn name(i: usize, op: &dyn core::fmt::Debug) {
    if unsafe { (*(&raw const NAMES))[i].is_some() } {
        return;
    }
    let full = format!("{op:?}");
    let end = full.find([' ', '{', '(']).unwrap_or(full.len());
    let short: &'static str = Box::leak(full[..end].to_string().into_boxed_str());
    unsafe { (*(&raw mut NAMES))[i] = Some(short) };
}

// One bytecode method entered, with "ops" ops in its optimized body.
#[inline(always)]
pub fn bytecode_call(ops: usize) {
    let b = match ops {
        0..=4 => 0,
        5..=8 => 1,
        9..=16 => 2,
        17..=32 => 3,
        33..=64 => 4,
        _ => 5,
    };
    unsafe { (*(&raw mut BODY))[b] += 1 };
}

#[inline(always)]
pub fn native_call() {
    unsafe { *(&raw mut NATIVE_CALLS) += 1 };
}

// Totals since the last report over "frames" frames, then reset.
pub fn report(frames: u64) -> String {
    let frames = frames.max(1);
    unsafe {
        let counts = &mut *(&raw mut COUNTS);
        let pairs = &mut *(&raw mut PAIRS);
        let names = &*(&raw const NAMES);
        let body = &mut *(&raw mut BODY);
        let native = &mut *(&raw mut NATIVE_CALLS);
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return String::new();
        }
        let nm = |i: usize| names[i].unwrap_or("?");
        let calls: u64 = body.iter().sum();
        let mut out = format!(
            "avm2ops: {} ops ({}/frame), {} bytecode calls ({}/frame, one per {} ops), {} native calls ({}/frame)\n",
            total,
            total / frames,
            calls,
            calls / frames,
            total / calls.max(1),
            *native,
            *native / frames
        );
        out.push_str(&format!(
            "avm2calls: callee body <=4 ops {:.1}%, <=8 {:.1}%, <=16 {:.1}%, <=32 {:.1}%, <=64 {:.1}%, more {:.1}%\n",
            body[0] as f64 * 100.0 / calls.max(1) as f64,
            body[1] as f64 * 100.0 / calls.max(1) as f64,
            body[2] as f64 * 100.0 / calls.max(1) as f64,
            body[3] as f64 * 100.0 / calls.max(1) as f64,
            body[4] as f64 * 100.0 / calls.max(1) as f64,
            body[5] as f64 * 100.0 / calls.max(1) as f64,
        ));
        let mut order: Vec<usize> = (0..N).filter(|&i| counts[i] > 0).collect();
        order.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
        for &i in order.iter().take(40) {
            out.push_str(&format!(
                "avm2op {:<22} n={:<11} share={:.2}%\n",
                nm(i),
                counts[i],
                counts[i] as f64 * 100.0 / total as f64
            ));
        }
        let mut top: Vec<(u32, usize)> = pairs.iter().enumerate().filter(|&(_, &n)| n > 0).map(|(p, &n)| (n, p)).collect();
        top.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        for &(n, p) in top.iter().take(50) {
            out.push_str(&format!(
                "avm2pair {:>22} > {:<22} n={:<11} share={:.2}%\n",
                nm(p / N),
                nm(p % N),
                n,
                n as f64 * 100.0 / total as f64
            ));
        }
        counts.iter_mut().for_each(|c| *c = 0);
        pairs.iter_mut().for_each(|c| *c = 0);
        body.iter_mut().for_each(|c| *c = 0);
        *native = 0;
        out
    }
}
