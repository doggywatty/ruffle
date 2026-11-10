
/*
    Per-method AVM2 profiler.

    The sampling profiler sees the interpreter ("run_actions", "op_call_prop_void")
    but never WHICH ActionScript function it is running.
    
    Super Smash Flash 2 spends 40-50 ms of a 60 ms fight frame inside its own code
    
    (its whole engine runs from "Event.RENDER"),
    and the first fight runs at 45-60 fps while the next ones run at 16-18:
    the question is which of the game's functions grow.

    Every call that goes through "function::exec" (bytecode and native methods) is timed with the CPU's counter:
    exclusive time (own code, callees removed), inclusive time, call count. "report()" prints the top entries and resets.

    Off unless "sdmc:/switch/FlashNX/as3prof.on" exists at game launch: when off,
    the cost is one relaxed atomic load per call.
*/

use fnv::FnvHashMap;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_as3_profiler(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    STATE.with(|s| {
        if let Ok(mut s) = s.try_borrow_mut() {
            *s = State::default();
        }
    });
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn now() -> u64 {
    let t: u64;
    unsafe { std::arch::asm!("mrs {}, cntpct_el0", out(reg) t, options(nomem, nostack)) };
    t
}

#[cfg(target_arch = "aarch64")]
fn ticks_per_us() -> f64 {
    let f: u64;
    unsafe { std::arch::asm!("mrs {}, cntfrq_el0", out(reg) f, options(nomem, nostack)) };
    f as f64 / 1_000_000.0
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn now() -> u64 {
    use std::sync::OnceLock;
    static T0: OnceLock<std::time::Instant> = OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

#[cfg(not(target_arch = "aarch64"))]
fn ticks_per_us() -> f64 {
    1000.0
}

struct Entry {
    name: String,
    native: bool,
    calls: u64,
    excl: u64,
    incl: u64,
}

struct Frame {
    entry: u32,
    start: u64,
    child: u64,
}

#[derive(Default)]
struct State {
    index: FnvHashMap<usize, u32>,
    entries: Vec<Entry>,
    stack: Vec<Frame>,
    /// Time spent in outermost calls only (no double counting).
    top_level: u64,
    calls: u64,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

// Ends the timing of one call when dropped (also on early "return Err").
pub struct Guard {
    active: bool,
}

impl Guard {
    // "key" identifies the method (its "Gc" address): "name" is only built the first time a method is seen.
    #[inline]
    pub fn enter(key: usize, native: bool, name: impl FnOnce() -> String) -> Guard {
        if !enabled() {
            return Guard { active: false };
        }
        let active = STATE.with(|s| {
            let Ok(mut s) = s.try_borrow_mut() else {
                return false;
            };
            let s = &mut *s;
            let entry = match s.index.get(&key) {
                Some(&i) => i,
                None => {
                    let i = s.entries.len() as u32;
                    s.entries.push(Entry {
                        name: name(),
                        native,
                        calls: 0,
                        excl: 0,
                        incl: 0,
                    });
                    s.index.insert(key, i);
                    i
                }
            };
            s.stack.push(Frame {
                entry,
                start: now(),
                child: 0,
            });
            true
        });
        Guard { active }
    }
}

impl Drop for Guard {
    #[inline]
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let t = now();
        STATE.with(|s| {
            let Ok(mut s) = s.try_borrow_mut() else {
                return;
            };
            let s = &mut *s;
            let Some(f) = s.stack.pop() else {
                return;
            };
            let elapsed = t.saturating_sub(f.start);
            let e = &mut s.entries[f.entry as usize];
            e.calls += 1;
            e.incl += elapsed;
            e.excl += elapsed.saturating_sub(f.child);
            s.calls += 1;
            match s.stack.last_mut() {
                Some(parent) => parent.child += elapsed,
                None => s.top_level += elapsed,
            }
        });
    }
}

// Top "n" methods by exclusive time since the last report, then reset the counters (names are kept). Empty when the profiler is off or idle.
pub fn report(n: usize) -> String {
    if !enabled() {
        return String::new();
    }
    STATE.with(|s| {
        let Ok(mut s) = s.try_borrow_mut() else {
            return String::new();
        };
        let s = &mut *s;
        if s.calls == 0 {
            return String::new();
        }
        let tpu = ticks_per_us();
        let ms = |t: u64| t as f64 / tpu / 1000.0;
        let mut order: Vec<usize> = (0..s.entries.len())
            .filter(|&i| s.entries[i].calls > 0)
            .collect();
        order.sort_by(|&a, &b| s.entries[b].excl.cmp(&s.entries[a].excl));
        let mut out = format!(
            "as3prof: {} calls, {:.0} ms in AS3 (outermost calls), {} methods seen\n",
            s.calls,
            ms(s.top_level),
            order.len()
        );
        for &i in order.iter().take(n) {
            let e = &s.entries[i];
            out.push_str(&format!(
                "as3prof:   excl {:7.1} ms  incl {:7.1} ms  calls {:8}  {:6.2} us/call  {}{}\n",
                ms(e.excl),
                ms(e.incl),
                e.calls,
                ms(e.excl) * 1000.0 / e.calls as f64,
                if e.native { "[native] " } else { "" },
                e.name
            ));
        }
        for e in &mut s.entries {
            e.calls = 0;
            e.excl = 0;
            e.incl = 0;
        }
        s.top_level = 0;
        s.calls = 0;
        out
    })
}
