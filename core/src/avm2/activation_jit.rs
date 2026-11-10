/*
	A baseline JIT for AVM2 bytecode (AArch64).

	Fireboy & Watergirl 2 runs 2.65 million AVM2 ops per frame at ~57 CPU cycles each:
	the interpreter loop pays one shared, badly predicted indirect branch per op,
	and reloads and stores its stack pointer from memory on every push and pop.
	Flash Player compiled ActionScript to machine code instead.

	This is the simplest compiler that removes both costs, the "baseline JIT" of the literature (template / copy-and-patch JITs):

	- The stack depth of every op is known before running (the verifier guarantees it, we recompute it), so each stack slot is a fixed offset from the frame base kept in x19: no stack pointer at all in compiled code.
	- Locals, constants, "Pop", "Dup", "Swap" and branches are a few inline instructions ("ldr q0" / "str q0" of the 16-byte "Value").
	- Every other op calls "h_op::<K>", one instantiation per op kind: it sets the interpreter's stack pointer, runs the very same "op_*" handler as the interpreter, and checks the depth it left against the one we computed.
	  Behaviour is the interpreter's by construction, and a wrong entry in "effect" is caught at the first run: the method goes back to the interpreter at the next op, and is never compiled again.

	Some ops also have an inline fast path, because their helper was the cost:
	"getslot" on an object (13 % of Fireboy's frame through its helper),
	"iftrue" / "iffalse" on a "Bool", and arithmetic, comparisons and int
	coercions when both operands are "Number"s or both "int"s (the exact esults of the interpreter's own fast paths, overflow and NaN included).
	They read Ruffle's data in place, so the layout they rely on is measured on real values first ("Layout"): a type of object whose layout was not checked takes the helper.

	"callmethod" from compiled code goes straight into the callee's compiled code when that is safe
	("fast_call", the way avmplus' JIT calls an early-bound method through its "_implGPR"):
	callee compiled, no "arguments" / "...rest", as many arguments as parameters, and arguments that already have their parameter's type (or only need "int" -> "Number" or "undefined" -> "null").
	It skips the generic "call_method_with_args" -> "exec" -> "init_from_method" path, which re-checks and re-coerces everything on each of Fireboy's 42 000 calls a frame.

	"getproperty" with an int index on a dense "Array" has its own helper (no dynamic dispatch, no borrow counting):
	"setslot" without coercion is inline when the object is not black for the GC (a black object takes the helper, whose "Gc::write" runs gc-arena's barrier).

	Methods with exception handlers, "lookupswitch" or unknown ops stay interpreted.
	Code memory comes from libnx's "jitCreate": CodeMemory, which hbloader allows on Mesosphère (see cpp/src/flashnx_jit.c)
*/

use super::*;
use crate::avm2::object::{Object, ScriptObjectData};
use fnv::FnvHashMap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
#[cfg(feature = "flashnx_instr")]
use std::sync::atomic::AtomicU64;

unsafe extern "C" {
    fn flashnx_jit_init(size: usize, rw: *mut *mut u8, rx: *mut *mut u8) -> u32;
    fn flashnx_jit_flush(rw: *mut u8, rx: *mut u8, len: usize);
}

const REJECTED: usize = usize::MAX;

// Compile on the second call: one-shot initializers stay interpreted.
const THRESHOLD: usize = 2;

// Code memory, taken once from the heap and reused by every game ("reset").
// Super Smash Flash 2, the biggest seen, compiles ~6 MB of it.
const CODE_SIZE: usize = 16 << 20;

// Helper table: 0..KINDS = "h_op::<K>", then the special helpers.
const KINDS: usize = 160;

// lookupswitch's index ("h_switch").
const H_SWITCH: usize = 247;

const H_ARRAY_GET: usize = 248;
const H_CALL: usize = 249;
const H_GENERIC: usize = 250;
const H_TRUTHY: usize = 251;
const H_TIMEOUT: usize = 252;
const H_RETURN: usize = 253;

static ENABLED: AtomicBool = AtomicBool::new(true);

// "callmethod" straight into compiled code ("fast_call"), the profiling build's A/B switches it.
static DIRECT_CALLS: AtomicBool = AtomicBool::new(true);

pub fn set_direct_calls(on: bool) {
    DIRECT_CALLS.store(on, AtomicOrdering::Relaxed);
}

// The one-entry stack cache of compiled code ("Tos"), off from the "jittos.off" marker.
static TOS_CACHE: AtomicBool = AtomicBool::new(true);

// Profiling build: every method is compiled twice, the second time without the stack cache, so the A/B windows can run either ("ALT").
static DUAL: AtomicBool = AtomicBool::new(false);

// Run the second compilation where there is one.
static ALT: AtomicBool = AtomicBool::new(false);

pub fn set_tos_cache(on: bool) {
    TOS_CACHE.store(on, AtomicOrdering::Relaxed);
}

pub fn set_dual(on: bool) {
    DUAL.store(on, AtomicOrdering::Relaxed);
}

pub fn set_alt(on: bool) {
    ALT.store(on, AtomicOrdering::Relaxed);
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, AtomicOrdering::Relaxed);
}

#[inline(always)]
pub fn enabled() -> bool {
    ENABLED.load(AtomicOrdering::Relaxed)
}

type Act = Activation<'static, 'static>;

/*
	What the compiled code and the helpers share.
	Only "frame" and "helpers" are read by machine code (offsets 0 and 8).
*/
#[repr(C)]
struct Ctx {
    frame: *mut Value<'static>,
    helpers: *const usize,
    act: *mut Act,
    ops: *const Op<'static>,
    num_locals: usize,
    result: Option<Result<Value<'static>, Error<'static>>>,
    resume_ip: usize,
}

pub(super) enum Outcome<'gc> {
    Done(Result<Value<'gc>, Error<'gc>>),
    Resume(usize),
    Interpret,
}

struct State {
    init: i32,
    init_rc: u32,
    rw: *mut u8,
    rx: *mut u8,
    used: usize,
    helpers: [usize; 256],
    compiled: u64,
    rejected: u64,
    reasons: FnvHashMap<String, u64>,
    deopts: u64,
    deopt_ops: FnvHashMap<String, u64>,
    runs: u64,
	
    // "callmethod" from compiled code: entered directly / through "exec".
    calls_direct: u64,
    calls_slow: u64,

    layout: Layout,

    /*
		Where each compiled op's code starts (address, op name), kept only hen "set_map" asked for it:
		the profiling build's attribution of samples in compiled code to the ops that produced it.
	*/
    map: Vec<(usize, String)>,

    // The name of each op kind met while compiling, for "HELPER_N".
    #[cfg(feature = "flashnx_instr")]
    kind_names: FnvHashMap<usize, String>,
}

/*
	Measuring builds: how many times compiled code handed each op kind to its helper ("h_op"),
	that is ran it through the interpreter's handler rather than as native code, since the last "report".

	Read with what a helper costs: ~15-20 ns, not the 150-250 ns guessed first.
	3D Mario Racing ran ~230 000 helpers a frame:
	making mixed int/Number arithmetic, Divide and Negate native removed ~57 000 of them and saved 1-3 ms of a 77 ms frame, in the noise, so it was taken out again.
	Its time is the sheer volume of ops and their memory traffic.
*/
#[cfg(feature = "flashnx_instr")]
static HELPER_N: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];

// Keep the op address map ("map"), off unless the profiling build asks.
static MAP: AtomicBool = AtomicBool::new(false);

pub fn set_map(on: bool) {
    MAP.store(on, AtomicOrdering::Relaxed);
}

// The op address map of the code compiled since the last "reset", one "<address hex> <op>" line per op, in address order.
pub fn map_text() -> String {
    let s = state();
    let mut out = String::with_capacity(s.map.len() * 24);
    for (addr, name) in &s.map {
        out.push_str(&format!("{addr:x} {name}
"));
    }
    out
}

// What the inline fast paths assume about Ruffle's data, measured.
#[derive(Clone, Default)]
struct Layout {
    // "Value::Bool": tag in byte 0, the bool in byte "bool_off".
    bool_ok: bool,
    bool_tag: u8,
    bool_off: usize,

    // First 8 bytes of a "Value::Bool" (the tag), when "bool_off" >= 8.
    bool_word: u64,

	/*
		"Value::Number": tag in byte 0, the f64 at "num_off"; "Value::Integer":
		tag in byte 0, the i32 at "int_off".
	*/
    num_ok: bool,
    num_tag: u8,
    num_off: usize,

    int_tag: u8,
    int_off: usize,

    // "Value::Object": tags of the object types checked so far, and where the slots' pointer and length are from the "Gc" box pointer at byte 8.
    obj_tags: Vec<u8>,
    slots_ptr_off: usize,
    slots_len_off: usize,
    probes: u32,
    obj_fail: &'static str,

    // Offset of "GcBoxHeader::tagged_vtable" from the box (its 2 low bits are the GC color, 3 = black), measured on two fresh arrays.
    header_ok: bool,
    header_probed: bool,
    vtable_word: usize,
}

static mut STATE: Option<State> = None;

fn state() -> &'static mut State {
    // Single-threaded: AVM2 only runs on the Ruffle worker.
    unsafe {
        let s = &mut *(&raw mut STATE);
        s.get_or_insert_with(|| State {
            init: 0,
            init_rc: 0,
            rw: std::ptr::null_mut(),
            rx: std::ptr::null_mut(),
            used: 0,
            helpers: [0; 256],
            compiled: 0,
            rejected: 0,
            reasons: FnvHashMap::default(),
            deopts: 0,
            deopt_ops: FnvHashMap::default(),
            runs: 0,
            calls_direct: 0,
            calls_slow: 0,
            layout: Layout::default(),
            map: Vec::new(),
            #[cfg(feature = "flashnx_instr")]
            kind_names: FnvHashMap::default(),
        })
    }
}

fn op_name(op: &Op<'_>) -> String {
    let full = format!("{op:?}");
    let end = full.find([' ', '{', '(']).unwrap_or(full.len());
    full[..end].to_string()
}

#[inline(always)]
fn kind(op: &Op<'_>) -> usize {
    core::intrinsics::discriminant_value(op) as usize
}

fn init(s: &mut State) {
    if s.init != 0 {
        return;
    }
    s.init = -1;
    if std::mem::size_of::<Value<'static>>() != 16 {
        s.init_rc = 0xBAD1;
        return;
    }
    let (mut rw, mut rx) = (std::ptr::null_mut(), std::ptr::null_mut());
    let rc = unsafe { flashnx_jit_init(CODE_SIZE, &mut rw, &mut rx) };
    s.init_rc = rc;
    if rc != 0 || rw.is_null() || rx.is_null() {
        return;
    }
    s.rw = rw;
    s.rx = rx;
    fill_helpers(&mut s.helpers);
    probe_bool(&mut s.layout);
    probe_numbers(&mut s.layout);
    s.init = 1;
}

fn bytes8(v: Value<'_>) -> [u8; 16] {
    let w = bytes_of(v);
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&w[0].to_le_bytes());
    b[8..].copy_from_slice(&w[1].to_le_bytes());
    b
}

fn primitive_tags() -> [u8; 5] {
    [
        bytes8(Value::Undefined)[0],
        bytes8(Value::Null)[0],
        bytes8(Value::Integer(0))[0],
        bytes8(Value::Number(0.0))[0],
        bytes8(Value::Bool(false))[0],
    ]
}

fn probe_bool(l: &mut Layout) {
    // The bytes around a payload are padding, with whatever was there: only the candidate places of the bool are compared (on Switch and on PC, Ruffle's "Value" keeps every payload at byte 8).
    let f = bytes8(Value::Bool(false));
    let t = bytes8(Value::Bool(true));
    let others = [bytes8(Value::Undefined)[0], bytes8(Value::Null)[0], bytes8(Value::Integer(0))[0], bytes8(Value::Number(0.0))[0]];
    if f[0] != t[0] || others.contains(&f[0]) {
        return;
    }
    let hits: Vec<usize> = [8usize, 1, 4].into_iter().filter(|&o| f[o] == 0 && t[o] == 1).collect();
    if hits.len() != 1 {
        return;
    }
    l.bool_tag = f[0];
    l.bool_off = hits[0];
    l.bool_ok = true;

    // A bool written by the JIT: this word (the tag), then the byte.
    l.bool_word = if l.bool_off >= 8 { f[0] as u64 } else { 0 };
}

fn probe_numbers(l: &mut Layout) {
    let n = bytes8(Value::Number(1.5));
    let i = bytes8(Value::Integer(0x1234_5678));
    let others = [bytes8(Value::Undefined)[0], bytes8(Value::Null)[0], bytes8(Value::Bool(false))[0]];
    if n[0] == i[0] || others.contains(&n[0]) || others.contains(&i[0]) {
        return;
    }
    let bits = 1.5f64.to_bits().to_le_bytes();
    let Some(num_off) = (8..=8).step_by(8).find(|&o| n[o..o + 8] == bits) else { return };
    let ib = 0x1234_5678i32.to_le_bytes();
    let Some(int_off) = (4..=12).step_by(4).find(|&o| i[o..o + 4] == ib) else { return };
    l.num_tag = n[0];
    l.num_off = num_off;
    l.int_tag = i[0];
    l.int_off = int_off;
    l.num_ok = l.bool_ok && l.bool_off >= 8;
}

/*
	Where the GC color lives in a box header:
	2 fresh objects of the same type share their (tagged) vtable pointer, not their "next" pointer, and gc-arena links each new box in front of the previous one ("allocate": "header.set_next(self.all)"), so the second one's "next" IS the first box.
	Anything else and "setslot" keeps its helper (and gc-arena's barrier).
*/
fn probe_header(l: &mut Layout, act: &mut Activation<'_, '_>) {
    l.header_probed = true;
    let a1 = ArrayObject::empty(act.context);
    let a2 = ArrayObject::empty(act.context);
    let b1 = bytes_of(Value::Object(a1.into()))[1] as usize;
    let b2 = bytes_of(Value::Object(a2.into()))[1] as usize;
    if b1 == 0 || b2 == 0 || b1 % 8 != 0 || b2 % 8 != 0 {
        return;
    }
    let w = |b: usize, k: usize| unsafe { *((b + 8 * k) as *const usize) };
    let mut hit = None;
    for k in 0..2 {
        let (x, y) = (w(b1, k) & !0xF, w(b2, k) & !0xF);
        let other = 1 - k;
        if x != 0 && x == y && w(b1, other) != w(b2, other) {
            if hit.is_some() {
                return;
            }
            hit = Some(k);
        }
    }
    if let Some(k) = hit
        && w(b2, 1 - k) == b1
    {
        l.vtable_word = 8 * k;
        l.header_ok = true;
    }
}

// Learn the layout of one more type of object from a live one: Ruffle's own accessors ("gc_base", the slots) against the raw bytes the JIT would read.
fn probe_object(l: &mut Layout, v: Value<'_>) {
    l.probes += 1;
    let Value::Object(o) = v else { return };
    let raw = bytes_of(v);
    let tag = (raw[0] & 0xff) as u8;
    if l.obj_tags.contains(&tag) {
        return;
    }
    if primitive_tags().contains(&tag) {
        l.obj_fail = "tag";
        return;
    }
    if std::mem::size_of::<gc_arena::lock::Lock<Value<'static>>>() != 16 {
        l.obj_fail = "slot size";
        return;
    }
    let box_ptr = raw[1] as usize;
    let base = o.gc_base();
    let value = Gc::as_ptr(base) as usize;
    if value < box_ptr || value - box_ptr > 64 {
        l.obj_fail = "box";
        return;
    }
    let (sp, sl) = base.flashnx_slots();
    if sl == 0 || sp == sl {
        return; // cannot tell the pointer from the length on this one
    }
    let field = value - box_ptr + ScriptObjectData::FLASHNX_SLOTS_OFFSET;
    let words = unsafe { *((box_ptr + field) as *const [usize; 2]) };
    let (ptr_off, len_off) = if words == [sp, sl] {
        (field, field + 8)
    } else if words == [sl, sp] {
        (field + 8, field)
    } else {
        l.obj_fail = "slots";
        return;
    };
    if ptr_off % 8 != 0 || ptr_off > 32000 {
        l.obj_fail = "offset";
        return;
    }
    if !l.obj_tags.is_empty() && (l.slots_ptr_off != ptr_off || l.slots_len_off != len_off) {
        l.obj_fail = "mismatch";
        return;
    }
    l.slots_ptr_off = ptr_off;
    l.slots_len_off = len_off;
    if l.obj_tags.len() < 4 {
        l.obj_tags.push(tag);
    }
}

// ---------------------------------------------------------------- assembler

#[derive(Clone, Copy)]
enum Target {
    Op(usize),
    Exit,

    // An instruction index of this method's code.
    Code(usize),
}

#[derive(Clone, Copy)]
enum Fix {
    B,

    // cbz (false) / cbnz (true) on w<rt>.
    Cb(bool, u32),
    BCond(u32),
}

const EQ: u32 = 0;
const NE: u32 = 1;
const MI: u32 = 4;
const VS: u32 = 6;
const LS: u32 = 9;
const GE: u32 = 10;
const LT: u32 = 11;
const GT: u32 = 12;
const LE: u32 = 13;

struct Asm {
    code: Vec<u32>,
    fixups: Vec<(usize, Target, Fix)>,
}

const X0: u32 = 0;
const X16: u32 = 16;
const X19: u32 = 19;
const X20: u32 = 20;
const X21: u32 = 21;
const XZR: u32 = 31;
const SP: u32 = 31;

impl Asm {
    fn e(&mut self, i: u32) {
        self.code.push(i);
    }

    // ldr q<rt>, [x<rn>, #off] (off multiple of 16, < 65536)
    fn ldr_q(&mut self, rt: u32, rn: u32, off: usize) {
        self.e(0x3DC0_0000 | ((off / 16) as u32) << 10 | rn << 5 | rt);
    }

    // str q<rt>, [x<rn>, #off]
    fn str_q(&mut self, rt: u32, rn: u32, off: usize) {
        self.e(0x3D80_0000 | ((off / 16) as u32) << 10 | rn << 5 | rt);
    }

    // str x<rt>, [x<rn>, #off] (off multiple of 8, < 32768)
    fn str_x(&mut self, rt: u32, rn: u32, off: usize) {
        self.e(0xF900_0000 | ((off / 8) as u32) << 10 | rn << 5 | rt);
    }

    // ldr x<rt>, [x<rn>, #off]
    fn ldr_x(&mut self, rt: u32, rn: u32, off: usize) {
        self.e(0xF940_0000 | ((off / 8) as u32) << 10 | rn << 5 | rt);
    }

    // ldrb w<rt>, [x<rn>, #off] (off < 4096)
    fn ldrb(&mut self, rt: u32, rn: u32, off: usize) {
        self.e(0x3940_0000 | (off as u32) << 10 | rn << 5 | rt);
    }

    // cmp w<rn>, #imm (imm < 4096)
    fn cmp_w(&mut self, rn: u32, imm: u32) {
        self.e(0x7100_001F | imm << 10 | rn << 5);
    }

    // cmp x<rn>, #imm (imm < 4096)
    fn cmp_x(&mut self, rn: u32, imm: u32) {
        self.e(0xF100_001F | imm << 10 | rn << 5);
    }

    // ldr d<rt>, [x<rn>, #off] (off multiple of 8)
    fn ldr_d(&mut self, rt: u32, rn: u32, off: usize) {
        self.e(0xFD40_0000 | ((off / 8) as u32) << 10 | rn << 5 | rt);
    }

    fn str_d(&mut self, rt: u32, rn: u32, off: usize) {
        self.e(0xFD00_0000 | ((off / 8) as u32) << 10 | rn << 5 | rt);
    }

    // ldr w<rt>, [x<rn>, #off] (off multiple of 4)
    fn ldr_w(&mut self, rt: u32, rn: u32, off: usize) {
        self.e(0xB940_0000 | ((off / 4) as u32) << 10 | rn << 5 | rt);
    }
    fn str_w(&mut self, rt: u32, rn: u32, off: usize) {
        self.e(0xB900_0000 | ((off / 4) as u32) << 10 | rn << 5 | rt);
    }

    // strb w<rt>, [x<rn>, #off] (off < 4096)
    fn strb(&mut self, rt: u32, rn: u32, off: usize) {
        self.e(0x3900_0000 | (off as u32) << 10 | rn << 5 | rt);
    }

    // cset w<rd>, cond
    fn cset(&mut self, rd: u32, cond: u32) {
        self.e(0x1A9F_07E0 | (cond ^ 1) << 12 | rd);
    }

    fn here(&self) -> usize {
        self.code.len()
    }

    // A branch whose target is set later with "set"; returns its fixup.
    fn fwd(&mut self, fix: Fix) -> usize {
        self.branch(fix, Target::Code(usize::MAX));
        self.fixups.len() - 1
    }

    fn set(&mut self, fixup: usize, to: usize) {
        self.fixups[fixup].1 = Target::Code(to);
    }

    // movz w<rd>, #imm16
    fn movz_w(&mut self, rd: u32, imm: u32) {
        self.e(0x5280_0000 | (imm & 0xFFFF) << 5 | rd);
    }

    // x<rd> = v; returns the register holding v (xzr for 0).
    fn mov_x(&mut self, rd: u32, v: u64) -> u32 {
        if v == 0 {
            return XZR;
        }
        let mut first = true;
        for hw in 0..4u32 {
            let part = ((v >> (hw * 16)) & 0xFFFF) as u32;
            if part == 0 {
                continue;
            }
            let base = if first { 0xD280_0000 } else { 0xF280_0000 };
            self.e(base | hw << 21 | part << 5 | rd);
            first = false;
        }
        rd
    }

    // Store the 16 bytes "v" at [x19 + off].
    fn store_const(&mut self, off: usize, v: [u64; 2]) {
        let r0 = self.mov_x(0, v[0]);
        let r1 = self.mov_x(1, v[1]);
        self.str_x(r0, X19, off);
        self.str_x(r1, X19, off + 8);
    }

    fn branch(&mut self, fix: Fix, t: Target) {
        self.fixups.push((self.code.len(), t, fix));
        self.e(match fix {
            Fix::B => 0x1400_0000,
            Fix::Cb(false, rt) => 0x3400_0000 | rt,
            Fix::Cb(true, rt) => 0x3500_0000 | rt,
            Fix::BCond(cond) => 0x5400_0000 | cond,
        });
    }

    // x0 = ctx, w1..w3 = args, call helpers[h]: on a non-zero status, leave.
    fn call(&mut self, h: usize, a1: usize, a2: usize, a3: usize, exit_on_status: bool) {
        self.e(0xAA00_03E0 | X21 << 16 | X0); // mov x0, x21
        self.movz_w(1, a1 as u32);
        self.movz_w(2, a2 as u32);
        self.movz_w(3, a3 as u32);
        self.ldr_x(X16, X20, h * 8);
        self.e(0xD63F_0000 | X16 << 5); // blr x16
        if exit_on_status {
            self.branch(Fix::Cb(true, 0), Target::Exit);
        }
    }

    fn prologue(&mut self) {
        self.e(0xA9BD_7BFD); // stp x29, x30, [sp, #-48]!
        self.e(0x9100_03FD); // mov x29, sp
        self.e(0xA901_53F3); // stp x19, x20, [sp, #16]
        self.str_x(X21, SP, 32);
        self.e(0xAA00_03E0 | X0 << 16 | X21); // mov x21, x0
        self.ldr_x(X19, X0, 0);
        self.ldr_x(X20, X0, 8);
    }

    fn epilogue(&mut self) {
        self.ldr_x(X21, SP, 32);
        self.e(0xA941_53F3); // ldp x19, x20, [sp, #16]
        self.e(0xA8C3_7BFD); // ldp x29, x30, [sp], #48
        self.e(0xD65F_03C0); // ret
    }

    fn resolve(&mut self, labels: &[usize], exit: usize) -> bool {
        for &(at, t, fix) in &self.fixups {
            let to = match t {
                Target::Op(i) => match labels.get(i) {
                    Some(&l) if l != usize::MAX => l,
                    _ => return false,
                },
                Target::Exit => exit,
                Target::Code(c) if c != usize::MAX => c,
                Target::Code(_) => return false,
            };
            let delta = to as i64 - at as i64;
			/*
				A branch that does not reach is a method that cannot be compiled, never a truncated offset:
				"cbz"/"cbnz"/"b.cond" reach +-1 MB and "b" +-128 MB, and the masks below would silently wrap.
				Every helper call is followed by "cbnz w0, <exit>", which jumps to the END of the method,
				so a method of more than 1 MB of code sent its first early return into whatever lay 1 MB away:
				haunt the house, loading a level, executed zeroed code memory.
			*/
            let (bits, ok) = match fix {
                Fix::B => (26, (-(1i64 << 25)..(1i64 << 25)).contains(&delta)),
                Fix::Cb(..) | Fix::BCond(_) => (19, (-(1i64 << 18)..(1i64 << 18)).contains(&delta)),
            };
            if !ok {
                return false;
            }
            let field = (delta as u32) & ((1u32 << bits) - 1);
            self.code[at] |= match fix {
                Fix::B => field,
                Fix::Cb(..) | Fix::BCond(_) => field << 5,
            };
        }
        true
    }
}

// The 16 bytes of "v" as the interpreter would store them in a stack slot.
fn bytes_of(v: Value<'_>) -> [u64; 2] {
    let mut buf = std::mem::MaybeUninit::<[u64; 2]>::zeroed();
    unsafe {
        (buf.as_mut_ptr() as *mut Value<'_>).write(v);
        buf.assume_init()
    }
}

fn lazy(m: &Gc<'_, Multiname<'_>>) -> usize {
    m.has_lazy_name() as usize + m.has_lazy_ns() as usize
}

// (values popped, values pushed). "None": never compiled.
fn effect(op: &Op<'_>) -> Option<(usize, usize)> {
    use Op::*;
    Some(match op {
        PushDouble { .. } | PushFalse | PushInt { .. } | PushNamespace { .. } | PushNull
        | PushString { .. } | PushTrue | PushUint { .. } | PushUndefined => (0, 1),
        Pop => (1, 0),
        Dup => (1, 2),
        Swap => (2, 2),
        GetLocal { .. } => (0, 1),
        SetLocal { .. } => (1, 0),
        StoreLocal { .. } => (1, 1),
        Kill { .. } | IncLocal { .. } | IncLocalI { .. } | DecLocal { .. } | DecLocalI { .. } => (0, 0),
        Call { num_args } => (2 + *num_args as usize, 1),
        CallMethod { num_args, push_return_value, .. }
        | CallNative { num_args, push_return_value, .. } => {
            (1 + *num_args as usize, *push_return_value as usize)
        }
        CallProperty { multiname, num_args } | CallPropLex { multiname, num_args } => {
            (1 + *num_args as usize + lazy(multiname), 1)
        }
        CallPropVoid { multiname, num_args } => (1 + *num_args as usize + lazy(multiname), 0),
        CallStatic { num_args, .. } => (1 + *num_args as usize, 1),
        CallSuper { multiname, num_args } => (1 + *num_args as usize + lazy(multiname), 1),
        GetPropertyStatic { multiname } | GetPropertyFast { multiname } | GetPropertySlow { multiname } => {
            (1 + lazy(multiname), 1)
        }
        SetPropertyStatic { multiname } | SetPropertyFast { multiname } | SetPropertySlow { multiname }
        | InitProperty { multiname } => (2 + lazy(multiname), 0),
        DeleteProperty { multiname } | GetSuper { multiname } | GetDescendants { multiname } => {
            (1 + lazy(multiname), 1)
        }
        SetSuper { multiname } => (2 + lazy(multiname), 0),
        FindProperty { multiname } | FindPropStrict { multiname } => (lazy(multiname), 1),
        FindDef { .. } => (0, 1),
        In | HasNext | NextName | NextValue | IsTypeLate | AsTypeLate | InstanceOf => (2, 1),
        PushScope | PushWith | SetGlobalSlot { .. } | DxnsLate => (1, 0),
        PopScope | Nop | Dxns { .. } | Debug { .. } | DebugFile { .. } | DebugLine { .. } | Bkpt
        | BkptLine { .. } => (0, 0),
        GetOuterScope { .. } | GetScopeObject { .. } | GetScriptGlobals { .. }
        | NewActivation { .. } | NewFunction { .. } | HasNext2 { .. } => (0, 1),
        GetSlot { .. } => (1, 1),
        SetSlot { .. } | SetSlotCoerceI { .. } | SetSlotNoCoerce { .. } => (2, 0),
        Construct { num_args } | ConstructSlot { num_args, .. } => (1 + *num_args as usize, 1),
        ConstructProp { multiname, num_args } => (1 + *num_args as usize + lazy(multiname), 1),
        ConstructSuper { num_args } => (1 + *num_args as usize, 0),
        NewObject { num_args } => (2 * *num_args as usize, 1),
        NewClass { .. } => (1, 1),
        ApplyType { num_types } => (1 + *num_types as usize, 1),
        NewArray { num_args } => (*num_args as usize, 1),
        CoerceA | CoerceB | CoerceD | CoerceI | CoerceO | CoerceS | CoerceU | ConvertO | ConvertS
        | Coerce { .. } | AsType { .. } | IsType { .. } | TypeOf | Not | Negate | NegateI | BitNot
        | Increment | IncrementI | Decrement | DecrementI | EscXAttr | EscXElem | CheckFilter
        | Lf32 | Lf64 | Li16 | Li32 | Li8 | Sxi1 | Sxi16 | Sxi8 => (1, 1),
        CoerceDSwapPop | CoerceISwapPop | CoerceUSwapPop | CoerceSwapPop { .. } => (2, 1),
        Add | AddI | BitAnd | BitOr | BitXor | Divide | LShift | Modulo | Multiply | MultiplyI
        | RShift | Subtract | SubtractI | URShift | StrictEquals | Equals | GreaterEquals
        | GreaterThan | LessEquals | LessThan => (2, 1),
        Sf32 | Sf64 | Si16 | Si32 | Si8 => (2, 0),
        Throw => (1, 0),
        Jump { .. } => (0, 0),
        IfTrue { .. } | IfFalse { .. } | PopJump { .. } | LookupSwitch(_) => (1, 0),
        ReturnValue { .. } => (1, 0),
        ReturnVoid { .. } => (0, 0),
        Timestamp | NewCatch { .. } => return None,
    })
}

// ---------------------------------------------------------------- compiler

fn compile(s: &mut State, method: Method<'_>) -> Result<usize, String> {
    init(s);
    if s.init != 1 {
        return Err("no code memory".into());
    }
    let info = method.get_verified_info();
    if !info.exceptions.is_empty() {
        return Err("try/catch".into());
    }
    let ops = &info.parsed_code;
    let body = method.body().ok_or("no body")?;
    let nl = body.num_locals as usize;
    let ms = body.max_stack as usize;
    let n = ops.len();
    if n == 0 || n > 0xFFFF || nl + ms > 2000 {
        return Err("size".into());
    }

    // Stack depth before every reachable op.
    let mut depth = vec![u32::MAX; n];
    depth[0] = 0;
    let mut work = vec![0usize];
    while let Some(i) = work.pop() {
        let d = depth[i] as usize;
        let op = &ops[i];
        let (pops, pushes) = effect(op).ok_or_else(|| op_name(op))?;
        if d < pops {
            return Err("underflow".into());
        }
        let nd = d - pops + pushes;
        if nd > ms {
            return Err("overflow".into());
        }
        let succ: std::vec::Vec<usize> = match op {
            Op::Jump { offset } | Op::PopJump { offset } => std::vec![*offset],
            Op::IfTrue { offset } | Op::IfFalse { offset } => std::vec![i + 1, *offset],
            Op::LookupSwitch(ls) => {
                // One compare per case, "cmp w0, #k": 12 bits of k.
                if ls.case_offsets.len() >= 4096 {
                    return Err("switch size".into());
                }
                let mut t: std::vec::Vec<usize> = ls.case_offsets.iter().map(|c| c.get()).collect();
                t.push(ls.default_offset.get());
                t
            }
            Op::ReturnValue { .. } | Op::ReturnVoid { .. } | Op::Throw => std::vec::Vec::new(),
            _ => std::vec![i + 1],
        };
        for t in succ {
            if t >= n {
                return Err("falls off".into());
            }
            if depth[t] == u32::MAX {
                depth[t] = nd as u32;
                work.push(t);
            } else if depth[t] as usize != nd {
                return Err("depth merge".into());
            }
        }
    }

    // Branch targets: the stack cache is empty there ("Tos").
    let mut target = vec![false; n];
    for op in ops.iter() {
        if let Op::Jump { offset } | Op::PopJump { offset } | Op::IfTrue { offset } | Op::IfFalse { offset } = op
            && *offset < n
        {
            target[*offset] = true;
        }
        if let Op::LookupSwitch(ls) = op {
            for t in ls.case_offsets.iter().map(|c| c.get()).chain(std::iter::once(ls.default_offset.get())) {
                if t < n {
                    target[t] = true;
                }
            }
        }
    }
    let lay = s.layout.clone();
    let cache = TOS_CACHE.load(AtomicOrdering::Relaxed);
    let (main, main_labels, main_exit) = emit(&lay, ops, &depth, &target, nl, cache)?;
    let alt = if DUAL.load(AtomicOrdering::Relaxed) {
        Some(emit(&lay, ops, &depth, &target, nl, false)?)
    } else {
        None
    };

    // A 16-byte header (the address of the code "ALT" runs: the 2nd compilation, or this one when there is none), the code, the other code.
    let main_bytes = (main.len() * 4 + 15) & !15;
    let alt_bytes = alt.as_ref().map_or(0, |a| a.0.len() * 4);
    let total = 16 + main_bytes + alt_bytes;
    if s.used + total > CODE_SIZE {
        return Err("code memory full".into());
    }
    let addr = s.rx as usize + s.used + 16;
    let alt_addr = if alt.is_some() { addr + main_bytes } else { addr };
    unsafe {
        let base = s.rw.add(s.used);
        (base as *mut usize).write(alt_addr);
        std::ptr::copy_nonoverlapping(main.as_ptr() as *const u8, base.add(16), main.len() * 4);
        if let Some((code, _, _)) = &alt {
            std::ptr::copy_nonoverlapping(code.as_ptr() as *const u8, base.add(16 + main_bytes), alt_bytes);
        }
        flashnx_jit_flush(base, s.rx.add(s.used), total);
    }
    s.used += (total + 15) & !15;

    // The names "report" gives the helper counts under ("HELPER_N").
    #[cfg(feature = "flashnx_instr")]
    for op in ops.iter() {
        s.kind_names.entry(kind(op)).or_insert_with(|| op_name(op));
    }
    if MAP.load(AtomicOrdering::Relaxed) {
        push_map(s, addr, ops, &depth, &main_labels, main_exit);
        if let Some((_, labels, exit)) = &alt {
            push_map(s, alt_addr, ops, &depth, labels, *exit);
        }
    }
    Ok(addr)
}

fn push_map(s: &mut State, addr: usize, ops: &[Op<'_>], depth: &[u32], labels: &[usize], exit: usize) {
    s.map.push((addr, "(prologue)".into()));
    for (i, op) in ops.iter().enumerate() {
        if depth[i] != u32::MAX {
            s.map.push((addr + labels[i] * 4, op_name(op)));
        }
    }
    s.map.push((addr + exit * 4, "(epilogue)".into()));
}

/*
	Where the top of the AVM2 operand stack is at the start of an op in compiled code:
	A one-entry stack cache, after the virtual stack of SpiderMonkey's baseline compiler (BaselineFrameInfo) and Ertl's stack caching.
	Without it every op wrote its result to the frame and the next op read it straight back:
	On Happy Wheels, "GetLocal -> GetSlot" alone was 10 % of the frame and "GetSlot -> GetSlot" 6 %, much of it a load waiting on the store just before it.

	Only the top entry is ever cached, and only between ops that know about it:
	every other op, every helper call, every branch and every branch target sees it written back first ("flush"),
	so the frame is exactly what the interpreter would hold wherever anything else can look at it.
	The registers only ever hold a value between two ops, where no collection can run.
*/
#[derive(Clone, Copy, PartialEq)]
enum Tos {
    // In its frame slot.
    Mem,

    // Not written yet: it is the local at this frame offset ("GetLocal").
    Local(usize),

    // Not written yet: its two words are in x14 (tag) and x15 (payload).
    Regs,
}

// Write the cached top of the stack to its frame slot at "at".
fn flush(a: &mut Asm, tos: Tos, at: usize) {
    match tos {
        Tos::Mem => {}
        Tos::Local(off) => {
            a.ldr_q(0, X19, off);
            a.str_q(0, X19, at);
        }
        Tos::Regs => {
            a.str_x(14, X19, at);
            a.str_x(15, X19, at + 8);
        }
    }
}

// The code of one method: words, the start of each op, the exit.
fn emit(
    lay: &Layout,
    ops: &[Op<'_>],
    depth: &[u32],
    target: &[bool],
    nl: usize,
    cache: bool,
) -> Result<(Vec<u32>, Vec<usize>, usize), String> {
    let n = ops.len();
    let slot = |d: usize| (nl + d) * 16;
    let mut a = Asm {
        code: Vec::with_capacity(n * 4 + 16),
        fixups: Vec::new(),
    };
    a.prologue();
    let mut labels = vec![usize::MAX; n];
    let mut tos = Tos::Mem;
    for i in 0..n {
        if depth[i] == u32::MAX {
            labels[i] = a.code.len();
            tos = Tos::Mem;
            continue;
        }
        let d = depth[i] as usize;
        // Jumps arrive with nothing cached: write it back on the way in.
        if target[i] && tos != Tos::Mem {
            flush(&mut a, tos, slot(d - 1));
            tos = Tos::Mem;
        }
        labels[i] = a.code.len();
        let op = &ops[i];
        let consumer = cache
            && match op {
                Op::GetLocal { .. } | Op::SetLocal { .. } | Op::StoreLocal { .. } | Op::Pop => true,
                Op::GetSlot { index } => !lay.obj_tags.is_empty() && slot(d - 1) < 4096 && *index < 4096,
                Op::SetSlotNoCoerce { index } => {
                    lay.header_ok && !lay.obj_tags.is_empty() && slot(d - 1) + 16 < 4096 && *index < 4096
                }
                _ => false,
            };
        if !consumer && tos != Tos::Mem {
            flush(&mut a, tos, slot(d - 1));
            tos = Tos::Mem;
        }
        match op {
            Op::GetLocal { index } if cache => {
                // The previous top is no longer the top: write it back.
                if tos != Tos::Mem {
                    flush(&mut a, tos, slot(d - 1));
                }
                tos = Tos::Local(*index as usize * 16);
            }
            Op::GetLocal { index } => {
                a.ldr_q(0, X19, *index as usize * 16);
                a.str_q(0, X19, slot(d));
            }
            Op::SetLocal { index } | Op::StoreLocal { index } => {
                let dst = *index as usize * 16;
                match tos {
                    Tos::Mem => {
                        a.ldr_q(0, X19, slot(d - 1));
                        a.str_q(0, X19, dst);
                    }
                    Tos::Local(src) => {
                        if src != dst {
                            a.ldr_q(0, X19, src);
                            a.str_q(0, X19, dst);
                        }
                    }
                    Tos::Regs => {
                        a.str_x(14, X19, dst);
                        a.str_x(15, X19, dst + 8);
                    }
                }
                // StoreLocal leaves the value on the stack, where it is.
                if matches!(op, Op::SetLocal { .. }) {
                    tos = Tos::Mem;
                }
            }
            Op::Kill { index } => a.store_const(*index as usize * 16, bytes_of(Value::Undefined)),
            Op::PushInt { value } => a.store_const(slot(d), bytes_of(Value::from(*value))),
            Op::PushUint { value } => a.store_const(slot(d), bytes_of(Value::from(*value))),
            Op::PushDouble { value } => a.store_const(slot(d), bytes_of(Value::from(*value))),
            Op::PushTrue => a.store_const(slot(d), bytes_of(Value::from(true))),
            Op::PushFalse => a.store_const(slot(d), bytes_of(Value::from(false))),
            Op::PushNull => a.store_const(slot(d), bytes_of(Value::Null)),
            Op::PushUndefined => a.store_const(slot(d), bytes_of(Value::Undefined)),
            Op::Pop => tos = Tos::Mem,
            Op::Nop | Op::CoerceA => {}
            Op::Dup => {
                a.ldr_q(0, X19, slot(d - 1));
                a.str_q(0, X19, slot(d));
            }
            Op::Swap => {
                a.ldr_q(0, X19, slot(d - 2));
                a.ldr_q(1, X19, slot(d - 1));
                a.str_q(1, X19, slot(d - 2));
                a.str_q(0, X19, slot(d - 1));
            }
            Op::Jump { offset } | Op::PopJump { offset } => {
                if *offset <= i {
                    a.call(H_TIMEOUT, 0, 0, 0, true);
                }
                a.branch(Fix::B, Target::Op(*offset));
            }
            Op::LookupSwitch(ls) => {
                /*
					The index as the interpreter takes it ("lookup_switch": the top as i32), then one compare per case, in the order of the table:
					Anything else, a negative index included, is the default.
					Switches are short (Stencyl's behaviours, #130's round: 93 methods refused for this alone in The Little Who).
				*/
                let cases: std::vec::Vec<usize> = ls.case_offsets.iter().map(|c| c.get()).collect();
                let default = ls.default_offset.get();
                if cases.iter().any(|&t| t <= i) || default <= i {
                    a.call(H_TIMEOUT, 0, 0, 0, true);
                }
                a.call(H_SWITCH, d, 0, 0, false);
                for (k, &t) in cases.iter().enumerate() {
                    a.cmp_w(0, k as u32);
                    a.branch(Fix::BCond(EQ), Target::Op(t));
                }
                a.branch(Fix::B, Target::Op(default));
            }
            Op::IfTrue { offset } | Op::IfFalse { offset } => {
                if *offset <= i {
                    a.call(H_TIMEOUT, 0, 0, 0, true);
                }
                let nz = matches!(op, Op::IfTrue { .. });
                let top = slot(d - 1);
                if lay.bool_ok && top + lay.bool_off < 4096 {
                    // A Bool decides in place: anything else asks the helper.
                    a.ldrb(9, X19, top);
                    a.cmp_w(9, lay.bool_tag as u32);
                    let to_slow = a.fwd(Fix::BCond(NE));
                    a.ldrb(9, X19, top + lay.bool_off);
                    a.branch(Fix::Cb(nz, 9), Target::Op(*offset));
                    let to_done = a.fwd(Fix::B);
                    let slow = a.here();
                    a.set(to_slow, slow);
                    a.call(H_TRUTHY, d, 0, 0, false);
                    a.branch(Fix::Cb(nz, 0), Target::Op(*offset));
                    let done = a.here();
                    a.set(to_done, done);
                } else {
                    a.call(H_TRUTHY, d, 0, 0, false);
                    a.branch(Fix::Cb(nz, 0), Target::Op(*offset));
                }
            }
            Op::Add | Op::Subtract | Op::Multiply | Op::LessThan | Op::GreaterThan | Op::LessEquals | Op::GreaterEquals | Op::Equals
			if lay.num_ok && d >= 2 && slot(d - 1) + 16 < 4096 => {
                // Both Numbers or both ints: the interpreter's fast paths, in place in the first operand's slot. Anything else: helper.
                let (v1, v2) = (slot(d - 2), slot(d - 1));
                let compare = !matches!(op, Op::Add | Op::Subtract | Op::Multiply);
                let mut to_slow = Vec::new();
                let mut to_done = Vec::new();
                let mut to_write = Vec::new();
                a.ldrb(9, X19, v1);
                a.ldrb(10, X19, v2);
                a.cmp_w(9, lay.num_tag as u32);
                let to_int = a.fwd(Fix::BCond(NE));
                a.cmp_w(10, lay.num_tag as u32);
                to_slow.push(a.fwd(Fix::BCond(NE)));
                a.ldr_d(0, X19, v1 + lay.num_off);
                a.ldr_d(1, X19, v2 + lay.num_off);
                if compare {
                    a.e(0x1E60_2000 | 1 << 16); // fcmp d0, d1
                    // Ordered conditions: a NaN makes every one of them false.
                    a.cset(12, match op {
                        Op::LessThan => MI,
                        Op::GreaterThan => GT,
                        Op::LessEquals => LS,
                        Op::GreaterEquals => GE,
                        _ => EQ,
                    });
                    to_write.push(a.fwd(Fix::B));
                } else {
                    a.e(match op {
                        Op::Add => 0x1E60_2800,
                        Op::Subtract => 0x1E60_3800,
                        _ => 0x1E60_0800,
                    } | 1 << 16); // fadd / fsub / fmul d0, d0, d1
                    a.str_d(0, X19, v1 + lay.num_off);
                    to_done.push(a.fwd(Fix::B));
                }
                let int_path = a.here();
                a.set(to_int, int_path);
                a.cmp_w(9, lay.int_tag as u32);
                to_slow.push(a.fwd(Fix::BCond(NE)));
                a.cmp_w(10, lay.int_tag as u32);
                to_slow.push(a.fwd(Fix::BCond(NE)));
                a.ldr_w(10, X19, v1 + lay.int_off);
                a.ldr_w(11, X19, v2 + lay.int_off);
                if compare {
                    a.e(0x6B0B_015F); // cmp w10, w11
                    a.cset(12, match op {
                        Op::LessThan => LT,
                        Op::GreaterThan => GT,
                        Op::LessEquals => LE,
                        Op::GreaterEquals => GE,
                        _ => EQ,
                    });
                    let write = a.here();
                    for f in to_write.drain(..) {
                        a.set(f, write);
                    }
                    // The Bool replaces the first operand: tag word, then the bool.
                    let r = a.mov_x(9, lay.bool_word);
                    a.str_x(r, X19, v1);
                    a.strb(12, X19, v1 + lay.bool_off);
                } else {
                    match op {
                        Op::Add => a.e(0x2B0B_014A), // adds w10, w10, w11
                        Op::Subtract => a.e(0x6B0B_014A), // subs w10, w10, w11
                        _ => {
                            a.e(0x9B2B_7D4A); // smull x10, w10, w11
                            a.e(0xEB2A_C15F); // cmp x10, w10, sxtw
                        }
                    }
                    // Overflow: the interpreter's answer is a Number.
                    to_slow.push(a.fwd(Fix::BCond(if matches!(op, Op::Multiply) { NE } else { VS })));
                    a.str_w(10, X19, v1 + lay.int_off);
                }
                to_done.push(a.fwd(Fix::B));
                let slow = a.here();
                for f in to_slow {
                    a.set(f, slow);
                }
                a.call(kind(op), d, i, d - 1, true);
                let done = a.here();
                for f in to_done {
                    a.set(f, done);
                }
            }
            Op::IncrementI | Op::DecrementI | Op::CoerceI | Op::CoerceD
            if lay.num_ok && slot(d - 1) + 16 < 4096 => {
                let v = slot(d - 1);
                a.ldrb(9, X19, v);
                let tag = if matches!(op, Op::CoerceD) { lay.num_tag } else { lay.int_tag };
                a.cmp_w(9, tag as u32);
                let to_slow = a.fwd(Fix::BCond(NE));
                match op {
                    Op::IncrementI | Op::DecrementI => {
                        a.ldr_w(10, X19, v + lay.int_off);
                        a.e(if matches!(op, Op::IncrementI) { 0x1100_054A } else { 0x5100_054A }); // add/sub w10, w10, #1
                        a.str_w(10, X19, v + lay.int_off);
                    }
                    _ => {} // already the right type
                }
                let to_done = a.fwd(Fix::B);
                let slow = a.here();
                a.set(to_slow, slow);
                a.call(kind(op), d, i, d, true);
                let done = a.here();
                a.set(to_done, done);
            }
            Op::GetSlot { index }
            if !lay.obj_tags.is_empty() && slot(d - 1) < 4096 && *index < 4096 =>
            {
                /*
					An object of a checked type: its slot, in place.
					Anything else (null, a primitive, another type, out of range) takes the helper, which raises the interpreter's own errors.
					The object comes from the stack cache when it is there (a local, or the previous GetSlot's x14/x15), and the slot's value goes to x14/x15, not to the frame.
				*/
				let top = slot(d - 1);
                if let Tos::Local(off) = tos
                    && off >= 4096
                {
                    flush(&mut a, tos, top);
                    tos = Tos::Mem;
                }
                let src = match tos {
                    Tos::Mem => Some(top),
                    Tos::Local(off) => Some(off),
                    Tos::Regs => None,
                };
                let mut to_slow = Vec::new();
                match src {
                    Some(off) => a.ldrb(9, X19, off),
                    None => a.e(0x5300_1C00 | 14 << 5 | 9), // uxtb w9, w14
                }
                if lay.obj_tags.len() == 1 {
                    a.cmp_w(9, lay.obj_tags[0] as u32);
                    to_slow.push(a.fwd(Fix::BCond(NE)));
                } else {
                    let mut to_fast = Vec::new();
                    for &t in &lay.obj_tags {
                        a.cmp_w(9, t as u32);
                        to_fast.push(a.fwd(Fix::BCond(EQ)));
                    }
                    to_slow.push(a.fwd(Fix::B));
                    let fast = a.here();
                    for f in to_fast {
                        a.set(f, fast);
                    }
                }
                match src {
                    Some(off) => a.ldr_x(10, X19, off + 8),
                    None => a.e(0xAA0F_03EA), // mov x10, x15
                }
                a.ldr_x(11, 10, lay.slots_len_off);
                a.cmp_x(11, *index as u32);
                to_slow.push(a.fwd(Fix::BCond(LS)));
                a.ldr_x(11, 10, lay.slots_ptr_off);
                let regs = cache && *index * 16 + 8 < 32768;
                if regs {
                    a.ldr_x(14, 11, *index * 16);
                    a.ldr_x(15, 11, *index * 16 + 8);
                } else {
                    a.ldr_q(0, 11, *index * 16);
                    a.str_q(0, X19, top);
                }
                let to_done = a.fwd(Fix::B);
                let slow = a.here();
                for f in to_slow {
                    a.set(f, slow);
                }
                // The helper works on the frame: the object goes there first, and the result comes back from there.
                flush(&mut a, tos, top);
                a.call(kind(op), d, i, d, true);
                if regs {
                    a.ldr_x(14, X19, top);
                    a.ldr_x(15, X19, top + 8);
                }
                let done = a.here();
                a.set(to_done, done);
                tos = if regs { Tos::Regs } else { Tos::Mem };
            }
            Op::ReturnValue { .. } | Op::ReturnVoid { .. } => a.call(H_RETURN, d, i, 0, true),
            Op::GetPropertyFast { multiname } if multiname.has_lazy_name() && !multiname.has_lazy_ns() => {
                a.call(H_ARRAY_GET, d, i, d - 1, true);
            }
            Op::SetSlotNoCoerce { index }
            if lay.header_ok && !lay.obj_tags.is_empty() && slot(d - 1) + 16 < 4096 && *index < 4096 =>
            {
                // Object of a checked type, not black for the GC, slot in range: store in place. Otherwise the helper (barrier, errors).
                let (obj, val) = (slot(d - 2), slot(d - 1));
                let mut to_slow = Vec::new();
                a.ldrb(9, X19, obj);
                if lay.obj_tags.len() == 1 {
                    a.cmp_w(9, lay.obj_tags[0] as u32);
                    to_slow.push(a.fwd(Fix::BCond(NE)));
                } else {
                    let mut to_fast = Vec::new();
                    for &t in &lay.obj_tags {
                        a.cmp_w(9, t as u32);
                        to_fast.push(a.fwd(Fix::BCond(EQ)));
                    }
                    to_slow.push(a.fwd(Fix::B));
                    let fast = a.here();
                    for f in to_fast {
                        a.set(f, fast);
                    }
                }
                a.ldr_x(10, X19, obj + 8);
                a.ldr_x(11, 10, lay.vtable_word);
                a.e(0x1200_056B); // and w11, w11, #3
                a.cmp_w(11, 3);
                to_slow.push(a.fwd(Fix::BCond(EQ)));
                a.ldr_x(12, 10, lay.slots_len_off);
                a.cmp_x(12, *index as u32);
                to_slow.push(a.fwd(Fix::BCond(LS)));
                a.ldr_x(12, 10, lay.slots_ptr_off);
                // The value is the top of the stack, maybe still cached (the object below it never is).
                let dst = *index * 16;
                match tos {
                    Tos::Mem => {
                        a.ldr_q(0, X19, val);
                        a.str_q(0, 12, dst);
                    }
                    Tos::Local(off) => {
                        a.ldr_q(0, X19, off);
                        a.str_q(0, 12, dst);
                    }
                    Tos::Regs if dst + 8 < 32768 => {
                        a.str_x(14, 12, dst);
                        a.str_x(15, 12, dst + 8);
                    }
                    Tos::Regs => {
                        flush(&mut a, tos, val);
                        a.ldr_q(0, X19, val);
                        a.str_q(0, 12, dst);
                    }
                }
                let to_done = a.fwd(Fix::B);
                let slow = a.here();
                for f in to_slow {
                    a.set(f, slow);
                }
                flush(&mut a, tos, val);
                a.call(kind(op), d, i, d - 2, true);
                let done = a.here();
                a.set(to_done, done);
                tos = Tos::Mem;
            }
            Op::CallMethod { num_args, push_return_value, .. } => {
                let post = d - 1 - *num_args as usize + *push_return_value as usize;
                a.call(H_CALL, d, i, post, true);
            }
            _ => {
                let (pops, pushes) = effect(op).unwrap();
                let k = kind(op);
                let h = if k < KINDS { k } else { H_GENERIC };
                a.call(h, d, i, d - pops + pushes, true);
            }
        }
    }

    // Verified code never gets here: leave with status 4 if it did.
    a.movz_w(0, 4);
    a.branch(Fix::B, Target::Exit);
    let exit = a.code.len();
    a.epilogue();
    if !a.resolve(&labels, exit) {
        return Err("bad branch".into());
    }
    Ok((a.code, labels, exit))
}

// ---------------------------------------------------------------- helpers

#[inline(always)]
unsafe fn finish(c: &mut Ctx, act: &mut Act, r: Result<(), Error<'static>>, idx: u32, post: u32) -> u32 {
    match r {
        Ok(()) => {
            if act.stack.len() == c.num_locals + post as usize {
                0
            } else {
                // "effect" is wrong for this op: the interpreter's own stack pointer is right, so it takes over at the next op.
                c.resume_ip = idx as usize + 1;
                let s = state();
                s.deopts += 1;
                let name = op_name(unsafe { &*c.ops.add(idx as usize) });
                *s.deopt_ops.entry(name).or_insert(0) += 1;
                3
            }
        }
        Err(e) => {
            c.result = Some(Err(e));
            1
        }
    }
}

unsafe extern "C" fn h_op<const K: usize>(ctx: *mut Ctx, depth: u32, idx: u32, post: u32) -> u32 {
    unsafe {
        let c = &mut *ctx;
        let act = &mut *c.act;
        act.stack.set_stack_pointer(c.num_locals + depth as usize);
        let op = &*c.ops.add(idx as usize);
        #[cfg(feature = "flashnx_instr")]
        HELPER_N[kind(op)].fetch_add(1, AtomicOrdering::Relaxed);
        if K < KINDS {
            // One copy per op kind: the match below folds to one arm, and each call site in compiled code has its own predictable branch.
            std::hint::assert_unchecked(kind(op) == K);
        }
        let r = act.flashnx_exec_simple(op);
        finish(c, act, r, idx, post)
    }
}

unsafe extern "C" fn h_truthy(ctx: *mut Ctx, depth: u32, _: u32, _: u32) -> u32 {
    unsafe {
        let c = &mut *ctx;
        let act = &mut *c.act;
        act.stack.set_stack_pointer(c.num_locals + depth as usize);
        act.check_if_true() as u32
    }
}

// "lookupswitch": pops the index (the op's operand, at "depth") and returns it as the interpreter reads it ("lookup_switch"), the compiled code comparing it with the cases.
unsafe extern "C" fn h_switch(ctx: *mut Ctx, depth: u32, _: u32, _: u32) -> u32 {
    unsafe {
        let c = &mut *ctx;
        let act = &mut *c.act;
        act.stack.set_stack_pointer(c.num_locals + depth as usize);
        act.pop_stack().as_i32() as u32
    }
}

unsafe extern "C" fn h_timeout(ctx: *mut Ctx, _: u32, _: u32, _: u32) -> u32 {
    unsafe {
        let c = &mut *ctx;
        let act = &mut *c.act;
        match act.timeout_check() {
            Ok(()) => 0,
            Err(e) => {
                c.result = Some(Err(e));
                1
            }
        }
    }
}

unsafe extern "C" fn h_return(ctx: *mut Ctx, depth: u32, idx: u32, _: u32) -> u32 {
    unsafe {
        let c = &mut *ctx;
        let act = &mut *c.act;
        act.stack.set_stack_pointer(c.num_locals + depth as usize);
        let r = match &*c.ops.add(idx as usize) {
            Op::ReturnValue { return_type } => act.return_value(*return_type),
            Op::ReturnVoid { return_type } => Ok(act.return_void(*return_type)),
            _ => unreachable!(),
        };
        let status = if r.is_ok() { 2 } else { 1 };
        c.result = Some(r);
        status
    }
}

/*
	"getproperty" with a runtime name:
	an int index into a dense "Array" is read here (what "op_get_property_fast" ends up doing for it, minus the dynamic dispatch and the borrow counting):
	anything else is that handler.
*/
unsafe extern "C" fn h_array_get(ctx: *mut Ctx, depth: u32, idx: u32, post: u32) -> u32 {
    unsafe {
        let c = &mut *ctx;
        let obj_slot = c.num_locals + depth as usize - 2;
        let o = *c.frame.add(obj_slot);
        let n = *c.frame.add(obj_slot + 1);
        if let (Value::Object(Object::ArrayObject(array)), Value::Integer(i)) = (o, n)
            && i >= 0
            && let Some(ArrayStorage::Dense { storage, .. }) = array.flashnx_storage_unguarded()
            && let Some(Some(v)) = storage.get(i as usize)
        {
            *c.frame.add(obj_slot) = *v;
            return 0;
        }
        let act = &mut *c.act;
        act.stack.set_stack_pointer(c.num_locals + depth as usize);
        let r = match &*c.ops.add(idx as usize) {
            Op::GetPropertyFast { multiname } => act.op_get_property_fast(*multiname),
            _ => unreachable!(),
        };
        finish(c, act, r, idx, post)
    }
}

unsafe extern "C" fn h_call_method(ctx: *mut Ctx, depth: u32, idx: u32, post: u32) -> u32 {
    unsafe {
        let c = &mut *ctx;
        let act = &mut *c.act;
        let (index, num_args, push) = match &*c.ops.add(idx as usize) {
            Op::CallMethod { index, num_args, push_return_value } => (*index, *num_args, *push_return_value),
            _ => unreachable!(),
        };
        let recv = c.num_locals + depth as usize - num_args as usize - 1;
        if let Some(r) = fast_call(act, c.frame, recv, index, num_args as usize) {
            state().calls_direct += 1;
            return match r {
                Ok(v) => {
                    if push {
                        *c.frame.add(recv) = v;
                    }
                    0
                }
                Err(e) => {
                    c.result = Some(Err(e));
                    1
                }
            };
        }
        state().calls_slow += 1;
        act.stack.set_stack_pointer(c.num_locals + depth as usize);
        let r = act.op_call_method(index, num_args, push);
        finish(c, act, r, idx, post)
    }
}

/*
	"callmethod" straight into the callee's compiled code, or "None" when any condition is not met
	(the caller then takes the interpreter's path, which does the very same thing in the general case).
	
	This is "call_method_with_args" + "exec" + "init_from_method" for the case the JIT can see is simple, step for step:
	same vtable lookup, same bound method check, same stack frame, same activation fields, same call stack, same cleanup.
	Arguments are only taken when "coerce_to_type" would give them back unchanged, or convert them in one of two trivial ways.
*/
unsafe fn fast_call(act: &mut Act, frame: *mut Value<'static>, recv: usize, index: usize, n: usize, ) -> Option<Result<Value<'static>, Error<'static>>> {
    if n > 16 || !DIRECT_CALLS.load(AtomicOrdering::Relaxed) || !fast_calls() || crate::flashnx_as3prof::enabled() {
        return None;
    }
    let receiver = unsafe { *frame.add(recv) };
    let Value::Object(obj) = receiver else { return None };
    if obj.get_bound_method(index).is_some() {
        return None;
    }
    let full = obj.vtable().get_full_method(index)?;
    let method = full.method;
    let code = method.flashnx_jit().get();
    if code == REJECTED || code <= 0xFFFF {
        return None;
    }
    if method.needs_arguments_object() || method.is_variadic() {
        return None;
    }
    let num_locals = method.body()?.num_locals as usize;
    let signature = method.resolved_param_config();
    if signature.len() != n || num_locals < n + 1 {
        return None;
    }
    // 0: as is, 1: int -> Number, 2: undefined -> null (what coerce_to_type does).
    let mut conv = [0u8; 16];
    for (i, param) in signature.iter().enumerate() {
        let v = unsafe { *frame.add(recv + 1 + i) };
        let Some(class) = param.param_type else { continue };
        conv[i] = if class.is_builtin_int() {
            if !matches!(v, Value::Integer(_)) {
                return None;
            }
            0
        } else if class.is_builtin_uint() {
            if !matches!(v, Value::Integer(x) if x >= 0) {
                return None;
            }
            0
        } else if class.is_builtin_number() {
            match v {
                Value::Number(_) => 0,
                Value::Integer(_) => 1,
                _ => return None,
            }
        } else if class.is_builtin_boolean() {
            if !matches!(v, Value::Bool(_)) {
                return None;
            }
            0
        } else if class.is_builtin_void() {
            return None;
        } else {
            match v {
                Value::Null => 0,
                Value::Undefined => 2,
                Value::String(_) if class.is_builtin_string() => 0,
                _ if class.is_builtin_object() => 0,
                Value::Object(o) if o.is_of_type(class) => 0,
                _ => return None,
            }
        };
    }

    // exec(), bytecode branch, with init_from_method's work inline.
    let mc = act.gc();
    let caller_dxns = act.default_xml_namespace();
    let stack = act.context.avm2.stack;
    let stack_frame = stack.get_stack_frame(method);
    let mut callee = Activation::from_nothing(act.context);
    callee.num_locals = num_locals;
    callee.outer = full.scope();
    callee.caller_domain = Some(callee.outer.domain());
    callee.caller_movie = CallerMovie::Method(method);
    callee.bound_superclass_object = full.super_class_obj;
    callee.stack = stack_frame;
    callee.scope_depth = callee.context.avm2.scope_stack.len();
    callee.is_interpreter = method.is_interpreted();
    callee.push_stack(receiver);
    for i in 0..n {
        let v = unsafe { *frame.add(recv + 1 + i) };
        callee.push_stack(match (conv[i], v) {
            (1, Value::Integer(x)) => Value::Number(x as f64),
            (2, _) => Value::Null,
            _ => v,
        });
    }
    callee.reset_stack();
    if !method.sets_dxns() {
        callee.default_xml_namespace = caller_dxns;
    }
    callee.context.avm2.push_call(mc, method);
    let result = match run_compiled(&mut callee, method, code) {
        Outcome::Done(r) => r,
        Outcome::Resume(ip) => callee.run_actions_from(method, ip),
        Outcome::Interpret => callee.run_actions_from(method, 0),
    };
    callee.cleanup();
    drop(callee);
    act.context.avm2.pop_call(mc);
    Some(result)
}

macro_rules! fill_kinds {
    ($t:ident; $($k:literal)*) => {
        $( $t[$k] = h_op::<$k> as unsafe extern "C" fn(*mut Ctx, u32, u32, u32) -> u32 as usize; )*
    };
}

fn fill_helpers(t: &mut [usize; 256]) {
	fill_kinds!(t; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159);
    t[H_GENERIC] = h_op::<H_GENERIC> as unsafe extern "C" fn(*mut Ctx, u32, u32, u32) -> u32 as usize;
    t[H_CALL] = h_call_method as unsafe extern "C" fn(*mut Ctx, u32, u32, u32) -> u32 as usize;
    t[H_ARRAY_GET] = h_array_get as unsafe extern "C" fn(*mut Ctx, u32, u32, u32) -> u32 as usize;
    t[H_TRUTHY] = h_truthy as unsafe extern "C" fn(*mut Ctx, u32, u32, u32) -> u32 as usize;
    t[H_SWITCH] = h_switch as unsafe extern "C" fn(*mut Ctx, u32, u32, u32) -> u32 as usize;
    t[H_TIMEOUT] = h_timeout as unsafe extern "C" fn(*mut Ctx, u32, u32, u32) -> u32 as usize;
    t[H_RETURN] = h_return as unsafe extern "C" fn(*mut Ctx, u32, u32, u32) -> u32 as usize;
}

// ---------------------------------------------------------------- entry

pub(super) fn run<'a, 'gc>(act: &mut Activation<'a, 'gc>, method: Method<'gc>) -> Outcome<'gc> {
    let cell = method.flashnx_jit();
    let st = cell.get();
    let code = if st == REJECTED {
        return Outcome::Interpret;
    } else if st > 0xFFFF {
        st
    } else if st + 1 < THRESHOLD {
        cell.set(st + 1);
        return Outcome::Interpret;
    } else {
        let s = state();
        if s.layout.obj_tags.len() < 4 && s.layout.probes < 100_000 {
            probe_object(&mut s.layout, act.stack.value_at(0));
        }
        if !s.layout.header_probed {
            probe_header(&mut s.layout, act);
        }
        match compile(s, method) {
            Ok(addr) => {
                s.compiled += 1;
                cell.set(addr);
                addr
            }
            Err(reason) => {
                s.rejected += 1;
                *s.reasons.entry(reason).or_insert(0) += 1;
                cell.set(REJECTED);
                return Outcome::Interpret;
            }
        }
    };
    run_compiled(act, method, code)
}

// Run method's compiled code at "code" in "act" (set up as "exec" does).
fn run_compiled<'a, 'gc>(act: &mut Activation<'a, 'gc>, method: Method<'gc>, code: usize) -> Outcome<'gc> {
    let cell = method.flashnx_jit();
    // The profiling build's other A/B window: the code in the header (see "compile"), which is this one when there is no second compilation.
    let code = if ALT.load(AtomicOrdering::Relaxed) {
        unsafe { *((code - 16) as *const usize) }
    } else {
        code
    };
    if let Err(e) = act.timeout_check() {
        return Outcome::Done(Err(e));
    }
    state().runs += 1;
    let ops = method.get_verified_info().parsed_code.as_ptr();
    let mut ctx = Ctx {
        frame: act.stack.base_ptr() as *mut Value<'static>,
        helpers: state().helpers.as_ptr(),
        act: act as *mut Activation<'a, 'gc> as *mut Act,
        ops: ops as *const Op<'static>,
        num_locals: act.num_locals,
        result: None,
        resume_ip: 0,
    };
    let status = unsafe {
        let f: unsafe extern "C" fn(*mut Ctx) -> u32 = std::mem::transmute(code);
        f(&mut ctx)
    };
    match status {
        1 | 2 => {
            let r = ctx.result.take().expect("JIT status without a result");
            Outcome::Done(unsafe {
                std::mem::transmute::<Result<Value<'static>, Error<'static>>, Result<Value<'gc>, Error<'gc>>>(r)
            })
        }
        3 => {
            cell.set(REJECTED);
            Outcome::Resume(ctx.resume_ip)
        }
        _ => {
            cell.set(REJECTED);
            Outcome::Done(Err("FlashNX JIT: compiled code ran off its end".into()))
        }
    }
}

/*
	A new game starts: the previous one's player, and every method compiled for it, are gone, so its code memory is free again.
	The measured layout is Ruffle's, not the game's: it stays.
*/
pub fn reset() {
    let s = state();
    s.used = 0;
    s.compiled = 0;
    s.rejected = 0;
    s.reasons.clear();
    s.deopts = 0;
    s.deopt_ops.clear();
    s.runs = 0;
    s.map.clear();
}

// Counters since the last call, then reset (the totals stay).
pub fn report() -> String {
    let s = state();
    if s.init == 0 {
        return String::new();
    }
    let top = |m: &FnvHashMap<String, u64>| {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        v.iter().take(8).map(|(k, n)| format!("{k} x{n}")).collect::<Vec<_>>().join(", ")
    };
    let l = &s.layout;
    let out = format!(
        "{} (rc=0x{:x}, bool {}, numbers {}, setslot {}, getslot {:?} slots@{}/{} {}), {} methods compiled ({} KB), {} rejected [{}], {} deopts [{}], {} runs, calls {} direct / {} slow",
        if s.init == 1 { "on" } else { "UNAVAILABLE" },
        s.init_rc,
        if l.bool_ok { "inline" } else { "helper" },
        if l.num_ok { "inline" } else { "helper" },
        if l.header_ok { "inline" } else { "helper" },
        l.obj_tags,
        l.slots_ptr_off,
        l.slots_len_off,
        l.obj_fail,
        s.compiled,
        s.used / 1024,
        s.rejected,
        top(&s.reasons),
        s.deopts,
        top(&s.deopt_ops),
        s.runs,
        s.calls_direct,
        s.calls_slow
    );
    s.runs = 0;
    s.calls_direct = 0;
    s.calls_slow = 0;
    // Measuring builds: the op kinds compiled code ran through their helper most often since the last report ("HELPER_N").
    #[cfg(feature = "flashnx_instr")]
    let out = {
        let mut counts: Vec<(u64, usize)> = (0..256)
            .map(|k| (HELPER_N[k].swap(0, AtomicOrdering::Relaxed), k))
            .filter(|(n, _)| *n > 0)
            .collect();
        counts.sort_by(|a, b| b.0.cmp(&a.0));
        let total: u64 = counts.iter().map(|(n, _)| n).sum();
        let top = counts.iter().take(14).map(|(n, k)| format!("{} x{n}", s.kind_names.get(k).map_or("?", |n| n.as_str()))).collect::<Vec<_>>().join(", ");
        format!("{out}, helpers {total} [{top}]")
    };
    out
}

impl<'a, 'gc> Activation<'a, 'gc> {
    // Every op that is not control flow, exactly as "run_actions_from" runs it.
    #[inline(always)]
    fn flashnx_exec_simple(&mut self, op: &Op<'gc>) -> Result<(), Error<'gc>> {
        match op {
                Op::PushDouble { value } => self.op_push_double(*value),
                Op::PushFalse => self.op_push_false(),
                Op::PushInt { value } => self.op_push_int(*value),
                Op::PushNamespace { namespace } => self.op_push_namespace(*namespace),
                Op::PushNull => self.op_push_null(),
                Op::PushString { string } => self.op_push_string(*string),
                Op::PushTrue => self.op_push_true(),
                Op::PushUint { value } => self.op_push_uint(*value),
                Op::PushUndefined => self.op_push_undefined(),
                Op::Pop => self.op_pop(),
                Op::Dup => self.op_dup(),
                Op::GetLocal { index } => self.op_get_local(*index),
                Op::SetLocal { index } => self.op_set_local(*index),
                Op::StoreLocal { index } => self.op_store_local(*index),
                Op::Kill { index } => self.op_kill(*index),
                Op::Call { num_args } => self.op_call(*num_args),
                Op::CallMethod {
                    index,
                    num_args,
                    push_return_value,
                } => self.op_call_method(*index, *num_args, *push_return_value),
                Op::CallNative {
                    method,
                    num_args,
                    push_return_value,
                } => self.op_call_native(*method, *num_args, *push_return_value),
                Op::CallProperty {
                    multiname,
                    num_args,
                } => self.op_call_property(*multiname, *num_args),
                Op::CallPropLex {
                    multiname,
                    num_args,
                } => self.op_call_prop_lex(*multiname, *num_args),
                Op::CallPropVoid {
                    multiname,
                    num_args,
                } => self.op_call_prop_void(*multiname, *num_args),
                Op::CallStatic { method, num_args } => self.op_call_static(*method, *num_args),
                Op::CallSuper {
                    multiname,
                    num_args,
                } => self.op_call_super(*multiname, *num_args),
                Op::GetPropertyStatic { multiname } => self.op_get_property_static(*multiname),
                Op::GetPropertyFast { multiname } => self.op_get_property_fast(*multiname),
                Op::GetPropertySlow { multiname } => self.op_get_property_slow(*multiname),
                Op::SetPropertyStatic { multiname } => self.op_set_property_static(*multiname),
                Op::SetPropertyFast { multiname } => self.op_set_property_fast(*multiname),
                Op::SetPropertySlow { multiname } => self.op_set_property_slow(*multiname),
                Op::InitProperty { multiname } => self.op_init_property(*multiname),
                Op::DeleteProperty { multiname } => self.op_delete_property(*multiname),
                Op::GetSuper { multiname } => self.op_get_super(*multiname),
                Op::SetSuper { multiname } => self.op_set_super(*multiname),
                Op::In => self.op_in(),
                Op::PushScope => self.op_push_scope(),
                Op::NewCatch { .. } => unreachable!("NewCatch is never compiled"),
                Op::PushWith => self.op_push_with(),
                Op::PopScope => self.op_pop_scope(),
                Op::GetOuterScope { index } => self.op_get_outer_scope(*index),
                Op::GetScopeObject { index } => self.op_get_scope_object(*index),
                Op::FindDef { multiname } => self.op_find_def(*multiname),
                Op::FindProperty { multiname } => self.op_find_property(*multiname),
                Op::FindPropStrict { multiname } => self.op_find_prop_strict(*multiname),
                Op::GetScriptGlobals { script } => self.op_get_script_globals(*script),
                Op::GetDescendants { multiname } => self.op_get_descendants(*multiname),
                Op::GetSlot { index } => self.op_get_slot(*index),
                Op::SetSlot { index } => self.op_set_slot(*index),
                Op::SetSlotCoerceI { index } => self.op_set_slot_coerce_i(*index),
                Op::SetSlotNoCoerce { index } => self.op_set_slot_no_coerce(*index),
                Op::SetGlobalSlot { index } => self.op_set_global_slot(*index),
                Op::Construct { num_args } => self.op_construct(*num_args),
                Op::ConstructProp {
                    multiname,
                    num_args,
                } => self.op_construct_prop(*multiname, *num_args),
                Op::ConstructSlot { index, num_args } => self.op_construct_slot(*index, *num_args),
                Op::ConstructSuper { num_args } => self.op_construct_super(*num_args),
                Op::NewActivation { activation_class } => self.op_new_activation(*activation_class),
                Op::NewObject { num_args } => self.op_new_object(*num_args),
                Op::NewFunction { method } => self.op_new_function(*method),
                Op::NewClass { class } => self.op_new_class(*class),
                Op::ApplyType { num_types } => self.op_apply_type(*num_types),
                Op::NewArray { num_args } => self.op_new_array(*num_args),
                Op::CoerceA => Ok(()),
                Op::CoerceB => self.op_coerce_b(),
                Op::CoerceD => self.op_coerce_d(),
                Op::CoerceDSwapPop => self.op_coerce_d_swap_pop(),
                Op::CoerceI => self.op_coerce_i(),
                Op::CoerceISwapPop => self.op_coerce_i_swap_pop(),
                Op::CoerceO => self.op_coerce_o(),
                Op::CoerceS => self.op_coerce_s(),
                Op::CoerceU => self.op_coerce_u(),
                Op::CoerceUSwapPop => self.op_coerce_u_swap_pop(),
                Op::ConvertO => self.op_convert_o(),
                Op::ConvertS => self.op_convert_s(),
                Op::Add => self.op_add(),
                Op::AddI => self.op_add_i(),
                Op::BitAnd => self.op_bitand(),
                Op::BitNot => self.op_bitnot(),
                Op::BitOr => self.op_bitor(),
                Op::BitXor => self.op_bitxor(),
                Op::DecLocal { index } => self.op_declocal(*index),
                Op::DecLocalI { index } => self.op_declocal_i(*index),
                Op::Decrement => self.op_decrement(),
                Op::DecrementI => self.op_decrement_i(),
                Op::Divide => self.op_divide(),
                Op::IncLocal { index } => self.op_inclocal(*index),
                Op::IncLocalI { index } => self.op_inclocal_i(*index),
                Op::Increment => self.op_increment(),
                Op::IncrementI => self.op_increment_i(),
                Op::LShift => self.op_lshift(),
                Op::Modulo => self.op_modulo(),
                Op::Multiply => self.op_multiply(),
                Op::MultiplyI => self.op_multiply_i(),
                Op::Negate => self.op_negate(),
                Op::NegateI => self.op_negate_i(),
                Op::RShift => self.op_rshift(),
                Op::Subtract => self.op_subtract(),
                Op::SubtractI => self.op_subtract_i(),
                Op::Swap => self.op_swap(),
                Op::URShift => self.op_urshift(),
                Op::StrictEquals => self.op_strict_equals(),
                Op::Equals => self.op_equals(),
                Op::GreaterEquals => self.op_greater_equals(),
                Op::GreaterThan => self.op_greater_than(),
                Op::LessEquals => self.op_less_equals(),
                Op::LessThan => self.op_less_than(),
                Op::Nop => Ok(()),
                Op::Not => self.op_not(),
                Op::HasNext => self.op_has_next(),
                Op::HasNext2 {
                    object_register,
                    index_register,
                } => self.op_has_next_2(*object_register, *index_register),
                Op::NextName => self.op_next_name(),
                Op::NextValue => self.op_next_value(),
                Op::IsType { class } => self.op_is_type(*class),
                Op::IsTypeLate => self.op_is_type_late(),
                Op::AsType { class } => self.op_as_type(*class),
                Op::AsTypeLate => self.op_as_type_late(),
                Op::InstanceOf => self.op_instance_of(),
                Op::Debug {
                    is_local_register,
                    register_name,
                    register,
                } => self.op_debug(*is_local_register, *register_name, *register),
                Op::DebugFile { file_name } => self.op_debug_file(*file_name),
                Op::DebugLine { line_num } => self.op_debug_line(*line_num),
                Op::Bkpt => self.op_bkpt(),
                Op::BkptLine { line_num } => self.op_bkpt_line(*line_num),
                Op::Timestamp => self.op_timestamp(),
                Op::TypeOf => self.op_type_of(),
                Op::Dxns { string } => self.op_dxns(*string),
                Op::DxnsLate => self.op_dxns_late(),
                Op::EscXAttr => self.op_esc_xattr(),
                Op::EscXElem => self.op_esc_elem(),
                Op::Coerce { class } => self.op_coerce(*class),
                Op::CoerceSwapPop { class } => self.op_coerce_swap_pop(*class),
                Op::CheckFilter => self.op_check_filter(),
                Op::Si8 => self.op_si8(),
                Op::Si16 => self.op_si16(),
                Op::Si32 => self.op_si32(),
                Op::Sf32 => self.op_sf32(),
                Op::Sf64 => self.op_sf64(),
                Op::Li8 => self.op_li8(),
                Op::Li16 => self.op_li16(),
                Op::Li32 => self.op_li32(),
                Op::Lf32 => self.op_lf32(),
                Op::Lf64 => self.op_lf64(),
                Op::Sxi1 => self.op_sxi1(),
                Op::Sxi8 => self.op_sxi8(),
                Op::Sxi16 => self.op_sxi16(),
                Op::Throw => self.op_throw(),

                _ => unreachable!("control flow is compiled inline"),
            }
        }
    }
