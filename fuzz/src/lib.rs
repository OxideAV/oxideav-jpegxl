//! Runtime libjxl interop for the cross-decode fuzz harness.
//!
//! libjxl is loaded via `dlopen` at first call — there is no
//! `jxl-sys`-style build-script dep that would pull libjxl source
//! into the workspace's cargo dep tree (workspace policy bars
//! external library code as reference, no exceptions). The harness
//! checks [`libjxl::available`] up front and `return`s early when the
//! shared library isn't installed, so a fuzz binary built on a host
//! without libjxl simply does nothing instead of panicking.
//!
//! Install libjxl with `brew install jpeg-xl` (macOS) or
//! `apt install libjxl-dev` (Debian/Ubuntu). The loader probes the
//! conventional shared-object names for both platforms.
//!
//! libjxl is fairly young — at the time of writing the conventional
//! release on Homebrew is 0.11 and 0.10 is still common on Linux
//! distros. The candidate list covers both.

#![allow(unsafe_code)]

pub mod libjxl {
    use libloading::{Library, Symbol};
    use std::sync::OnceLock;

    /// Conventional libjxl shared-object names the loader will try
    /// in order. Covers macOS (`.dylib`), Linux (versioned + plain
    /// `.so`), and Windows (`.dll`).
    const CANDIDATES: &[&str] = &[
        "libjxl.dylib",
        "libjxl.0.11.dylib",
        "libjxl.0.10.dylib",
        "libjxl.so.0.11",
        "libjxl.so.0.10",
        "libjxl.so",
        "jxl.dll",
    ];

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: `Library::new` is documented as unsafe because
                // the loaded library may run code at load time. We
                // accept that risk for fuzz tooling — libjxl is a
                // well-behaved shared library.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    /// True iff a libjxl shared library was successfully loaded.
    /// The cross-decode fuzz harness early-returns when this is false
    /// so the binary still runs without an oracle (the assertions just
    /// don't fire).
    pub fn available() -> bool {
        lib().is_some()
    }

    // --- libjxl C ABI types we touch -----------------------------------------
    //
    // These mirror the public headers (`<jxl/encode.h>`, `<jxl/types.h>`,
    // `<jxl/codestream_header.h>`, `<jxl/color_encoding.h>`). We do NOT
    // pull in libjxl source — only the ABI shape, which is part of the
    // documented public API.

    /// `JxlPixelFormat` from `<jxl/types.h>`. RGBA / 8-bit / native-endian
    /// / no-alignment is what we feed the encoder.
    #[repr(C)]
    struct JxlPixelFormat {
        num_channels: u32,
        data_type: u32,  // JxlDataType (JXL_TYPE_UINT8 = 2)
        endianness: u32, // JxlEndianness (JXL_NATIVE_ENDIAN = 0)
        align: usize,
    }

    const JXL_TYPE_UINT8: u32 = 2;
    const JXL_NATIVE_ENDIAN: u32 = 0;
    const JXL_TRUE: i32 = 1;
    const JXL_FALSE: i32 = 0;

    const JXL_ENC_SUCCESS: i32 = 0;
    #[allow(dead_code)]
    const JXL_ENC_ERROR: i32 = 1;
    const JXL_ENC_NEED_MORE_OUTPUT: i32 = 2;

    /// `JxlBasicInfo` from `<jxl/codestream_header.h>` is large (~272 B
    /// including 100-byte tail padding). We call `JxlEncoderInitBasicInfo`
    /// to populate defaults, then poke a handful of fields by absolute
    /// offset. Field offsets here come from the 0.11 header layout and
    /// the libjxl ABI contract (the struct ends in 100 bytes of padding
    /// reserved for forwards-compat). We keep the buffer over-sized so
    /// any future field additions still fit safely.
    ///
    /// We DON'T poke fields by offset — too brittle. Instead we mirror
    /// the front part of the struct as a `#[repr(C)]` type that ends
    /// with a generous `tail` buffer. `JxlEncoderInitBasicInfo` will
    /// initialise the head, we patch the head, and the tail is zeroed
    /// on construction.
    #[repr(C)]
    struct JxlBasicInfoHead {
        have_container: i32,
        xsize: u32,
        ysize: u32,
        bits_per_sample: u32,
        exponent_bits_per_sample: u32,
        intensity_target: f32,
        min_nits: f32,
        relative_to_max_display: i32,
        linear_below: f32,
        uses_original_profile: i32,
        have_preview: i32,
        have_animation: i32,
        orientation: u32,
        num_color_channels: u32,
        num_extra_channels: u32,
        alpha_bits: u32,
        alpha_exponent_bits: u32,
        alpha_premultiplied: i32,
        // JxlPreviewHeader { uint32_t xsize; uint32_t ysize; }
        preview_xsize: u32,
        preview_ysize: u32,
        // JxlAnimationHeader { tps_num, tps_den, num_loops, have_timecodes }
        anim_tps_num: u32,
        anim_tps_den: u32,
        anim_num_loops: u32,
        anim_have_timecodes: i32,
        intrinsic_xsize: u32,
        intrinsic_ysize: u32,
    }

    /// Wraps the head plus libjxl's reserved 100-byte forwards-compat
    /// padding, plus extra slack for safety. Total >= 256 bytes.
    #[repr(C)]
    struct JxlBasicInfoBuf {
        head: JxlBasicInfoHead,
        padding: [u8; 160],
    }

    /// `JxlColorEncoding` is small but the exact layout depends on the
    /// libjxl version. We over-allocate and let `JxlColorEncodingSetToSRGB`
    /// populate it.
    #[repr(C)]
    struct JxlColorEncodingBuf {
        bytes: [u8; 256],
    }

    /// Encode an RGBA image losslessly via the libjxl C API.
    /// Returns `None` if libjxl isn't available or any step of the
    /// encoding pipeline fails.
    pub fn encode_lossless_rgba(rgba: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
        let l = lib()?;
        let expected = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)?;
        if rgba.len() != expected || width == 0 || height == 0 {
            return None;
        }

        // Symbol typedefs (kept at call site for readability).
        type EncCreateFn = unsafe extern "C" fn(*const std::ffi::c_void) -> *mut std::ffi::c_void;
        type EncDestroyFn = unsafe extern "C" fn(*mut std::ffi::c_void);
        type EncUseContainerFn = unsafe extern "C" fn(*mut std::ffi::c_void, i32) -> i32;
        type EncInitBasicInfoFn = unsafe extern "C" fn(*mut JxlBasicInfoBuf);
        type EncSetBasicInfoFn =
            unsafe extern "C" fn(*mut std::ffi::c_void, *const JxlBasicInfoBuf) -> i32;
        type ColorSetSrgbFn = unsafe extern "C" fn(*mut JxlColorEncodingBuf, i32);
        type EncSetColorEncodingFn =
            unsafe extern "C" fn(*mut std::ffi::c_void, *const JxlColorEncodingBuf) -> i32;
        type EncFrameSettingsCreateFn = unsafe extern "C" fn(
            *mut std::ffi::c_void,
            *const std::ffi::c_void,
        ) -> *mut std::ffi::c_void;
        type EncSetFrameLosslessFn = unsafe extern "C" fn(*mut std::ffi::c_void, i32) -> i32;
        type EncAddImageFrameFn = unsafe extern "C" fn(
            *const std::ffi::c_void,
            *const JxlPixelFormat,
            *const u8,
            usize,
        ) -> i32;
        type EncCloseInputFn = unsafe extern "C" fn(*mut std::ffi::c_void);
        type EncProcessOutputFn =
            unsafe extern "C" fn(*mut std::ffi::c_void, *mut *mut u8, *mut usize) -> i32;

        unsafe {
            let create: Symbol<EncCreateFn> = l.get(b"JxlEncoderCreate").ok()?;
            let destroy: Symbol<EncDestroyFn> = l.get(b"JxlEncoderDestroy").ok()?;
            let use_container: Symbol<EncUseContainerFn> = l.get(b"JxlEncoderUseContainer").ok()?;
            let init_basic: Symbol<EncInitBasicInfoFn> = l.get(b"JxlEncoderInitBasicInfo").ok()?;
            let set_basic: Symbol<EncSetBasicInfoFn> = l.get(b"JxlEncoderSetBasicInfo").ok()?;
            let set_srgb: Symbol<ColorSetSrgbFn> = l.get(b"JxlColorEncodingSetToSRGB").ok()?;
            let set_color: Symbol<EncSetColorEncodingFn> =
                l.get(b"JxlEncoderSetColorEncoding").ok()?;
            let fs_create: Symbol<EncFrameSettingsCreateFn> =
                l.get(b"JxlEncoderFrameSettingsCreate").ok()?;
            let set_lossless: Symbol<EncSetFrameLosslessFn> =
                l.get(b"JxlEncoderSetFrameLossless").ok()?;
            let add_frame: Symbol<EncAddImageFrameFn> = l.get(b"JxlEncoderAddImageFrame").ok()?;
            let close_input: Symbol<EncCloseInputFn> = l.get(b"JxlEncoderCloseInput").ok()?;
            let process: Symbol<EncProcessOutputFn> = l.get(b"JxlEncoderProcessOutput").ok()?;

            let enc = create(std::ptr::null());
            if enc.is_null() {
                return None;
            }

            // RAII-ish guard: ensure JxlEncoderDestroy runs even on early
            // return. We capture the symbol's function pointer by copy
            // (it's `Copy` once dereferenced from the `Symbol<T>`).
            let destroy_fn: EncDestroyFn = *destroy;
            struct EncGuard {
                enc: *mut std::ffi::c_void,
                destroy: EncDestroyFn,
            }
            impl Drop for EncGuard {
                fn drop(&mut self) {
                    unsafe { (self.destroy)(self.enc) };
                }
            }
            let _guard = EncGuard {
                enc,
                destroy: destroy_fn,
            };

            // Force container framing — keeps the codestream cleanly
            // wrapped in ISOBMFF, matching what oxideav-jpegxl's
            // container::detect handles best.
            if use_container(enc, JXL_TRUE) != JXL_ENC_SUCCESS {
                return None;
            }

            let mut info = JxlBasicInfoBuf {
                head: std::mem::zeroed(),
                padding: [0u8; 160],
            };
            init_basic(&mut info);
            info.head.xsize = width;
            info.head.ysize = height;
            info.head.bits_per_sample = 8;
            info.head.exponent_bits_per_sample = 0;
            info.head.num_color_channels = 3;
            info.head.num_extra_channels = 1;
            info.head.alpha_bits = 8;
            info.head.alpha_exponent_bits = 0;
            // Lossless requires uses_original_profile=true so the
            // encoder doesn't transform into XYB.
            info.head.uses_original_profile = JXL_TRUE;

            if set_basic(enc, &info) != JXL_ENC_SUCCESS {
                return None;
            }

            let mut ce = JxlColorEncodingBuf { bytes: [0u8; 256] };
            set_srgb(&mut ce, JXL_FALSE);
            if set_color(enc, &ce) != JXL_ENC_SUCCESS {
                return None;
            }

            let settings = fs_create(enc, std::ptr::null());
            if settings.is_null() {
                return None;
            }
            if set_lossless(settings, JXL_TRUE) != JXL_ENC_SUCCESS {
                return None;
            }

            let pf = JxlPixelFormat {
                num_channels: 4,
                data_type: JXL_TYPE_UINT8,
                endianness: JXL_NATIVE_ENDIAN,
                align: 0,
            };
            if add_frame(settings, &pf, rgba.as_ptr(), rgba.len()) != JXL_ENC_SUCCESS {
                return None;
            }
            close_input(enc);

            // Drain the encoder into a growable buffer.
            let mut out: Vec<u8> = Vec::with_capacity(64);
            loop {
                let chunk = 4096usize;
                let prev_len = out.len();
                out.resize(prev_len + chunk, 0);
                let mut next_out: *mut u8 = out.as_mut_ptr().add(prev_len);
                let mut avail_out: usize = chunk;
                let st = process(enc, &mut next_out, &mut avail_out);
                let written = chunk - avail_out;
                out.truncate(prev_len + written);
                match st {
                    JXL_ENC_SUCCESS => break,
                    JXL_ENC_NEED_MORE_OUTPUT => continue,
                    _ => return None,
                }
            }

            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
    }
}
