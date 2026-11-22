use std::cell::RefCell;
use std::ffi::{CStr, c_char};
use std::fs::File;
use std::io::BufReader;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::atomic::AtomicBool;

use common::universal_io::MmapFs;
use quantization::encoded_storage::{
    EncodedStorageWrite, TestEncodedStorage, TestEncodedStorageBuilder,
};
use quantization::encoded_vectors::{DistanceType, EncodedVectors, VectorParameters};
use quantization::encoded_vectors_tq::{
    self, EncodedVectorsTQ, EncoderTQ, Metadata, new_turbo_quantizer_from_metadata,
};
use quantization::turboquant::{EncodedQueryTQ, TQBits, TQMode, TQRotation};
use rayon::prelude::*;

type Index = EncodedVectorsTQ<TestEncodedStorage>;

pub struct QdrantTqIndex {
    encoded: Index,
    distance: DistanceType,
    meta_path: PathBuf,
}

pub struct QdrantTqQuery {
    encoded: EncodedQueryTQ,
}

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn set_error(message: impl ToString) {
    LAST_ERROR.with(|slot| {
        let mut bytes = message.to_string().into_bytes();
        bytes.retain(|byte| *byte != 0);
        bytes.push(0);
        *slot.borrow_mut() = bytes;
    });
}

fn ffi_ptr<T>(f: impl FnOnce() -> Result<T, String>) -> *mut T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => Box::into_raw(Box::new(value)),
        Ok(Err(error)) => {
            set_error(error);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error("panic in Qdrant TurboQuant FFI");
            std::ptr::null_mut()
        }
    }
}

fn ffi_bool(f: impl FnOnce() -> Result<(), String>) -> bool {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => true,
        Ok(Err(error)) => {
            set_error(error);
            false
        }
        Err(_) => {
            set_error("panic in Qdrant TurboQuant FFI");
            false
        }
    }
}

fn c_path(pointer: *const c_char) -> Result<PathBuf, String> {
    if pointer.is_null() {
        return Err("null path".to_string());
    }
    let text = unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map_err(|error| format!("path is not UTF-8: {error}"))?;
    Ok(PathBuf::from(text))
}

fn distance_from_code(code: i32) -> Result<DistanceType, String> {
    match code {
        0 => Ok(DistanceType::Cosine),
        1 => Ok(DistanceType::L2),
        _ => Err(format!("unsupported distance code {code}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qdrant_tq_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        let bytes = slot.borrow();
        if bytes.is_empty() {
            c"".as_ptr()
        } else {
            bytes.as_ptr().cast()
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qdrant_tq_build(
    data: *const f32,
    rows: usize,
    dim: usize,
    distance_code: i32,
    num_threads: usize,
    data_path: *const c_char,
    meta_path: *const c_char,
) -> *mut QdrantTqIndex {
    ffi_ptr(|| {
        if data.is_null() || rows == 0 || dim == 0 || num_threads == 0 {
            return Err("invalid build arguments".to_string());
        }
        let distance = distance_from_code(distance_code)?;
        let data_path = c_path(data_path)?;
        let meta_path = c_path(meta_path)?;
        let values = unsafe { slice::from_raw_parts(data, rows.checked_mul(dim).ok_or("size overflow")?) };
        let vector_parameters = VectorParameters {
            dim,
            distance_type: distance,
            invert: false,
            deprecated_count: None,
        };
        let bits = TQBits::Bits1;
        let mode = TQMode::Plus;
        let stride = encoded_vectors_tq::get_quantized_vector_size(
            &vector_parameters,
            bits,
            mode,
        );
        let vectors = (0..rows).map(|row| &values[row * dim..(row + 1) * dim]);
        let encoded = EncodedVectorsTQ::encode(
            vectors,
            std::iter::repeat_n(true, rows),
            TestEncodedStorageBuilder::new(Some(&data_path), stride),
            &vector_parameters,
            bits,
            mode,
            TQRotation::Padded,
            false,
            num_threads,
            Some(&meta_path),
            &AtomicBool::new(false),
        )
        .map_err(|error| format!("Qdrant TurboQuant encode failed: {error}"))?;
        Ok(QdrantTqIndex {
            encoded,
            distance,
            meta_path,
        })
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn qdrant_tq_load(
    data_path: *const c_char,
    meta_path: *const c_char,
) -> *mut QdrantTqIndex {
    ffi_ptr(|| {
        let data_path = c_path(data_path)?;
        let meta_path = c_path(meta_path)?;
        let metadata: Metadata = serde_json::from_reader(BufReader::new(
            File::open(&meta_path).map_err(|error| error.to_string())?,
        ))
        .map_err(|error| format!("metadata parse failed: {error}"))?;
        let quantizer = new_turbo_quantizer_from_metadata(&metadata)
            .map_err(|error| format!("metadata validation failed: {error}"))?;
        let stride = quantizer.quantized_size();
        let storage = TestEncodedStorage::from_file(Path::new(&data_path), stride)
            .map_err(|error| format!("code load failed: {error}"))?;
        let distance = metadata.vector_parameters.distance_type;
        let encoded = EncodedVectorsTQ::load(&MmapFs, storage, &meta_path)
            .map_err(|error| format!("Qdrant TurboQuant load failed: {error}"))?;
        Ok(QdrantTqIndex {
            encoded,
            distance,
            meta_path,
        })
    })
}

/// Encode and append rows using the metadata fitted by the original build.
///
/// Encoding is parallel, while publication into the contiguous storage is
/// ordered so logical IDs remain [old rows, appended rows]. The in-memory
/// storage deliberately is not flushed back to the base code file: dynamic
/// insertion checkpoints own their appended codes only for the process
/// lifetime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdrant_tq_append(
    index: *mut QdrantTqIndex,
    data: *const f32,
    rows: usize,
    dim: usize,
    num_threads: usize,
) -> bool {
    ffi_bool(|| {
        let index = unsafe { index.as_mut() }.ok_or("null index")?;
        if rows == 0 {
            return Ok(());
        }
        if data.is_null() || dim == 0 || num_threads == 0 {
            return Err("invalid append arguments".to_string());
        }
        let expected_dim = index.encoded.get_metadata().vector_parameters.dim;
        if dim != expected_dim {
            return Err(format!(
                "append dimension mismatch: got {dim}, expected {expected_dim}",
            ));
        }
        let value_count = rows.checked_mul(dim).ok_or("append size overflow")?;
        let values = unsafe { slice::from_raw_parts(data, value_count) };
        let meta_path = index.meta_path.clone();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|error| format!("failed to create TQ append pool: {error}"))?;
        let encoded_rows: Vec<Vec<u8>> = pool.install(|| {
            values
                .par_chunks_exact(dim)
                .map_init(
                    || {
                        EncoderTQ::load(&MmapFs, &meta_path)
                            .map_err(|error| error.to_string())
                    },
                    |encoder, vector| match encoder {
                        Ok(encoder) => Ok(encoder.encode(vector)),
                        Err(error) => Err(error.clone()),
                    },
                )
                .collect::<Result<Vec<_>, String>>()
        })?;
        let stride = index.encoded.quantized_vector_size();
        let first = index.encoded.storage().as_bytes().len() / stride;
        for (local, encoded) in encoded_rows.iter().enumerate() {
            let id = u32::try_from(first + local)
                .map_err(|_| "TQ append row id exceeds u32".to_string())?;
            index
                .encoded
                .storage_mut()
                .upsert_vector(id, encoded)
                .map_err(|error| format!("TQ append publication failed: {error}"))?;
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdrant_tq_free(index: *mut QdrantTqIndex) {
    if !index.is_null() {
        unsafe { drop(Box::from_raw(index)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdrant_tq_query_new(
    index: *const QdrantTqIndex,
    query: *const f32,
    dim: usize,
) -> *mut QdrantTqQuery {
    ffi_ptr(|| {
        let index = unsafe { index.as_ref() }.ok_or("null index")?;
        if query.is_null() {
            return Err("null query".to_string());
        }
        let query = unsafe { slice::from_raw_parts(query, dim) };
        Ok(QdrantTqQuery {
            encoded: index.encoded.encode_query(query),
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdrant_tq_query_free(query: *mut QdrantTqQuery) {
    if !query.is_null() {
        unsafe { drop(Box::from_raw(query)) };
    }
}

/// Export the query representation used by the TQ+ Bits1 asymmetric scorer.
/// This keeps Qdrant's canonical rotation/quantization while allowing a C++
/// HNSW traversal to execute the per-candidate dot product without an FFI
/// crossing for every distance evaluation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdrant_tq_query_export_bits1_wide(
    query: *const QdrantTqQuery,
    signed_query: *mut i16,
    signed_query_len: usize,
    postprocess_scale: *mut f32,
    bias_correction: *mut i64,
    ec_correction: *mut f32,
    l2_norm: *mut f32,
) -> bool {
    let Some(query) = (unsafe { query.as_ref() }) else {
        return false;
    };
    if signed_query.is_null()
        || postprocess_scale.is_null()
        || bias_correction.is_null()
        || ec_correction.is_null()
        || l2_norm.is_null()
    {
        return false;
    }
    let output = unsafe { slice::from_raw_parts_mut(signed_query, signed_query_len) };
    let Some(params) = query.encoded.export_bits1_wide(output) else {
        return false;
    };
    unsafe {
        *postprocess_scale = params.postprocess_scale;
        *bias_correction = params.bias_correction;
        *ec_correction = params.ec_correction;
        *l2_norm = params.l2_norm.unwrap_or(0.0);
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdrant_tq_query_export_bits1_wide_planes(
    query: *const QdrantTqQuery,
    query_planes: *mut i8,
    query_planes_len: usize,
    postprocess_scale: *mut f32,
    bias_correction: *mut i64,
    ec_correction: *mut f32,
    l2_norm: *mut f32,
) -> bool {
    let Some(query) = (unsafe { query.as_ref() }) else {
        return false;
    };
    if query_planes.is_null()
        || postprocess_scale.is_null()
        || bias_correction.is_null()
        || ec_correction.is_null()
        || l2_norm.is_null()
    {
        return false;
    }
    let output = unsafe { slice::from_raw_parts_mut(query_planes, query_planes_len) };
    let Some(params) = query.encoded.export_bits1_wide_planes(output) else {
        return false;
    };
    unsafe {
        *postprocess_scale = params.postprocess_scale;
        *bias_correction = params.bias_correction;
        *ec_correction = params.ec_correction;
        *l2_norm = params.l2_norm.unwrap_or(0.0);
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdrant_tq_score(
    index: *const QdrantTqIndex,
    query: *const QdrantTqQuery,
    id: u32,
) -> f32 {
    let Some(index) = (unsafe { index.as_ref() }) else {
        return f32::NAN;
    };
    let Some(query) = (unsafe { query.as_ref() }) else {
        return f32::NAN;
    };
    let vector = index.encoded.get_quantized_vector(id);
    let score = index.encoded.score(&query.encoded, &vector);
    match index.distance {
        DistanceType::Cosine | DistanceType::Dot => -score,
        DistanceType::L2 | DistanceType::L1 => score,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdrant_tq_score_internal(
    index: *const QdrantTqIndex,
    left: u32,
    right: u32,
) -> f32 {
    let Some(index) = (unsafe { index.as_ref() }) else {
        return f32::NAN;
    };
    let score = index.encoded.score_internal(left, right);
    match index.distance {
        DistanceType::Cosine | DistanceType::Dot => -score,
        DistanceType::L2 | DistanceType::L1 => score,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdrant_tq_code_size(index: *const QdrantTqIndex) -> usize {
    unsafe { index.as_ref() }
        .map(|value| value.encoded.quantized_vector_size())
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdrant_tq_codes(
    index: *const QdrantTqIndex,
    byte_len: *mut usize,
) -> *const u8 {
    let Some(index) = (unsafe { index.as_ref() }) else {
        return std::ptr::null();
    };
    if byte_len.is_null() {
        return std::ptr::null();
    }
    let bytes = index.encoded.storage().as_bytes();
    unsafe { *byte_len = bytes.len() };
    bytes.as_ptr()
}

