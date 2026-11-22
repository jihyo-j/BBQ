#!/usr/bin/env python3
"""Fail-fast layout checks for SPANN + RaBitQ experimental artifacts."""

from __future__ import annotations

import argparse
import configparser
import math
import struct
from dataclasses import dataclass
from pathlib import Path


PAGE_BYTES = 4096


@dataclass(frozen=True)
class MatrixHeader:
    rows: int
    dim: int


def read_matrix_header(path: Path) -> MatrixHeader:
    with path.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"short matrix header: {path}")
    rows, dim = struct.unpack("<ii", raw)
    if rows <= 0 or dim <= 0:
        raise RuntimeError(f"invalid matrix shape ({rows}, {dim}): {path}")
    return MatrixHeader(rows, dim)


def validate_dense_matrix(path: Path, bytes_per_value: int) -> MatrixHeader:
    header = read_matrix_header(path)
    expected = 8 + header.rows * header.dim * bytes_per_value
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"matrix size mismatch: {path}: {actual} != {expected}")
    return header


def expected_rabitq_stride(padded_dim: int, bits: int) -> tuple[int, int, int]:
    if padded_dim <= 0 or padded_dim % 64:
        raise RuntimeError(f"invalid RaBitQ padded dimension: {padded_dim}")
    if not 1 <= bits <= 8:
        raise RuntimeError(f"invalid RaBitQ bit width: {bits}")
    bin_stride = padded_dim // 8 + 3 * 4
    ex_bits = bits - 1
    ex_stride = padded_dim * ex_bits // 8 + 2 * 4 if ex_bits else 0
    return bin_stride, ex_stride, bin_stride + ex_stride


def validate_quantizer(path: Path, base: MatrixHeader, expected_bits: int) -> int:
    raw = path.read_bytes()
    fixed_size = 2 + 7 * 4
    if len(raw) < fixed_size + 4:
        raise RuntimeError(f"short RaBitQ quantizer: {path}")

    quantizer_type, reconstruct_type = struct.unpack_from("<BB", raw, 0)
    dim, padded_dim, bits, ex_bits, num_points, bin_stride, ex_stride = struct.unpack_from(
        "<7i", raw, 2
    )
    if quantizer_type != 3:
        raise RuntimeError(f"not a RaBitQ quantizer (type={quantizer_type}): {path}")
    if reconstruct_type != 3:
        raise RuntimeError(f"unexpected reconstruct type {reconstruct_type}: {path}")
    if (dim, bits, ex_bits, num_points) != (
        base.dim,
        expected_bits,
        expected_bits - 1,
        base.rows,
    ):
        raise RuntimeError(
            "RaBitQ quantizer header mismatch: "
            f"got dim={dim}, bits={bits}, ex_bits={ex_bits}, N={num_points}; "
            f"expected dim={base.dim}, bits={expected_bits}, "
            f"ex_bits={expected_bits - 1}, N={base.rows}"
        )

    expected_bin, expected_ex, code_stride = expected_rabitq_stride(padded_dim, bits)
    if (bin_stride, ex_stride) != (expected_bin, expected_ex):
        raise RuntimeError(
            f"RaBitQ stride mismatch: got ({bin_stride}, {ex_stride}), "
            f"expected ({expected_bin}, {expected_ex})"
        )

    rotator_size_offset = fixed_size + padded_dim * 4
    if rotator_size_offset + 4 > len(raw):
        raise RuntimeError(f"truncated RaBitQ centroid: {path}")
    (rotator_size,) = struct.unpack_from("<i", raw, rotator_size_offset)
    if rotator_size <= 0:
        raise RuntimeError(f"invalid RaBitQ rotator size {rotator_size}: {path}")
    expected_bytes = rotator_size_offset + 4 + rotator_size
    if len(raw) != expected_bytes:
        raise RuntimeError(f"RaBitQ quantizer payload mismatch: {len(raw)} != {expected_bytes}")
    return code_stride


def validate_quantized_vectors(path: Path, base: MatrixHeader, code_stride: int) -> None:
    header = validate_dense_matrix(path, 1)
    if header != MatrixHeader(base.rows, code_stride):
        raise RuntimeError(
            f"quantized vector header mismatch: got {header}, "
            f"expected N={base.rows}, stride={code_stride}"
        )


def validate_spann_ssd(path: Path, base: MatrixHeader, code_stride: int) -> None:
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(16)
        if len(raw) != 16:
            raise RuntimeError(f"short SPANN SSD header: {path}")
        list_count, total_docs, vector_dim, list_page_offset = struct.unpack("<4i", raw)
        if list_count < 0 or list_page_offset <= 0:
            raise RuntimeError(f"invalid SPANN SSD header values: {path}")
        if total_docs != base.rows or vector_dim != code_stride:
            raise RuntimeError(
                "SPANN SSD header mismatch: "
                f"docs={total_docs}/{base.rows}, dim={vector_dim}/{code_stride}"
            )
        header_bytes = list_page_offset * PAGE_BYTES
        metadata_bytes = 16 + list_count * 12
        if metadata_bytes > header_bytes or header_bytes > file_bytes:
            raise RuntimeError(
                f"SPANN header/page boundary mismatch: metadata={metadata_bytes}, "
                f"posting_start={header_bytes}, file={file_bytes}"
            )

        vector_bytes = code_stride + 4
        max_end = header_bytes
        for list_id in range(list_count):
            entry = handle.read(12)
            if len(entry) != 12:
                raise RuntimeError(f"truncated SPANN list metadata at list {list_id}")
            page_num, page_offset, element_count, page_count = struct.unpack("<iHiH", entry)
            if page_num < 0 or not 0 <= page_offset < PAGE_BYTES:
                raise RuntimeError(f"invalid SPANN list offset at list {list_id}")
            if element_count < 0:
                raise RuntimeError(f"negative SPANN element count at list {list_id}")
            payload_bytes = element_count * vector_bytes
            expected_pages = (
                math.ceil((page_offset + payload_bytes) / PAGE_BYTES) if payload_bytes else 0
            )
            if page_count != expected_pages:
                raise RuntimeError(
                    f"SPANN page-count mismatch at list {list_id}: "
                    f"header={page_count}, required={expected_pages}"
                )
            start = header_bytes + page_num * PAGE_BYTES + page_offset
            end = start + payload_bytes
            if end > file_bytes:
                raise RuntimeError(
                    f"SPANN posting exceeds file at list {list_id}: {end} > {file_bytes}"
                )
            max_end = max(max_end, end)

    print(
        f"SPANN_SSD_OK lists={list_count} docs={total_docs} "
        f"code_stride={code_stride} max_payload_end={max_end} file_bytes={file_bytes}"
    )


def validate_fp_sidecar(path: Path, base: MatrixHeader) -> None:
    with path.open("rb") as handle:
        raw = handle.read(32)
    record_bytes = math.ceil(base.dim * 4 / PAGE_BYTES) * PAGE_BYTES
    expected_header = (base.rows, base.dim, record_bytes, PAGE_BYTES)
    if len(raw) != 32 or struct.unpack("<QQQQ", raw) != expected_header:
        raise RuntimeError(f"FP sidecar header mismatch: {path}")
    expected = PAGE_BYTES + base.rows * record_bytes
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"FP sidecar size mismatch: {actual} != {expected}")


def validate_index_loader(path: Path, quantizer: Path) -> None:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    if not parser.read(path, encoding="utf-8"):
        raise RuntimeError(f"cannot read SPANN index loader: {path}")
    expected = quantizer.name
    actual = parser.get("Quantizer", "QuantizerFilePath", fallback="")
    if actual != expected:
        raise RuntimeError(
            f"SPANN loader quantizer mismatch: got {actual!r}, expected {expected!r}"
        )
    if parser.get("Index", "IndexAlgoType", fallback="") != "SPANN":
        raise RuntimeError(f"SPANN loader index type mismatch: {path}")
    if parser.get("Index", "ValueType", fallback="") != "Float":
        raise RuntimeError(f"SPANN loader value type mismatch: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--quantizer", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=1)
    parser.add_argument("--spann-ssd", type=Path)
    parser.add_argument("--fp-sidecar", type=Path)
    parser.add_argument("--index-loader", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = validate_dense_matrix(args.base, 4)
    code_stride = validate_quantizer(args.quantizer, base, args.bits)
    validate_quantized_vectors(args.quantized, base, code_stride)
    if args.spann_ssd:
        validate_spann_ssd(args.spann_ssd, base, code_stride)
    if args.fp_sidecar:
        validate_fp_sidecar(args.fp_sidecar, base)
    if args.index_loader:
        validate_index_loader(args.index_loader, args.quantizer)
    print(
        f"SPANN_RABITQ_LAYOUT_OK N={base.rows} dim={base.dim} "
        f"bits={args.bits} code_stride={code_stride}"
    )


if __name__ == "__main__":
    main()
