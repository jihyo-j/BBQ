#!/usr/bin/env python3
"""Run the four disk baselines with a one-bit-equivalent routing-code budget.

The campaign covers four corpora (MS MARCO, DBpedia, Quora, and NQ) and four
embedding families (OpenAI text-embedding-3-small 1536D, BGE 1024D, Qwen
1024D, and E5-Mistral 4096D).  DiskANN-PQ and SPANN-PQ receive exactly
``dimension / 8`` PQ bytes per vector.  DiskANN-RaBitQ and PipeANN keep their
native one-bit database codes; their per-vector metadata is reported
separately.

The runner is restartable and imports old measurements only when the dataset,
held-out query count, code budget, search grid, thread count, and round count
are identical.  Large indexes are built one dataset at a time and deleted by
the delegated runners after durable result CSVs have been written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiment"
RESULT_ROOT = EXP / "onebit_disk_baselines_t1"
EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/onebit_disk_baselines_t1")
OPENAI_ROOT = Path("/home/smrc/samsung-nvme/jihyo/openai_beir_11methods_t1")
EXTRA_MANIFEST = OPENAI_ROOT / "dataset_manifest.json"
PREPARE = EXP / "prepare_openai_beir_11methods_inputs.py"
OLD_ROOT = EXP / "diskveclab_full_matrix"
PYTHON = Path("/home/jihyo/anaconda3/envs/rabitq-cfr/bin/python")
SYSTEM_PYTHON = Path("/usr/bin/python3")

OPENAI_KEYS = (
    "sieun_msmarco_openai1536",
    "sieun_dbpedia_entity_openai1536",
    "sieun_quora_openai1536",
    "sieun_nq_openai1536",
)
DBPEDIA_BGE_KEY = "sieun_dbpedia_entity_bge_large1024_beir400"
NEW_KEYS = (*OPENAI_KEYS, DBPEDIA_BGE_KEY)

DATASET_KEYS = (
    "sieun_msmarco_openai1536",
    "msmarco_bge1024",
    "msmarco_qwen1024",
    "msmarco_e5mistral4096",
    "sieun_dbpedia_entity_openai1536",
    DBPEDIA_BGE_KEY,
    "dbpedia_qwen1024",
    "dbpedia_e5mistral4096",
    "sieun_quora_openai1536",
    "quora_bge_m3_vanilla",
    "quora_qwen1024",
    "quora_e5mistral4096",
    "sieun_nq_openai1536",
    "nq_bge_m3_vanilla",
    "nq_qwen06_1024",
    "nq_e5mistral4096",
)

EXISTING_KEYS = tuple(key for key in DATASET_KEYS if key not in NEW_KEYS)
SPANN_REUSABLE_KEYS = (
    "msmarco_bge1024",
    "msmarco_qwen1024",
    "dbpedia_qwen1024",
    "quora_bge_m3_vanilla",
    "quora_qwen1024",
    "nq_bge_m3_vanilla",
    "nq_qwen06_1024",
)

DISKANN_L = (
    80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 224, 256, 288, 320, 384, 448, 512, 768, 1024,
)
SPANN_HEADS = (
    32, 36, 40, 44, 48, 52, 56, 64, 72, 80,
    96, 112, 128, 144, 160, 192, 224, 256, 384, 512,
)
PIPEANN_L = DISKANN_L


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def valid_curve(
    path: Path,
    *,
    knob: str,
    expected: Iterable[int],
    rounds: int,
    dim: int,
    nq: int,
    threads: int = 1,
) -> bool:
    try:
        rows = read_rows(path)
        return (
            sorted(int(row[knob]) for row in rows) == sorted(expected)
            and all(int(float(row.get("rounds", -1))) == rounds for row in rows)
            and all(int(float(row.get("dim", -1))) == dim for row in rows)
            and all(int(float(row.get("nq", -1))) == nq for row in rows)
            and all(
                int(float(row.get("search_threads", row.get("threads", -1)))) == threads
                for row in rows
            )
        )
    except (OSError, KeyError, ValueError):
        return False


def prepare_inputs(threads: int) -> None:
    command = [
        str(PYTHON), "-u", str(PREPARE),
        "--output-root", str(OPENAI_ROOT),
        "--dims", "1536",
        "--threads", str(threads),
    ]
    print(f"RUN {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def validate_extra_manifest() -> dict[str, object]:
    payload = json.loads(EXTRA_MANIFEST.read_text(encoding="utf-8"))
    specs = payload.get("datasets", [])
    keys = {str(item["key"]) for item in specs}
    if keys != set(NEW_KEYS):
        raise RuntimeError(
            f"OpenAI/BEIR manifest must contain exactly {sorted(NEW_KEYS)}, got {sorted(keys)}"
        )
    by_key = {str(item["key"]): item for item in specs}
    for key in OPENAI_KEYS:
        spec = by_key[key]
        if spec.get("model") != "OpenAI text-embedding-3-small" or int(spec["dim"]) != 1536:
            raise RuntimeError(f"not latest OpenAI 1536 input: {key}: {spec}")
    dbpedia_bge = by_key[DBPEDIA_BGE_KEY]
    if "BEIR-400" not in str(dbpedia_bge.get("model", "")) or int(dbpedia_bge["dim"]) != 1024:
        raise RuntimeError(f"not BEIR DBpedia BGE input: {dbpedia_bge}")
    return payload


def campaign_env() -> dict[str, str]:
    return {
        **os.environ,
        "DISKVECLAB_EXTRA_DATASETS_MANIFEST": str(EXTRA_MANIFEST),
        "DISKVECLAB_DATASETS": ",".join(DATASET_KEYS),
        "DISKVECLAB_RESULT_ROOT": str(RESULT_ROOT),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def dataset_metadata(
    env: dict[str, str], dataset_keys: Iterable[str] = DATASET_KEYS
) -> dict[str, dict[str, object]]:
    old = os.environ.copy()
    os.environ.update(env)
    try:
        import sys

        sys.path.insert(0, str(EXP))
        import diskveclab_full_matrix as matrix
        import run_v051_ssd_odirect_multidataset as registry

        metadata: dict[str, dict[str, object]] = {}
        for key in dataset_keys:
            dataset = registry.DATASETS[key]
            heldout = matrix.heldout_dataset(key)
            nb, dim = registry.matrix_shape(dataset.train)
            nq, _ = registry.matrix_shape(heldout.query)
            metadata[key] = {
                "dataset": key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "train": str(dataset.train),
                "query": str(dataset.query),
                "gt": str(dataset.gt),
                "heldout_query": str(heldout.query),
                "heldout_gt": str(heldout.gt),
                "nb": nb,
                "nq": nq,
                "dim": dim,
                "pq_bytes_per_vector": dim // 8,
            }
        return metadata
    finally:
        os.environ.clear()
        os.environ.update(old)


def copy_reusable(
    source: Path,
    target: Path,
    *,
    csv_path: Path,
    validation: dict[str, object],
) -> bool:
    if target.exists():
        return False
    if not source.is_dir() or not csv_path.is_file():
        return False
    if not valid_curve(csv_path, **validation):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    copied_csv = target / csv_path.relative_to(source)
    atomic_json(
        target / "reuse_manifest.json",
        {
            "reused_at": now(),
            "source": str(source),
            "source_csv": str(csv_path),
            "source_csv_sha256": sha256(csv_path),
            "compatibility": validation,
        },
    )
    print(f"REUSED {source} -> {target}", flush=True)
    if not copied_csv.is_file():
        raise RuntimeError(f"copy lost result CSV: {copied_csv}")
    return True


def import_compatible_results(
    metadata: dict[str, dict[str, object]], rounds: int, dataset_keys: Iterable[str]
) -> list[dict[str, str]]:
    reused: list[dict[str, str]] = []
    selected = set(dataset_keys)
    for key in EXISTING_KEYS:
        if key not in selected:
            continue
        spec = metadata[key]
        validation = {
            "knob": "Lsearch",
            "expected": DISKANN_L,
            "rounds": rounds,
            "dim": int(spec["dim"]),
            "nq": int(spec["nq"]),
        }
        source = OLD_ROOT / "diskann" / key / "diskann_rabitq_1bit_adaptive_fp"
        target = RESULT_ROOT / "diskann" / key / "diskann_rabitq_1bit_adaptive_fp"
        if copy_reusable(source, target, csv_path=source / "search_t1.csv", validation=validation):
            reused.append({"dataset": key, "method": "diskann_rabitq", "source": str(source)})

        validation = {
            "knob": "Lsearch",
            "expected": PIPEANN_L,
            "rounds": rounds,
            "dim": int(spec["dim"]),
            "nq": int(spec["nq"]),
        }
        source = OLD_ROOT / "spann_pipeann" / key / "pipeann"
        target = RESULT_ROOT / "spann_pipeann" / key / "pipeann"
        manifest = source / "build_manifest.json"
        manifest_ok = False
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_ok = int(payload.get("data_bits", -1)) == 1 and int(payload.get("rabitq_c", -1)) == 16
        if manifest_ok and copy_reusable(
            source, target, csv_path=source / "search_t1.csv", validation=validation
        ):
            reused.append({"dataset": key, "method": "pipeann", "source": str(source)})

    for key in SPANN_REUSABLE_KEYS:
        if key not in selected:
            continue
        spec = metadata[key]
        if int(spec["dim"]) // 8 != 128:
            raise RuntimeError(f"bad SPANN reuse declaration for {key}")
        validation = {
            "knob": "internal_heads",
            "expected": SPANN_HEADS,
            "rounds": rounds,
            "dim": int(spec["dim"]),
            "nq": int(spec["nq"]),
        }
        source = OLD_ROOT / "spann_pipeann" / key / "spann"
        target = RESULT_ROOT / "spann_pipeann" / key / "spann"
        manifest = source / "build_manifest.json"
        manifest_ok = False
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_ok = int(payload.get("pq_bytes_per_vector", -1)) == 128
        if manifest_ok and copy_reusable(
            source, target, csv_path=source / "search_t1.csv", validation=validation
        ):
            reused.append({"dataset": key, "method": "spann", "source": str(source)})
    return reused


def run_logged(phase: str, command: list[str], env: dict[str, str]) -> None:
    log = RESULT_ROOT / "logs" / f"{phase}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {shlex.join(command)}", flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] RUN {shlex.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write(f"[{now()}] returncode={completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(f"phase failed: {phase}; see {log}")


def run_diskann(
    rounds: int,
    build_threads: int,
    env: dict[str, str],
    dataset_keys: Iterable[str],
) -> None:
    run_logged(
        "diskann_onebit_budget",
        [
            str(SYSTEM_PYTHON), "-u",
            str(EXP / "run_diskveclab_full_matrix_diskann.py"),
            "--datasets", *dataset_keys,
            "--rounds", str(rounds),
            "--build-threads", str(build_threads),
            "--pq-code-budget", "one-bit",
            "--external-root", str(EXTERNAL_ROOT / "diskann"),
        ],
        env,
    )


def run_spann_pipeann(
    rounds: int, env: dict[str, str], dataset_keys: Iterable[str]
) -> None:
    command = [
        str(PYTHON), "-u", str(EXP / "run_nq_quora_spann_pipeann.py"),
        "--method", "all",
        "--rounds", str(rounds),
        "--search-threads", "1",
        "--heldout-matrix",
        "--pq-code-budget", "one-bit",
        "--result-root", str(RESULT_ROOT / "spann_pipeann"),
        "--external-root", str(EXTERNAL_ROOT / "spann_pipeann"),
    ]
    for key in dataset_keys:
        command.extend(("--dataset", key))
    run_logged("spann_pipeann_onebit_budget", command, env)


def method_specs() -> tuple[tuple[object, ...], ...]:
    return (
        (
            "diskann_pq_onebit_budget",
            RESULT_ROOT / "diskann",
            "diskann_classic",
            "Lsearch",
            DISKANN_L,
            0,
            "all database vectors",
        ),
        (
            "diskann_rabitq_1bit",
            RESULT_ROOT / "diskann",
            "diskann_rabitq_1bit_adaptive_fp",
            "Lsearch",
            DISKANN_L,
            16,
            "all database vectors",
        ),
        (
            "spann_pq_onebit_budget",
            RESULT_ROOT / "spann_pipeann",
            "spann",
            "internal_heads",
            SPANN_HEADS,
            0,
            "selected SPANN heads; posting codes remain on SSD",
        ),
        (
            "pipeann_rabitq_1bit",
            RESULT_ROOT / "spann_pipeann",
            "pipeann",
            "Lsearch",
            PIPEANN_L,
            9,
            "all database vectors",
        ),
    )


def curves_complete(
    metadata: dict[str, dict[str, object]], rounds: int, dataset_keys: Iterable[str]
) -> bool:
    for key in dataset_keys:
        spec = metadata[key]
        for _, root, directory, knob, expected, _, _ in method_specs():
            if not valid_curve(
                root / key / directory / "search_t1.csv",
                knob=knob,
                expected=expected,
                rounds=rounds,
                dim=int(spec["dim"]),
                nq=int(spec["nq"]),
            ):
                return False
    return True


def collect(
    metadata: dict[str, dict[str, object]],
    rounds: int,
    dataset_keys: Iterable[str],
    destination: Path,
) -> Path:
    output: list[dict[str, object]] = []
    selected = tuple(dataset_keys)
    for key in selected:
        spec = metadata[key]
        for canonical, root, directory, knob, expected, metadata_bytes, resident_scope in method_specs():
            path = root / key / directory / "search_t1.csv"
            if not valid_curve(
                path,
                knob=knob,
                expected=expected,
                rounds=rounds,
                dim=int(spec["dim"]),
                nq=int(spec["nq"]),
            ):
                raise RuntimeError(f"missing/incompatible final curve: {path}")
            for row in read_rows(path):
                merged: dict[str, object] = dict(row)
                merged.update(
                    {
                        "canonical_method": canonical,
                        "code_budget_policy": "one-bit-per-original-dimension",
                        "nominal_code_bytes_per_vector": int(spec["dim"]) // 8,
                        "metadata_bytes_per_vector": metadata_bytes,
                        "resident_code_scope": resident_scope,
                        "result_source": str(path),
                    }
                )
                output.append(merged)
    preferred = [
        "dataset", "corpus", "model", "canonical_method", "family", "dim", "nb", "nq",
        "code_budget_policy", "nominal_code_bytes_per_vector", "metadata_bytes_per_vector",
        "resident_code_scope", "recall_at_10", "qps", "rounds", "result_source",
    ]
    all_fields = {key for row in output for key in row}
    fields = [key for key in preferred if key in all_fields]
    fields.extend(sorted(all_fields - set(fields)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    os.replace(temporary, destination)
    atomic_json(
        RESULT_ROOT / "dataset_manifest.json",
        {
            "format": "onebit-disk-baselines-t1-v1",
            "created_at": now(),
            "rounds": rounds,
            "datasets": [metadata[key] for key in selected],
            "methods": [item[0] for item in method_specs()],
            "important_scope_note": (
                "DiskANN and PipeANN keep all routing codes resident. SPANN keeps only selected head codes resident; "
                "its PQ posting codes remain on SSD. The d/8 rule matches code size, not total process RSS."
            ),
        },
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--skip-reuse", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_KEYS)
    parser.add_argument(
        "--segment-name",
        help="durable restart marker for a selected dataset segment",
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.build_threads != 24:
        parser.error("rounds must be positive and build threads must be 24")

    selected = tuple(dict.fromkeys(args.datasets or DATASET_KEYS))
    segment_name = args.segment_name or ("all16" if selected == DATASET_KEYS else "selected")
    if not segment_name or any(not (character.isalnum() or character in "_-") for character in segment_name):
        parser.error("segment-name must contain only letters, digits, '_' or '-'")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    status = RESULT_ROOT / "campaign_status.json"
    segment_status = RESULT_ROOT / "segments" / f"{segment_name}.json"
    segment_csv = RESULT_ROOT / "segments" / f"{segment_name}_results.csv"
    if args.dry_run:
        compatible_rounds = args.rounds == 1
        print(
            json.dumps(
                {
                    "datasets": selected,
                    "segment_name": segment_name,
                    "new_inputs": NEW_KEYS,
                    "reuse": {
                        "diskann_rabitq": [key for key in selected if key in EXISTING_KEYS] if compatible_rounds else [],
                        "pipeann": [key for key in selected if key in EXISTING_KEYS] if compatible_rounds else [],
                        "spann": [key for key in selected if key in SPANN_REUSABLE_KEYS] if compatible_rounds else [],
                        "diskann_pq": [],
                    },
                    "existing_compatible_curves_have_rounds": 1,
                    "requested_rounds": args.rounds,
                    "pq_bytes": {"1024D": 128, "1536D": 192, "4096D": 512},
                },
                indent=2,
            )
        )
        return 0

    try:
        atomic_json(
            segment_status,
            {
                "stage": "running",
                "updated_at": now(),
                "datasets": list(selected),
                "rounds": args.rounds,
            },
        )
        atomic_json(
            status,
            {
                "stage": "preparing_inputs",
                "updated_at": now(),
                "current_segment": segment_name,
                "datasets": list(selected),
            },
        )
        if not args.skip_prepare:
            prepare_inputs(args.build_threads)
        validate_extra_manifest()
        env = campaign_env()
        metadata = dataset_metadata(env, DATASET_KEYS)
        atomic_json(status, {"stage": "importing_compatible_results", "updated_at": now()})
        reused = (
            []
            if args.skip_reuse
            else import_compatible_results(metadata, args.rounds, selected)
        )
        atomic_json(
            RESULT_ROOT / "reuse_summary.json",
            {"created_at": now(), "count": len(reused), "results": reused},
        )
        atomic_json(
            status,
            {"stage": "running_diskann", "updated_at": now(), "reused_results": len(reused)},
        )
        run_diskann(args.rounds, args.build_threads, env, selected)
        atomic_json(status, {"stage": "running_spann_pipeann", "updated_at": now()})
        run_spann_pipeann(args.rounds, env, selected)
        output = collect(metadata, args.rounds, selected, segment_csv)
        atomic_json(
            segment_status,
            {
                "stage": "complete",
                "updated_at": now(),
                "datasets": list(selected),
                "methods": 4,
                "rounds": args.rounds,
                "result_csv": str(output),
            },
        )
        if curves_complete(metadata, args.rounds, DATASET_KEYS):
            combined = collect(
                metadata,
                args.rounds,
                DATASET_KEYS,
                RESULT_ROOT / "all_results_t1.csv",
            )
            atomic_json(
                status,
                {
                    "stage": "complete",
                    "updated_at": now(),
                    "datasets": list(DATASET_KEYS),
                    "methods": 4,
                    "reused_results": len(reused),
                    "combined_csv": str(combined),
                },
            )
        else:
            atomic_json(
                status,
                {
                    "stage": "partial_complete",
                    "updated_at": now(),
                    "completed_segment": segment_name,
                    "datasets": list(selected),
                    "methods": 4,
                    "reused_results": len(reused),
                    "segment_csv": str(output),
                },
            )
    except BaseException as error:
        atomic_json(
            segment_status,
            {
                "stage": "failed",
                "updated_at": now(),
                "datasets": list(selected),
                "rounds": args.rounds,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        atomic_json(
            status,
            {"stage": "failed", "updated_at": now(), "error": f"{type(error).__name__}: {error}"},
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
