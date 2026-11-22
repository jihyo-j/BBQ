#!/usr/bin/env python3
"""Restartable OpenAI/BGE BEIR main experiment with 12 methods."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "experiment"
RESULT_ROOT = EXPERIMENT / "openai_beir_11methods_t1"
EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/openai_beir_11methods_t1")
MANIFEST = EXTERNAL_ROOT / "dataset_manifest.json"
PREPARE = EXPERIMENT / "prepare_openai_beir_11methods_inputs.py"
HNSW_PYTHON = Path("/home/jihyo/anaconda3/envs/rabitq-cfr/bin/python")
RABITQ_PYTHON = Path("/home/jihyo/anaconda3/envs/rabitq-cfr/bin/python")
# The vendored Elasticsearch bulk serializer contains a CPython 3.12 orjson
# extension.  Keep this phase on the matching system interpreter; the other
# native runners continue to use the dedicated RaBitQ environment above.
ANN_PYTHON = Path("/usr/bin/python3")
SYSTEM_PYTHON = Path("/usr/bin/python3")
EFS = (
    64, 72, 80, 88, 96, 104, 112, 120, 128, 144,
    160, 176, 192, 208, 224, 240, 256, 320, 512, 1024,
)
ONEBIT_ROOT = EXPERIMENT / "onebit_disk_baselines_t1"
ONEBIT_STATUS = ONEBIT_ROOT / "campaign_status.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.selected_keys = tuple(dict.fromkeys(args.datasets))
        self.state_path = RESULT_ROOT / "campaign_status.json"
        self.log_root = RESULT_ROOT / "logs"
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            previous = tuple(self.state.get("datasets", ()))
            if previous and previous != self.selected_keys:
                raise RuntimeError(
                    f"campaign dataset selection changed: {previous} -> {self.selected_keys}"
                )
        else:
            self.state = {
                "format": "openai-beir-12methods-t1-v3",
                "completed": [],
                "datasets": list(self.selected_keys),
            }
            self.save()

    def save(self, **updates: object) -> None:
        self.state.update(updates)
        self.state["updated_at"] = now()
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)

    def complete(self, phase: str) -> bool:
        return phase in self.state.get("completed", [])

    def mark(self, phase: str) -> None:
        completed = self.state.setdefault("completed", [])
        if phase not in completed:
            completed.append(phase)
        self.save(stage="checkpoint", last_completed=phase)

    def mark_reused(self, phase: str, sources: list[str]) -> None:
        reused = self.state.setdefault("reused_external_phases", {})
        reused[phase] = {"campaign": str(ONEBIT_STATUS), "sources": sources}
        self.mark(phase)

    def environment(self) -> dict[str, str]:
        if not MANIFEST.is_file():
            raise FileNotFoundError(MANIFEST)
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        available = {str(item["key"]) for item in payload["datasets"]}
        missing = set(self.selected_keys) - available
        if missing:
            raise RuntimeError(f"selected datasets missing from manifest: {sorted(missing)}")
        return {
            **os.environ,
            "DISKVECLAB_EXTRA_DATASETS_MANIFEST": str(MANIFEST),
            "DISKVECLAB_DATASETS": ",".join(self.selected_keys),
            "DISKVECLAB_RESULT_ROOT": str(RESULT_ROOT / "native_matrix"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }

    def run(self, phase: str, command: list[str], env: dict[str, str] | None = None) -> None:
        if self.complete(phase):
            print(f"SKIP completed phase={phase}", flush=True)
            return
        log_path = self.log_root / f"{phase}.log"
        print(f"[{now()}] START {phase}: {shlex.join(command)}", flush=True)
        self.save(stage="running", current_phase=phase, command=command)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{now()}] RUN {shlex.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env or os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            returncode = process.wait()
            log.write(f"[{now()}] returncode={returncode}\n")
        if returncode:
            self.save(stage="failed", current_phase=phase, returncode=returncode)
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
            raise RuntimeError(f"phase failed: {phase} rc={returncode}\n{tail}")
        self.mark(phase)
        print(f"[{now()}] DONE {phase}", flush=True)

    def prepare(self) -> None:
        if self.args.skip_prepare:
            self.environment()
            if not self.complete("prepare_inputs_and_exact_gt"):
                self.state["reused_input_manifest"] = str(MANIFEST)
                self.mark("prepare_inputs_and_exact_gt")
            return
        self.run(
            "prepare_inputs_and_exact_gt",
            [
                str(HNSW_PYTHON), str(PREPARE),
                "--output-root", str(EXTERNAL_ROOT),
                "--threads", str(self.args.build_threads),
            ],
        )

    def four_methods(self) -> None:
        phase = "adcbq_turboquant_rabitq1_rabitq7"
        if self.complete(phase):
            print(f"SKIP completed phase={phase}", flush=True)
            return
        self.save(stage="running", current_phase=phase)
        os.environ.update(self.environment())
        import run_four_methods_series_t1 as four

        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for spec in payload["datasets"]:
            if str(spec["key"]) not in self.selected_keys:
                continue
            four.run_one(
                dict(spec),
                result_base=RESULT_ROOT / "four_methods",
                external_base=EXTERNAL_ROOT,
                efs=EFS,
                rounds=self.args.rounds,
                pilot_rounds=self.args.pilot_rounds,
                build_threads=self.args.build_threads,
                cpu=self.args.cpu,
                cleanup_indexes=False,
            )
            # The two RaBitQ RAM indexes are no longer needed by later phases.
            for name in ("b1_index", "b7_index"):
                path = Path(str(spec[name]))
                if path.is_file():
                    four.remove_owned(path, EXTERNAL_ROOT)
        self.mark(phase)

    def cleanup_shared_graphs(self) -> None:
        phase = "cleanup_shared_adcbq_graphs"
        if self.complete(phase):
            return
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for spec in payload["datasets"]:
            if str(spec["key"]) not in self.selected_keys:
                continue
            path = Path(str(spec["adcbq_index"]))
            resolved = path.resolve()
            if EXTERNAL_ROOT.resolve() not in resolved.parents:
                raise RuntimeError(f"refusing cleanup outside OpenAI root: {resolved}")
            if path.is_file():
                size = path.stat().st_size
                path.unlink()
                print(f"REMOVED shared graph bytes={size}: {path}", flush=True)
            sidecar = Path(str(spec["fp_sidecar"]))
            resolved_sidecar = sidecar.resolve()
            if EXTERNAL_ROOT.resolve() not in resolved_sidecar.parents:
                raise RuntimeError(f"refusing sidecar cleanup outside OpenAI root: {resolved_sidecar}")
            if sidecar.is_file() and sidecar != Path(str(spec["train"])):
                size = sidecar.stat().st_size
                sidecar.unlink()
                print(f"REMOVED completed FP sidecar bytes={size}: {sidecar}", flush=True)
        self.mark(phase)

    def native_methods(self) -> None:
        env = self.environment()
        keys = env["DISKVECLAB_DATASETS"].split(",")
        self.run(
            "faiss_pq_opq_calibrated_fixed_r",
            [
                str(HNSW_PYTHON),
                str(EXPERIMENT / "run_openai_pq_opq_calibrated_selected_r_t1.py"),
                "--rounds", str(self.args.rounds),
                "--pilot-rounds", str(self.args.pilot_rounds),
                "--build-threads", str(self.args.build_threads),
                "--cpu", str(self.args.cpu),
                "--result-root", str(RESULT_ROOT / "faiss_pq_opq_calibrated"),
                "--external-root", str(EXTERNAL_ROOT / "faiss_pq_opq_work"),
            ],
            env,
        )
        self.cleanup_shared_graphs()
        onebit_complete = False
        onebit_status = (
            ONEBIT_ROOT / "segments" / f"{self.args.onebit_segment_name}.json"
            if self.args.onebit_segment_name
            else ONEBIT_STATUS
        )
        if onebit_status.is_file():
            try:
                onebit_payload = json.loads(onebit_status.read_text(encoding="utf-8"))
                onebit_complete = (
                    onebit_payload.get("stage") == "complete"
                    and set(keys).issubset(set(onebit_payload.get("datasets", ())))
                    and int(onebit_payload.get("rounds", self.args.rounds)) == self.args.rounds
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        diskann_sources = [
            str(ONEBIT_ROOT / "diskann" / key / method / "search_t1.csv")
            for key in keys
            for method in ("diskann_classic", "diskann_rabitq_1bit_adaptive_fp")
        ]
        if onebit_complete and all(Path(path).is_file() for path in diskann_sources):
            self.mark_reused("diskann_pq_and_diskann_rabitq", diskann_sources)
        else:
            self.run(
                "diskann_pq_and_diskann_rabitq",
                [
                    str(SYSTEM_PYTHON),
                    str(EXPERIMENT / "run_diskveclab_full_matrix_diskann.py"),
                    "--datasets", *keys,
                    "--rounds", str(self.args.rounds),
                    "--build-threads", str(self.args.build_threads),
                    "--pq-code-budget", "one-bit",
                    "--external-root", str(EXTERNAL_ROOT / "diskann_onebit_work"),
                ],
                env,
            )
        spann_command = [
            str(RABITQ_PYTHON),
            str(EXPERIMENT / "run_nq_quora_spann_pipeann.py"),
            "--method", "spann",
            "--rounds", str(self.args.rounds),
            "--search-threads", "1",
            "--heldout-matrix",
            "--result-root", str(RESULT_ROOT / "spann"),
            "--external-root", str(EXTERNAL_ROOT / "spann_work"),
        ]
        for key in keys:
            spann_command.extend(("--dataset", key))
        spann_sources = [
            str(ONEBIT_ROOT / "spann_pipeann" / key / "spann" / "search_t1.csv")
            for key in keys
        ]
        if onebit_complete and all(Path(path).is_file() for path in spann_sources):
            self.mark_reused("spann", spann_sources)
        else:
            spann_command.extend(("--pq-code-budget", "one-bit"))
            self.run("spann", spann_command, env)
        pipeann_command = [
            str(RABITQ_PYTHON),
            str(EXPERIMENT / "run_nq_quora_spann_pipeann.py"),
            "--method", "pipeann",
            "--rounds", str(self.args.rounds),
            "--search-threads", "1",
            "--heldout-matrix",
            "--result-root", str(RESULT_ROOT / "pipeann_rabitq"),
            "--external-root", str(EXTERNAL_ROOT / "pipeann_rabitq_work"),
        ]
        for key in keys:
            pipeann_command.extend(("--dataset", key))
        pipeann_sources = [
            str(ONEBIT_ROOT / "spann_pipeann" / key / "pipeann" / "search_t1.csv")
            for key in keys
        ]
        if onebit_complete and all(Path(path).is_file() for path in pipeann_sources):
            self.mark_reused("pipeann_rabitq", pipeann_sources)
        else:
            pipeann_command.extend(("--pq-code-budget", "one-bit"))
            self.run("pipeann_rabitq", pipeann_command, env)
        elastic_command = [
            str(ANN_PYTHON),
            str(EXPERIMENT / "run_nq_quora_elasticsearch_bbq_rerank80.py"),
            "--phase", "all",
            "--rounds", str(self.args.rounds),
            "--search-concurrency", "1",
            "--heldout-matrix",
            "--cleanup-after-search",
            "--result-root", str(RESULT_ROOT / "elasticsearch_bbq"),
            "--external-root", str(EXTERNAL_ROOT / "elasticsearch_bbq"),
        ]
        for key in keys:
            elastic_command.extend(("--dataset", key))
        self.run("elasticsearch_bbq", elastic_command, env)
        self.run(
            "rabitq_ssd",
            [
                str(HNSW_PYTHON),
                str(EXPERIMENT / "run_diskveclab_full_matrix_mrq_rabitq_ssd.py"),
                "--datasets", *keys,
                "--methods", "rabitq",
                "--rounds", str(self.args.rounds),
                "--build-threads", "20",
            ],
            env,
        )

    def execute(self) -> None:
        self.prepare()
        self.four_methods()
        self.native_methods()
        self.save(
            stage="complete",
            methods=[
                "calibrated_adcbq", "elasticsearch_bbq", "turboquant",
                "rabitq_ssd", "diskann_rabitq", "diskann_pq",
                "faiss_opq_d_over_8", "rabitq_7bit", "rabitq_1bit_adaptive",
                "spann", "pipeann_rabitq", "faiss_pq_d_over_8",
            ],
            datasets=list(self.selected_keys),
        )
        print(f"COMPLETE OpenAI/BGE 12-method campaign: {self.state_path}", flush=True)


def main() -> int:
    global RESULT_ROOT, EXTERNAL_ROOT, MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pilot-rounds", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--cpu", type=int, default=39)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "sieun_msmarco_openai1536",
            "sieun_dbpedia_entity_openai1536",
        ],
    )
    parser.add_argument(
        "--onebit-segment-name",
        help="reuse a completed d/8 segment for the selected datasets",
    )
    args = parser.parse_args()
    if min(args.rounds, args.pilot_rounds, args.build_threads, args.cpu) < 1:
        parser.error("numeric arguments must be positive")
    RESULT_ROOT = args.result_root.resolve()
    EXTERNAL_ROOT = args.external_root.resolve()
    MANIFEST = args.manifest.resolve()
    Campaign(args).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
