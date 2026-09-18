#!/mnt/shared-storage-user/guoshengyu/envs/vla-interpretability/bin/python
"""Download official LIBERO hdf5 suites onto TOS without filling GPFS.

Repo: yifengzhu-hf/LIBERO-datasets
Dest: /data/tos/guoshengyu/vla/libero/<suite>/

Scheme:
  Hugging Face (proxy) -> local home staging (one hdf5) -> inplace TOS copy
  -> delete local file. s3mount cannot rename, so TOS is opened with 'wb'.

Resume: skip a TOS file whose size matches the Hugging Face manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = "yifengzhu-hf/LIBERO-datasets"
EXPECTED = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}
TOS_ROOT = Path("/data/tos/guoshengyu/vla/libero")
STAGE = Path("/home/guoshengyu/.cache/vla_dl/libero_datasets")
STATUS_DIR = Path("/mnt/shared-storage-user/guoshengyu/vla_rjob_runs/libero_download")
CHUNK = 8 * 1024 * 1024
HTTP_TIMEOUT = 1200.0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(payload: dict) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = utc_now()
    (STATUS_DIR / "status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"updated_at: {payload['updated_at']}",
        f"phase:      {payload.get('phase', '')}",
        f"suite:      {payload.get('suite', '')}",
        f"current:    {payload.get('current', '')}",
        f"progress:   {payload.get('done_files', 0)}/{payload.get('total_files', 0)} files",
        f"bytes:      {payload.get('done_bytes', 0)}/{payload.get('total_bytes', 0)}",
        f"last_error: {payload.get('last_error', '')}",
        "",
        "per-suite TOS hdf5:",
    ]
    for suite, info in payload.get("suites", {}).items():
        lines.append(
            f"  {suite}: {info.get('tos', 0)}/{info.get('expected', 0)} hdf5  {info.get('state', '')}"
        )
    (STATUS_DIR / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_hdf5(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob("*_demo.hdf5"))


def suite_snapshot() -> dict:
    return {
        suite: {
            "expected": expected,
            "tos": count_hdf5(TOS_ROOT / suite),
            "state": "complete" if count_hdf5(TOS_ROOT / suite) >= expected else "incomplete",
        }
        for suite, expected in EXPECTED.items()
    }


def load_manifest() -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    rows = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if not name.endswith("_demo.hdf5"):
            continue
        suite = name.split("/", 1)[0]
        if suite not in EXPECTED:
            continue
        rows.append({"rel": name, "suite": suite, "size": int(sibling.size or 0)})
    rows.sort(key=lambda row: (list(EXPECTED).index(row["suite"]), row["rel"]))
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    (STATUS_DIR / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def stream_download(rel: str, dest: Path, expected: int) -> None:
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import get_session

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    url = hf_hub_url(repo_id=REPO, filename=rel, repo_type="dataset")
    session = get_session()
    last_error = None
    for attempt in range(1, 6):
        written = 0
        try:
            with session.stream("GET", url, follow_redirects=True, timeout=HTTP_TIMEOUT) as response:
                response.raise_for_status()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_bytes(CHUNK):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
            if expected and written != expected:
                raise RuntimeError(f"size mismatch downloaded={written} expected={expected}")
            tmp.replace(dest)
            print(f"  downloaded {rel} ({written / 1024**2:.1f} MiB) attempt={attempt}", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(f"  retry {attempt}/5 {rel}: {exc}", flush=True)
            time.sleep(min(30, 2 ** attempt))
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"{rel}: download failed: {last_error}")


def copy_inplace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as inf, open(dst, "wb") as out:
        while True:
            buf = inf.read(CHUNK)
            if not buf:
                break
            out.write(buf)
    src_size = src.stat().st_size
    dst_size = dst.stat().st_size
    if src_size != dst_size:
        raise RuntimeError(f"TOS copy size mismatch {dst}: {dst_size} != {src_size}")


def write_source(dest: Path, suite: str, count: int) -> None:
    (dest / "SOURCE.txt").write_text(
        f"repo={REPO}\nsubset={suite}\nfiles={count} *_demo.hdf5\n",
        encoding="utf-8",
    )


def print_status() -> int:
    status_txt = STATUS_DIR / "STATUS.txt"
    if status_txt.exists():
        sys.stdout.write(status_txt.read_text(encoding="utf-8"))
    else:
        print("no STATUS.txt yet; download has not started")
    print("---- live TOS counts ----")
    for suite, expected in EXPECTED.items():
        got = count_hdf5(TOS_ROOT / suite)
        print(f"  {suite}: {got}/{expected}")
    return 0


def pending_rows(rows: list[dict]) -> list[dict]:
    pending = []
    for row in rows:
        if row["suite"] == "libero_spatial":
            continue
        tos_path = TOS_ROOT / row["suite"] / Path(row["rel"]).name
        if tos_path.exists() and tos_path.stat().st_size == row["size"]:
            continue
        pending.append(row)
    return pending


def select_rows(rows: list[dict], test: bool) -> list[dict]:
    pending = pending_rows(rows)
    if test:
        if not pending:
            raise SystemExit("test: every missing non-spatial file is already on TOS")
        return pending[:1]
    return pending


def run(test: bool) -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    work = select_rows(rows, test=test)
    total_files = len(work)
    total_bytes = sum(row["size"] for row in work)
    done_files = 0
    done_bytes = 0
    payload = {
        "phase": "test" if test else "download",
        "suite": "",
        "current": "",
        "done_files": 0,
        "total_files": total_files,
        "done_bytes": 0,
        "total_bytes": total_bytes,
        "last_error": "",
        "suites": suite_snapshot(),
    }
    write_status(payload)

    for row in work:
        rel = row["rel"]
        suite = row["suite"]
        filename = Path(rel).name
        tos_path = TOS_ROOT / suite / filename
        payload.update(suite=suite, current=rel, phase="copy" if test else "download")
        write_status(payload)
        if tos_path.exists() and tos_path.stat().st_size == row["size"]:
            print(f"[skip] TOS {rel}", flush=True)
            done_files += 1
            done_bytes += row["size"]
            payload.update(done_files=done_files, done_bytes=done_bytes, suites=suite_snapshot())
            write_status(payload)
            continue

        local = STAGE / suite / filename
        print(f"[fetch] {rel}", flush=True)
        if not (local.exists() and local.stat().st_size == row["size"]):
            stream_download(rel, local, row["size"])
        print(f"[tos]   {tos_path}", flush=True)
        copy_inplace(local, tos_path)
        local.unlink()
        leftover = STAGE / suite
        if leftover.exists() and not any(leftover.iterdir()):
            leftover.rmdir()
        write_source(TOS_ROOT / suite, suite, count_hdf5(TOS_ROOT / suite))
        done_files += 1
        done_bytes += row["size"]
        payload.update(
            done_files=done_files,
            done_bytes=done_bytes,
            last_error="",
            suites=suite_snapshot(),
        )
        write_status(payload)

    payload.update(phase="done", current="", suites=suite_snapshot())
    write_status(payload)
    print("==== TOS suite counts ====", flush=True)
    for suite, expected in EXPECTED.items():
        got = count_hdf5(TOS_ROOT / suite)
        print(f"  {suite}: {got}/{expected}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="download one missing hdf5 then exit")
    parser.add_argument("--status", action="store_true", help="print STATUS.txt and live TOS counts")
    args = parser.parse_args()
    if args.status:
        return print_status()
    try:
        return run(test=args.test)
    except Exception as exc:
        payload = {
            "phase": "error",
            "suite": "",
            "current": "",
            "done_files": 0,
            "total_files": 0,
            "done_bytes": 0,
            "total_bytes": 0,
            "last_error": str(exc),
            "suites": suite_snapshot(),
        }
        try:
            write_status(payload)
        except Exception:
            pass
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
