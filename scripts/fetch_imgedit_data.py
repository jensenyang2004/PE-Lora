#!/usr/bin/env python3
"""
Fetch a small (~1k sample) single-instance edit dataset from the ImgEdit HF
dataset (sysuyy/ImgEdit) for PE-injection LoRA training on Flux2 Klein.
Restricted to `replace` and `adjust_canny` — both keep the object in the same
place (unlike `add`/`remove`, where the object is only present in one of the
two images), which keeps the before/after pair maximally aligned around the
injected centroid.

Why this exists instead of `datasets.load_dataset(...)`:
  - The released Parquet files only have {input_images, output_images, prompt} —
    no mask/bbox, so they can't give us a per-instance centroid.
  - The mask/bbox lives in the raw `Singleturn/*.tar.split.NNN` archives, which
    are split into ~10GB chunks. We only need ~1k samples total, so this script
    streams tar entries directly over HTTP (no local copy of the multi-hundred-GB
    archives) and stops once each task's quota is filled.

Verified against a live sample (2026-09-07, results_add_laion_part0):
  - Each sample is a folder with original.png (input), result.png (output),
    mask.png, result.json, judge.json.
  - GOTCHA: result.json's "resolution" field (and mask.png's own pixel size) is
    the ORIGINAL source-image resolution. original.png/result.png are resized by
    the ComfyUI inpaint workflow to a *different* resolution, and are not even
    pixel-identical to each other (e.g. 1632x1024 vs 1639x1024 in one sample).
    So bbox/mask must be treated as normalized-fraction data, not raw pixels,
    and remapped onto whatever resolution is actually used downstream.
  - mask.png convention (from ImgEdit's rle_to_mask): edited/object region is
    DARK (~0), background is LIGHT (~255) — inverted from the usual
    white-is-hole inpainting-mask convention.
  - judge.json holds a GPT-4o quality string like "...Score: 3".
  - Object bbox/mask lives in `edit_obj` for both replace and adjust_canny.

Output layout (under --out-dir):
  images/<sample_id>/original.png   (as shipped, untouched)
  images/<sample_id>/result.png     (as shipped, untouched)
  images/<sample_id>/mask.png       (resized onto result.png's pixel size)
  manifest.jsonl                    (one row per accepted sample; see below)

Manifest row:
  {
    "id": str, "task": str, "class_name": str, "edit_prompt": str,
    "judge_score": float,
    "input_image": "images/<id>/original.png",
    "output_image": "images/<id>/result.png",
    "mask": "images/<id>/mask.png",
    "mask_resolution": {"height": int, "width": int},  # = result.png's size
    "bbox_norm": [x1, y1, x2, y2],       # fractions of width/height, 0..1
    "centroid_norm": [cx, cy],           # fraction of width/height, 0..1
    "source_tar": str,
  }

Run this on the training server (network + disk there, not on this machine):
  pip install huggingface_hub requests pillow
  python scripts/fetch_imgedit_data.py --out-dir ./data/imgedit_single_instance

Intentionally NOT executed by the assistant — network + multi-GB streaming
belongs on the training box.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi, get_token

from _hf_tar_stream import MultiUrlReader, iter_grouped_samples, list_split_urls

EXPECTED_FILES = {"original.png", "result.png", "mask.png", "result.json", "judge.json"}

# Which task tars to pull from, and which result.json field holds the
# object's bbox/mask for that task type. Picked for: has a real per-instance
# mask (unlike background_change/style_transfer), and is a single edit
# (unlike compose_*/omit_*, which are 2-3 round multi-turn tasks).
TASK_SPECS = {
    "replace": {"tar_bases": ["results_replace_part0"], "object_key": "edit_obj"},
    "adjust_canny": {"tar_bases": ["results_adjust_canny_laion_part0"], "object_key": "edit_obj"},
}

SCORE_RE = re.compile(r"Score:\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass
class Filters:
    min_judge_score: float = 3.0
    min_mask_frac: float = 0.003
    max_mask_frac: float = 0.6


@dataclass
class Counters:
    seen: int = 0
    accepted: int = 0
    rejected_score: int = 0
    rejected_mask_frac: int = 0
    rejected_missing_obj: int = 0
    rejected_parse: int = 0


def parse_judge_score(judge_bytes: bytes) -> float | None:
    try:
        text = json.loads(judge_bytes)["score"]
    except Exception:
        return None
    m = SCORE_RE.search(text)
    return float(m.group(1)) if m else None


def extract_object(meta: dict, object_key: str) -> dict | None:
    obj = meta.get(object_key)
    if not obj or not isinstance(obj, dict):
        return None
    if "bbox" not in obj or "class_name" not in obj:
        return None
    return obj


def process_sample(
    task: str,
    sample_dir: str,
    files: dict[str, bytes],
    source_tar: str,
    filters: Filters,
    counters: Counters,
    out_images_dir: Path,
) -> dict | None:
    counters.seen += 1
    object_key = TASK_SPECS[task]["object_key"]

    try:
        meta = json.loads(files["result.json"])
    except Exception:
        counters.rejected_parse += 1
        return None

    obj = extract_object(meta, object_key)
    if obj is None:
        counters.rejected_missing_obj += 1
        return None

    score = parse_judge_score(files["judge.json"])
    if score is None or score < filters.min_judge_score:
        counters.rejected_score += 1
        return None

    edit_prompt = meta.get("edit_prompt")
    if not edit_prompt:
        counters.rejected_parse += 1
        return None

    resolution = meta.get("resolution") or {}
    res_h, res_w = resolution.get("height"), resolution.get("width")
    if not res_h or not res_w:
        counters.rejected_parse += 1
        return None

    try:
        result_img = Image.open(io.BytesIO(files["result.png"])).convert("RGB")
        mask_img = Image.open(io.BytesIO(files["mask.png"])).convert("L")
    except Exception:
        counters.rejected_parse += 1
        return None

    # Mask convention: edited/object region is DARK (~0), background LIGHT (~255).
    mask_frac = 1.0 - (sum(mask_img.histogram()[128:]) / (mask_img.width * mask_img.height))
    if not (filters.min_mask_frac <= mask_frac <= filters.max_mask_frac):
        counters.rejected_mask_frac += 1
        return None

    x1, y1, x2, y2 = obj["bbox"]
    bbox_norm = [x1 / res_w, y1 / res_h, x2 / res_w, y2 / res_h]
    centroid_norm = [(bbox_norm[0] + bbox_norm[2]) / 2.0, (bbox_norm[1] + bbox_norm[3]) / 2.0]

    sample_id = f"{task}__{sample_dir.split('/', 1)[1]}"
    sample_out_dir = out_images_dir / sample_id
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    (sample_out_dir / "original.png").write_bytes(files["original.png"])
    (sample_out_dir / "result.png").write_bytes(files["result.png"])
    # Resize mask onto result.png's actual pixel space (see module docstring
    # gotcha: mask.png ships at the *source* resolution, not the saved PNG's).
    mask_resized = mask_img.resize(result_img.size, Image.NEAREST)
    mask_resized.save(sample_out_dir / "mask.png")

    counters.accepted += 1
    return {
        "id": sample_id,
        "task": task,
        "class_name": obj.get("class_name"),
        "edit_prompt": edit_prompt,
        "judge_score": score,
        "input_image": f"images/{sample_id}/original.png",
        "output_image": f"images/{sample_id}/result.png",
        "mask": f"images/{sample_id}/mask.png",
        "mask_resolution": {"height": result_img.height, "width": result_img.width},
        "bbox_norm": bbox_norm,
        "centroid_norm": centroid_norm,
        "source_tar": source_tar,
    }


def fetch_task(
    task: str,
    quota: int,
    max_splits: int,
    filters: Filters,
    out_dir: Path,
    manifest_fh,
    session: requests.Session,
    api: HfApi,
) -> None:
    import tarfile

    counters = Counters()
    out_images_dir = out_dir / "images"
    tar_bases = TASK_SPECS[task]["tar_bases"]

    for tar_base in tar_bases:
        if counters.accepted >= quota:
            break
        urls = list_split_urls(api, tar_base)[:max_splits]
        print(f"[{task}] streaming {tar_base} ({len(urls)} split file(s) available, capped at {max_splits})")
        reader = MultiUrlReader(urls, session)
        try:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                for sample_dir, files in iter_grouped_samples(tf, EXPECTED_FILES):
                    row = process_sample(
                        task, sample_dir, files, tar_base, filters, counters, out_images_dir
                    )
                    if row is not None:
                        manifest_fh.write(json.dumps(row) + "\n")
                        manifest_fh.flush()
                    if counters.seen % 50 == 0:
                        print(
                            f"[{task}] seen={counters.seen} accepted={counters.accepted}/{quota} "
                            f"(rej: score={counters.rejected_score} mask_frac={counters.rejected_mask_frac} "
                            f"missing_obj={counters.rejected_missing_obj} parse={counters.rejected_parse})",
                            file=sys.stderr,
                        )
                    if counters.accepted >= quota:
                        break
        except tarfile.ReadError:
            # Expected once we hit the end of the last streamed split (or, if
            # max_splits < total splits for this tar, the truncation point).
            pass
        finally:
            reader.close()

    print(f"[{task}] done: accepted {counters.accepted}/{quota} (saw {counters.seen} samples)")
    if counters.accepted < quota:
        print(
            f"[{task}] WARNING: quota not reached — raise --max-splits-per-tar, "
            f"loosen filters, or add more tar_bases in TASK_SPECS for {task!r}.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("./data/imgedit_single_instance"))
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASK_SPECS.keys()),
        help=f"comma-separated subset of {list(TASK_SPECS.keys())}",
    )
    parser.add_argument("--total-samples", type=int, default=1200, help="target across all tasks combined")
    parser.add_argument("--per-task-quota", type=int, default=None, help="override total-samples // num_tasks")
    parser.add_argument("--max-splits-per-tar", type=int, default=3, help="safety cap on .tar.split.NNN files read per tar")
    parser.add_argument("--min-judge-score", type=float, default=3.0)
    parser.add_argument("--min-mask-frac", type=float, default=0.003)
    parser.add_argument("--max-mask-frac", type=float, default=0.6)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in TASK_SPECS:
            parser.error(f"unknown task {t!r}, choose from {list(TASK_SPECS.keys())}")

    per_task_quota = args.per_task_quota or max(1, args.total_samples // len(tasks))
    filters = Filters(args.min_judge_score, args.min_mask_frac, args.max_mask_frac)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(exist_ok=True)

    token = args.hf_token or get_token()
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    api = HfApi(token=token)

    manifest_path = args.out_dir / "manifest.jsonl"
    with open(manifest_path, "a") as manifest_fh:
        for task in tasks:
            fetch_task(
                task=task,
                quota=per_task_quota,
                max_splits=args.max_splits_per_tar,
                filters=filters,
                out_dir=args.out_dir,
                manifest_fh=manifest_fh,
                session=session,
                api=api,
            )

    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
