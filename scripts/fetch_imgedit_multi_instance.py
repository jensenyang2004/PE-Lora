#!/usr/bin/env python3
"""
Fetch the "hybrid" (internally `results_compose_part*`) two-instance subset of
the ImgEdit HF dataset (sysuyy/ImgEdit) as a held-out MULTI-INSTANCE EVAL SET
for the PE-injection LoRA on Flux2 Klein.

Do NOT use this as training data: the project trains only on single-instance
edits ([[project_pe_lora_goal]]) and measures whether PE injection generalizes
to multi-instance editing. This set is the generalization test, not more
training signal.

Verified 2026-09-07 by live-streaming samples from all three hybrid tar parts:
  - Externally the tar is named `results_hybrid_part{0,2,6}`, but on disk each
    sample folder is `results_compose_part*/<id>/`, matching schemas.py's
    ComposeTask: edit_obj1 + edit_obj2 (two objects, two bboxes), edit_type as
    a 2-element list (e.g. ["add", "adjust"]), and ONE combined edit_prompt
    describing both edits (e.g. "add a sign 'EXIT' ... and adjust the
    device...").
  - Files per sample: origin_0.png, result_0.png, mask_0.png, origin_1.png,
    result_1.png, mask_1.png, result.json, result_fixed.json (byte-identical
    duplicate of result.json). No judge.json (unlike add/remove/replace/
    adjust_canny) — there's no per-sample quality score to filter on here.
  - GOTCHA (confirmed by MD5): origin_1.png == result_0.png. This is a
    SEQUENTIAL two-round chain (edit object 1 on origin_0 -> result_0, then
    edit object 2 on that same result -> result_1), not a joint one-shot
    multi-instance generation. The valid before/after pair is
    (origin_0.png, result_1.png); edit_type[0]/edit_obj1 is the first round's
    edit, edit_type[1]/edit_obj2 is the second round's.
  - Checked bbox IoU across 484 sampled results_json: 0.0 in every case — the
    two instances are never spatially the same object, even the 14% of the
    time they share a class_name (e.g. two separate motorcycles on opposite
    sides of the image).
  - Same resolution gotcha as the single-instance data: result.json's
    "resolution" field is the bbox coordinate space; mask_0.png/mask_1.png may
    not match origin_0.png/result_1.png's actual saved pixel size, so bbox and
    masks are remapped per-sample rather than trusted as raw pixels.

Filtering: by default keeps samples where `edit_type` contains "remove" and
does NOT contain "add" (the ~9k bucket discussed with the user: no object
appears/disappears out of thin air, but the object count still changes so
it's not just a pure appearance-only edit either). Use --exclude-types /
--require-any-type to switch to the other buckets that were sized up from a
484-sample pilot (2026-09-07, all three hybrid tar parts, ~1.6% of the total
28,390 hybrid rows):
    exclude=add                              -> ~46.5% of 28390 ~= 13,200
    exclude=add,        require-any=remove   -> ~33.3% of 28390 ~=  9,450  (default)
    exclude=add,remove                       -> ~13.2% of 28390 ~=  3,750

IMPORTANT — data volume: matches are scattered essentially uniformly through
the stream, so collecting most of a ~33%-hit-rate bucket means reading nearly
all of the source data. The three hybrid tars total ~277GB combined
(results_hybrid_part0 ~96GB, part2 ~90GB, part6 ~91GB, confirmed via the HF
API's file listing). There is no way to skip non-matching samples in a
sequential, unindexed tar stream — expect this to take a long time and pull
a large amount of network traffic regardless of --max-total. Run it on the
training server, not here.

Output layout (under --out-dir), one subdir per source part so parallel runs
(one process per --parts value) never collide:
  images/<sample_id>/origin_0.png    (as shipped, untouched — the "before")
  images/<sample_id>/result_1.png    (as shipped, untouched — the "after", both edits applied)
  images/<sample_id>/mask_0.png      (resized onto result_1.png's pixel size)
  images/<sample_id>/mask_1.png      (resized onto result_1.png's pixel size)
  manifest_<part>.jsonl              (one row per accepted sample from that part)

Merge the per-part manifests afterward with e.g.
  cat manifest_part*.jsonl > manifest.jsonl

Manifest row:
  {
    "id": str, "edit_prompt": str, "edit_type": [str, str],
    "input_image": "images/<id>/origin_0.png",
    "output_image": "images/<id>/result_1.png",
    "resolution": {"height": int, "width": int},  # = result_1.png's size
    "instances": [
      {"index": 0, "edit_type": str, "class_name": str, "mask": "images/<id>/mask_0.png",
       "bbox_norm": [x1,y1,x2,y2], "centroid_norm": [cx,cy]},
      {"index": 1, "edit_type": str, "class_name": str, "mask": "images/<id>/mask_1.png",
       "bbox_norm": [x1,y1,x2,y2], "centroid_norm": [cx,cy]},
    ],
    "source_tar": str,
  }

Run this on the training server:
  pip install huggingface_hub requests pillow
  python scripts/fetch_imgedit_multi_instance.py --out-dir ./data/imgedit_multi_instance_eval

To cut wall-clock time, run one part per process in parallel, e.g.:
  python scripts/fetch_imgedit_multi_instance.py --parts part0 --max-total 3000 &
  python scripts/fetch_imgedit_multi_instance.py --parts part2 --max-total 3000 &
  python scripts/fetch_imgedit_multi_instance.py --parts part6 --max-total 3000 &
  wait

Intentionally NOT executed by the assistant — network + multi-hundred-GB
streaming belongs on the training box.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi, get_token
from PIL import Image

from _hf_tar_stream import MultiUrlReader, iter_grouped_samples, list_split_urls

EXPECTED_FILES = {"origin_0.png", "result_1.png", "mask_0.png", "mask_1.png", "result.json"}

PARTS = {
    "part0": "results_hybrid_part0",
    "part2": "results_hybrid_part2",
    "part6": "results_hybrid_part6",
}


@dataclass
class Filters:
    exclude_types: frozenset[str]
    require_any_type: frozenset[str]
    min_mask_frac: float = 0.001
    max_mask_frac: float = 0.7

    def edit_type_passes(self, edit_type: list[str]) -> bool:
        et = set(edit_type)
        if self.exclude_types & et:
            return False
        if self.require_any_type and not (self.require_any_type & et):
            return False
        return True


@dataclass
class Counters:
    seen: int = 0
    accepted: int = 0
    rejected_edit_type: int = 0
    rejected_mask_frac: int = 0
    rejected_missing_obj: int = 0
    rejected_parse: int = 0


def mask_area_frac(mask_img: Image.Image) -> float:
    # Mask convention (from ImgEdit's rle_to_mask): edited region is DARK
    # (~0), background LIGHT (~255) — same as the single-instance data.
    return 1.0 - (sum(mask_img.histogram()[128:]) / (mask_img.width * mask_img.height))


def bbox_and_centroid_norm(bbox: list[float], res_w: float, res_h: float) -> tuple[list[float], list[float]]:
    x1, y1, x2, y2 = bbox
    bbox_norm = [x1 / res_w, y1 / res_h, x2 / res_w, y2 / res_h]
    centroid_norm = [(bbox_norm[0] + bbox_norm[2]) / 2.0, (bbox_norm[1] + bbox_norm[3]) / 2.0]
    return bbox_norm, centroid_norm


def process_sample(
    sample_dir: str,
    files: dict[str, bytes],
    source_tar: str,
    filters: Filters,
    counters: Counters,
    out_images_dir: Path,
) -> dict | None:
    counters.seen += 1

    try:
        meta = json.loads(files["result.json"])
    except Exception:
        counters.rejected_parse += 1
        return None

    edit_type = meta.get("edit_type")
    obj1, obj2 = meta.get("edit_obj1"), meta.get("edit_obj2")
    edit_prompt = meta.get("edit_prompt")
    resolution = meta.get("resolution") or {}
    res_h, res_w = resolution.get("height"), resolution.get("width")

    if not (isinstance(edit_type, list) and len(edit_type) == 2):
        counters.rejected_parse += 1
        return None
    if not obj1 or not obj2 or "bbox" not in obj1 or "bbox" not in obj2:
        counters.rejected_missing_obj += 1
        return None
    if not edit_prompt or not res_h or not res_w:
        counters.rejected_parse += 1
        return None

    if not filters.edit_type_passes(edit_type):
        counters.rejected_edit_type += 1
        return None

    try:
        after_img = Image.open(io.BytesIO(files["result_1.png"])).convert("RGB")
        mask0_img = Image.open(io.BytesIO(files["mask_0.png"])).convert("L")
        mask1_img = Image.open(io.BytesIO(files["mask_1.png"])).convert("L")
    except Exception:
        counters.rejected_parse += 1
        return None

    frac0, frac1 = mask_area_frac(mask0_img), mask_area_frac(mask1_img)
    if not (filters.min_mask_frac <= frac0 <= filters.max_mask_frac):
        counters.rejected_mask_frac += 1
        return None
    if not (filters.min_mask_frac <= frac1 <= filters.max_mask_frac):
        counters.rejected_mask_frac += 1
        return None

    sample_id = f"{sample_dir.split('/', 1)[1]}"
    sample_out_dir = out_images_dir / sample_id
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    (sample_out_dir / "origin_0.png").write_bytes(files["origin_0.png"])
    (sample_out_dir / "result_1.png").write_bytes(files["result_1.png"])
    # Resize onto result_1.png's actual pixel space, same gotcha as the
    # single-instance fetch script: mask.png dimensions can drift from the
    # saved PNG's actual size.
    mask0_img.resize(after_img.size, Image.NEAREST).save(sample_out_dir / "mask_0.png")
    mask1_img.resize(after_img.size, Image.NEAREST).save(sample_out_dir / "mask_1.png")

    instances = []
    for idx, (obj, et) in enumerate([(obj1, edit_type[0]), (obj2, edit_type[1])]):
        bbox_norm, centroid_norm = bbox_and_centroid_norm(obj["bbox"], res_w, res_h)
        instances.append(
            {
                "index": idx,
                "edit_type": et,
                "class_name": obj.get("class_name"),
                "mask": f"images/{sample_id}/mask_{idx}.png",
                "bbox_norm": bbox_norm,
                "centroid_norm": centroid_norm,
            }
        )

    counters.accepted += 1
    return {
        "id": sample_id,
        "edit_prompt": edit_prompt,
        "edit_type": edit_type,
        "input_image": f"images/{sample_id}/origin_0.png",
        "output_image": f"images/{sample_id}/result_1.png",
        "resolution": {"height": after_img.height, "width": after_img.width},
        "instances": instances,
        "source_tar": source_tar,
    }


def fetch_part(
    part: str,
    filters: Filters,
    out_dir: Path,
    manifest_fh,
    session: requests.Session,
    api: HfApi,
    max_total: int | None,
    global_accepted: list[int],
) -> None:
    counters = Counters()
    out_images_dir = out_dir / "images"
    tar_base = PARTS[part]

    urls = list_split_urls(api, tar_base)
    print(f"[{part}] streaming {tar_base} ({len(urls)} split file(s), no cap — see module docstring)")
    reader = MultiUrlReader(urls, session)
    try:
        with tarfile.open(fileobj=reader, mode="r|") as tf:
            for sample_dir, files in iter_grouped_samples(tf, EXPECTED_FILES):
                row = process_sample(sample_dir, files, tar_base, filters, counters, out_images_dir)
                if row is not None:
                    manifest_fh.write(json.dumps(row) + "\n")
                    manifest_fh.flush()
                    global_accepted[0] += 1
                if counters.seen % 100 == 0:
                    print(
                        f"[{part}] seen={counters.seen} accepted={counters.accepted} "
                        f"(global={global_accepted[0]}) "
                        f"(rej: edit_type={counters.rejected_edit_type} mask_frac={counters.rejected_mask_frac} "
                        f"missing_obj={counters.rejected_missing_obj} parse={counters.rejected_parse})",
                        file=sys.stderr,
                    )
                if max_total is not None and global_accepted[0] >= max_total:
                    print(f"[{part}] global --max-total {max_total} reached, stopping")
                    break
    except tarfile.ReadError:
        pass
    finally:
        reader.close()

    print(f"[{part}] done: accepted {counters.accepted} (saw {counters.seen} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("./data/imgedit_multi_instance_eval"))
    parser.add_argument(
        "--parts",
        type=str,
        default=",".join(PARTS.keys()),
        help=f"comma-separated subset of {list(PARTS.keys())} (run one per process to parallelize)",
    )
    parser.add_argument("--max-total", type=int, default=9000, help="stop across all --parts once this many accepted; None/0 for unlimited")
    parser.add_argument("--exclude-types", type=str, default="add", help="comma-separated edit_type values that disqualify a sample if either instance has one")
    parser.add_argument("--require-any-type", type=str, default="remove", help="comma-separated edit_type values; at least one must be present (empty string to disable)")
    parser.add_argument("--min-mask-frac", type=float, default=0.001)
    parser.add_argument("--max-mask-frac", type=float, default=0.7)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    for p in parts:
        if p not in PARTS:
            parser.error(f"unknown part {p!r}, choose from {list(PARTS.keys())}")

    max_total = args.max_total if args.max_total and args.max_total > 0 else None
    exclude_types = frozenset(t.strip() for t in args.exclude_types.split(",") if t.strip())
    require_any_type = frozenset(t.strip() for t in args.require_any_type.split(",") if t.strip())
    filters = Filters(exclude_types, require_any_type, args.min_mask_frac, args.max_mask_frac)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(exist_ok=True)

    token = args.hf_token or get_token()
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    api = HfApi(token=token)

    # Shared mutable counter across parts run sequentially in this process
    # (a separate --parts invocation per process, as suggested in the
    # docstring, gets its own independent counter/manifest and doesn't need
    # this to be cross-process safe).
    global_accepted = [0]

    for part in parts:
        if max_total is not None and global_accepted[0] >= max_total:
            break
        manifest_path = args.out_dir / f"manifest_{part}.jsonl"
        with open(manifest_path, "a") as manifest_fh:
            fetch_part(
                part=part,
                filters=filters,
                out_dir=args.out_dir,
                manifest_fh=manifest_fh,
                session=session,
                api=api,
                max_total=max_total,
                global_accepted=global_accepted,
            )
        print(f"[{part}] manifest written to {manifest_path}")

    print(f"\nTotal accepted across {parts}: {global_accepted[0]}")


if __name__ == "__main__":
    main()
