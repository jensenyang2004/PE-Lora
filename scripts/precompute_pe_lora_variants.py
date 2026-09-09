#!/usr/bin/env python3
"""
Offline precompute pass for the PE-injection LoRA ([[project_pe_lora_goal]]).

Reads the 2-INSTANCE manifest.jsonl from fetch_imgedit_multi_instance.py
(one combined edit_prompt covering two edits, two masks/bboxes -- see that
script's docstring for the exact schema). This IS the training data (see
[[project_pe_lora_goal]]'s 2026-09-09 correction) -- training is on 2
instances, generalizing to ~8 average at inference.

For every accepted row, writes K randomly-augmented (image, two mask-derived
centroids, prompt) variants to disk, fully encoded (VAE latents + Qwen3
prompt embeddings + baked RoPE ids + a per-text-token instance_ids vector).
scripts/train_pe_lora.py then loads *only* this cache plus the small Flux2
transformer + LoRA -- no VAE, no text encoder resident during training.

Two kinds of variants, as before, generated separately because only the
noun-dropout coin flip changes the encoded text -- geometric augmentation
never touches it:
  - "text variants" (<=2 per sample): both instances' clauses in full, and
    (if pe_lora_common.maybe_drop_source_noun matches on either clause) a
    noun-dropped version of both.
  - "geometric variants" (K per sample, default 8): a random resize+crop+
    flip applied identically to the target image, the source image, and
    BOTH masks -- both centroids are always recomputed from the
    *transformed* masks, never assumed to shift affinely with the crop.

Text segmentation: ImgEdit's combined edit_prompt has no guaranteed clean
boundary between the two instances' clauses. pe_lora_common.find_hybrid_split
implements the user's rule (split on the "and" immediately followed by a
known edit verb, or the "and" closest to the string's midpoint if none/many
qualify) after inspecting real hybrid prompts. Falls back to a synthesized
"{edit_type} the {class_name}" per instance when a sample has no "and" at
all (logged, expected rare).

Run this on the training server (needs the VAE + Qwen3 text encoder loaded):
  python scripts/precompute_pe_lora_variants.py \
      --manifest ./data/imgedit_multi_instance_eval/manifest.jsonl \
      --out-dir ./data/pe_lora_cache \
      --pretrained_model_name_or_path <repo id, e.g. black-forest-labs/FLUX.2-klein>

Try a tiny slice first:
  python scripts/precompute_pe_lora_variants.py --limit 5 --num-variants 3 ...
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
from diffusers.training_utils import find_nearest_bucket, generate_aspect_ratio_buckets

from pe_lora_common import (
    compute_instance_ids,
    find_hybrid_split,
    inject_centroids,
    mask_centroid_px,
    maybe_drop_source_noun,
    px_to_patched_grid,
)

DOWNSAMPLE = 16  # vae_scale_factor(8) * patchify(2) -- pixel space -> patched-latent-grid units
IMAGE_TEMPORAL_SCALE = 10  # matches the base img2img script's Flux2KleinPipeline._prepare_image_ids default
AND_GLUE = " and "


def resolve_instance_segments(
    edit_prompt: str,
    instances: list[dict],
    rng: random.Random,
    apply_dropout: bool,
) -> tuple[str, list[tuple[int, int]], bool, bool]:
    """Build the text actually fed to the encoder plus each instance's char
    span within it. Returns (text, [span0, span1], used_template_fallback,
    any_dropout_matched).

    Always reconstructs as "{seg0}{AND_GLUE}{seg1}" (rather than reusing
    edit_prompt verbatim when a split succeeds) so the full/dropout/fallback
    paths share one code path and span-computation is trivial -- the only
    cost is losing whatever exact connecting wording ImgEdit's LLM used
    between the two clauses, not any real content.
    """
    split = find_hybrid_split(edit_prompt)
    used_fallback = split is None
    if split is not None:
        seg0_end, seg1_start = split
        seg0, seg1 = edit_prompt[:seg0_end].strip(), edit_prompt[seg1_start:].strip()
    else:
        seg0 = f"{instances[0]['edit_type']} the {instances[0]['class_name']}"
        seg1 = f"{instances[1]['edit_type']} the {instances[1]['class_name']}"

    any_matched = False
    if apply_dropout:
        seg0, m0 = maybe_drop_source_noun(seg0, instances[0].get("class_name"), rng)
        seg1, m1 = maybe_drop_source_noun(seg1, instances[1].get("class_name"), rng)
        any_matched = m0 or m1

    text = f"{seg0}{AND_GLUE}{seg1}"
    spans = [(0, len(seg0)), (len(seg0) + len(AND_GLUE), len(text))]
    return text, spans, used_fallback, any_matched


def encode_prompt_with_offsets(
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    text: str,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
    hidden_states_layers: tuple[int, ...] = (9, 18, 27),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Re-implements Flux2KleinPipeline._get_qwen3_prompt_embeds (same chat
    template + padding + layer-stacking convention) but also returns the
    tokenizer's attention_mask and a per-token offset_mapping *relative to
    `text` itself* (not the chat-template-wrapped string) -- letting callers
    know exactly which tokens fall inside a given char span of the original
    instruction text, via pe_lora_common.compute_instance_ids.
    Returns (prompt_embeds [L, D], attention_mask [L], offset_mapping [L, 2], was_truncated).
    """
    messages = [{"role": "user", "content": text}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    content_start = chat_text.index(text)  # the template inserts content verbatim

    untruncated_len = len(tokenizer(chat_text, truncation=False)["input_ids"])
    was_truncated = untruncated_len > max_sequence_length

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
        return_offsets_mapping=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    offset_mapping = inputs["offset_mapping"][0] - content_start  # -> relative to `text`

    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
        )
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)  # [1, nlayers, L, hdim]
    out = out.to(dtype=dtype, device=device)
    _, num_layers, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(seq_len, num_layers * hidden_dim)

    return prompt_embeds.cpu(), attention_mask[0].cpu(), offset_mapping.cpu(), was_truncated


def _sample_crop_offset(
    new_h: int,
    new_w: int,
    target_h: int,
    target_w: int,
    rng: random.Random,
    bbox_px: tuple[float, float, float, float] | None,
) -> tuple[int, int]:
    """Pick a crop top-left (i, j). With bbox_px=None, uniformly random over
    the valid range. With bbox_px set (x1,y1,x2,y2 in the new_h,new_w scaled
    space), constrained so the crop window fully contains the bbox -- falls
    back to centering on the bbox if the bbox itself is larger than the crop
    in some dimension."""
    max_i, max_j = max(0, new_h - target_h), max(0, new_w - target_w)
    if bbox_px is None:
        i = rng.randint(0, max_i) if max_i > 0 else 0
        j = rng.randint(0, max_j) if max_j > 0 else 0
        return i, j

    x1, y1, x2, y2 = bbox_px
    lo_i, hi_i = max(0, int(y2) - target_h + 1), min(max_i, int(y1))
    i = rng.randint(lo_i, hi_i) if hi_i > lo_i else max(0, min(max_i, int(round((y1 + y2) / 2 - target_h / 2))))
    lo_j, hi_j = max(0, int(x2) - target_w + 1), min(max_j, int(x1))
    j = rng.randint(lo_j, hi_j) if hi_j > lo_j else max(0, min(max_j, int(round((x1 + x2) / 2 - target_w / 2))))
    return i, j


def augment_variant_multi(
    output_image: Image.Image,
    input_image_aligned: Image.Image,
    mask_images: list[Image.Image],
    bucket: tuple[int, int],
    bboxes_norm: list[list[float]],
    rng: random.Random,
    min_object_frac: float,
    retries: int,
) -> tuple[Image.Image, Image.Image, list[Image.Image], list[tuple[float, float]], bool] | None:
    """Returns (cropped output, cropped input, [cropped mask0, mask1],
    [(cx0,cy0), (cx1,cy1)] centroids in the cropped image's pixel space,
    used_fallback), or None if even the union-bbox fallback couldn't keep
    both objects visible after 2 attempts -- caller should skip this variant
    and log it, not silently reduce K without a record."""
    target_h, target_w = bucket
    width, height = output_image.size
    scale = max(target_h / height, target_w / width)
    new_h, new_w = round(height * scale), round(width * scale)

    out_r = TF.resize(output_image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    in_r = TF.resize(input_image_aligned, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    mask_rs = [TF.resize(m, [new_h, new_w], interpolation=InterpolationMode.NEAREST) for m in mask_images]

    def crop_all(i: int, j: int, flip: bool):
        out_c = TF.crop(out_r, i, j, target_h, target_w)
        in_c = TF.crop(in_r, i, j, target_h, target_w)
        mask_cs = [TF.crop(m, i, j, target_h, target_w) for m in mask_rs]
        if flip:
            out_c, in_c = TF.hflip(out_c), TF.hflip(in_c)
            mask_cs = [TF.hflip(m) for m in mask_cs]
        return out_c, in_c, mask_cs, [np.array(m) for m in mask_cs]

    for _ in range(retries):
        flip = rng.random() < 0.5
        i, j = _sample_crop_offset(new_h, new_w, target_h, target_w, rng, bbox_px=None)
        out_c, in_c, mask_cs, mask_arrs = crop_all(i, j, flip)
        centroids = [mask_centroid_px(a) for a in mask_arrs]
        fracs = [(a < 128).mean() for a in mask_arrs]
        if all(c is not None for c in centroids) and all(f >= min_object_frac for f in fracs):
            return out_c, in_c, mask_cs, centroids, False

    # Fallback: constrain the crop to contain the union of both instances' bboxes.
    xs1 = [b[0] for b in bboxes_norm]
    ys1 = [b[1] for b in bboxes_norm]
    xs2 = [b[2] for b in bboxes_norm]
    ys2 = [b[3] for b in bboxes_norm]
    bbox_px = (min(xs1) * new_w, min(ys1) * new_h, max(xs2) * new_w, max(ys2) * new_h)

    for _ in range(2):  # up to 2 fallback attempts before giving up on this variant
        flip = rng.random() < 0.5
        i, j = _sample_crop_offset(new_h, new_w, target_h, target_w, rng, bbox_px=bbox_px)
        out_c, in_c, mask_cs, mask_arrs = crop_all(i, j, flip)
        centroids = [mask_centroid_px(a) for a in mask_arrs]
        if all(c is not None for c in centroids):
            return out_c, in_c, mask_cs, centroids, True

    return None


def to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(image), [0.5], [0.5])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--num-variants", type=int, default=8, help="K geometric variants per sample")
    parser.add_argument("--max-sequence-length", type=int, default=160)
    parser.add_argument(
        "--noun-dropout-prob", type=float, default=0.3, help="probability a geometric variant uses the noun-dropped text"
    )
    parser.add_argument("--resolution", type=int, default=1024, help="base resolution fed to generate_aspect_ratio_buckets")
    parser.add_argument("--bucket-divisibility", type=int, default=16)
    parser.add_argument("--min-mask-frac", type=float, default=0.003, help="minimum object-pixel fraction of a crop before it's accepted")
    parser.add_argument("--crop-retries", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N manifest rows (smoke testing)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("WARNING: no CUDA device found, running on CPU -- this will be extremely slow for a real run.")

    manifest_dir = args.manifest.parent
    rows = [json.loads(line) for line in args.manifest.open()]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} manifest rows from {args.manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print("Loading VAE + Qwen3 text encoder...")
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device=device, dtype=dtype)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, torch_dtype=dtype
    ).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    index_fh = (args.out_dir / "index.jsonl").open("a")
    total_bytes = 0
    n_truncated = 0
    n_dropout_matched = 0
    n_template_fallback = 0
    n_fallback_crops = 0
    n_skipped_variants = 0
    n_accepted = 0
    t0 = time.time()

    bucket_list = generate_aspect_ratio_buckets(args.resolution, divisibility=args.bucket_divisibility)

    for row_idx, row in enumerate(rows):
        sample_id = row["id"]
        sample_dir = args.out_dir / sample_id
        instances = row["instances"]
        if len(instances) != 2:
            print(f"[{sample_id}] skipping, expected exactly 2 instances, got {len(instances)}", file=sys.stderr)
            continue

        try:
            output_image = exif_transpose(Image.open(manifest_dir / row["output_image"])).convert("RGB")
            input_image = exif_transpose(Image.open(manifest_dir / row["input_image"])).convert("RGB")
            mask_images = [
                exif_transpose(Image.open(manifest_dir / inst["mask"])).convert("L") for inst in instances
            ]
        except Exception as exc:
            print(f"[{sample_id}] skipping, failed to load images: {exc}", file=sys.stderr)
            continue

        target_w, target_h = output_image.size  # PIL size is (W, H)
        if input_image.size != (target_w, target_h):
            input_image = input_image.resize((target_w, target_h), Image.BILINEAR)
        mask_images = [
            m if m.size == (target_w, target_h) else m.resize((target_w, target_h), Image.NEAREST)
            for m in mask_images
        ]

        bucket = bucket_list[find_nearest_bucket(target_h, target_w, bucket_list)]

        # --- text variants (<=2, shared across all K geometric variants) ---
        text_variants = []
        text0, spans0, used_fallback0, _ = resolve_instance_segments(
            row["edit_prompt"], instances, rng, apply_dropout=False
        )
        n_template_fallback += int(used_fallback0)
        embeds0, amask0, offsets0, trunc0 = encode_prompt_with_offsets(
            text_encoder, tokenizer, text0, device, dtype, args.max_sequence_length
        )
        n_truncated += int(trunc0)
        iids0 = compute_instance_ids(offsets0, spans0)
        text_variants.append({"prompt_embeds": embeds0, "attention_mask": amask0, "instance_ids": iids0})

        text1, spans1, _, dropout_matched = resolve_instance_segments(
            row["edit_prompt"], instances, rng, apply_dropout=True
        )
        if dropout_matched:
            n_dropout_matched += 1
            embeds1, amask1, offsets1, trunc1 = encode_prompt_with_offsets(
                text_encoder, tokenizer, text1, device, dtype, args.max_sequence_length
            )
            n_truncated += int(trunc1)
            iids1 = compute_instance_ids(offsets1, spans1)
            text_variants.append({"prompt_embeds": embeds1, "attention_mask": amask1, "instance_ids": iids1})

        sample_dir.mkdir(parents=True, exist_ok=True)
        for ti, tv in enumerate(text_variants):
            path = sample_dir / f"text_{ti}.pt"
            torch.save(tv, path)
            total_bytes += path.stat().st_size

        # --- geometric variants ---
        print(f"[{row_idx + 1}/{len(rows)}] {sample_id}: text encoded, building {args.num_variants} geometric variants...", file=sys.stderr)
        bboxes_norm = [inst["bbox_norm"] for inst in instances]
        geo_idx = 0
        for _ in range(args.num_variants):
            result = augment_variant_multi(
                output_image, input_image, mask_images, bucket, bboxes_norm, rng, args.min_mask_frac, args.crop_retries
            )
            if result is None:
                n_skipped_variants += 1
                continue
            out_c, in_c, mask_cs, centroids_px, used_fallback = result
            n_fallback_crops += int(used_fallback)

            centroids_grid = [px_to_patched_grid(cx, cy, downsample=DOWNSAMPLE) for cx, cy in centroids_px]

            text_idx = 0
            if len(text_variants) > 1 and rng.random() < args.noun_dropout_prob:
                text_idx = 1
            text_ids = inject_centroids(text_variants[text_idx]["instance_ids"], centroids_grid)

            with torch.no_grad():
                out_t = to_normalized_tensor(out_c).unsqueeze(0).to(device=device, dtype=dtype)
                in_t = to_normalized_tensor(in_c).unsqueeze(0).to(device=device, dtype=dtype)
                model_input = vae.encode(out_t).latent_dist.mode()
                cond_model_input = vae.encode(in_t).latent_dist.mode()

                model_input = Flux2KleinPipeline._patchify_latents(model_input)
                model_input = (model_input - bn_mean) / bn_std
                cond_model_input = Flux2KleinPipeline._patchify_latents(cond_model_input)
                cond_model_input = (cond_model_input - bn_mean) / bn_std

                main_ids = Flux2KleinPipeline._prepare_latent_ids(model_input)
                cond_ids = Flux2KleinPipeline._prepare_image_ids([cond_model_input], scale=IMAGE_TEMPORAL_SCALE)
                img_ids = torch.cat([main_ids, cond_ids], dim=1)[0]  # [S_main+S_cond, 4]

            record = {
                "model_input": model_input[0].to(torch.bfloat16).cpu(),
                "cond_model_input": cond_model_input[0].to(torch.bfloat16).cpu(),
                "text_ids": text_ids,
                "img_ids": img_ids.cpu(),
                "text_variant_idx": text_idx,
            }
            path = sample_dir / f"geo_{geo_idx}.pt"
            torch.save(record, path)
            total_bytes += path.stat().st_size
            geo_idx += 1

        if geo_idx == 0:
            print(f"[{sample_id}] skipping, no geometric variant survived augmentation", file=sys.stderr)
            continue

        index_fh.write(
            json.dumps(
                {
                    "id": sample_id,
                    "bucket": list(bucket),
                    "num_geo_variants": geo_idx,
                    "has_dropout_variant": len(text_variants) > 1,
                    "used_template_fallback": used_fallback0,
                    "instances": [
                        {"class_name": inst.get("class_name"), "edit_type": inst.get("edit_type")} for inst in instances
                    ],
                }
            )
            + "\n"
        )
        index_fh.flush()
        n_accepted += 1

        elapsed = time.time() - t0
        s_per_row = elapsed / (row_idx + 1)
        eta = s_per_row * (len(rows) - row_idx - 1)
        print(
            f"[{row_idx + 1}/{len(rows)}] accepted={n_accepted} "
            f"dropout_matched={n_dropout_matched} template_fallback={n_template_fallback} "
            f"truncated={n_truncated} fallback_crops={n_fallback_crops} "
            f"skipped_variants={n_skipped_variants} disk={total_bytes / 1e9:.2f}GB "
            f"elapsed={elapsed:.0f}s ({s_per_row:.1f}s/sample, ETA {eta / 60:.0f}min)",
            file=sys.stderr,
        )

    index_fh.close()
    (args.out_dir / "meta.json").write_text(
        json.dumps(
            {
                "max_sequence_length": args.max_sequence_length,
                "num_variants": args.num_variants,
                "bucket_divisibility": args.bucket_divisibility,
                "downsample": DOWNSAMPLE,
                "image_temporal_scale": IMAGE_TEMPORAL_SCALE,
                "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
                "dtype": args.dtype,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print(
        f"\nDone: {n_accepted}/{len(rows)} samples, {total_bytes / 1e9:.2f}GB total, "
        f"{n_dropout_matched} noun-dropout matches, {n_template_fallback} template-fallback prompts, "
        f"{n_truncated} truncated, {n_fallback_crops} fallback crops, "
        f"{n_skipped_variants} skipped geometric variants. Cache at {args.out_dir}"
    )


if __name__ == "__main__":
    main()
