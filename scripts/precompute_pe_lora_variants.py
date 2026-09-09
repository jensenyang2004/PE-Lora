#!/usr/bin/env python3
"""
Offline precompute pass for the PE-injection LoRA ([[project_pe_lora_goal]]).

Reads a single-instance manifest.jsonl (from fetch_imgedit_data.py -- NOT the
multi-instance manifest from fetch_imgedit_multi_instance.py, which is a
held-out eval set, never training data), and for every accepted row writes
K randomly-augmented (image, mask-derived centroid, prompt) variants to
disk, fully encoded (VAE latents + Qwen3 prompt embeddings + baked RoPE
ids). scripts/train_pe_lora.py then loads *only* this cache plus the small
Flux2 transformer + LoRA -- no VAE, no text encoder resident during
training -- to keep training-loop VRAM low and predictable.

Two kinds of variants, generated separately because only the source-noun
dropout coin flip changes the encoded prompt -- geometric augmentation never
touches text, so text is encoded at most twice per sample and geometric
variants reference it by index (dedup matters: prompt_embeds is by far the
largest tensor here; see the disk-size note in the project plan):
  - "text variants" (<=2 per sample): the full edit_prompt, and (if
    pe_lora_common.maybe_drop_source_noun finds a literal match) the
    source-noun-dropped version.
  - "geometric variants" (K per sample, default 8): a random resize+crop+
    flip applied identically to the target image, the source image (already
    aligned onto the target's pixel grid), and the mask -- the centroid is
    always recomputed from the *transformed* mask, never assumed to shift
    affinely with the crop.

Run this on the training server (needs the VAE + Qwen3 text encoder loaded,
a GPU strongly recommended, though nothing here needs the diffusion
transformer itself):
  python scripts/precompute_pe_lora_variants.py \
      --manifest ./data/imgedit_single_instance/manifest.jsonl \
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

from pe_lora_common import build_rope_ids, mask_centroid_px, maybe_drop_source_noun, px_to_patched_grid

DOWNSAMPLE = 16  # vae_scale_factor(8) * patchify(2) -- pixel space -> patched-latent-grid units
IMAGE_TEMPORAL_SCALE = 10  # matches the base img2img script's Flux2KleinPipeline._prepare_image_ids default


def encode_prompt_with_mask(
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
    hidden_states_layers: tuple[int, ...] = (9, 18, 27),
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Re-implements Flux2KleinPipeline._get_qwen3_prompt_embeds (same chat
    template + padding + layer-stacking convention) but also returns the
    tokenizer's attention_mask, which the pipeline's own encode_prompt()
    computes internally and discards -- we need it to know which text-token
    positions are real (get the injected centroid) vs pad (stay at 0,0).
    Returns (prompt_embeds [L, D], attention_mask [L], was_truncated).
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    untruncated_len = len(tokenizer(text, truncation=False)["input_ids"])
    was_truncated = untruncated_len > max_sequence_length

    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_sequence_length
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
        )
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)  # [1, nlayers, L, hdim]
    out = out.to(dtype=dtype, device=device)
    _, num_layers, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(seq_len, num_layers * hidden_dim)

    return prompt_embeds.cpu(), attention_mask[0].cpu(), was_truncated


def _sample_crop_offset(
    new_h: int, new_w: int, target_h: int, target_w: int, rng: random.Random, bbox_px: tuple[float, float, float, float] | None
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


def augment_variant(
    output_image: Image.Image,
    input_image_aligned: Image.Image,
    mask_image: Image.Image,
    bucket: tuple[int, int],
    bbox_norm: list[float],
    rng: random.Random,
    min_object_frac: float,
    retries: int,
) -> tuple[Image.Image, Image.Image, Image.Image, tuple[float, float], bool]:
    """Returns (cropped output, cropped input, cropped mask, (cx_px, cy_px)
    centroid in the *cropped* image's pixel space, used_fallback)."""
    target_h, target_w = bucket
    width, height = output_image.size
    scale = max(target_h / height, target_w / width)
    new_h, new_w = round(height * scale), round(width * scale)

    out_r = TF.resize(output_image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    in_r = TF.resize(input_image_aligned, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    mask_r = TF.resize(mask_image, [new_h, new_w], interpolation=InterpolationMode.NEAREST)

    def crop_triple(i: int, j: int, flip: bool) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray]:
        out_c = TF.crop(out_r, i, j, target_h, target_w)
        in_c = TF.crop(in_r, i, j, target_h, target_w)
        mask_c = TF.crop(mask_r, i, j, target_h, target_w)
        if flip:
            out_c, in_c, mask_c = TF.hflip(out_c), TF.hflip(in_c), TF.hflip(mask_c)
        return out_c, in_c, mask_c, np.array(mask_c)

    for _ in range(retries):
        flip = rng.random() < 0.5
        i, j = _sample_crop_offset(new_h, new_w, target_h, target_w, rng, bbox_px=None)
        out_c, in_c, mask_c, mask_arr = crop_triple(i, j, flip)
        centroid = mask_centroid_px(mask_arr)
        if centroid is not None and (mask_arr < 128).mean() >= min_object_frac:
            return out_c, in_c, mask_c, centroid, False

    # Fallback: constrain the crop to fully contain the original bbox.
    x1, y1, x2, y2 = bbox_norm
    bbox_px = (x1 * new_w, y1 * new_h, x2 * new_w, y2 * new_h)
    flip = rng.random() < 0.5
    i, j = _sample_crop_offset(new_h, new_w, target_h, target_w, rng, bbox_px=bbox_px)
    out_c, in_c, mask_c, mask_arr = crop_triple(i, j, flip)
    centroid = mask_centroid_px(mask_arr) or (target_w / 2.0, target_h / 2.0)
    return out_c, in_c, mask_c, centroid, True


def to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(image), [0.5], [0.5])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--num-variants", type=int, default=8, help="K geometric variants per sample")
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--noun-dropout-prob", type=float, default=0.3, help="probability a geometric variant uses the noun-dropped prompt, when available")
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
    n_fallback_crops = 0
    n_accepted = 0
    t0 = time.time()

    for row_idx, row in enumerate(rows):
        sample_id = row["id"]
        sample_dir = args.out_dir / sample_id
        try:
            output_image = exif_transpose(Image.open(manifest_dir / row["output_image"])).convert("RGB")
            input_image = exif_transpose(Image.open(manifest_dir / row["input_image"])).convert("RGB")
            mask_image = exif_transpose(Image.open(manifest_dir / row["mask"])).convert("L")
        except Exception as exc:
            print(f"[{sample_id}] skipping, failed to load images: {exc}", file=sys.stderr)
            continue

        target_w, target_h = output_image.size  # PIL size is (W, H)
        if input_image.size != (target_w, target_h):
            input_image = input_image.resize((target_w, target_h), Image.BILINEAR)
        if mask_image.size != (target_w, target_h):
            mask_image = mask_image.resize((target_w, target_h), Image.NEAREST)

        bucket = generate_aspect_ratio_buckets(args.resolution, divisibility=args.bucket_divisibility)
        bucket = bucket[find_nearest_bucket(target_h, target_w, bucket)]

        # --- text variants (<=2, shared across all K geometric variants) ---
        text_variants = []
        with torch.no_grad():
            embeds, amask, truncated = encode_prompt_with_mask(
                text_encoder, tokenizer, row["edit_prompt"], device, dtype, args.max_sequence_length
            )
        n_truncated += int(truncated)
        text_variants.append({"prompt_embeds": embeds, "attention_mask": amask})

        dropped_prompt, matched = maybe_drop_source_noun(row["edit_prompt"], row.get("class_name"), rng)
        if matched:
            n_dropout_matched += 1
            with torch.no_grad():
                embeds2, amask2, truncated2 = encode_prompt_with_mask(
                    text_encoder, tokenizer, dropped_prompt, device, dtype, args.max_sequence_length
                )
            n_truncated += int(truncated2)
            text_variants.append({"prompt_embeds": embeds2, "attention_mask": amask2})

        sample_dir.mkdir(parents=True, exist_ok=True)
        for ti, tv in enumerate(text_variants):
            path = sample_dir / f"text_{ti}.pt"
            torch.save(tv, path)
            total_bytes += path.stat().st_size

        # --- geometric variants ---
        num_geo_variants = 0
        for k in range(args.num_variants):
            out_c, in_c, mask_c, (cx_px, cy_px), used_fallback = augment_variant(
                output_image, input_image, mask_image, bucket, row["bbox_norm"], rng, args.min_mask_frac, args.crop_retries
            )
            n_fallback_crops += int(used_fallback)

            h_coord, w_coord = px_to_patched_grid(cx_px, cy_px, downsample=DOWNSAMPLE)

            text_idx = 0
            if len(text_variants) > 1 and rng.random() < args.noun_dropout_prob:
                text_idx = 1
            attn_mask = text_variants[text_idx]["attention_mask"]
            text_ids = build_rope_ids(attn_mask, h_coord=h_coord, w_coord=w_coord)

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
            path = sample_dir / f"geo_{k}.pt"
            torch.save(record, path)
            total_bytes += path.stat().st_size
            num_geo_variants += 1

        index_fh.write(
            json.dumps(
                {
                    "id": sample_id,
                    "bucket": list(bucket),
                    "num_geo_variants": num_geo_variants,
                    "has_dropout_variant": len(text_variants) > 1,
                    "class_name": row.get("class_name"),
                    "task": row.get("task"),
                }
            )
            + "\n"
        )
        index_fh.flush()
        n_accepted += 1

        if (row_idx + 1) % 20 == 0 or row_idx == len(rows) - 1:
            elapsed = time.time() - t0
            print(
                f"[{row_idx + 1}/{len(rows)}] accepted={n_accepted} "
                f"dropout_matched={n_dropout_matched} truncated={n_truncated} "
                f"fallback_crops={n_fallback_crops} disk={total_bytes / 1e9:.2f}GB "
                f"elapsed={elapsed:.0f}s",
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
        f"{n_dropout_matched} noun-dropout matches, {n_truncated} truncated prompts, "
        f"{n_fallback_crops} fallback crops. Cache at {args.out_dir}"
    )


if __name__ == "__main__":
    main()
