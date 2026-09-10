#!/usr/bin/env python3
"""
Inference for the PE-injection LoRA ([[project_pe_lora_goal]]).

Flux2KleinPipeline.__call__ CANNOT do centroid injection: it always builds
txt_ids itself via encode_prompt -> _prepare_text_ids, which pins every text
token's (H, W) to (0, 0) and exposes no parameter to override it. So this
script does not call the pipeline directly -- it loads the same components
the pipeline would (vae, tokenizer, text_encoder, transformer, scheduler)
and reimplements __call__'s denoising loop by hand, substituting:
  - txt_ids: built via pe_lora_common.inject_centroids, the exact same
    function scripts/precompute_pe_lora_variants.py uses, from a per-token
    instance_ids vector (pe_lora_common.compute_instance_ids) and each
    instance's desired centroid (caller-supplied here, since at inference
    the desired object *placement* is the whole point -- there's no ground
    truth mask to derive it from like there is at train time).
  - attention_kwargs={"attention_mask": ...}: the same cross-instance mask
    scripts/train_pe_lora.py builds via
    pe_lora_common.build_cross_instance_attention_mask, so text tokens
    belonging to different instances still can't attend to each other.

Everything else (image conditioning, latent prep, scheduler timesteps,
final unpack/decode) is copied from
diffusers/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py's __call__
as closely as possible -- diff against that file if diffusers is upgraded.

Trained on 2-instance samples ([[project_pe_lora_goal]]); this script
accepts N >= 1 instances since generalizing to ~8 at inference is the
experiment's whole point. No classifier-free guidance branch: Flux2 Klein
is a distilled model (do_classifier_free_guidance is gated on
`not config.is_distilled`), so it's omitted for simplicity -- add it back
(see __call__'s negative_prompt_embeds/negative_text_ids handling) if you
point this at a non-distilled checkpoint with guidance_scale > 1.

Example (2 instances):
  python scripts/infer_pe_lora.py \
      --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein \
      --lora-dir ./out/pe_lora_run \
      --source-image ./examples/source.png \
      --segment "add a red exit sign above the door" --center 0.72 0.18 \
      --segment "darken the window on the left" --center 0.15 0.55 \
      --output ./out/edited.png

Example (4 instances, testing generalization beyond the 2-instance training data):
  python scripts/infer_pe_lora.py \
      --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein \
      --lora-dir ./out/pe_lora_run --source-image ./examples/source.png \
      --segment "add a red exit sign above the door" --center 0.72 0.18 \
      --segment "darken the window on the left" --center 0.15 0.55 \
      --segment "remove the trash can" --center 0.50 0.85 \
      --segment "replace the poster with a map" --center 0.30 0.40 \
      --output ./out/edited.png

--center is (cx, cy) normalized to [0, 1] of the OUTPUT image (--height/--width)
-- i.e. where you want that instance's edit to be anchored, not where it
currently is in the source image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

from pe_lora_common import build_cross_instance_attention_mask, compute_instance_ids, inject_centroids, px_to_patched_grid
from precompute_pe_lora_variants import DOWNSAMPLE, encode_prompt_with_offsets

AND_GLUE = " and "


def build_prompt_and_spans(segments: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Joins instance clauses the same way precompute_pe_lora_variants.py's
    resolve_instance_segments does ("{seg0}{AND_GLUE}{seg1}...") so spans
    are trivial to track by construction -- no find_hybrid_split parsing
    needed here since we're authoring the prompt, not recovering its
    structure from ImgEdit's free text."""
    text = AND_GLUE.join(segments)
    spans, pos = [], 0
    for seg in segments:
        spans.append((pos, pos + len(seg)))
        pos += len(seg) + len(AND_GLUE)
    return text, spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--segment", action="append", required=True, help="one instance's edit clause; repeat per instance")
    parser.add_argument(
        "--center", action="append", nargs=2, type=float, required=True,
        help="(cx, cy) normalized to [0,1] of the OUTPUT image for this instance; repeat once per --segment, same order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--max-sequence-length", type=int, default=160, help="match the value precompute was run with")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    if len(args.segment) != len(args.center):
        raise ValueError(f"got {len(args.segment)} --segment but {len(args.center)} --center, must match 1:1 in order")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device found, running on CPU -- this will be extremely slow.")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print("Loading pipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=dtype
    ).to(device)
    pipe.transformer.eval()
    print(f"Loading LoRA weights from {args.lora_dir}...")
    pipe.load_lora_weights(str(args.lora_dir))

    # --- text: encode + compute per-token instance_ids + inject centroids ---
    text, spans = build_prompt_and_spans(args.segment)
    print(f"Prompt: {text!r}")
    embeds, _amask, offsets, was_truncated = encode_prompt_with_offsets(
        pipe.text_encoder, pipe.tokenizer, text, device, dtype, args.max_sequence_length
    )
    if was_truncated:
        print(f"WARNING: prompt truncated at max_sequence_length={args.max_sequence_length}")
    prompt_embeds = embeds.unsqueeze(0).to(device=device, dtype=dtype)  # [1, L, D]
    instance_ids = compute_instance_ids(offsets, spans)  # [L]

    centroids_grid = [
        px_to_patched_grid(cx_norm * args.width, cy_norm * args.height, downsample=DOWNSAMPLE)
        for cx_norm, cy_norm in args.center
    ]
    text_ids = inject_centroids(instance_ids, centroids_grid).unsqueeze(0).to(device)  # [1, L, 4]

    # --- image conditioning (source image to edit) ---
    source_image = Image.open(args.source_image).convert("RGB")
    multiple_of = pipe.vae_scale_factor * 2
    height = (args.height // multiple_of) * multiple_of
    width = (args.width // multiple_of) * multiple_of
    cond_image = pipe.image_processor.preprocess(source_image, height=height, width=width, resize_mode="crop")
    image_latents, image_latent_ids = pipe.prepare_image_latents(
        images=[cond_image], batch_size=1, generator=generator, device=device, dtype=pipe.vae.dtype
    )

    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_ids = pipe.prepare_latents(
        batch_size=1,
        num_latents_channels=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
    )

    num_img_tokens = latents.shape[1] + image_latents.shape[1]  # main + cond, matches training's packed_input.shape[1]
    attention_mask = build_cross_instance_attention_mask(instance_ids, num_img_tokens).to(device)

    # --- timesteps (copied from Flux2KleinPipeline.__call__) ---
    sigmas = np.linspace(1.0, 1 / args.num_inference_steps, args.num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=args.num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, args.num_inference_steps, device, sigmas=sigmas, mu=mu
    )
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(0)

    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], args.guidance_scale, device=device).expand(latents.shape[0])

    # --- denoising loop (copied from Flux2KleinPipeline.__call__, minus the CFG branch) ---
    print(f"Denoising, {num_inference_steps} steps...")
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        latent_model_input = torch.cat([latents, image_latents], dim=1).to(pipe.transformer.dtype)
        latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={"attention_mask": attention_mask},
                return_dict=False,
            )[0]
        noise_pred = noise_pred[:, : latents.size(1), :]

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        print(f"  step {i + 1}/{num_inference_steps}", end="\r")
    print()

    # --- unpack + decode (copied from Flux2KleinPipeline.__call__) ---
    latent_height = 2 * (height // (pipe.vae_scale_factor * 2))
    latent_width = 2 * (width // (pipe.vae_scale_factor * 2))
    latents = pipe._unpack_latents_with_ids(latents, latent_ids, latent_height // 2, latent_width // 2)

    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    latents = latents * bn_std + bn_mean
    latents = pipe._unpatchify_latents(latents)

    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
