"""Shared, dependency-light helpers for the PE-injection LoRA pipeline
(scripts/precompute_pe_lora_variants.py + scripts/train_pe_lora.py).

RoPE axis convention (confirmed against diffusers' Flux2 code, see
Flux2Transformer2DModel / Flux2KleinPipeline._prepare_text_ids /
Flux2KleinPipeline._prepare_latent_ids): 4 axes, in order (T, H, W, L) --
T=time/ref-image id, H/W=spatial position in *patched-latent-grid* units
(vae_scale_factor=8, x2 for 2x2 patchify => 16x downsample from pixel
space), L=token/sequence index. Text tokens normally get H=W=0; this
project's whole mechanism is injecting the target instance's mask centroid
into H,W for a sample's real (non-pad) text tokens instead ([[project_pe_lora_goal]]).
"""

from __future__ import annotations

import re

import numpy as np
import torch


def build_rope_ids(
    attention_mask: torch.Tensor,
    h_coord: float,
    w_coord: float,
    t_coord: float = 0.0,
) -> torch.Tensor:
    """Build [L, 4] (T, H, W, L) RoPE ids for one sample's text tokens.

    Real (attention_mask == 1) positions -- which include the chat-template
    tokens, not just the edit-instruction span, since isolating that span
    would require brittle template-string introspection -- get
    (t_coord, h_coord, w_coord). Pad positions stay at (t_coord, 0, 0),
    matching Flux2KleinPipeline._prepare_text_ids's own default-zero
    convention. The L axis is always arange(L), pad or not.
    """
    length = attention_mask.shape[0]
    ids = torch.zeros(length, 4, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, dtype=torch.float32)
    real = attention_mask.bool()
    ids[real, 0] = t_coord
    ids[real, 1] = h_coord
    ids[real, 2] = w_coord
    return ids


def mask_centroid_px(
    mask_arr: np.ndarray,
    dark_is_object: bool = True,
    threshold: int = 128,
) -> tuple[float, float] | None:
    """Mean pixel coordinates (cx_px, cy_px) of the object region in an HxW
    mask array, in the array's own pixel space. Mask convention (ImgEdit's
    rle_to_mask, matches [[project_imgedit_dataset_gotchas]]): edited/object
    region is DARK (~0), background LIGHT (~255). Returns None if the
    thresholded region is empty (e.g. fully cropped away).
    """
    obj = mask_arr < threshold if dark_is_object else mask_arr >= threshold
    ys, xs = np.nonzero(obj)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def px_to_patched_grid(cx_px: float, cy_px: float, downsample: int = 16) -> tuple[float, float]:
    """Pixel-space centroid -> (h_coord, w_coord) in patched-latent-grid
    units. H is the row/height axis (uses cy, vertical); W is the
    column/width axis (uses cx, horizontal). Fractional on purpose -- RoPE
    handles continuous positions, no need to snap to an integer grid point.
    """
    return cy_px / downsample, cx_px / downsample


_LEADING_ARTICLE_RE = r"(?:a|an|the)\s+"


def maybe_drop_source_noun(edit_prompt: str, class_name: str | None, rng) -> tuple[str, bool]:
    """Strip a (leading-article +) class_name occurrence from edit_prompt.

    ImgEdit's edit_prompt strings are LLM-generated free text (the object
    name is usually, but not grammatically guaranteed, echoed literally --
    see ImgEdit/tools/5_generate_prompt.py's few-shot templates), so this
    can miss. Returns (possibly-modified prompt, whether a match was
    found/removed) -- callers must check the flag and log the miss rate
    rather than assume every call succeeds. `rng` is accepted but unused
    here (kept in the signature so callers don't need a special case) --
    the probability gate for *whether* to apply the drop belongs to the
    caller, not this pure string function.
    """
    if not class_name:
        return edit_prompt, False
    pattern = re.compile(rf"\b(?:{_LEADING_ARTICLE_RE})?{re.escape(class_name)}\b", re.IGNORECASE)
    new_prompt, n = pattern.subn("", edit_prompt, count=1)
    if n == 0:
        return edit_prompt, False
    new_prompt = re.sub(r"\s{2,}", " ", new_prompt).strip()
    return new_prompt, True
