"""Shared, dependency-light helpers for the PE-injection LoRA pipeline
(scripts/precompute_pe_lora_variants.py + scripts/train_pe_lora.py).

RoPE axis convention (confirmed against diffusers' Flux2 code, see
Flux2Transformer2DModel / Flux2KleinPipeline._prepare_text_ids /
Flux2KleinPipeline._prepare_latent_ids): 4 axes, in order (T, H, W, L) --
T=time/ref-image id, H/W=spatial position in *patched-latent-grid* units
(vae_scale_factor=8, x2 for 2x2 patchify => 16x downsample from pixel
space), L=token/sequence index. Text tokens normally get H=W=0; this
project's whole mechanism is injecting each instance's mask centroid into
H,W for that instance's own text-token span instead ([[project_pe_lora_goal]]).

Training is on the 2-instance ImgEdit hybrid/compose data (one combined
prompt covering two edits), not single-instance data -- see
[[project_pe_lora_goal]]'s 2026-09-09 correction. That means two centroids
per sample, injected onto *different* text-token spans, and (since training
now has real cross-instance structure to overfit to) a real attention_mask
blocking one instance's text tokens from attending to the other's.
"""

from __future__ import annotations

import re

import numpy as np
import torch

# Verbs observed to mark the start of the second instance's clause in
# ImgEdit's hybrid edit_prompt strings (e.g. "... and adjust the device...").
# Used by find_hybrid_split to disambiguate when a prompt contains more than
# one literal "and".
EDIT_VERBS = {"modify", "remove", "replace", "enhance", "change", "darken", "adjust", "brighten"}


def find_hybrid_split(edit_prompt: str) -> tuple[int, int] | None:
    """Find the char-offset boundary between a hybrid sample's two instance
    clauses, in the convention "{instr0} and {verb1} {instr1}".

    Returns (seg0_end, seg1_start): edit_prompt[:seg0_end] is instance 0's
    clause, edit_prompt[seg1_start:] is instance 1's ("and " itself belongs
    to neither). Returns None if edit_prompt contains no "and" at all --
    callers must fall back (e.g. to a synthesized per-instance template)
    rather than guess.

    Among all "and" occurrences, prefers one immediately followed by a known
    edit verb (EDIT_VERBS); if none (or several) qualify, or as the general
    tiebreak, picks whichever candidate sits closest to the string's
    midpoint. This is the exact rule the user specified after inspecting
    real hybrid prompts.
    """
    candidates = [m.start() for m in re.finditer(r"\band\b", edit_prompt, re.IGNORECASE)]
    if not candidates:
        return None

    def verb_follows(idx: int) -> bool:
        m = re.match(r"and\s+(\w+)", edit_prompt[idx:], re.IGNORECASE)
        return bool(m) and m.group(1).lower() in EDIT_VERBS

    verb_candidates = [idx for idx in candidates if verb_follows(idx)]
    pool = verb_candidates or candidates
    mid = len(edit_prompt) / 2
    best = min(pool, key=lambda idx: abs(idx - mid))

    seg1_lead = re.match(r"and\s+", edit_prompt[best:], re.IGNORECASE)
    seg1_start = best + (seg1_lead.end() if seg1_lead else len("and "))
    return best, seg1_start


def compute_instance_ids(
    offset_mapping: torch.Tensor,
    instance_spans: list[tuple[int, int]],
) -> torch.Tensor:
    """[L] instance_ids for one sample's tokenized (offset-mapped) text --
    a fixed property of the tokenization + instance-span split, independent
    of any particular geometric crop's centroids, so this is computed once
    per text variant (see precompute_pe_lora_variants.py) and cached.

    offset_mapping: [L, 2] (char_start, char_end) per token, as returned by
    a fast tokenizer's return_offsets_mapping=True (special/pad tokens
    conventionally report a zero-length span -- explicitly skipped below so
    they never get misclassified as belonging to a span that happens to
    start at char 0).
    instance_spans: one (char_start, char_end) per instance, in the SAME
    string offset_mapping was computed against.

    A token whose offsets fall fully inside instance i's span gets
    instance_ids[token] = i. Every other token (chat-template boilerplate,
    the "and" glue text, pad) gets -1 -- deliberately treated identically,
    since none of them belong to a specific instance and none should be
    blocked from attending to anything by build_cross_instance_attention_mask.
    """
    length = offset_mapping.shape[0]
    instance_ids = torch.full((length,), -1, dtype=torch.int64)

    for tok_idx in range(length):
        start, end = int(offset_mapping[tok_idx, 0]), int(offset_mapping[tok_idx, 1])
        if start == end:
            continue  # special/pad token, no real content
        for inst_idx, (span_start, span_end) in enumerate(instance_spans):
            if start >= span_start and end <= span_end:
                instance_ids[tok_idx] = inst_idx
                break

    return instance_ids


def inject_centroids(
    instance_ids: torch.Tensor,
    centroids: list[tuple[float, float]],
    t_coord: float = 0.0,
) -> torch.Tensor:
    """[L, 4] (T, H, W, L) RoPE ids from a (cached) instance_ids vector and
    this geometric variant's per-instance (h_coord, w_coord) centroids --
    called fresh per geometric variant, since the centroid changes with the
    crop even though instance_ids doesn't. Pure tensor indexing, no
    re-tokenization needed."""
    length = instance_ids.shape[0]
    ids = torch.zeros(length, 4, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, dtype=torch.float32)
    ids[:, 0] = t_coord
    for inst_idx, (h_coord, w_coord) in enumerate(centroids):
        owned = instance_ids == inst_idx
        ids[owned, 1] = h_coord
        ids[owned, 2] = w_coord
    return ids


def build_cross_instance_attention_mask(instance_ids: torch.Tensor, num_img_tokens: int) -> torch.Tensor:
    """Build the [L+S, L+S] boolean attention mask (True = attend, matching
    torch.nn.functional.scaled_dot_product_attention's convention -- see
    dispatch_attention_fn's "native" backend, the diffusers default) for one
    training step. Blocks exactly (i, j) pairs where i and j are both real,
    instance-owned text tokens (instance_ids >= 0) belonging to *different*
    instances. Image tokens, and text tokens belonging to no instance
    (boilerplate/pad/glue), are never blocked from anything.

    Sequence order matches every Flux2 block (double- and single-stream
    alike, confirmed by reading transformer_flux2.py): text tokens first,
    then image tokens (main + cond, already concatenated by the caller).
    """
    num_txt = instance_ids.shape[0]
    total = num_txt + num_img_tokens
    mask = torch.ones(total, total, dtype=torch.bool)

    owned = instance_ids >= 0
    different = instance_ids.unsqueeze(1) != instance_ids.unsqueeze(0)
    block_txt = owned.unsqueeze(1) & owned.unsqueeze(0) & different
    mask[:num_txt, :num_txt] = ~block_txt

    return mask


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
