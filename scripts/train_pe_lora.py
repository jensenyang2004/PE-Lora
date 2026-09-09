#!/usr/bin/env python3
"""
PE-injection LoRA training loop for Flux2 Klein ([[project_pe_lora_goal]]).

Loads ONLY the Flux2 transformer (bf16) + LoRA adapters + the (tiny)
flow-matching noise scheduler -- no VAE, no Qwen3 text encoder, no full
Flux2KleinPipeline resident. All per-sample data (VAE latents, Qwen3 prompt
embeddings, and the RoPE ids with the target-instance centroid already baked
into the text tokens' H,W) comes from a cache built by
scripts/precompute_pe_lora_variants.py -- see that script's docstring and
the project plan for why encoding is done offline rather than online.

Physical batch_size is hard-pinned to 1 (not just defaulted): Flux2Transformer2DModel.forward
squeezes img_ids/txt_ids to batch index 0 whenever they arrive with a batch
dimension (`ids = ids[0]` in transformer_flux2.py), silently reusing ONE
sample's ids for a whole batch -- since every sample here needs its own
injected centroid, a physical batch > 1 would silently corrupt training.
Use --gradient_accumulation_steps for a larger effective batch, and
`accelerate launch --num_processes N --multi_gpu` for data parallelism
(each replica still physical batch_size=1).

LoRA scope ("QK only", see project plan for the full rationale):
  Stage 1 (default): double-stream Flux2Attention only -- to_q, to_k
  (image-side) and add_q_proj, add_k_proj (text-side, RoPE is applied to
  the *concatenated* text+image Q/K so the text-side projections are just
  as much "the QK layer"). Excludes to_v/add_v_proj/to_out/to_add_out.
  Stage 2 (--enable-single-stream-lora): also adapts the single-stream
  blocks' fused to_qkv_mlp_proj (Q/K/V/MLP-in packed into one Linear --
  PEFT can't target a slice of a fused Linear natively), then zeros the
  LoRA-B rows corresponding to V and MLP after every optimizer step so only
  the Q/K rows actually train. Verify with:
    assert lora_B.weight[2*inner_dim:].abs().max() == 0
  after a few steps before trusting a real run.

Run (single GPU, smoke test):
  python scripts/train_pe_lora.py \
      --pretrained_model_name_or_path <repo id> \
      --cache_dir ./data/pe_lora_cache --output_dir ./out/pe_lora_run \
      --max_train_steps 5 --gradient_checkpointing

Run (2x GPU DDP):
  accelerate launch --num_processes 2 --multi_gpu scripts/train_pe_lora.py \
      --pretrained_model_name_or_path <repo id> \
      --cache_dir ./data/pe_lora_cache --output_dir ./out/pe_lora_run \
      --gradient_accumulation_steps 8 --gradient_checkpointing --mixed_precision bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)

logger = get_logger(__name__)


class PELoraVariantBankDataset(Dataset):
    """Reads scripts/precompute_pe_lora_variants.py's on-disk cache. Every
    __getitem__ call re-rolls which of a sample's K geometric variants to
    use, so re-augmentation happens naturally across epochs -- there is
    nothing cached-once here to go stale, unlike the base dreambooth
    script's __init__-time-fixed crop/flip."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.index = [json.loads(line) for line in (cache_dir / "index.jsonl").open()]
        if not self.index:
            raise ValueError(f"No entries in {cache_dir / 'index.jsonl'} -- did precompute finish?")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        row = self.index[i]
        k = random.randrange(row["num_geo_variants"])
        sample_dir = self.cache_dir / row["id"]
        geo = torch.load(sample_dir / f"geo_{k}.pt", weights_only=True)
        text = torch.load(sample_dir / f"text_{geo['text_variant_idx']}.pt", weights_only=True)
        return {
            "model_input": geo["model_input"],
            "cond_model_input": geo["cond_model_input"],
            "text_ids": geo["text_ids"],
            "img_ids": geo["img_ids"],
            "prompt_embeds": text["prompt_embeds"],
        }


def collate_fn(examples: list[dict]) -> dict:
    if len(examples) != 1:
        raise ValueError(
            f"Physical batch_size must be 1 (got {len(examples)}) -- see module docstring: "
            "Flux2Transformer2DModel.forward silently reuses one sample's RoPE ids for the "
            "whole batch, and every sample here needs its own injected centroid."
        )
    ex = examples[0]
    return {
        "model_input": ex["model_input"].unsqueeze(0),
        "cond_model_input": ex["cond_model_input"].unsqueeze(0),
        "text_ids": ex["text_ids"],
        "img_ids": ex["img_ids"],
        "prompt_embeds": ex["prompt_embeds"].unsqueeze(0),
    }


def zero_single_stream_non_qk_rows(transformer: torch.nn.Module, inner_dim: int) -> None:
    """Stage-2 masking: after every optimizer.step(), zero the lora_B rows
    of every single-stream to_qkv_mlp_proj LoRA adapter that correspond to
    V and MLP-in (packed layout [Q | K | V | MLP], each of the first two
    `inner_dim` wide) so only Q/K actually receive a trained update. PEFT
    already zero-inits lora_B, so this only maintains that zero state. Pure
    post-step data manipulation (no autograd hook) -- trivially DDP-safe,
    each rank operates on its own already-synced replica."""
    with torch.no_grad():
        for name, module in transformer.named_modules():
            if not name.endswith("to_qkv_mlp_proj") or not hasattr(module, "lora_B"):
                continue
            for lora_b in module.lora_B.values():
                lora_b.weight[2 * inner_dim :].zero_()


def get_sigmas(noise_scheduler_copy, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=Path, required=True, help="output of precompute_pe_lora_variants.py")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--enable-single-stream-lora",
        action="store_true",
        help="Stage 2: also adapt single-stream to_qkv_mlp_proj (Q/K rows only, via post-step masking). "
        "Leave off until Stage 1 (double-stream only) is verified working end-to-end.",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--train_batch_size", type=int, default=1, help="must be 1 -- see module docstring")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weighting_scheme", type=str, default="none", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")
    args = parser.parse_args()

    if args.train_batch_size != 1:
        parser.error("--train_batch_size must be 1 -- see module docstring for why this is enforced, not a default.")
    return args


def main() -> None:
    args = parse_args()

    logging_dir = args.output_dir / args.logging_dir
    accelerator_project_config = ProjectConfiguration(project_dir=str(args.output_dir), logging_dir=str(logging_dir))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[accelerator.mixed_precision]

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
    )
    import copy

    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, torch_dtype=weight_dtype
    )
    transformer.requires_grad_(False)
    transformer.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    target_modules = ["to_q", "to_k", "add_q_proj", "add_k_proj"]
    if args.enable_single_stream_lora:
        target_modules += ["to_qkv_mlp_proj"]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(lora_config)
    inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    if args.enable_single_stream_lora:
        zero_single_stream_non_qk_rows(transformer, inner_dim)

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    def unwrap_model(model):
        return accelerator.unwrap_model(model)

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers = get_peft_model_state_dict(unwrap_model(models[0]))
            if weights:
                weights.pop()
            Flux2KleinPipeline.save_lora_weights(output_dir, transformer_lora_layers=transformer_lora_layers)

    def load_model_hook(models, input_dir):
        # Matches save_model_hook: accelerator.save_state normally checkpoints a full model
        # state dict, but save_model_hook pops that and writes LoRA-only weights instead, so
        # the load side has to mirror it -- accelerator.load_state() would otherwise look for a
        # full-model file that was never written.
        model = models.pop()
        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            k.replace("transformer.", ""): v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        incompatible_keys = set_peft_model_state_dict(model, transformer_state_dict, adapter_name="default")
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logger.warning(f"Loading adapter weights led to unexpected keys: {unexpected_keys}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    train_dataset = PELoraVariantBankDataset(args.cache_dir)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    lora_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("pe-lora-flux2-klein", config=vars(args) | {"output_dir": str(args.output_dir), "cache_dir": str(args.cache_dir)})

    total_batch_size = accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num samples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Physical batch size per device = 1 (enforced)")
    logger.info(f"  Effective batch size (parallel x accumulation) = {total_batch_size}")
    logger.info(f"  LoRA target_modules = {target_modules}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            dirs = sorted(
                (d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")),
                key=lambda x: int(x.split("-")[1]),
            )
            path = dirs[-1] if dirs else None
        if path is not None:
            accelerator.load_state(os.path.join(args.output_dir, os.path.basename(str(path))))
            global_step = int(os.path.basename(str(path)).split("-")[1])

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        transformer.train()
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                model_input = batch["model_input"].to(dtype=weight_dtype)
                cond_model_input = batch["cond_model_input"].to(dtype=weight_dtype)
                text_ids = batch["text_ids"].to(device=accelerator.device)
                img_ids = batch["img_ids"].to(device=accelerator.device)
                prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype)

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                sigmas = get_sigmas(noise_scheduler_copy, timesteps, model_input.device, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
                packed_cond_model_input = Flux2KleinPipeline._pack_latents(cond_model_input)
                orig_main_len = packed_noisy_model_input.shape[1]

                packed_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=1)

                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.full([1], args.guidance_scale, device=accelerator.device).expand(bsz)
                else:
                    guidance = None

                model_pred = transformer(
                    hidden_states=packed_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                model_pred = model_pred[:, :orig_main_len, :]
                main_img_ids = img_ids[:orig_main_len].unsqueeze(0)
                model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, main_img_ids)

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)

                optimizer.step()
                if args.enable_single_stream_lora:
                    zero_single_stream_non_qk_rows(unwrap_model(transformer), inner_dim)
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            (d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")),
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for old in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(args.output_dir / old)
                    save_path = args.output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_path))
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        Flux2KleinPipeline.save_lora_weights(
            save_directory=str(args.output_dir), transformer_lora_layers=transformer_lora_layers
        )
        logger.info(f"Final LoRA weights saved to {args.output_dir}")

    accelerator.end_training()
    free_memory()


if __name__ == "__main__":
    main()
