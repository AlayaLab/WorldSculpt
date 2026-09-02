"""LoRA helpers for Pixal3D denoiser models.

Two responsibilities:
1. Load a base checkpoint into a freshly-built denoiser BEFORE wrapping with
   peft — so the LoRA adapters start from the right pretrained weights.
2. Wrap the model with peft.LoraConfig + get_peft_model, freezing the base
   and making only LoRA params trainable.

The optimizer in BasicTrainer.init_models_and_more then auto-selects only
`requires_grad=True` params, so no further trainer surgery is needed.
"""
from typing import Optional, Dict, Any
import torch


def _load_state_dict_any(path: str, device='cpu'):
    """Load state_dict from .pt or .safetensors."""
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path, device=str(device))
    return torch.load(path, map_location=device, weights_only=True)


def load_base_into(model: torch.nn.Module, ckpt_path: str, strict: bool = False):
    """Load `ckpt_path`'s state_dict into `model` in-place.

    Tolerates missing/unexpected keys (we set strict=False) since the upstream
    image_conditioned_proj.py shape-remap logic isn't needed for same-resolution
    LoRA finetune (both base and ft config are 1024).
    """
    state = _load_state_dict_any(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[LoRA base load] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"[LoRA base load] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    return model


def wrap_with_lora(model: torch.nn.Module, lora_config: Dict[str, Any]) -> torch.nn.Module:
    """Apply LoRA to `model` via peft. Returns the wrapped PEFT model.

    Args:
        model: an instantiated nn.Module (e.g. SparseStructureFlowModel).
        lora_config: dict matching `peft.LoraConfig` constructor args.
                     Typical keys: r, lora_alpha, target_modules, lora_dropout, bias.

    Effect:
        - All non-LoRA params have `requires_grad=False`.
        - Only LoRA A/B matrices are trainable (typically ~0.3-1% of total params).
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(**lora_config)
    peft_model = get_peft_model(model, cfg)
    # Sanity print: trainable param count
    n_total = sum(p.numel() for p in peft_model.parameters())
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[LoRA] target_modules={lora_config.get('target_modules')}")
    print(f"[LoRA] rank={lora_config.get('r')}, alpha={lora_config.get('lora_alpha')}")
    print(f"[LoRA] trainable / total = {n_train:,} / {n_total:,} ({100*n_train/n_total:.3f}%)")
    return peft_model


def prepare_lora_model(
    model: torch.nn.Module,
    base_ckpt_path: Optional[str],
    lora_config: Dict[str, Any],
) -> torch.nn.Module:
    """One-shot: load base ckpt (if given) + wrap with LoRA."""
    if base_ckpt_path:
        load_base_into(model, base_ckpt_path)
    return wrap_with_lora(model, lora_config)
