import torch
import torch.nn as nn
import re
import types
from typing import Optional, Tuple, List, Union
import os

try:
    from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
    try:
        from peft import TaskType  # optional in older versions
    except Exception:
        TaskType = None
    PEFT_AVAILABLE = True
except Exception:
    TaskType = None
    PEFT_AVAILABLE = False


def is_peft_available() -> bool:
    return PEFT_AVAILABLE


def _collect_linear_targets(module: nn.Module) -> List[str]:
    targets: List[str] = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            targets.append(name)
    return targets


def _collect_conv_targets(module: nn.Module) -> List[str]:
    targets: List[str] = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            targets.append(name)
    return targets


def _collect_last_linear_target(module: nn.Module) -> List[str]:
    linear_names = _collect_linear_targets(module)
    if not linear_names:
        return []
    # pick the deepest (longest name) as the last linear in head sequential
    linear_names.sort(key=lambda n: (n.count('.'), len(n)))
    return [linear_names[-1]]


def build_lora_cfg(r: int, alpha: int, dropout: float, target_modules: list):
    common_kwargs = dict(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules, bias="none")
    # Prefer declaring task type for CV feature extractors if available in PEFT version
    if 'TaskType' in globals() and TaskType is not None:
        try:
            return LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, **common_kwargs)
        except Exception:
            # Fall back if current PEFT version doesn't accept task_type
            pass
    return LoraConfig(**common_kwargs)


def apply_peft_lora_to_model(model: nn.Module, target: str, r: int, alpha: int, dropout: float) -> Tuple[nn.Module, list]:
    if not PEFT_AVAILABLE:
        return model, []

    target_modules: List[Union[str, re.Pattern]] = []
    name_targets: List[str] = []
    if target in ("head", "both"):
        if hasattr(model, 'head'):
            # only last linear in head; prefix with 'head.' to get fully-qualified name
            local_names = _collect_last_linear_target(model.head)
            name_targets += [f"head.{n}" for n in local_names]
    if target in ("encoder", "both"):
        if hasattr(model, 'encoder'):
            local_convs = [n for n in _collect_conv_targets(model.encoder) if n.startswith('layer4.')]
            name_targets += [f"encoder.{n}" for n in local_convs]
    if target in ("encoder_all", "both_all"):
        if hasattr(model, 'encoder'):
            local_convs = [n for n in _collect_conv_targets(model.encoder) if n.startswith('layer')]
            name_targets += [f"encoder.{n}" for n in local_convs]

    # deduplicate and keep non-empty (use string names for broader PEFT compatibility)
    name_targets = sorted(set([t for t in name_targets if t]))
    target_modules = name_targets
    if len(target_modules) == 0:
        return model, []

    cfg = build_lora_cfg(r=r, alpha=alpha, dropout=dropout, target_modules=target_modules)
    peft_model = get_peft_model(model, cfg)
    # Ensure forward signature is CV-friendly: avoid mapping positional tensor to 'input_ids'
    def _cv_forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    try:
        peft_model.forward = types.MethodType(_cv_forward, peft_model)
    except Exception:
        # Fallback binding
        peft_model.forward = lambda *args, **kwargs: peft_model.base_model(*args, **kwargs)
    # alias commonly used attributes to preserve existing access patterns
    try:
        if hasattr(peft_model, 'base_model'):
            base = peft_model.base_model
            if hasattr(base, 'encoder'):
                setattr(peft_model, 'encoder', getattr(base, 'encoder'))
            if hasattr(base, 'head'):
                setattr(peft_model, 'head', getattr(base, 'head'))
            if hasattr(base, 'fc'):
                setattr(peft_model, 'fc', getattr(base, 'fc'))
    except Exception:
        pass
    peft_model.train()
    return peft_model, target_modules


def save_peft_adapter(model: nn.Module, path: str):
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT is not available; install `peft` package.")
    if isinstance(model, PeftModel):
        model.save_pretrained(path)
    else:
        raise ValueError("Model is not a PeftModel; cannot save PEFT adapter.")


def load_peft_adapter(model: nn.Module, path: str) -> nn.Module:
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT is not available; install `peft` package.")
    # If already peft-wrapped, load adapter weights
    if isinstance(model, PeftModel):
        model.load_adapter(path, adapter_name="default")
        model.set_adapter("default")
        return model
    # Otherwise, attempt to reconstruct from config and wrap
    cfg = PeftConfig.from_pretrained(path)
    peft_model = PeftModel.from_pretrained(model, path)
    # Patch forward for CV models as well when loading from disk
    def _cv_forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    try:
        peft_model.forward = types.MethodType(_cv_forward, peft_model)
    except Exception:
        peft_model.forward = lambda *args, **kwargs: peft_model.base_model(*args, **kwargs)
    peft_model.set_adapter(cfg.base_model_name_or_path if hasattr(cfg, 'base_model_name_or_path') else "default")
    return peft_model


def load_peft_adapters(model: nn.Module, paths: List[str]) -> Tuple[nn.Module, List[str]]:
    """Load multiple PEFT adapters into a PeftModel with unique names (derived from directory basenames).
    Returns the model and the list of adapter names loaded.
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT is not available; install `peft` package.")
    if not isinstance(model, PeftModel):
        raise ValueError("Model is not a PeftModel; call apply_peft_lora_to_model first.")
    loaded_names: List[str] = []
    for p in paths:
        if not p:
            continue
        try:
            base = os.path.basename(os.path.normpath(p))
            # sanitize name to be adapter-safe
            name = base.replace('.', '_').replace('-', '_')
            model.load_adapter(p, adapter_name=name)
            loaded_names.append(name)
            print(f"[peft] loaded adapter '{name}' from {p}")
        except Exception as e:
            print(f"[peft] failed to load adapter from {p}: {e}")
    return model, loaded_names


def add_trainable_peft_adapter(model: nn.Module, adapter_name: str, target_modules: List[str], r: int, alpha: int, dropout: float) -> nn.Module:
    """Add a new trainable adapter with given name and target modules to an existing PeftModel."""
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT is not available; install `peft` package.")
    if not isinstance(model, PeftModel):
        raise ValueError("Model is not a PeftModel; call apply_peft_lora_to_model first.")
    cfg = build_lora_cfg(r=r, alpha=alpha, dropout=dropout, target_modules=target_modules)
    try:
        model.add_adapter(adapter_name, cfg)
        print(f"[peft] added trainable adapter '{adapter_name}'")
    except Exception as e:
        print(f"[peft] failed to add adapter '{adapter_name}': {e}")
    return model


def freeze_peft_adapters(model: nn.Module, adapter_names: List[str]):
    """Freeze LoRA parameters (A/B) for the given adapter names in a PeftModel."""
    if not isinstance(model, PeftModel):
        return
    names_set = set(adapter_names)
    for n, p in model.named_parameters():
        # Typical PEFT param names include substrings like '.lora_A.<name>.' or '.lora_B.<name>.'
        if ('.lora_A.' in n) or ('.lora_B.' in n):
            for an in names_set:
                key = f".{an}."
                if key in n:
                    p.requires_grad = False
                    break


def set_active_peft_adapters(model: nn.Module, adapter_names: List[str]):
    """Set active adapters for forward. Tries to activate a list; falls back to the last one if needed."""
    if not isinstance(model, PeftModel):
        return
    try:
        # Some PEFT versions accept a list to compose adapters
        model.set_adapter(adapter_names if isinstance(adapter_names, (list, tuple)) else adapter_names)
        print(f"[peft] active adapters set to: {adapter_names}")
    except Exception as e:
        print(f"[peft] set_adapter(list) failed: {e}; fallback to last adapter")
        if isinstance(adapter_names, (list, tuple)) and len(adapter_names) > 0:
            try:
                model.set_adapter(adapter_names[-1])
                print(f"[peft] active adapter set to: {adapter_names[-1]}")
            except Exception as ee:
                print(f"[peft] set_adapter fallback failed: {ee}")


