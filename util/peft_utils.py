import torch
import torch.nn as nn
import re
import types
from typing import Optional, Tuple, List, Union

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


