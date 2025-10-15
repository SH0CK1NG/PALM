import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # frozen base weight
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        if r > 0:
            self.A = nn.Parameter(torch.zeros(self.in_features, r))
            self.B = nn.Parameter(torch.zeros(r, self.out_features))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            x_d = self.dropout(x)
            y = y + (x_d @ self.A @ self.B) * self.scaling
        return y


def apply_lora_to_resnet_head(model: nn.Module, r: int, alpha: int, dropout: float):
    # PALMResNet has self.head as MLP; replace the last Linear with LoRA-wrapped
    if hasattr(model, 'head') and isinstance(model.head, nn.Sequential):
        for i, m in enumerate(model.head):
            if isinstance(m, nn.Linear) and i == len(model.head) - 1:
                model.head[i] = LoRALinear(m, r=r, alpha=alpha, dropout=dropout)
                return True
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head = LoRALinear(model.head, r=r, alpha=alpha, dropout=dropout)
        return True
    return False


def freeze_for_lora(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if m.A is not None:
                m.A.requires_grad = True
            if m.B is not None:
                m.B.requires_grad = True


def extract_lora_state_dict(model: nn.Module) -> dict:
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if module.A is not None:
                sd[f"{name}.A"] = module.A.detach().cpu()
            if module.B is not None:
                sd[f"{name}.B"] = module.B.detach().cpu()
        if isinstance(module, LoRAConv2d):
            if module.A is not None:
                sd[f"{name}.A.weight"] = module.A.weight.detach().cpu()
            if module.B is not None:
                sd[f"{name}.B.weight"] = module.B.weight.detach().cpu()
    return sd


def load_lora_state_dict(model: nn.Module, state_dict: dict):
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.A"
            b_key = f"{name}.B"
            if a_key in state_dict and module.A is not None:
                module.A.data.copy_(state_dict[a_key].to(module.A.device, dtype=module.A.dtype))
                loaded += 1
            if b_key in state_dict and module.B is not None:
                module.B.data.copy_(state_dict[b_key].to(module.B.device, dtype=module.B.dtype))
                loaded += 1
        if isinstance(module, LoRAConv2d):
            a_key = f"{name}.A.weight"
            b_key = f"{name}.B.weight"
            if a_key in state_dict and module.A is not None:
                module.A.weight.data.copy_(state_dict[a_key].to(module.A.weight.device, dtype=module.A.weight.dtype))
                loaded += 1
            if b_key in state_dict and module.B is not None:
                module.B.weight.data.copy_(state_dict[b_key].to(module.B.weight.device, dtype=module.B.weight.dtype))
                loaded += 1
    return loaded


class LoRAConv2d(nn.Module):
    """LoRA for Conv2d using parallel 1x1 low-rank path.
    Output: y = base(x) + scaling * B(A(dropout(x)))
    A: 1x1 conv (in_channels -> r), B: 1x1 conv (r -> out_channels)
    """
    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        if r > 0:
            # A follows base stride to match spatial size; B keeps stride=1
            self.A = nn.Conv2d(base.in_channels, r, kernel_size=1, stride=base.stride, padding=0, bias=False)
            self.B = nn.Conv2d(r, base.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
        else:
            self.A = None
            self.B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            xd = self.dropout(x)
            y = y + self.B(self.A(xd)) * self.scaling
        return y


def apply_lora_to_resnet_layer4(model: nn.Module, r: int, alpha: int, dropout: float) -> bool:
    """Wrap conv layers in encoder.layer4 blocks with LoRAConv2d.
    Returns True if any layer replaced.
    """
    replaced = False
    if not hasattr(model, 'encoder'):
        return False
    layer4 = getattr(model.encoder, 'layer4', None)
    if layer4 is None:
        return False
    for block in layer4:
        # BasicBlock has conv1/conv2; Bottleneck has conv1/conv2/conv3
        for conv_name in ['conv1', 'conv2', 'conv3']:
            if hasattr(block, conv_name):
                conv = getattr(block, conv_name)
                if isinstance(conv, nn.Conv2d):
                    wrapped = LoRAConv2d(conv, r=r, alpha=alpha, dropout=dropout)
                    setattr(block, conv_name, wrapped)
                    replaced = True
    return replaced


