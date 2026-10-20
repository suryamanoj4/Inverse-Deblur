"""Direct data-only networks (M1). Requires PyTorch."""
from .common import pad_to_multiple, restore
from .nafnet import NAFNet
from .unet import UNet

MODELS = {"nafnet": NAFNet, "unet": UNet}


def build_model(name: str, **kwargs):
    """Instantiate an M1 network by name (``nafnet`` or ``unet``)."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


__all__ = ["MODELS", "NAFNet", "UNet", "build_model", "pad_to_multiple", "restore"]
