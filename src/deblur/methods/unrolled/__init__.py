"""Unrolled physics-aware networks (M3). Requires PyTorch."""
from .ista_net import ISTANet
from .lista import LearnedISTA
from .placement import KNAFNet, KUNet, NAFNetConcat, NAFNetDC, NAFNetFiLM, NAFNetUnrolled, UNetConcat, UNetFiLM
from .usrnet import USRNet

MODELS = {
    "usrnet": USRNet, "ista_net": ISTANet, "lista": LearnedISTA,
    # Phase 3 kernel-placement ladder
    "nafnet_concat": NAFNetConcat, "nafnet_film": NAFNetFiLM, "knafnet": KNAFNet,
    "nafnet_dc": NAFNetDC,
    "nafnet_unrolled": NAFNetUnrolled,
    "unet_concat": UNetConcat, "unet_film": UNetFiLM, "kunet": KUNet,
}


def build_model(name: str, **kwargs):
    """Instantiate an M3 network by name."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


__all__ = ["ISTANet", "KNAFNet", "KUNet", "UNetConcat", "UNetFiLM", "NAFNetConcat", "NAFNetDC", "NAFNetFiLM", "NAFNetUnrolled", "LearnedISTA", "MODELS", "USRNet", "build_model"]
