"""Unrolled physics-aware networks (M3). Requires PyTorch."""
from .ista_net import ISTANet
from .lista import LearnedISTA
from .usrnet import USRNet

MODELS = {"usrnet": USRNet, "ista_net": ISTANet, "lista": LearnedISTA}


def build_model(name: str, **kwargs):
    """Instantiate an M3 network by name."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


__all__ = ["ISTANet", "LearnedISTA", "MODELS", "USRNet", "build_model"]
