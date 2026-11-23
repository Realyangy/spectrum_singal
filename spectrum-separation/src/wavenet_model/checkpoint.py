"""Load trusted RF Challenge checkpoints with their original class paths."""
import sys
import types

import torch

from . import config_torchwavenet


def load_legacy_checkpoint(path):
    names = ("src", "src.config_torchwavenet")
    previous = {name: sys.modules.get(name) for name in names}
    package = types.ModuleType("src")
    package.__path__ = []
    package.config_torchwavenet = config_torchwavenet
    sys.modules["src"] = package
    sys.modules["src.config_torchwavenet"] = config_torchwavenet
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
