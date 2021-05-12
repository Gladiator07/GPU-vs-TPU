import torch
from tqdm import tqdm

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

except ImportError:
    _xla_available = False


class Engine:
    def train(
        data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        tpu=False
    ):
        