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
        model.train()
        if tpu:
            para_loader = pl.ParallelLoader(data_loader, [device])
            tk0 = tqdm(para_loader.per_device_loader(device), total=len(data_loader))
        else:    
            tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, data in enumerate(tk0):
            for k,v in data.items():
                data[k] = v.to(device)
        
        _, loss = model(**data)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        