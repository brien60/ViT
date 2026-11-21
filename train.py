from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.profiler import profile

import torch
import time

from tqdm import tqdm


# TODO: enclose optimizer, loss_fn, device, scheduler, mixup, and grad_accum in TrainingConfig
def train_model(
        model: Module,
        train_DL: DataLoader,
        optimizer: Optimizer,
        loss_fn: Module,
        device: str,
        pbar: tqdm,
        plot_metrics: dict[str, list],
        scheduler: LRScheduler | None = None,
        mixup: v2.Transform | None = None,
        grad_accum_steps: int = 1,
        autocast: bool = True
) -> None:
    """Trains the given model on one epoch of the given train dataloader, computing and recording the training loss.

    Can optionally apply MixUp on the training batches and/or use a learning rate scheduler.

    Args:
        model: Model to train on

        train_DL: DataLoader containing the (possibly augmented) training batches

        optimizer: Optimizer to use to update the model's parameters

        loss_fn: Loss function used to evaluate the model

        device: Device on which the model is stored

        pbar: Tqdm progress bar to update

        plot_metrics: Dictionary mapping the name of the metric to a list containing the metric's value for every previous epoch 

        scheduler: Learning rate scheduler to use

        mixup: Mixup transformation to apply

        grad_accum_steps: Specifies the number of gradient accumulation steps to be performed with a micro-batch size of `train_DL.batch_size`

        autocast: Whether to use bf16 autocast mixed precision
    """

    assert grad_accum_steps >= 1, "grad_accum_steps must be greater than or equal to 1"
    train_loss = 0.0
    batch_group_samples = 0
    window_samples = 0

    optimizer.zero_grad()
    model.train()

    # GPU-only benchmark data
    # dummy_x_train = torch.rand(
    #     256, 3, 224, 224,
    #     device=device,
    #     dtype=torch.float32,
    # )

    # dummy_y_train = torch.randint(
    #     low=0,
    #     high=1000,
    #     size=(256,),
    #     device=device,
    #     dtype=torch.long,
    # )

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # train_iter = iter(train_DL)
    # access_times_all = []
    # rgb_times_all = []
    # transform_times_all = []

    if device == "cuda": torch.cuda.synchronize()
    start = time.perf_counter()

    for i, (x_train, y_train) in enumerate(train_DL):
    # for i in range(len(train_DL)):

        # x_train, y_train, access_times, rgb_times, transform_times = next(train_iter)

        # access_times_all.extend(access_times.tolist())
        # rgb_times_all.extend(rgb_times.tolist())
        # transform_times_all.extend(transform_times.tolist())


        x_train = x_train.to(device, non_blocking=True)
        y_train = y_train.to(device, non_blocking=True)

        x_train = x_train.float().div_(255.0)
        x_train.sub_(mean).div_(std)
        # x_train, y_train = dummy_x_train.to(device, non_blocking=True), dummy_y_train.to(device, non_blocking=True)

        # x_train = x_train.contiguous(
        #     memory_format=torch.channels_last
        # )
        
        if mixup is not None:
            x_train, y_train = mixup(x_train, y_train)  
        batch_samples = len(x_train)

        with torch.autocast(device, torch.bfloat16, enabled=autocast):
            logits = model(x_train)
            loss = loss_fn(logits, y_train) * batch_samples

        train_loss += loss.detach()

        loss.backward()

        batch_group_samples += batch_samples
        window_samples += batch_samples

        if (i+1) % grad_accum_steps == 0 or i+1 == len(train_DL):
            # torch.cuda.synchronize()
            # start = time.perf_counter()

            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(batch_group_samples)

            # torch.cuda.synchronize()
            # print(f"Gradient normalization: {(time.perf_counter() - start) * 1000:.2f} ms")


            # torch.cuda.synchronize()
            # start = time.perf_counter()

            optimizer.step()

            # torch.cuda.synchronize()
            # step_time = time.perf_counter() - start

            optimizer.zero_grad()

            # print(f"Optimizer step: {step_time * 1000:.2f} ms")
            
            batch_group_samples = 0

        if (i+1) % 512 == 0:
            if device == "cuda": torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            print(f"{window_samples / elapsed:.0f} images/s\n")
            window_samples = 0

            start = time.perf_counter()

        pbar.update(len(x_train))

    pbar.close()

    if scheduler is not None:
        scheduler.step()

    plot_metrics["train_losses"].append(train_loss.item() / len(train_DL.dataset))
    plot_metrics["train_epochs"].append(len(plot_metrics["train_epochs"]) + 1)

    # import numpy as np


    # access_times_all_ms = np.array(access_times_all) * 1000

    # print("acesss times\n")
    # print(f"Mean: {access_times_all_ms.mean():.2f} ms")
    # print(f"Median: {np.median(access_times_all_ms):.2f} ms")
    # print(f"P95: {np.percentile(access_times_all_ms, 95):.2f} ms")
    # print(f"P99: {np.percentile(access_times_all_ms, 99):.2f} ms")
    # print(f"Max: {access_times_all_ms.max():.2f} ms")
    # print(f">10 ms: {(access_times_all_ms > 10).sum()}")
    # print(f">100 ms: {(access_times_all_ms > 100).sum()}")

    # rgb_times_all_ms = np.array(rgb_times_all) * 1000
    
    # print("rgb times\n")
    # print(f"Mean: {rgb_times_all_ms.mean():.2f} ms")
    # print(f"Median: {np.median(rgb_times_all_ms):.2f} ms")
    # print(f"P95: {np.percentile(rgb_times_all_ms, 95):.2f} ms")
    # print(f"P99: {np.percentile(rgb_times_all_ms, 99):.2f} ms")
    # print(f"Max: {rgb_times_all_ms.max():.2f} ms")
    # print(f">10 ms: {(rgb_times_all_ms > 10).sum()}")
    # print(f">100 ms: {(rgb_times_all_ms > 100).sum()}")

    # transform_times_all_ms = np.array(transform_times_all) * 1000
        
    # print("transform times\n")
    # print(f"Mean: {transform_times_all_ms.mean():.2f} ms")
    # print(f"Median: {np.median(transform_times_all_ms):.2f} ms")
    # print(f"P95: {np.percentile(transform_times_all_ms, 95):.2f} ms")
    # print(f"P99: {np.percentile(transform_times_all_ms, 99):.2f} ms")
    # print(f"Max: {transform_times_all_ms.max():.2f} ms")
    # print(f">10 ms: {(transform_times_all_ms > 10).sum()}")
    # print(f">100 ms: {(transform_times_all_ms > 100).sum()}")