from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.transforms import v2

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
        grad_accum_steps: int = 1
) -> None:
    """Trains the given model on one epoch of the given train dataloader, computing and recording the training loss.

    Can optionally apply MixUp on the training batches and/or use a learning rate scheduler

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
    """

    assert grad_accum_steps >= 1, "grad_accum_steps must be greater than or equal to 1"
    train_loss = 0.0
    batch_group_samples = 0

    optimizer.zero_grad()
    model.train()

    for i, (x_train, y_train) in enumerate(train_DL):
        x_train, y_train = x_train.to(device), y_train.to(device)

        # original_images = x_train.clone()
        # original_labels = y_train.clone()
        if mixup is not None:
            x_train, y_train = mixup(x_train, y_train)

        # plt.figure(figsize = (4,   16))

        # for i in range(10):
        #     idx = int(batch_size * torch.rand(1))

        #     image_1 = original_images[idx].cpu()
        #     image_2 = original_images[(idx-1) % batch_size].cpu()

        #     label_1 = original_labels[idx].cpu()
        #     label_2 = original_labels[(idx-1) % batch_size].cpu()

        #     mixed_img = x_train[idx].cpu()

        #     mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        #     std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

        #     lambda_1 = y_train[idx][label_1].item()
        #     lambda_2 = y_train[idx][label_2].item()

        #     for j, info in enumerate(zip(
        #         [image_1, image_2, mixed_img],
        #         [classes[label_1], classes[label_2], f"({lambda_1:.2f}, {lambda_2:.2f})"]
        #     )):
        #         img, title = info
        #         img = img * std + mean

        #         plt.subplot(10, 3, 3*i+(j+1))
        #         plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
        #         plt.axis("off")
        #         plt.title(title)

        # plt.show()

        # sys.exit(0)
        
        logits = model(x_train)

        batch_samples = len(x_train)

        loss = loss_fn(logits, y_train) * batch_samples
        train_loss += loss.detach()

        loss.backward()

        batch_group_samples += batch_samples

        if (i+1) % grad_accum_steps == 0 or i+1 == len(train_DL):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(batch_group_samples)

            optimizer.step()
            optimizer.zero_grad()
            batch_group_samples = 0

        pbar.update(len(x_train))   

    pbar.close()

    if scheduler is not None:
        scheduler.step()

    plot_metrics["train_losses"].append(train_loss.item() / len(train_DL.dataset))
    plot_metrics["train_epochs"].append(len(plot_metrics["train_epochs"]) + 1)