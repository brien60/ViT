import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import Tensor


def evaluate_train_and_val(
        model: Module, 
        clean_train_DL: DataLoader, 
        val_DL: DataLoader,
        num_train_samples: int,
        num_val_samples: int,
        loss_fn: Module,
        device: str,
        epoch: int,
        plot_metrics: dict[str, list]
    ):
    """Computes the training accuracy, validation accuracy, and the validation loss
    on the given model using the clean (no data augmentations) train and validation 
    dataloaders. 

    Args:
        model: Model that the dataloaders will be evaluated on

        clean_train_DL: DataLoader containing training data batches. Does not apply data augmentation transformations applied these batches.

        val_DL: DataLoader containing validation data batches. Similarly does not apply data augmentations.

        num_train_samples: Number of total samples in the training dataset

        num_val_samples: Number of total samples in the validation dataset

        loss_fn: Loss function used to evaluate the model

        device: Device on which the model is stored

        epoch: Current epoch in the training loop

        plot_metrics: Dictionary mapping the name of the metric to a list containing the metric's value for every previous epoch 
    """
    train_correct_preds = 0
    
    val_correct_preds = 0
    val_loss = 0.0

    model.eval()

    # must go here because dropout should be disabled
    with torch.inference_mode():
        for train_batch in clean_train_DL:
            x_train, y_train = train_batch
            x_train, y_train = x_train.to(device), y_train.to(device)

            logits = model(x_train)
            train_correct_preds += correct_preds(logits, y_train)

        for val_batch in val_DL:
            x_val, y_val = val_batch
            x_val, y_val = x_val.to(device), y_val.to(device)

            logits = model(x_val)

            loss = loss_fn(logits, y_val)
            val_loss += loss.item()
            
            val_correct_preds += correct_preds(logits, y_val)

    plot_metrics["val_losses"].append(val_loss / len(val_DL))
    plot_metrics["train_accs"].append(100 * train_correct_preds / num_train_samples)
    plot_metrics["val_accs"].append(100 * val_correct_preds / num_val_samples)
    plot_metrics["val_epochs"].append(epoch+1)



def correct_preds(logits: Tensor, labels: Tensor) -> int:
    """Computes the number of correct predictions in the batch given `logits` and `labels`
    
    Args:
        logits: Tensor of shape (B, 10)
        labels: Tensor of shape B

    Returns:
        An integer representing the number of correct predictions
    """
    return (logits.argmax(dim=1) == labels).sum().item()
    