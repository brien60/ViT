import torch

from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from torch import Tensor


def evaluate_train_and_val(
        model: Module, 
        clean_train_DL: DataLoader, 
        val_DL: DataLoader,
        num_train_samples: int,
        num_val_samples: int,
        loss_fn: Module,
        device: str,
        plot_metrics: dict[str, list],
        train_eval_samples: int = None,
    ) -> None:
    """Computes the training accuracy, validation accuracy, and the validation loss on the given model using the clean 

    (no data augmentations) train and validation dataloaders. 
    
    Can optionally pass in `train_eval_samples` to set the number of training samples to compute accuracy on.

    Args:
        model: Model that the dataloaders will be evaluated on

        clean_train_DL: DataLoader containing training data batches. Does not apply data augmentation transformations applied these batches.

        val_DL: DataLoader containing validation data batches. Similarly does not apply data augmentations.

        num_train_samples: Number of total samples in the training dataset

        num_val_samples: Number of total samples in the validation dataset

        loss_fn: Loss function used to evaluate the model

        device: Device on which the model is stored

        plot_metrics: Dictionary mapping the name of the metric to a list containing the metric's value for every previous epoch 

        train_eval_samples: Number of training samples to evaluate on
    """
    train_eval_DL = clean_train_DL

    if train_eval_samples is not None:
        indices = torch.randperm(num_train_samples)[:train_eval_samples].tolist()

        train_eval_subset = Subset(clean_train_DL.dataset, indices)

        train_eval_DL = DataLoader(
            train_eval_subset,
            batch_size=clean_train_DL.batch_size,
            shuffle=False,
            num_workers=clean_train_DL.num_workers,
            pin_memory=clean_train_DL.pin_memory
        )

    train_correct_preds = 0
    
    val_correct_preds = 0
    val_loss = 0.0

    model.eval()

    # must go here because dropout should be disabled
    with torch.inference_mode():
        for train_batch in train_eval_DL:
            x_train, y_train = train_batch
            x_train, y_train = x_train.to(device), y_train.to(device)

            logits = model(x_train)
            train_correct_preds += correct_preds(logits, y_train)

        for val_batch in val_DL:
            x_val, y_val = val_batch
            x_val, y_val = x_val.to(device), y_val.to(device)

            logits = model(x_val)

            loss = loss_fn(logits, y_val)
            val_loss += loss.item() * len(x_val)
            
            val_correct_preds += correct_preds(logits, y_val)

    plot_metrics["val_losses"].append(val_loss / len(val_DL.dataset))
    plot_metrics["train_accs"].append(100 * train_correct_preds / len(train_eval_DL.dataset))
    plot_metrics["val_accs"].append(100 * val_correct_preds / num_val_samples)
    plot_metrics["val_epochs"].append(len(plot_metrics["val_epochs"]) + 1)



def test_model(
        model: Module, 
        test_DL: DataLoader, 
        loss_fn: Module,
        device: str,
    ) -> None:
    """Computes and prints the test accuracy and test loss on the given model using the clean (no data augmentations) test dataloader.

    Args:
        model: Model that the dataloaders will be evaluated on

        test_DL: DataLoader containing test data batches. Does not apply data augmentation transformations applied these batches.

        loss_fn: Loss function used to evaluate the model

        device: Device on which the model is stored
    """

    test_correct_preds = 0
    test_loss = 0.0

    model.eval()
    
    with torch.inference_mode():
        for test_batch in test_DL:
            x_test, y_test = test_batch
            x_test, y_test = x_test.to(device), y_test.to(device)

            logits = model(x_test)
            loss = loss_fn(logits, y_test)
            test_loss += loss.item() * len(x_test)
            
            test_correct_preds += correct_preds(logits, y_test)

    print(test_correct_preds / len(test_DL.dataset))
    print(test_loss / len(test_DL.dataset))



def correct_preds(logits: Tensor, labels: Tensor) -> int:
    """Computes the number of correct predictions in the batch given `logits` and `labels`
    
    Args:
        logits: Tensor of shape (B, 10)
        labels: Tensor of shape B

    Returns:
        An integer representing the number of correct predictions
    """
    return (logits.argmax(dim=1) == labels).sum().item()

    