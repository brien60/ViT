from torch import Tensor

def correct_preds(logits: Tensor, labels: Tensor) -> int:
    """Computes the number of correct predictions given `logits` and `labels`
    
    Args:
        logits: Tensor of shape (B, 10)
        labels: Tensor of shape B

    Returns:
        An integer representing the number of correct predictions
    """
    return (logits.argmax(dim=1) == labels).sum()
    