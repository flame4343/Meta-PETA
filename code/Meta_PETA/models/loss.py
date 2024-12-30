import torch
import torch.nn as nn


def SquareError(predictions, targets):
    """
    Compute the squared error between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Squared error for each prediction.
    """
    squared_error = torch.square(predictions - targets)
    return squared_error


def AbsolError(predictions, targets):
    """
    Compute the absolute error between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Absolute error for each prediction.
    """
    absolute_error = torch.abs(predictions - targets)
    return absolute_error


def AbsolPercentageError(predictions, targets):
    """
    Compute the absolute percentage error between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Absolute percentage error for each prediction.
    """
    # Ensure targets are absolute to avoid division by zero or negative percentages
    targets_abs = torch.abs(targets)
    absolute_error = torch.abs(predictions - targets)
    percentage_error = absolute_error / targets_abs

    return percentage_error
