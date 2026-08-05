import math

def sigmoid(x: float) -> float:
    """AI: Activation Function"""
    return 1 / (1 + math.exp(-x))

def relu(x: float) -> float:
    """AI: ReLU Activation"""
    return max(0.0, x)

def mse_loss(y_true: list, y_pred: list) -> float:
    """ML: Mean Squared Error"""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
