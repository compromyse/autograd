from .scalar import Scalar

def mse(ys, preds) -> Scalar:
    losses = [ (y - pred) ** 2 for y, pred in zip(ys, preds) ]
    return sum(losses)
