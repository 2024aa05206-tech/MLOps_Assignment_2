import torch
from src.model import BaselineCNN

def test_model_forward():
    model = BaselineCNN()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 1)
