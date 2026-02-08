import torch
from src.data.preprocess import normalize_image

def test_normalize_image():
    img = torch.rand(3, 224, 224)
    out = normalize_image(img)

    assert out.shape == img.shape
    assert torch.max(out) <= 1.0
    assert torch.min(out) >= 0.0
