import torch
import pytest
from src.models.model_file import BaselineCNN
import torch.nn as nn


def test_model_instance():
    model = BaselineCNN()
    assert isinstance(model, torch.nn.Module)


def test_model_output_shape_single():
    model = BaselineCNN()
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    assert output.shape == (1, 1)


def test_model_output_shape_batch():
    model = BaselineCNN()
    dummy = torch.randn(8, 3, 224, 224)
    output = model(dummy)
    assert output.shape == (8, 1)


def test_output_range_sigmoid():
    model = BaselineCNN()
    dummy = torch.randn(4, 3, 224, 224)
    output = model(dummy)
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)


def test_model_parameters_not_empty():
    model = BaselineCNN()
    params = list(model.parameters())
    assert len(params) > 0


def test_backward_pass():
    model = BaselineCNN()
    dummy = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    loss = output.mean()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None

def test_model_has_conv_layers():
    model = BaselineCNN()
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert len(conv_layers) >= 2


def test_model_has_sigmoid():
    model = BaselineCNN()
    sigmoid_layers = [m for m in model.modules() if isinstance(m, nn.Sigmoid)]
    assert len(sigmoid_layers) == 1
