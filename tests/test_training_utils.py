import torch
from src.models.training import calculate_accuracy


def test_accuracy_perfect():
    outputs = torch.tensor([[0.9], [0.8], [0.1]])
    labels = torch.tensor([[1.0], [1.0], [0.0]])

    acc = calculate_accuracy(outputs, labels)
    assert acc == 1.0


def test_accuracy_zero():
    outputs = torch.tensor([[0.1], [0.2], [0.9]])
    labels = torch.tensor([[1.0], [1.0], [0.0]])

    acc = calculate_accuracy(outputs, labels)
    assert acc == 0.0


def test_accuracy_partial():
    outputs = torch.tensor([[0.9], [0.2], [0.9]])
    labels = torch.tensor([[1.0], [0.0], [0.0]])

    acc = calculate_accuracy(outputs, labels)
    assert 0 < acc < 1
