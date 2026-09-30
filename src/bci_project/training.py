"""PyTorch training, validation, and inference utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainingHistory:
    """Store per-epoch training loss and validation accuracy."""

    train_loss: list[float]
    val_accuracy: list[float]


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap NumPy EEG arrays in a PyTorch DataLoader."""

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    batch_size: int = 64,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_size: float = 0.2,
    patience: int = 8,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> TrainingHistory:
    """Optimize a neural network with validation-based early stopping."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be positive")
    if validation_data is None:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=validation_size,
            random_state=seed, stratify=y_train,
        )
    else:
        X_fit, y_fit = X_train, y_train
        X_val, y_val = validation_data

    train_loader = _make_loader(X_fit, y_fit, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    history = TrainingHistory(train_loss=[], val_accuracy=[])

    for _epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Standard optimization step: forward pass, loss, backpropagation, update.
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

            # The EEGNet reproduction exposes the max-norm constraints used by
            # the reference Keras implementation. Other models skip this step.
            if hasattr(model, "apply_max_norm"):
                model.apply_max_norm()

            running_loss += loss.item() * batch_X.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_accuracy = evaluate_accuracy(model, val_loader, device)
        history.train_loss.append(train_loss)
        history.val_accuracy.append(val_accuracy)
        print(f"epoch={_epoch + 1} loss={train_loss:.4f} val_acc={val_accuracy:.4f}", flush=True)

        # Retain the best validation state and stop after sustained stagnation.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy for a DataLoader."""

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_X).argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.numel()

    return correct / max(total, 1)


def predict_model(model: nn.Module, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Generate batched class predictions for an EEG array."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dummy_y = np.zeros(len(X), dtype=np.int64)
    loader = _make_loader(X, dummy_y, batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            predictions.append(model(batch_X).argmax(dim=1).cpu().numpy())

    return np.concatenate(predictions)
