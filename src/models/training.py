import os
import torch
import torch.nn as nn
import torch.optim as optim


# Configuration
PROCESSED_DIR = "data/processed"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# Model Definition
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 54 * 54, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# Utility Function
def calculate_accuracy(outputs, labels):
    preds = (outputs >= 0.5).float()
    correct = (preds == labels).sum()
    return correct / labels.size(0)


# Training Function
def train():
    # Heavy imports moved inside function (important for fast testing)
    import mlflow
    import mlflow.pytorch
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    from sklearn.metrics import confusion_matrix

    # MLflow Setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("cats_vs_dogs_baseline")

    # Data Preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(PROCESSED_DIR, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Dataset loaded and split successfully.")

    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    # Training Loop
    with mlflow.start_run():

        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("epochs", NUM_EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)

        for epoch in range(NUM_EPOCHS):

            # Training
            model.train()
            running_loss = 0.0
            running_acc = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_acc += calculate_accuracy(outputs, labels)

            train_loss = running_loss / len(train_loader)
            train_acc = running_acc / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            val_acc = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.float().unsqueeze(1).to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_acc += calculate_accuracy(outputs, labels)

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        # Log Final Metrics
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_accuracy", val_acc)

        # Save Model
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/baseline_cnn.pt")

        example_input = torch.randn(1, 3, 224, 224).cpu().numpy()

        mlflow.pytorch.log_model(
            model,
            "baseline_cnn",
            input_example=example_input
        )

    # Evaluation - Confusion Matrix
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            preds = (outputs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    cm = confusion_matrix(all_labels, all_preds)

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Cat", "Dog"],
        yticklabels=["Cat", "Dog"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Loss Curve
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("outputs/loss_curve.png")
    plt.close()

    print("Training complete. Model and artifacts saved successfully.")


# Main Entry Point
if __name__ == "__main__":
    train()
