import torch.nn as nn
import os
import mlflow.pytorch
import torch.optim as optim
import torch # Keep existing imports for context
from torchvision import datasets, transforms # Add transforms import
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = "data/processed" # Define PROCESSED_DIR in this cell

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(PROCESSED_DIR, transform=transform)

try:
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    print("Dataset loaded and split successfully.")
except ValueError as e:
    print(f"Error splitting dataset: {e}. This might happen if the dataset is empty or too small.")
    train_loader = None
    val_loader = None


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 54 * 54, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# Moved definitions from oylX1pe7oZtM to ensure they are defined before use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineCNN().to(device);

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

def calculate_accuracy(outputs, labels):
    preds = (outputs >= 0.5).float()
    correct = (preds == labels).sum()
    return correct / labels.size(0)

os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), "models/baseline_cnn_initial.pt") # Renamed to avoid confusion with final model save


mlflow.set_experiment("cats_vs_dogs_baseline")

with mlflow.start_run(): # Moved mlflow.start_run() to encompass the training loop
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", num_epochs) # Log num_epochs correctly
    mlflow.log_param("batch_size", 32)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
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

        # ---------- Validation ----------
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
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # All these lines should be correctly indented inside the with mlflow.start_run(): block
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_cnn.pt") # Saving the final model

    # Log final metrics after the training loop completes
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)


    mlflow.pytorch.log_model(model, "baseline_cnn")

    # Make sure 'outputs' directory exists if logging artifacts
    os.makedirs('outputs', exist_ok=True)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        preds = (outputs >= 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


#Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

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

os.makedirs("outputs", exist_ok=True)
cm_path = "outputs/confusion_matrix.png"
plt.savefig(cm_path)
plt.close()
#Loss Curve

plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

loss_curve_path = "outputs/loss_curve.png"
plt.savefig(loss_curve_path)
plt.close()