import torch
from sklearn.metrics import f1_score, accuracy_score

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from .model import MLP
from ..dataset.dataset import get_mnist_loaders
import os
from tqdm import tqdm
import numpy as np

import mlflow

MLFLOW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mlruns"))
BASE_DIR = os.path.dirname(os.path.dirname("/mnt/c/Users/Марсель/Documents/co/PMLDL_A1/dataset"))

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

mlflow.set_tracking_uri(MLFLOW_DIR)

torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, test_loader = get_mnist_loaders(DATA_DIR, batch_size=64)

print(device)

def train(
    model,
    loss_fn,
    optim,
    train_loader,
    val_loader,
    device,
    epochs,
    checkpoint=None
):
    model.to(device)

    best_acc = 0

    try:
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
            with open(MODEL_DIR + "/metrics.txt", "r") as f:
                best_acc = float(f.read().split()[1])
            print(f"[Checkpoint] Loading checkpoint with accuracy {best_acc:.4f}")
    except:
        print(MODEL_DIR + "/metrics.txt")
        print("[Checkpoint] No checkpoint or metrics found. Skipping")

    with mlflow.start_run():

        mlflow.log_params({
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": type(optim).__name__,
            "loss": type(loss_fn).__name__
        })

        for epoch in range(1, epochs + 1):
            model.train()
            train_preds = []
            train_labels = []
            train_loss = 0.0

            for image, label in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}"):
                image, label = image.to(device), label.to(device)

                optim.zero_grad()
                out = model(image)
                loss = loss_fn(out, label)
                loss.backward()
                optim.step()

                train_loss += loss.item()

                preds = torch.argmax(out, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(label.cpu().numpy())

            train_accuracy = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_f1": train_f1
            }, step=epoch)

            model.eval()
            val_preds = []
            val_labels = []
            val_loss = 0.0

            with torch.inference_mode():
                for image, label in tqdm(val_loader, desc=f"[Validation] Epoch {epoch}/{epochs}"):
                    image, label = image.to(device), label.to(device)
                    out = model(image)
                    loss = loss_fn(out, label)
                    val_loss += loss.item()

                    preds = torch.argmax(out, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(label.cpu().numpy())

            val_accuracy = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

            mlflow.log_metrics({
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1
            }, step=epoch)

            if val_accuracy > best_acc:
                print("[Checkpoint] Saving new best model")
                best_acc = val_accuracy
                torch.save(model.state_dict(), MODEL_DIR + "/model.pth")
                with open(MODEL_DIR + "/metrics.txt", "w") as f:
                    f.write(f"Accuracy: {best_acc}")

            print(
                f"[Metrics] Epoch {epoch}/{epochs}: "
                f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}"
            )
        
        example_inputs = (torch.randn(1, 1, 28, 28).to(device))
        torch.onnx.export(model, example_inputs, MODEL_DIR + "/model.onnx")

model = MLP()
optimizer = Adam(model.parameters())
loss_function = CrossEntropyLoss()

train(
    model=model,
    loss_fn=loss_function,
    optim=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=5,
    checkpoint=MODEL_DIR + "/model.pth"
)
