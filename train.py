import argparse
import json
from io import BytesIO

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from interruptions import (EC2Interruption, detect_ec2_interruption,
                           is_spot_instance_terminating,
                           remove_interrupted_file, touch_empty_file)


class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0
        return False


class Trainer:
    def __init__(self, model, train_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in self.train_loader:
            if is_spot_instance_terminating():
                raise EC2Interruption()

            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)

        return total_loss / total_samples

    def validate(self, val_dataset):
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        total_loss = 0.0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                total_samples += y_batch.size(0)

        return total_loss / total_samples


class ModelHandler:
    @staticmethod
    def save_model(model, filename="model.pt"):
        torch.save(model.state_dict(), filename)

    @staticmethod
    def load_model(model):
        with dvc.api.open("model.pt", mode="rb") as model_file:
            model_buffer = BytesIO(model_file.read())

        state_dict = torch.load(model_buffer)
        model.load_state_dict(state_dict)


def evaluate(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    total_correct = 0
    total_samples = 0

    all_y_true = []
    all_y_pred = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            scores = model(x_batch)
            _, labels = torch.max(scores, 1)
            total_correct += (labels == y_batch).sum().item()
            total_samples += y_batch.size(0)

            all_y_true.extend(y_batch.numpy())
            all_y_pred.extend(labels.numpy())

    accuracy = total_correct / total_samples
    return {"acc": accuracy}, all_y_true, all_y_pred


def save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig(filename)


def save_metrics(metrics):
    with open("metrics.json", "w") as outfile:
        outfile.write(json.dumps(metrics))


def save_loss_plot(train_losses, val_losses, filename="loss.png"):
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def update_params_yaml(starting_epoch):
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    params["starting_epoch"] = starting_epoch

    with open("params.yaml", "w") as file:
        yaml.safe_dump(params, file)


def load_data(features_file, labels_file):
    # Read in data
    X = np.genfromtxt(features_file)
    y = np.genfromtxt(labels_file)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor


def main(args):
    torch.manual_seed(0)

    training_interrupted = False

    x_train, y_train = load_data("data/train_features.csv", "data/train_labels.csv")
    x_test, y_test = load_data("data/test_features.csv", "data/test_labels.csv")

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    train_dataset = TabularDataset(x_train, y_train)
    val_dataset = TabularDataset(x_val, y_val)
    test_dataset = TabularDataset(x_test, y_test)

    input_size = x_train.shape[1]  # Number of features
    hidden_size = 16
    output_size = len(y_train.unique())

    model = FeedForwardNN(input_size, hidden_size, output_size)

    if args.starting_epoch > 1:
        print(
            f"Resuming training. Downloading checkpoint from epoch {args.starting_epoch}"
        )
        ModelHandler.load_model(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print(
        f"Training for {args.epochs} epochs. Starting from epoch {args.starting_epoch}."
    )

    # Instantiate EarlyStopper with the desired patience and min_delta
    early_stopper = EarlyStopper(patience=1, min_delta=0)

    trainer = Trainer(model, train_loader, optimizer, criterion, device)

    train_losses = []
    val_losses = []

    for epoch in range(args.starting_epoch, args.epochs + 1):
        try:
            train_loss = trainer.train()
            train_losses.append(train_loss)

            val_loss = trainer.validate(val_dataset)
            val_losses.append(val_loss)
            print(
                f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
            )

            if early_stopper.early_stop(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        except EC2Interruption:
            print(f"EC2 interrupted, saving model at epoch {epoch}")
            training_interrupted = True
            touch_empty_file("interrupted")
            break

    if not training_interrupted:
        remove_interrupted_file()

    ModelHandler.save_model(model)
    metrics, y_true, y_pred = evaluate(model, test_dataset)
    update_params_yaml(epoch)
    save_metrics(metrics)
    save_loss_plot(train_losses, val_losses)
    save_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model with the given parameters."
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--starting_epoch",
        type=int,
        default=1,
        help="Epoch number to start training from.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="How many examples in a batch.",
    )
    args = parser.parse_args()

    main(args)
