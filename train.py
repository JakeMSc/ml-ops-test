import argparse
import json
import os
from io import BytesIO

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


from interruptions import EC2Interruption, detect_ec2_interruption, touch_empty_file


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
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model, x, y, optimizer, criterion):
    model.train()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def predict(model, x):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
    return y_pred


def get_metrics(y, y_pred_label):
    metrics = {}
    metrics["acc"] = (y_pred_label == y).sum().item() / len(y)
    return metrics


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


def validate(model, val_dataset, criterion):
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    total_loss = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)

    return total_loss / total_samples


def save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig(filename)


def save_metrics(metrics):
    with open("metrics.json", "w") as outfile:
        outfile.write(json.dumps(metrics))


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
        # Use DVC API to open and read the model file
        with dvc.api.open("model.pt", mode="rb") as model_file:
            model_buffer = BytesIO(model_file.read())

        # Load the state_dict using PyTorch
        state_dict = torch.load(model_buffer)
        # Load the state_dict into the model
        model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print(
        f"Training for {args.epochs} epochs. Starting from epoch {args.starting_epoch}."
    )

    # Instantiate EarlyStopper with the desired patience and min_delta
    early_stopper = EarlyStopper(patience=3, min_delta=0.001)

    for epoch in range(args.starting_epoch, args.epochs + 1):
        try:
            for x_batch, y_batch in train_loader:
                if detect_ec2_interruption():
                    raise EC2Interruption()

                train(model, x_batch, y_batch, optimizer, criterion)

            val_loss = validate(model, val_dataset, criterion)
            print(f"Epoch {epoch}, Validation Loss: {val_loss}")

            if early_stopper.early_stop(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        except EC2Interruption:
            print(f"EC2 interrupted, saving model at epoch {epoch}")
            touch_empty_file("interrupted")
            break

    torch.save(model.state_dict(), "model.pt")
    metrics, y_true, y_pred = evaluate(model, test_dataset)
    update_params_yaml(epoch)
    save_metrics(metrics)
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
