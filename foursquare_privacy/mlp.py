import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd

device = "cpu"


class PoiMLP(nn.Module):
    def __init__(self, inp_size, out_size, dropout_rate=0, act=nn.functional.softmax):
        super(PoiMLP, self).__init__()
        self.linear_1 = nn.Linear(inp_size, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear_3 = nn.Linear(128, out_size)
        self.final_act = act

    def forward(self, x):
        hidden = self.dropout1(torch.relu(self.linear_1(x)))
        hidden = self.dropout2(torch.relu(self.linear_2(hidden)))
        out = self.final_act(self.linear_3(hidden))
        return out


class PoiDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, ind):
        x = self.features[ind]
        y = self.labels[ind]
        return x, y


def train_model(
    train_set_nn_x,
    train_set_nn_y,
    val_set_nn_x,
    val_set_nn_y,
    batch_size=8,
    epochs=10,
    learning_rate=1e-3,
    save_path=os.path.join("trained_models", "test"),
    **kwargs,
):
    # create dataset and dataloader
    train_set_nn_torch = PoiDataset(train_set_nn_x, train_set_nn_y)
    val_set_nn_torch = PoiDataset(val_set_nn_x, val_set_nn_y)
    train_loader = DataLoader(train_set_nn_torch, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_set_nn_torch, batch_size=batch_size, shuffle=False)

    # model
    model = PoiMLP(inp_size=train_set_nn_x.shape[1], out_size=train_set_nn_y.shape[1])
    os.makedirs(save_path, exist_ok=True)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # get ground truth test labels for accuracy evaluation
    gt_test_labels = np.argmax(val_set_nn_y, axis=1)
    uni, counts = np.unique(gt_test_labels, return_counts=True)
    print("Test label distribution:", {u: c for u, c in zip(uni, counts)})

    model.train()
    best_performance = np.inf
    epoch_test_loss, epoch_train_loss = [], []
    for epoch in range(epochs):
        losses = []
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device).float()

            output = model(x)
            # loss
            softmax_loss = criterion(output, y)
            entropy_loss = torch.mean(torch.sum(output * torch.log(output), dim=-1)) + 3
            loss = softmax_loss + entropy_loss * 10
            loss.backward()
            # if batch_num == 10:
            #     # print(model.linear_3.weight.grad)
            #     print("train loss at batch 10:", round(loss.item(), 2))
            losses.append(loss.item())

            optimizer.step()

            # if batch_num == 10:
            #     print("\tEpoch %d | Batch %d | Loss %6.2f" % (epoch, batch_num, np.median(losses)))

        # TESTING
        model.eval()
        with torch.no_grad():
            test_losses, test_pred = [], []
            for batch_num, input_data in enumerate(test_loader):
                x, y = input_data
                x = x.to(device).float()
                y = y.to(device).float()
                output = model(x)
                loss = criterion(output, y)
                test_losses.append(loss.item())
                test_pred.extend(output.numpy().tolist())
        model.train()
        test_pred = np.argmax(np.array(test_pred), axis=1)
        uni, counts = np.unique(test_pred, return_counts=True)
        print(
            f"\n Epoch {epoch} | TRAIN Loss {round(sum(losses) / len(losses), 2)}\
                 | TEST loss {round(sum(test_losses) / len(test_losses), 2)} \n"
        )
        print(
            "Accuracy:",
            round(np.sum(test_pred == gt_test_labels) / len(test_pred), 2),
            {u: c for u, c in zip(uni, counts)},
        )
        if sum(test_losses) < best_performance:
            best_performance = sum(test_losses)
            torch.save(model.state_dict(), os.path.join(save_path, "model"))
            print("Saved model")
        print()
        # print(
        #     f"\n Epoch {epoch} (median) | TRAIN Loss {round(np.median(losses), 3)}\
        #  | TEST loss {round(np.median(test_losses), 3)} \n"
        # )
        epoch_test_loss.append(np.mean(test_losses))
        epoch_train_loss.extend(list(average_batches(losses)))
    plot_losses(epoch_train_loss, epoch_test_loss, save_path)
    return model


def plot_losses(losses, test_losses, save_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.subplot(1, 2, 2)
    plt.plot(test_losses)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "losses.png"))


def average_batches(loss, n_average=1000):
    return np.array(loss)[: -(len(loss) % n_average)].reshape((-1, n_average)).mean(axis=1)
