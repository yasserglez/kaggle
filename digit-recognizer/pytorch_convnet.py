import torch
from torch import nn
from torch.nn import functional as func
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import numpy as np

from util import load_train_data, load_test_data, save_predictions


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        hidden_units = (7 * 7 * 32 + 10) // 2
        self.hidden = nn.Linear(7 * 7 * 32, hidden_units)
        self.output = nn.Linear(hidden_units, 10)

    def forward(self, x):
        x = func.max_pool2d(func.relu(self.conv1(x)), kernel_size=2)
        x = func.max_pool2d(func.relu(self.conv2(x)), kernel_size=2)
        x = x.view(-1, 7 * 7 * 32)
        x = self.hidden(func.dropout(x, 0.5))
        x = self.output(func.dropout(x, 0.5))
        return func.log_softmax(x)


def train_model(model, optimizer, loader):
    model.train()
    loss_sum = correct = 0
    for images, targets in loader:
        images = Variable(images.float(), requires_grad=False)
        targets = Variable(targets, requires_grad=False)
        optimizer.zero_grad()
        outputs = model.forward(images)
        loss = func.nll_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.data[0] * outputs.size(0)
        correct += (torch.max(outputs.data, 1)[1] == targets.data).sum()
    n = len(loader.dataset)
    return loss_sum / n, correct / n


def evaluate_model(model, loader):
    model.eval()
    loss_sum = correct = 0
    for images, targets in loader:
        images = Variable(images.float(), requires_grad=False)
        targets = Variable(targets, requires_grad=False)
        outputs = model.forward(images)
        loss = func.nll_loss(outputs, targets)
        loss_sum += loss.data[0] * outputs.size(0)
        correct += (torch.max(outputs.data, 1)[1] == targets.data).sum()
    n = len(loader.dataset)
    return loss_sum / n, correct / n


def main():
    model = ConvNet()
    print(model)

    images, targets = load_train_data()
    train_images, val_images, train_targets, val_targets = train_test_split(images, targets, test_size=0.1)

    train_images = torch.from_numpy(train_images).unsqueeze(1)
    train_targets = torch.from_numpy(train_targets)
    train_dataset = TensorDataset(train_images, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=64)

    val_images = torch.from_numpy(val_images).unsqueeze(1)
    val_targets = torch.from_numpy(val_targets)
    val_dataset = TensorDataset(val_images, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=64)

    optimizer = Adam(model.parameters(), lr=1e-3)

    best_val_acc = -1
    patience_count = 0
    for epoch in range(1, 1001):
        loss, acc = train_model(model, optimizer, train_loader)
        val_loss, val_acc = evaluate_model(model, val_loader)
        patience_count += 1
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save(model, 'pytorch_convnet')
        msg = 'Epoch {:04d} - loss: {:.6g} - acc: {:.6g} - val_loss: {:.6g} - val_acc: {:.6g}'
        print(msg.format(epoch, loss, acc, val_loss, val_acc))
        if patience_count > 3:
            break

    model = torch.load('pytorch_convnet')
    images = load_test_data()
    images = torch.from_numpy(images).unsqueeze(1)
    test_dataset = TensorDataset(images, torch.zeros(images.size(0)))
    test_loader = DataLoader(test_dataset)
    labels = []
    for images, _ in test_loader:
        images = Variable(images.float(), requires_grad=False)
        outputs = model.forward(images)
        labels.extend(torch.max(outputs.data, 1)[1])
    save_predictions(np.array(labels), 'pytorch_convnet.csv')


if __name__ == '__main__':
    main()
