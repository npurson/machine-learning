import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm


arch = 'googlenet'
num_classes = 10
lr = 1e-2
batch = 224
epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(arch, num_classes):
    model = getattr(torchvision.models, arch)(pretrained=True)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, device):
    bar = tqdm(dataloader)
    bar.set_description(f'epoch {epoch:2}')
    correct, total = 0, 0
    for X, y in bar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += sum(torch.argmax(pred, axis=1) == y)
        total += len(X)
        bar.set_postfix_str(f'acc={correct / total * 100:.2f} loss={loss.item():.2f}')
    scheduler.step()
    return


def test_epoch(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        correct += sum(torch.argmax(pred, axis=1) == y)
        total += len(X)
    print(f'test acc: {correct / total * 100:.2f}')
    return


def main():
    transforms = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='~/data/', train=True, transform=transforms, download=True)
    testset = torchvision.datasets.CIFAR10(root='~/data/', transform=transforms, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch)

    model = build_model(arch, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_epoch(model, trainloader, criterion, optimizer, scheduler, epoch + 1, device)
        test_epoch(model, testloader, device)


if __name__ == '__main__':
    main()
