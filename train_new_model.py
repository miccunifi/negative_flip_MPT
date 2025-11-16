import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import datasets
import os
import argparse
from torchvision.models.resnet import *
torch.backends.cudnn.benchmark = True
from loss import * 

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs to train')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class CIFAR100Subset(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super(CIFAR100Subset, self).__init__(*args, **kwargs)
        self.classes = self.classes[:50]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data, self.targets = self._filter_classes(self.data, self.targets)

    def _filter_classes(self, data, targets):
        mask = torch.tensor([target in range(50) for target in targets])
        return data[mask], torch.tensor(targets)[mask]


trainset = datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

testset_old = CIFAR100Subset(
    root='./data', train=False, download=True, transform=transform_test)
testloader_old = torch.utils.data.DataLoader(
    testset_old, batch_size=128, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = resnet18(num_classes=100)
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) \r'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) \r'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print("New Classes Accuracy: ", acc)

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt_new_100.pth')

    
def test_old():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader_old):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(batch_idx, len(testloader_old), 'Loss: %.3f | Acc: %.3f%% (%d/%d) \r'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print("Old Classes Accuracy: ", acc)


for epoch in range(0, args.epochs):
    train(epoch)
    test(epoch)
    test_old()
    scheduler.step()