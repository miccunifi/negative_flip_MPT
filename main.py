'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import os
import argparse
import numpy as np
from torchvision.models.resnet import *
from loss import * 
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs to train')
parser.add_argument('--method', default='MPT_KL', type=str,
                    choices=['PCT_KL', 'PCT_LM', 'PCT_Naive', 'ELODI', 'ELODI_topk', 'MPT_KL', 'MPT_LM', 'MPT_nodistillation', 'MPT_LM_nobias', 'MPT_KL_nobias', 'No_Treatment'],
                    help='method for negative flip')
parser.add_argument('--b', default=4, type=float, help='logit bias term')
parser.add_argument('--topk', default=10, type=int, help='top-k classes to consider (ELODI Only)')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# deterministic training
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)


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

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(
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

# Model
print('==> Building old model..')
net_old = resnet18(num_classes=50)
net_old = net_old.to(device)
# Load checkpoint.
print('==> Resuming from checkpoint old net..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt_old_50.pth')
net_old.load_state_dict(checkpoint['net'])

# Model
print('==> Building new model..')
net_new = resnet18(num_classes=100)
net_new = net_new.to(device)
# Load checkpoint.
print('==> Resuming from checkpoint new net..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt_new_100.pth')
net_new.load_state_dict(checkpoint['net'])

if 'elodi' in args.method:
    list_net_new = []
    for i in range(8):
        net_new = resnet18(num_classes=100)
        net_new = net_new.to(device)
        if args.dataset == 'cifar100':
            checkpoint = torch.load(f'./checkpoint/ensamble/ckpt_{i}.pth')
        elif args.dataset == 'cifar10':
            checkpoint = torch.load(f'./checkpoint/ensamble/ckpt_{i}.pth')
        net_new.load_state_dict(checkpoint['net'])
        net_new.eval()
        list_net_new.append(net_new)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

net_old.eval()
net_new.eval()

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

        if 'elodi' in args.method:
            with torch.no_grad():
                outputs_old = net_old(inputs)

                outputs_new = torch.zeros_like(outputs)
                for current_net_new in list_net_new:
                    outputs_new_ensemble = current_net_new(inputs)
                    outputs_new += outputs_new_ensemble
                outputs_new /= len(list_net_new)
        else:
            with torch.no_grad():
                outputs_old = net_old(inputs)
                outputs_new = net_new(inputs)

        
        if args.method == 'PCT_KL':
            loss_pcd = focal_distillation_fdkl_loss(outputs[:,:50], outputs_old, targets, alpha=1.0, beta=5.0, tau=5.0)
        elif args.method == 'PCT_LM':
            loss_pcd = 0.4 * focal_distillation_fdlm_loss(outputs[:,:50], outputs_old, targets, alpha=1.0, beta=5.0)
        elif args.method == 'PCT_Naive':
            loss_pcd = focal_distillation_naive_loss(outputs, outputs_old, targets)

        elif args.method == 'ELODI':
            if epoch > 5:
                loss_pcd += 0.6 * lm_loss(outputs[:,:50], outputs_old)
                loss_pcd += 0.4 * lm_loss(outputs, outputs_new)

        elif args.method == 'ELODI_topk':
            if epoch > 5:
                _, topk_indices_old = torch.topk(outputs_old, args.topk, dim=1)
                mask_old = torch.zeros_like(outputs_old)
                mask_old.scatter_(1, topk_indices_old, 1)
                loss_pcd += 0.6 * lm_loss(outputs[:,:50] * mask_old, outputs_old * mask_old)
                _, topk_indices = torch.topk(outputs_new, args.topk, dim=1)
                mask = torch.zeros_like(outputs_new)
                mask.scatter_(1, topk_indices, 1)
                loss_pcd += 0.4 * lm_loss(outputs * mask, outputs_new * mask)

        elif args.method == 'MPT_KL_nobias':
            loss_pcd = focal_distillation_fdkl_loss(outputs[:,:50], outputs_old, targets, alpha=1.0, beta=5.0, tau=5.0)
            loss_pcd += focal_distillation_fdkl_loss(outputs, outputs_new, targets, alpha=1.0, beta=5.0, tau=5.0)
        elif args.method == 'MPT_LM_nobias':
            loss_pcd = 0.4 * focal_distillation_fdlm_loss(outputs[:,:50], outputs_old, targets, alpha=1.0, beta=5.0)
            loss_pcd += 0.4 * focal_distillation_fdlm_loss(outputs, outputs_new, targets, alpha=1.0, beta=5.0)
        elif args.method == 'MPT_nodistillation':
            outputs[:,50:] = outputs[:,50:] + args.b
            loss_pcd = 0

        elif args.method == 'MPT_LM':
            loss_pcd = 0.4 * focal_distillation_fdlm_loss(outputs[:,:50], outputs_old, targets, alpha=1.0, beta=5.0)
            loss_pcd += 0.4 * focal_distillation_fdlm_loss(outputs, outputs_new, targets, alpha=1.0, beta=5.0)
            outputs[:,50:] = outputs[:,50:] + args.b
        elif args.method == 'MPT_KL':
            loss_pcd = focal_distillation_fdkl_loss(outputs[:,:50], outputs_old, targets, alpha=1.0, beta=5.0, tau=5.0)
            loss_pcd += focal_distillation_fdkl_loss(outputs, outputs_new, targets, alpha=1.0, beta=5.0, tau=5.0)
            outputs[:,50:] = outputs[:,50:] + args.b
        else:
            loss_pcd = 0
        
        loss_ce = criterion(outputs, targets)
        loss = loss_ce + loss_pcd

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) \r'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    outputs_old_list = []
    outputs_list = []
    labels_mask = []

    negative_flips = [0] * 50  # Initialize a list to count negative flips for the first 50 classes

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs_old = net_old(inputs)  # Get the outputs from net_old
            loss = criterion(outputs, targets)

            # Calculate negative flips for the first 50 classes
            mask = targets < 50  # Only consider the first 50 classes

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted_old = outputs_old.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            outputs_old_list.append(outputs_old[mask])
            outputs_list.append(outputs[mask])
            labels_mask.append(targets[mask])

            old_correct = predicted_old[mask] == targets[mask]
            new_wrong = predicted[mask] != targets[mask]
            for target in targets[mask][old_correct & new_wrong]:
                negative_flips[target] += 1

            if batch_idx % 100 == 0:
                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) \r'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    outputs_old = torch.cat(outputs_old_list, dim=0).cpu().numpy()
    outputs = torch.cat(outputs_list, dim=0).cpu().numpy()
    labels_mask = torch.cat(labels_mask, dim=0).cpu().numpy()

    NFR = compute_negative_flip_rate(outputs, outputs_old, labels_mask)
    RNFR = compute_relative_negative_flip_rate(outputs, outputs_old, labels_mask)

    print(f"Negative flip rate: {NFR}")
    print(f"Relative negative flip rate: {RNFR}")
    acc = 100.*correct/total
    print(f"Test All Classes Accuracy: {acc}")

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
    test()
    test_old()
    scheduler.step()

    