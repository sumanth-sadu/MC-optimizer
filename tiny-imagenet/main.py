import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from AdaBelief import AdaBelief
from RAdam import RAdam
from Adam import Adam
from AdaBelief_GC import AdaBelief_GC
from RAdam_GC import RAdam_GC
from Adam_GC import Adam_GC
from AdaBelief_MC import AdaBelief_MC
from RAdam_MC import RAdam_MC
from Adam_MC import Adam_MC

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch TinyImagenet Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--beta1', default=0.9, type=float, help='moment centralization coefficients beta_1')
parser.add_argument('--beta2', default=0.999, type=float, help='moment centralization coefficients beta_2')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--eps', default=1e-8, type=float, help='eps for var adam')
parser.add_argument('--name', default='/content/drive/MyDrive/pytorch-tiny/checkpoints/tiny_vgg16_run1_adabelief_bat128', type=str, help='name of the file')
parser.add_argument('--data_dir', default='/content/tiny-imagenet-200', type=str, help='path to data')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

data_dir = args.data_dir

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

trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Model
print('==> Building model..')

num = int(input('enter model : 1.vgg16  2. Resnet18 : '))

if num == 1:
  model = VGG('VGG16')
elif num == 2:
  model = ResNet18()
else:
  print('Wrong model option')
  sys.exit()

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

folder_name = args.name

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(folder_name), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(folder_name + '/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    acc = checkpoint['acc']
    best_acc = checkpoint['best_acc']
    args.start_epoch = checkpoint['epoch']
    print('Best Accuracy until now : "{}"'.format(best_acc))
    print("=> loaded checkpoint '{}' (epoch {})".format(args.name, checkpoint['epoch']))
    

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)

opt = int(input('Enter optimizer : 1.Adabelief 2.RAdam 3.Adam 4.Adabelief_GC 5.RAdam_GC 6.Adam_GC 7.Adabelief_MC 8.RAdam_MC 9.Adam_MC : '))

if opt == 1:
    optimizer = AdaBelief(model.parameters(), args.lr, betas=(args.beta1, args.beta2),  weight_decay=args.weight_decay, eps=args.eps)
elif opt == 2:
    optimizer = RAdam(model.parameters(), args.lr, betas=(args.beta1, args.beta2),
                        weight_decay=args.weight_decay, eps=args.eps)
elif opt == 3: 
    optimizer = Adam(model.parameters(), args.lr)
if opt == 4:
    optimizer = AdaBelief_GC(model.parameters(), args.lr, betas=(args.beta1, args.beta2),  weight_decay=args.weight_decay, eps=args.eps)
elif opt == 5:
    optimizer = RAdam_GC(model.parameters(), args.lr, betas=(args.beta1, args.beta2),
                        weight_decay=args.weight_decay, eps=args.eps)
elif opt == 6: 
    optimizer = Adam_GC(model.parameters(), args.lr)
if opt == 7:
    optimizer = AdaBelief_MC(model.parameters(), args.lr, betas=(args.beta1, args.beta2),  weight_decay=args.weight_decay, eps=args.eps)
elif opt == 8:
    optimizer = RAdam_MC(model.parameters(), args.lr, betas=(args.beta1, args.beta2),
                        weight_decay=args.weight_decay, eps=args.eps)
elif opt == 9: 
    optimizer = Adam_MC(model.parameters(), args.lr)
else:
    print("Invalid choice")
    sys.exit()

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1, last_epoch=-1, verbose=False)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    state = {
            'model': model.state_dict(),
            'acc': acc,
            'best_acc':best_acc,
            'epoch': epoch,
        }
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    filepath = folder_name + '/ckpt.pth'
    torch.save(state, filepath)
    print("Saving model ===> ")
    print(filepath)
    print()

for epoch in range(args.start_epoch, args.epochs):
    train(epoch)
    test(epoch)
    print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    scheduler.step()
