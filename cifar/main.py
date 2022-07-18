import argparse
import os
import time
import shutil
import sys

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

import pandas as pd
# from optimizers.Adam import Adam

import torchvision
import torchvision.transforms as transforms

from models import *

parser = argparse.ArgumentParser(description='PyTorch Cifar100 Training')

parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
# Path to store the checkpoint. 
parser.add_argument('--start', default='/content/drive/MyDrive/pytorch-cifar-models-master_20_01_21/checkpoints/cifar100_vgg16_radam_bat128_graph/checkpoint.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='100', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 100)')

parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--batchsize', type=int, default=128, help='batch size')
parser.add_argument('--eps', default=1e-8, type=float, help='eps for var adam')

best_prec = 0

def main():
    global args, best_prec
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Model building
    print('=> Building model...')
    if use_gpu:
        # model can be set to anyone that I have defined in models folder
        # note the model should match to the cifar type !
        model_num = int(input("Enter model number : 1.Vgg16 2.ResNet18 : "))
        
        if model_num == 1:  
            model = VGG('VGG16')
        elif model_num == 2:
            model = ResNet18()
        else:
            print("Invalid choice")
            sys.exit()
      
        fdir = args.start
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        
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

        # optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        cudnn.benchmark = True
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1, last_epoch=-1, verbose=False)

        # epoch_counter = 0
        
    else:
        print('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.start):
            print('=> loading checkpoint "{}"'.format(args.start))
            checkpoint = torch.load(args.start)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Best Accuracy until now : "{}"'.format(best_prec))
            print("=> loaded checkpoint '{}' (epoch {})".format(args.start, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.start))

    # Data loading and preprocessing
    # CIFAR10
    if args.cifar_type == 10:
        print('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    # CIFAR100
    else:
        print('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    if args.evaluate:
        validate(testloader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(trainloader, model, criterion, optimizer, epoch)

        # evaluate on test set
        prec = validate(testloader, model, criterion)
        scheduler.step()
        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec,best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    
    end = time.time()

    for i, (input, target) in enumerate(trainloader):
      # measure data loading time
      data_time.update(time.time() - end)

      input, target = input.cuda(), target.cuda()

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      prec = accuracy(output, target)[0]
      losses.update(loss.item(), input.size(0))
      top1.update(prec.item(), input.size(0))

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
       
      if i % args.print_freq == 0:
          print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                epoch, i, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    print("Saving model ===> ")
    print(filepath)
    print()
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    main()
