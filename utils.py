# from calibration import ECE_Loss
import torch
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
import time
from collections import OrderedDict
import os


class ECE_Loss(nn.Module):
    """
    This calculates the ECE Loss metric for calibration.
    Make sure to provide the logits of the network!
    From Guo et al.
    re-adapted by: Ib. A.
    """

    def __init__(self, number_bins=10):
        super(ECE_Loss, self).__init__()
        interval = torch.linspace(0, 1, number_bins + 1)
        self.lower = interval[:-1]
        self.upper = interval[1:]

    def forward(self, logits, targets):
        all_softmaxes = F.softmax(logits, dim=1)
        confidences, preds = torch.max(all_softmaxes, 1)
        accuracies = preds.eq(targets)

        ece = torch.zeros(1, device=logits.device)
        for low, up in zip(self.lower, self.upper):
            inside_bin = confidences.gt(low.item()) * confidences.le(up.item())
            n_inside = inside_bin.float().mean()
            if n_inside.item() > 0:
                accuracy_per_bin = accuracies[inside_bin].float().mean()
                avg_conf_per_bin = confidences[inside_bin].mean()
                ece += torch.abs(avg_conf_per_bin - accuracy_per_bin) * n_inside
        return ece

def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print('Start validating...')
    list_confidence_st = []
    list_all_labels = []
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            target = target.to(device)
            out = model(img)
            list_confidence_st.append(out)
            list_all_labels.append(target)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
        logits_all = torch.cat(list_confidence_st,dim=0)
        targets_all = torch.cat(list_all_labels,dim=0)
        ece = class_ece(logits_all,targets_all)
        ece = ece.detach().cpu().numpy()
    return correct / total, ece

def class_ece(logits_tensor,targets_tensor ):
    my_ECELoss= ECE_Loss(10)
    ece_value= my_ECELoss(logits_tensor, targets_tensor)
    print("the ece loss is : " , ece_value)
    return ece_value

def get_dataloader_cifar(dataset, image_size, batch_size):
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_TRAIN_STD = (0.247, 0.243, 0.261)


    if dataset == 10:
        if image_size == 32:
            train_loader = torch.utils.data.DataLoader(
                CIFAR10('./data', train=True, transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                ]), download=True),batch_size=batch_size, num_workers=2)
            test_loader = torch.utils.data.DataLoader(
                CIFAR10('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                ]),download=True),batch_size=batch_size, num_workers=2)
        if image_size ==224:
            train_loader = torch.utils.data.DataLoader(
                CIFAR10('./data', train=True, transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                    transforms.Resize(224),
                ]), download=True),batch_size=batch_size, num_workers=2)
            test_loader = torch.utils.data.DataLoader(
                CIFAR10('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                    transforms.Resize(224),
                ]),download=True),batch_size=batch_size, num_workers=2)
    


    if dataset == 100:
        if image_size == 32:
            train_loader = torch.utils.data.DataLoader(
                CIFAR100('./data', train=True, transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
                ]), download=True),batch_size=batch_size, num_workers=2)
            test_loader = torch.utils.data.DataLoader(
                CIFAR100('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
                ]),download=True),batch_size=batch_size, num_workers=2)
        if image_size ==224:
            train_loader = torch.utils.data.DataLoader(
                CIFAR100('./data', train=True, transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
                    transforms.Resize(224),
                ]), download=True),batch_size=batch_size, num_workers=2)
            test_loader = torch.utils.data.DataLoader(
                CIFAR100('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
                    transforms.Resize(224),
                ]),download=True),batch_size=batch_size, num_workers=2)

    return train_loader, test_loader

def get_dataloader_imagenet(b, w):
    traindir = os.path.join('/home/ruofengl/projects/def-jjclark/shared_data/imagenet', 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=b, shuffle=True,
        num_workers=w, pin_memory=True)

    valdir = os.path.join('/home/ruofengl/projects/def-jjclark/shared_data/imagenet', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=b, shuffle=False,
        num_workers=w, pin_memory=True)
    return train_loader, val_loader

class AverageMeter(object):
    #Computes and stores the average and current value
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    a, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].contiguous().view(-1).float().sum(0) * 100. / batch_size for k in topk]


def eval_imagenet(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    print('Start validating...')
    list_confidence_st = []
    list_all_labels = []
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            target = target.to(device)
            out = model(img)
            acc1, acc5 = accuracy(out, target, topk=(1, 5))
            list_confidence_st.append(out)
            list_all_labels.append(target)
            top1_m.update(acc1.item(), out.size(0))
            top5_m.update(acc5.item(), out.size(0))
        logits_all = torch.cat(list_confidence_st,dim=0)
        targets_all = torch.cat(list_all_labels,dim=0)
        ece = class_ece(logits_all,targets_all)
        ece = ece.detach().cpu().numpy()
        metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])
        print(metrics)
    return top1_m.avg, top5_m.avg, ece

