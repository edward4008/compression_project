import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_dataloader_cifar, class_ece
import cifar_resnet as resnet
import argparse

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--path1', type=str)
parser.add_argument('--path2', type=str)
parser.add_argument('--model', type=str, default='resnet18')

args = parser.parse_args()

def cascaded_cifar100_eval(conf_t, test_loader):
    if args.model == 'resnet18':
        net2 = resnet.ResNet18(num_classes=100)
    if args.model == 'resnet34':
        net2 = resnet.ResNet34(num_classes=100)
    if args.model == 'resnet50':
        net2 = resnet.ResNet50(num_classes=100)
    if args.model == 'resnet101':
        net2 = resnet.ResNet101(num_classes=100)
    if args.model == 'resnet152':
        net2 = resnet.ResNet152(num_classes=100)
    net2.load_state_dict(torch.load(args.path1))
    #acc, ece = eval(net1, test_loader)
    #print('Level 1 Acc: ', acc, ' ece: ', ece )

    net1 = torch.load(args.path2)
    #print('Level 2 Acc: ', acc, ' ece: ', ece )


    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1.to(device)
    net1.eval()
    net2.to(device)
    net2.eval()
    list_confidence_st = []
    list_all_labels = []
    n1 = 0
    n2 = 0
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            if i % 1000 == 0:
                print('Image ', i)
            img = img.to(device)
            target = target.to(device)
            #print('Evaluating by level 1...')
            n1 = n1 + 1
            out = net1(img)
            out_sf = torch.softmax(out, dim=1)
            #print(out_sf)
            conf = torch.max(out_sf)
            #print(conf)
            if conf < conf_t:
                #print('Evaluating by level 2...')
                n2 = n2 + 1
                out = net2(img)
                out_sf = torch.softmax(out, dim=1)
                conf = torch.max(out_sf)
            list_confidence_st.append(out)
            list_all_labels.append(target)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
            # if i == 2:
            #     break
        logits_all = torch.cat(list_confidence_st,dim=0)
        targets_all = torch.cat(list_all_labels,dim=0)
        ece = class_ece(logits_all,targets_all)
        ece = ece.detach().cpu().numpy()
        print('n1:', n1)
        print('n2:', n2)
    return correct/total, ece

if __name__ == '__main__':
    threshold = args.threshold
    train_set, test_set = get_dataloader_cifar(100, 32, 1)
    acc, ece = cascaded_cifar100_eval(threshold, test_set)
    print('Acc: ', acc, 'ece: ', ece)
    print(threshold)