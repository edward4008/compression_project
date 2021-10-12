import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_dataloader_imagenet, eval_imagenet, class_ece, accuracy, AverageMeter
import imagenet_resnet as resnet
import argparse
from collections import OrderedDict

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.5)

args = parser.parse_args()

def cascaded_resnets_imagenet_eval(conf_t, test_loader):
    net1 = resnet.resnet18(pretrained=True)
    #acc, ece = eval(net1, test_loader)
    #print('Level 1 Acc: ', acc, ' ece: ', ece )

    net2 = resnet.resnet34(pretrained=True)
    #print('Level 2 Acc: ', acc, ' ece: ', ece )

    net3 = resnet.resnet50(pretrained=True)
    #net3.load_state_dict(torch.load('trained_resnets_cifar100/resnet50-best.pth'))
    #print('Level 3 Acc: ', acc, ' ece: ', ece )

    net4 = resnet.resnet101(pretrained=True)
    #net4.load_state_dict(torch.load('trained_resnets_cifar100/resnet101-best.pth'))
    #print('Level 4 Acc: ', acc, ' ece: ', ece )

    net5 = resnet.resnet152(pretrained=True)
    #net5.load_state_dict(torch.load('trained_resnets_cifar100/resnet152-best.pth'))
    #print('Level 5 Acc: ', acc, ' ece: ', ece )

    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1.to(device)
    net1.eval()
    net2.to(device)
    net2.eval()
    net3.to(device)
    net3.eval()
    net4.to(device)
    net4.eval()
    net5.to(device)
    net5.eval()
    list_confidence_st = []
    list_all_labels = []
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            if i % 100 == 0:
                print('Image ', i)
            img = img.to(device)
            target = target.to(device)
            #print('Evaluating by level 1...')
            n1 = n1+1
            out = net1(img)
            out_sf = torch.softmax(out, dim=1)
            #print(out_sf)
            conf = torch.max(out_sf)
            #print(conf)
            if conf < conf_t:
                #print('Evaluating by level 2...')
                n2 = n2+1
                out = net2(img)
                out_sf = torch.softmax(out, dim=1)
                conf = torch.max(out_sf)
                #print(conf)
                if conf < conf_t:
                    #print('Evaluating by level 3...')
                    n3 = n3+1
                    out = net3(img)
                    out_sf = torch.softmax(out, dim=1)
                    conf = torch.max(out_sf)
                    #print(conf)
                    if conf < conf_t:
                        #print('Evaluating by level 4...')
                        n4 = n4+1
                        out = net4(img)
                        out_sf = torch.softmax(out, dim=1)
                        conf = torch.max(out_sf)
                        #print(conf)
                        if conf < conf_t:
                            #print('Evaluating by level 5...')
                            n5 = n5+1
                            out = net5(img)
                            out_sf = torch.softmax(out, dim=1)
                            conf = torch.max(out_sf)
                            #print(conf)
            list_confidence_st.append(out)
            list_all_labels.append(target)
            #pred = out.max(1)[1].detach().cpu().numpy()
            # if i == 2:
            #     break
            acc1, acc5 = accuracy(out, target, topk=(1, 5))
            top1_m.update(acc1.item(), out.size(0))
            top5_m.update(acc5.item(), out.size(0))
            metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])
        logits_all = torch.cat(list_confidence_st,dim=0)
        targets_all = torch.cat(list_all_labels,dim=0)
        ece = class_ece(logits_all,targets_all)
        ece = ece.detach().cpu().numpy()
        print('n1:', n1)
        print('n2:', n2)
        print('n3:', n3)
        print('n4:', n4)
        print('n5:', n5)
        print('Top1: ', top1_m.avg)
        print('Top5: ', top5_m.avg)
        print('ece: ', ece)
    return top1_m.avg, top5_m.avg, ece

if __name__ == '__main__':
    threshold = args.threshold
    train_set, test_set = get_dataloader_imagenet(1, 1)
    acc1, acc5, ece = cascaded_resnets_imagenet_eval(threshold, test_set)
    print('Acc1: ', acc1, 'Acc2: ', acc5, 'ece: ', ece)
    print(threshold)