import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# from cifar_resnet import ResNet18
import cifar_resnet as resnet
 
from utils import eval, get_dataloader_cifar

import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
import time

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--ratio', type=float, default=1.0)

args = parser.parse_args()

# def eval(model, test_loader):
#     correct = 0
#     total = 0
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#     print('Start validating...')
#     with torch.no_grad():
#         for i, (img, target) in enumerate(test_loader):
#             img = img.to(device)
#             out = model(img)
#             pred = out.max(1)[1].detach().cpu().numpy()
#             target = target.cpu().numpy()
#             correct += (pred==target).sum()
#             total += len(target)
#     return correct / total

def train_model(model, train_loader, test_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    best_epoch = 0
    for epoch in range(args.total_epochs):
        t = time.time()
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i%10==0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc, ece = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.total_epochs, acc))
        if best_acc<acc:
            torch.save( model, 'pruned-resnet18-cifar100-l2norm/resnet18-cifar100-ratio%.2f-best.pth'%(args.ratio) )
            best_acc=acc
            best_epoch = epoch+1
            best_ece = ece
        scheduler.step()
        torch.save( model, 'pruned-resnet18-cifar100-l2norm/resnet18-cifar100-ratio%.2f-last.pth'%(args.ratio) )
        print('Epoch time ', time.time()-t)
        print("Best Acc=%.4f, best epoch=%d"%(best_acc, best_epoch) )
        print('Best epoch ece: ', best_ece)
        print()

def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
    def prune_conv(conv, amount=0.2):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        strategy = tp.strategy.L2Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
    block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    block_prune_probs = np.asarray(block_prune_probs) * args.ratio
    blk_id = 0
    for m in model.modules():
        if isinstance( m, resnet.BasicBlock ):
            prune_conv( m.conv1, block_prune_probs[blk_id] )
            prune_conv( m.conv2, block_prune_probs[blk_id] )
            blk_id+=1
    return model    

def main():
    dataset = 100
    size = 32
    batch_size = 256
    train_loader, test_loader = get_dataloader_cifar(dataset, size, batch_size)
    if args.mode=='train':
        args.round=0
        model = resnet.ResNet18(num_classes=100)
        train_model(model, train_loader, test_loader)
    elif args.mode=='prune':
        previous_ckpt = 'resnet18-cifar100.pth'
        print("Load model from %s"%( previous_ckpt ))
        model = resnet.ResNet18(num_classes=100)
        model.load_state_dict(torch.load( previous_ckpt ))
        prune_model(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        print('Start training...')
        train_model(model, train_loader, test_loader)
        print("Number of Parameters: %.1fM"%(params/1e6))
        print(args.ratio)
    elif args.mode=='test':
        ckpt = 'resnet18-round%d.pth'%(args.round)
        print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc))

if __name__=='__main__':
    main()
