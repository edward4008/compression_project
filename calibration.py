import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class ECE_Loss(nn.Module):
    """ 
    This calculates the ECE Loss metric for calibration.
    Make sure to provide the logits of the network! 
    From Guo et al. 
    re-adapted by: Ib. A. 
    """

    def __init__(self, number_bins=10):
        super(ECE_Loss,self).__init__()
        interval=torch.linspace(0,1,number_bins +1)
        self.lower = interval[:-1]
        self.upper = interval[1:]
    
    def forward(self, logits, targets):
        all_softmaxes = F.softmax(logits,dim=1)
        confidences, preds = torch.max(all_softmaxes, 1)
        accuracies = preds.eq(targets)

        ece = torch.zeros(1,device = logits.device)
        for low, up in zip(self.lower, self.upper):
            inside_bin = confidences.gt(low.item())* confidences.le(up.item())
            n_inside = inside_bin.float().mean()
            if n_inside.item() >0: 
                accuracy_per_bin= accuracies[inside_bin].float().mean()
                avg_conf_per_bin = confidences[inside_bin].mean()
                ece +=torch.abs(avg_conf_per_bin - accuracy_per_bin) * n_inside 
        return ece
        
class Calibration_curve(nn.Module):
    def __init__(self, number_bins=10):
        super(Calibration_curve,self).__init__()
        interval=torch.linspace(0,1,number_bins +1)
        self.lower = interval[:-1]
        self.upper = interval[1:]
        self.number_bins = number_bins
    
    def forward(self, logits, targets):
        all_softmaxes = F.softmax(logits,dim=1)
        confidences, preds = torch.max(all_softmaxes, 1)
        accuracies = preds.eq(targets)
        values_perbin = torch.zeros(self.number_bins,1)
        count=0
        for low, up in zip(self.lower, self.upper):
            inside_bin = confidences.gt(low.item())* confidences.le(up.item())
            sum_in_bin = inside_bin.float().sum()
            if sum_in_bin.item() >0: 
                accuracy_per_bin= accuracies[inside_bin].float().mean()
                values_perbin[count] = accuracy_per_bin
                count+=1
            else:
                values_perbin[count]= 0
                count+=1
              
        return values_perbin


        
        
# if __name__=='__main__':

# # *********** EXAMPLE ************
#     logits= np.loadtxt('predictions_cifar10.csv', delimiter=',')
#     targets = np.loadtxt('targets_cifar10.csv',delimiter=',')
#     # turn these numpy arrays to tensors 
#     logits_tensor = torch.tensor(logits)
#     targets_tensor = torch.LongTensor(targets)
#     #*************** ECE LOSS ***************
#     my_ECELoss= ECE_Loss(10)
#     ece_value= my_ECELoss(logits_tensor, targets_tensor)
#     print("the ece loss is : " , ece_value)
    
    '''
    #*************** CALIBRATION CURVES *********
    my_calib_curves= Calibration_curve(10)
    #get the accuracies per bin. 
    out_=my_calib_curves(logits_tensor, targets_tensor) 
    points=('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1.0')
    ypos = np.arange(len(points))
    plt.figure()
    plt.plot(ypos, np.squeeze(out_.cpu().numpy()), '-mo', label='Alexnet')
    plt.plot([0.0,10.0],[0.0,1.0], "k:",label='perfect calibration')
    plt.legend()
    plt.xlim([0,1])
    plt.xticks(ypos, points)
    plt.ylabel('accuracy')
    plt.xlabel('confidence')
    plt.savefig("calib_.jpg")
    '''
