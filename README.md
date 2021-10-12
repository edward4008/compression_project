# compression_project

This project is about analyzing the calibration performance of compressed networks and building cascaded classifier architecture based on different compressed networks.

## Dependencies
```
python >= 3.7
numpy == 1.21.0
pytorch == 1.9.1+cu102
torchvision == 0.10.0
tensorboard == 2.6.0
torch-pruning == 0.2.7
nni == 2.0
```

## Performance Analysis of Compressed Networks
The first part of the project is about the performance analysis of compressing methods. The code for this part of the project can be found in the `pruning` folder. CUrrently there are six different compressing methods implemented:
```
Level Pruning
L1 Filter Pruning
L2 Filter Pruning
Filter Pruning via Geometric Medium (FPGM)
Stripe-wise Pruning (SWP)
Random Filter Pruning
```

## Cascaded Classifier Design
The second part of the project is about the deisgn of cascaded classifier architecture based on compressed networks. The code for this part of the project can be found in the `cascade` folder.
