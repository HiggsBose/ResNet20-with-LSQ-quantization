'''

ResNet18 for clarification of FashionMNIST

Created by Zelun Pan
04/04/2023

'''
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

batch_size = 64
learning_rate = 0.001
num_epoch = 20



