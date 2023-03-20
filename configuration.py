# all the configurations and hyperparameters
# 07/28/2022

import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Predefine some of the hyperparameters
config = {"batch_size": 64,         # batch size
          "num_epochs": 200,        # number of epochs
          "lr": 0.1,                # learning rate
          "weight_decay": 1e-4,     # weight decay rate (regularization)
          "device": device,         # device
          "w_bit": 2,               # weight bit
          "add_noise": False,       # choose whether add noise or not
          "m_of_dispersion": 0.05,   # measure of dispersion
          "additive": True,         # decide whether the noise is additive or multiplicative
          "inference_noise": True   # decide whether inference noise is added to the model
          }