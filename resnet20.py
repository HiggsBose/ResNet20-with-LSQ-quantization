# Attempt to write a ResNet-20 on CIFAR-10 dataset
# 07/21/2022 by pzl

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn import functional as F
import BuildingBlock as B
import QuantizationScheme
import NoiseScheme as N
import configuration as cfg
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import train_test as T
import DataLoader
import Model


def save_model(model):
    '''
    save the model as .pth file

    :param model: the model to be saved
    :return: None
    '''
    torch.save(model.cpu().state_dict(), './result/saved_model/resnet.pth')
    return None


def plot_param_hist_fig(model):
    '''
    use matplotlib to draw the histogram of model parameters, and save the different figures in files

    :param model: the required model
    :return: None
    '''
    for name, param in model.named_parameters():
        if ('layer' in name) and ('conv' in name) and ('quantize' not in name):
            fake_param = param.to('cpu').detach().view(1, -1)
            plt.hist(fake_param, bins=100)
            plt.savefig('./result/ori_params/{}.jpg'.format(name))
            # plt.show()
            plt.clf()
            quantize = QuantizationScheme.weight_quantize_fn(cfg.config["w_bit"])
            param = quantize(param)
            noise_param = N.add_noise(param, cfg.config["m_of_dispersion"], cfg.config["additive"])
            param = param.to('cpu').detach().view(1, -1)
            plt.hist(param, bins=100)
            plt.savefig('./result/quantized_params/{}.jpg'.format(name))
            # plt.show()
            plt.clf()
            noise_param = noise_param.to('cpu').detach().view(1, -1)
            plt.hist(noise_param, bins=100)
            plt.savefig('./result/noise_params/{}.jpg'.format(name))
            # plt.show()
            plt.clf()
    return None


def plot_param_hist(writer1, writer2, model):
    '''
    use tensorboard to draw the histogram of the parameters

    :param writer1: writers to draw parameters before quantization
    :param writer2: wrietrs to draw parameters after quantization
    :param model: the required model
    :return: None
    '''
    for name, param in model.named_parameters():
        quantize = QuantizationScheme.weight_quantize_fn(cfg.config["w_bit"])
        writer2.add_histogram(f"{name}", param, 0)
        param = quantize(param)
        writer1.add_histogram(f"{name}", param, 0)

    return None


writer = SummaryWriter(log_dir='./result/log/')
net = Model.ResNet20()
train_loader, test_loader = DataLoader.load_data()
net = T.train_net(net, train_loader, writer, test_set=test_loader, evaluation=True)
plot_param_hist_fig(net)
save_model(net)
