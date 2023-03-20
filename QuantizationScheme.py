import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import NoiseScheme as N
import configuration as cfg
import math


def uniform_quantize(k):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, i):
            if k == 32:
                out = i
            elif k == 1:
                out = torch.sign(i)
            else:
                n = float(2 ** k - 1)
                out = torch.round(i * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = self.uniform_q.apply(x / E) * E
        else:
            weight = torch.tanh(x)
            max_w = torch.max(torch.abs(weight)).detach()
            weight = weight / 2 / max_w + 0.5
            weight_q = max_w * (2 * self.uniform_q.apply(weight) - 1)
        return weight_q


# quantization class for convolution layer
def conv2d_Q_fn(w_bit):
    class Conv2d_Q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=False):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)
            self.w_bit = w_bit
            # self.quantize_fn = weight_quantize_fn(w_bit=w_bit)
            self.quantize_fn = LsqQuan(w_bit)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            if cfg.config["add_noise"]:
                weight_q = N.add_noise(weight_q, m_of_dispersion=cfg.config["m_of_dispersion"],
                                       additive=cfg.config["additive"])

            return F.conv2d(input, weight_q, bias=None, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, groups=self.groups)
    return Conv2d_Q


# LSQ quantization scheme
def grad_scale(x, s):
    y = x
    y_grad = x * s
    return y.detach() - y_grad.detach() + y_grad


def roundpass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, is_activation=False):
        super(Quantizer, self).__init__()

        if all_positive:
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            self.thd_neg = - 2 ** (bit - 1) + 1
            self.thd_pos = 2 ** (bit - 1) - 1

        self.s = nn.Parameter(torch.ones(1).to(cfg.config['device']))
        self.init_state = torch.zeros(1)
        self.is_activation = is_activation

    def forward(self, x):
        s_grad_scale = 1.0 / (math.sqrt(self.thd_pos * x.numel()))
        if self.training and not self.is_activation and self.init_state == 0:
            self.s.data.copy_(2 * x.abs().mean() / math.sqrt(self.thd_pos))
            self.init_state.fill_(1)

        s_scale = grad_scale(self.s, s_grad_scale)
        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = roundpass(x)
        x = x * s_scale
        return x
        # return FunLSQ.apply(x, self.s, s_grad_scale, self.thd_neg, self.thd_pos)


class FunLSQ(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha >0
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_big - indicate_small
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle
                       * (-q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        return grad_weight, grad_alpha, None, None, None

