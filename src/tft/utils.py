import torch
import torch.nn as nn

class Quantize(torch.nn.Module):
    def __init__(self, bits=8, eps=1e-3):
        super(Quantize, self).__init__()
        self.scale = 2**bits/2 - 1 - eps
    def forward(self, x):
        x *= self.scale
        if self.training:
            x+=torch.rand_like(x) - 0.5
        else:
            x = torch.round(x)
        x /= self.scale
        return x

def compand(x, eps=0.1, power=0.4):
    return x.sign() * ((x.abs() + eps) ** power - eps**power)

def decompand(y, eps=0.1, power=0.4):
    return y.sign() * ((y.abs() + eps**power) ** (1 / power) - eps)

class CompandGN(nn.Module):
    def __init__(self, num_features, eps=1e-7, affine=True, num_groups=8, compand_pow=0.4, compand_eps=0.1, rescale=5):
        super().__init__()
        if num_features % num_groups != 0:
            raise ValueError(f"num_features={num_features} must be divisible by num_groups={num_groups}.")
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        self.compand_pow = compand_pow
        self.compand_eps = compand_eps
        self.rescale = rescale
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("saved_mean", None)
        self.register_buffer("saved_var", None)
    def forward(self, x):
        x_c = compand(x, self.compand_eps, self.compand_pow)
        N, C, *sd = x_c.shape
        G = self.num_groups
        gs = C // G
        xr = x_c.view(N, G, gs, *sd)
        df = list(range(2, xr.dim()))
        m = xr.mean(dim=df, keepdim=True)
        v = xr.var(dim=df, keepdim=True, unbiased=False)
        self.saved_mean = m
        self.saved_var = v
        xn = (xr - m) / torch.sqrt(v + self.eps)
        xn = xn.view(N, C, *sd)
        if self.affine and self.weight is not None and self.bias is not None:
            xn = xn * self.weight.view(1, -1, *[1]*len(sd)) + self.bias.view(1, -1, *[1]*len(sd))
        return xn / self.rescale

class InvCompandGN(nn.Module):
    def __init__(self, gn_module: CompandGN):
        super().__init__()
        self.gn_module = gn_module
    def forward(self, y):
        y = y * self.gn_module.rescale
        m = self.gn_module.saved_mean
        v = self.gn_module.saved_var
        e = self.gn_module.eps
        w = self.gn_module.weight
        b = self.gn_module.bias
        cp = self.gn_module.compand_pow
        ce = self.gn_module.compand_eps
        N, C, *sd = y.shape
        G = self.gn_module.num_groups
        gs = C // G
        if self.gn_module.affine and w is not None and b is not None:
            y = (y - b.view(1, -1, *[1]*len(sd))) / w.view(1, -1, *[1]*len(sd))
        yr = y.view(N, G, gs, *sd)
        x = (yr * torch.sqrt(v + e) + m).view(N, C, *sd)
        return decompand(x, ce, cp)
