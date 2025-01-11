import torch
from torchvision.transforms.v2 import RandomCrop

def shrink_1d(x, batch=False):
    if not batch:
        x = x.unsqueeze(0)
    ℓ = x.shape[2]
    s = 0.2*torch.rand(1).item()
    x = torch.nn.ReflectionPad1d(padding=int(ℓ*s))(x)
    x = torch.nn.functional.interpolate(x,ℓ,mode='linear')
    if not batch:
        x = x[0]
    return x

def shrink_2d(x, batch=False):
    if not batch:
        x = x.unsqueeze(0)
    h = x.shape[2]; w = x.shape[3]
    sh = 0.5*torch.rand(1).item(); T = int(h*sh)//2; B = int(h*sh) - T
    sw = 0.5*torch.rand(1).item(); L = int(w*sw)//2; R = int(w*sw) - L
    x = torch.nn.ReflectionPad2d(padding=(L,R,T,B))(x)
    x = torch.nn.functional.interpolate(x,(h,w),mode='bilinear')
    if not batch:
        x = x[0]
    return x

def expand_1d(x, batch=False):
    if not batch:
        x = x.unsqueeze(0)
    ℓ = x.shape[2]
    s = 0.2*torch.rand(1).item()
    x = torch.nn.functional.interpolate(x,int(ℓ*(1+s)),mode='linear')
    x = RandomCrop((1,ℓ))(x.unsqueeze(1))[:,0]
    if not batch:
        x = x[0]
    return x

def expand_2d(x, batch=False):
    if not batch:
        x = x.unsqueeze(0)
    h = x.shape[2]; w = x.shape[3]
    sh = 0.5*torch.rand(1).item(); H = int(h*(1+sh))
    sw = 0.5*torch.rand(1).item(); W = int(w*(1+sw))
    x = torch.nn.functional.interpolate(x,(H,W),mode='bilinear')
    x = RandomCrop((h,w))(x)
    if not batch:
        x = x[0]
    return x

def filter_1d(x, batch=False):
    if not batch:
        x = x.unsqueeze(0)
    g = 2*torch.rand(5)-1
    g = torch.cat([g,torch.zeros(1),-g])
    g = g.view(1, 1, -1)
    B, C, L = x.shape
    x = torch.nn.functional.conv1d(
        x,
        weight=g.repeat(C, 1, 1),
        bias=None,
        stride=1,
        padding='same',
        groups=C
    )
    if not batch:
        x = x[0]
    return x


def filter_2d(x, batch=False):
    if not batch:
        x = x.unsqueeze(0)
    horizontal_filter = 0.2 + torch.rand(5)-0.5
    horizontal_filter = horizontal_filter.view(1, 1, 1, -1)
    vertical_filter = 0.2 + torch.rand(5)-0.5
    vertical_filter = vertical_filter.view(1, 1, -1, 1)
    B, C, H, W = x.shape
    x = torch.nn.functional.conv2d(
        x,
        weight=horizontal_filter.repeat(C, 1, 1, 1),
        bias=None,
        stride=1,
        padding='same',
        groups=C
    )
    x = torch.nn.functional.conv2d(
        x,
        weight=vertical_filter.repeat(C, 1, 1, 1),
        bias=None,
        stride=1,
        padding='same',
        groups=C
    )
    if not batch:
        x = x[0]
    return x

def aug(x, batch=False):
    if not batch:
        x = x.unsqueeze(0)
    if x.ndim == 3:
        operation = torch.randint(0, 2, (1,)).item()
        if operation == 0:
            x = shrink_1d(x, batch=True)
        else:
            x = expand_1d(x, batch=True)
        x = filter_1d(x, batch=True)
    elif x.ndim == 4:
        operation = torch.randint(0, 2, (1,)).item()
        if operation == 0:
            x = shrink_2d(x, batch=True)
        else:
            x = expand_2d(x, batch=True)
        x = filter_2d(x, batch=True)
    else:
        raise ValueError("Input tensor must have 3 (1D) or 4 (2D) dimensions.")
    if not batch:
        x = x[0]
    return x

def crop_1d(x, ℓ, batch=False):
    if not batch:
        x = x.unsqueeze(0)
    if x.shape[-1]<ℓ:
        r = 2 + (ℓ - x.shape[-1])//x.shape[-1]
        x = x.repeat(1, 1, r)
        print(r)
        print (x.shape)
    x = RandomCrop((1,ℓ))(x.unsqueeze(1))[:,0]
    if not batch:
        x = x[0]
    return x


def reflect_pad_horizontal(x, W):
    _, _, h, w = x.shape
    while w < W:
        if torch.rand(1).item()<0.5:
            x = torch.cat([x,x.flip(dims=(3,))],dim=3)
        else:
            x = torch.cat([x.flip(dims=(3,)),x],dim=3)
        w *= 2
    return x

def reflect_pad_vertical(x, H):
    _, _, h, w = x.shape
    while h < H:
        if torch.rand(1).item()<0.5:
            x = torch.cat([x,x.flip(dims=(2,))],dim=2)
        else:
            x = torch.cat([x.flip(dims=(2,)),x],dim=2)
        h *= 2
    return x
    
def crop_2d(x, h, w, batch=False):
    if not batch:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    x = reflect_pad_horizontal(x,w)
    x = reflect_pad_vertical(x,h)
    x = RandomCrop((h, w))(x)
    if not batch:
        x = x[0]
    return x