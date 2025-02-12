import torch

def compand(x, eps=0.1, power=0.4):
    return x.sign() * ((x.abs()+eps)**power - eps**power)
def decompand(y, eps=0.1, power=0.4):
    return y.sign() * ((y.abs()+eps**power)**(1/power) - eps)