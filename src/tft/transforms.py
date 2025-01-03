import torch
import einops
from pytorch_wavelets import DWTForward, DWT1DForward, DWTInverse, DWT1DInverse

class WPT1D(torch.nn.Module):
    def __init__(self, wt=DWT1DForward(J=1, mode='periodization', wave='bior4.4'), J=4):
        super().__init__()
        self.wt = wt
        self.J = J

    def analysis_one_level(self,x):
        L, H = self.wt(x)
        X = torch.cat([L.unsqueeze(2),H[0].unsqueeze(2)],dim=2)
        X = einops.rearrange(X, 'b c f ℓ -> b (c f) ℓ')
        return X

    def wavelet_analysis(self, x, J):
        for _ in range(J):
            x = self.analysis_one_level(x)
        return x

    def forward(self, x):
        return self.wavelet_analysis(x, J=self.J)


class IWPT1D(torch.nn.Module):
    def __init__(self, iwt=DWT1DInverse(mode='periodization', wave='bior4.4'), J=4):
        super().__init__()
        self.iwt = iwt
        self.J = J

    def synthesis_one_level(self, X):
        X = einops.rearrange(X, 'b (c f) ℓ -> b c f ℓ', f=2)
        L, H = torch.split(X, [1, 1], dim=2)
        L = L.squeeze(2)
        H = [H.squeeze(2)]
        y = self.iwt((L, H))
        return y

    def wavelet_synthesis(self, x, J):
        for _ in range(J):
            x = self.synthesis_one_level(x)
        return x

    def forward(self, x):
        return self.wavelet_synthesis(x, J=self.J)

class WPT2D(torch.nn.Module):
    def __init__(self, wt=DWTForward(J=1, mode='periodization', wave='bior4.4'), J=4):
        super().__init__()
        self.wt  = wt
        self.J = J
    def analysis_one_level(self,x):
        L, H = self.wt(x)
        X = torch.cat([L.unsqueeze(2),H[0]],dim=2)
        X = einops.rearrange(X, 'b c f h w -> b (c f) h w')
        return X
    def wavelet_analysis(self,x,J):
        for _ in range(J):
            x = self.analysis_one_level(x)
        return x
    def forward(self, x):
        return self.wavelet_analysis(x,J=self.J)

        
class IWPT2D(torch.nn.Module):
    def __init__(self, iwt=DWTInverse(mode='periodization', wave='bior4.4'), J=4):
        super().__init__()
        self.iwt  = iwt
        self.J = J
    def synthesis_one_level(self,X):
        X = einops.rearrange(X, 'b (c f) h w -> b c f h w', f=4)
        L, H = torch.split(X, [1, 3], dim=2)
        L = L.squeeze(2)
        H = [H]
        y = self.iwt((L, H))
        return y
    def wavelet_synthesis(self,x,J):
        for _ in range(J):
            x = self.synthesis_one_level(x)
        return x
    def forward(self, x):
        return self.wavelet_synthesis(x,J=self.J)