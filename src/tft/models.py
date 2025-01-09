import torch
import einops
from timm.models.maxxvit import TransformerBlock2d, MaxxVitTransformerCfg, LayerScale, LayerScale2d

class RMSNormAct(torch.nn.Module):
    def __init__(self, normalized_features):
        super(RMSNormAct, self).__init__()
        self.norm = torch.nn.RMSNorm(normalized_features)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        return x

class InvertedResidual1D(torch.nn.Module):
    def __init__(self, in_dim, out_dim, sequence_dim, exp_ratio):
        super(InvertedResidual1D, self).__init__()
        self.exp_dim = int(in_dim * exp_ratio)
        self.pw_exp = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, self.exp_dim, kernel_size=1, stride=1, bias=False),
            RMSNormAct((self.exp_dim, sequence_dim))
        )
        self.dw_mid = torch.nn.Sequential(
            torch.nn.Conv1d(self.exp_dim, self.exp_dim, kernel_size=3, stride=1, padding=1, groups=self.exp_dim, bias=False),
            RMSNormAct((self.exp_dim, sequence_dim))
        )
        self.se = torch.nn.Identity()
        self.pw_proj = torch.nn.Sequential(
            torch.nn.Conv1d(self.exp_dim, out_dim, kernel_size=1, stride=1, bias=False),
            torch.nn.RMSNorm((out_dim, sequence_dim)) 
        )
        self.dw_end = torch.nn.Identity()
        self.layer_scale = LayerScale(out_dim)
        self.drop_path = torch.nn.Identity()
        
    def forward(self, x):
        shortcut = x if x.shape[1] == self.pw_proj[0].out_channels else None
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)
        if shortcut is not None:
            x += shortcut
        return x

class InvertedResidual2D(torch.nn.Module):
    def __init__(self, in_dim, out_dim, spatial_dim, exp_ratio):
        super(InvertedResidual2D, self).__init__()
        self.exp_dim = int(in_dim * exp_ratio)
        self.pw_exp = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, self.exp_dim, kernel_size=1, stride=1, bias=False),
            RMSNormAct((self.exp_dim, spatial_dim, spatial_dim))
        )
        self.dw_mid = torch.nn.Sequential(
            torch.nn.Conv2d(self.exp_dim, self.exp_dim, kernel_size=3, stride=1, padding=1, groups=self.exp_dim, bias=False),
            RMSNormAct((self.exp_dim, spatial_dim, spatial_dim))
        )
        self.se = torch.nn.Identity()
        self.pw_proj = torch.nn.Sequential(
            torch.nn.Conv2d(self.exp_dim, out_dim, kernel_size=1, stride=1, bias=False),
            torch.nn.RMSNorm((out_dim, spatial_dim, spatial_dim)) 
        )
        self.dw_end = torch.nn.Identity()
        self.layer_scale = LayerScale2d(out_dim)
        self.drop_path = torch.nn.Identity()
        
    def forward(self, x):
        shortcut = x if x.shape[1] == self.pw_proj[0].out_channels else None
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)
        if shortcut is not None:
            x += shortcut
        return x

class TransformerBlock1D(torch.nn.Module):
    def __init__(self, embed_dim, dim_feedforward, nhead):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
    def forward(self, x):
        x = einops.rearrange(x, 'b c s -> b s c')
        x = self.layer(x)
        x = einops.rearrange(x, 'b s c -> b c s')
        return x

class AsCAN1D(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, sequence_dim, dim_head, exp_ratio):
        super().__init__()
        C=lambda:InvertedResidual1D(embed_dim, embed_dim, sequence_dim, exp_ratio)
        T=lambda:TransformerBlock1D(embed_dim, int(exp_ratio*embed_dim), embed_dim//dim_head)
        self.layers=torch.nn.Sequential(
            torch.nn.Conv1d(input_dim,embed_dim,kernel_size=1),
            RMSNormAct((embed_dim, sequence_dim)),
            C(),C(),C(),T(),
            C(),C(),T(),T(),
            C(),T(),T(),T()
        )
    def forward(self,x):
        return self.layers(x)

class AsCAN2D(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, spatial_dim, dim_head, exp_ratio):
        super().__init__()
        cfg = MaxxVitTransformerCfg(dim_head=dim_head)
        C=lambda:InvertedResidual2D(embed_dim, embed_dim, spatial_dim, exp_ratio)
        T=lambda:TransformerBlock2d(embed_dim,embed_dim,cfg)
        self.layers=torch.nn.Sequential(
            torch.nn.Conv2d(input_dim,embed_dim,kernel_size=1),
            RMSNormAct((embed_dim, spatial_dim, spatial_dim)),
            C(),C(),C(),T(),
            C(),C(),T(),T(),
            C(),T(),T(),T()
        )
    def forward(self,x):
        return self.layers(x)

class WaveletPooling1D(torch.nn.Module):
    def __init__(self, embed_dim, sequence_dim, wpt, num_levels):
        super().__init__()
        self.wpt = wpt
        self.num_levels = num_levels
        current_sequence_dim = sequence_dim
        self.projection_down = torch.nn.ModuleList()
        for _ in range(num_levels):
            self.projection_down.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=1, padding=0),
                    torch.nn.RMSNorm((embed_dim // 2, current_sequence_dim))
                )
            )
            current_sequence_dim //= 2
    def forward(self, x):
        for i in range(self.num_levels):
            x = self.projection_down[i](x)
            x = self.wpt.analysis_one_level(x)
        return x
        
class WaveletPooling2D(torch.nn.Module):
    def __init__(self, embed_dim, spatial_dim, wpt, num_levels):
        super().__init__()
        self.wpt = wpt
        self.num_levels = num_levels
        current_spatial_dim = spatial_dim
        self.projection_down = torch.nn.ModuleList()
        for i in range(num_levels):
            self.projection_down.append(torch.nn.Sequential(
                    torch.nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1, padding=0),
                    torch.nn.RMSNorm((embed_dim // 4, current_spatial_dim, current_spatial_dim))
                ))
            current_spatial_dim = current_spatial_dim // 2
    def forward(self, x):
        for i in range(self.num_levels):
            x = self.projection_down[i](x)
            x = self.wpt.analysis_one_level(x)
        return x

class TFTClassifier1D(torch.nn.Module):
    def __init__(self, config, wpt):
        super().__init__()
        self.wpt = wpt
        self.ascan = AsCAN1D(
            input_dim=config.channels*(2**config.J),
            embed_dim=config.embed_dim,
            sequence_dim=config.crop_size//(2**config.J),
            dim_head=config.dim_head,
            exp_ratio=config.exp_ratio
        )
        self.pool = WaveletPooling1D(
            embed_dim=config.embed_dim,
            sequence_dim=config.crop_size//(2**config.J),
            wpt=wpt,
            num_levels=int(np.log2(config.crop_size) - config.J)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv1d(config.embed_dim, config.classifier_num_classes, kernel_size=1),
            torch.nn.Flatten()
        )
    def forward(self,x):
        x = self.wpt(x)
        x = self.ascan(x)
        x = self.pool(x)
        return self.classifier(x)

class TFTClassifier2D(torch.nn.Module):
    def __init__(self, config, wpt):
        super().__init__()
        self.wpt = wpt
        self.ascan = AsCAN2D(
            input_dim=config.channels*(4**config.J),
            embed_dim=config.embed_dim,
            spatial_dim=config.crop_size//(2**config.J),
            dim_head=config.dim_head,
            exp_ratio=config.exp_ratio
        )
        self.pool = WaveletPooling2D(
            embed_dim=config.embed_dim,
            spatial_dim=config.crop_size//(2**config.J),
            wpt=wpt,
            num_levels=(config.crop_size//(2**config.J))//4
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(config.embed_dim, config.classifier_num_classes, kernel_size=1),
            torch.nn.Flatten()
        )
    def forward(self,x):
        x = self.wpt(x)
        x = self.ascan(x)
        x = self.pool(x)
        return self.classifier(x)