import torch
import torch.nn as nn
from models_mamba.model_VSS.vmamba import VSSBlock, Permute

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=6, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  
        x1 = x[:, 1::2, 0::2, :]  
        x2 = x[:, 0::2, 1::2, :]  
        x3 = x[:, 1::2, 1::2, :]  

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  
        x = x.view(B, H//2, W//2, 4 * C)  

        x = self.norm(x)
        x = self.reduction(x)

        return x.permute(0, 3, 1, 2)

class Encoder(nn.Module):
    def __init__(self, channel_first=False, norm_layer="LN", ssm_act_layer="silu", mlp_act_layer="gelu", **kwargs):
        super(Encoder, self).__init__()

        self.patch_emb = PatchEmbed2D()
        self.downsample_1 = PatchMerging2D(96) 
        self.st_block_1 = nn.ModuleList()
        for _ in range(2):  
            self.st_block_1.extend([
                Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
                VSSBlock(
            hidden_dim=96, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']
                ),
                Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
                nn.Identity() if not channel_first else Permute(0, 2, 3, 1),
                VSSBlock(
            hidden_dim=32, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']
                ),
                nn.Identity() if not channel_first else Permute(0, 3, 1, 2),
                ])
            
        self.downsample_2 = PatchMerging2D(192) 
        self.st_block_2 = nn.ModuleList()
        for _ in range(2):  
            self.st_block_2.extend([
                Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
                VSSBlock(
            hidden_dim=192, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']
                ),
                Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
                nn.Identity() if not channel_first else Permute(0, 2, 3, 1),
                VSSBlock(
            hidden_dim=16, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'] ),
                nn.Identity() if not channel_first else Permute(0, 3, 1, 2),
                ])
        self.downsample_3 = PatchMerging2D(192*2) 
        self.st_block_3 = nn.ModuleList()
        for _ in range(2):  
            self.st_block_3.extend([
                Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
                VSSBlock(
            hidden_dim=192*2, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']
                ),
                Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
                nn.Identity() if not channel_first else Permute(0, 2, 3, 1),
                VSSBlock(
            hidden_dim=8, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'] ),
                nn.Identity() if not channel_first else Permute(0, 3, 1, 2),
                ])
        self.st_block_4 = nn.ModuleList()
        for _ in range(1):  
            self.st_block_4.extend([
                Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
                VSSBlock(
            hidden_dim=192*4, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
                Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
                nn.Identity() if not channel_first else Permute(0, 2, 3, 1),
                VSSBlock(
            hidden_dim=4, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
                nn.Identity() if not channel_first else Permute(0, 3, 1, 2),
                ])

    def forward(self, x):
        output = []
        x = self.patch_emb(x)
        for index, module in enumerate(self.st_block_1):
            x = module(x)
        output.append(x)
        x = self.downsample_1(x)
        for index, module in enumerate(self.st_block_2):
            x = module(x)
        output.append(x)
        x = self.downsample_2(x)
        for index, module in enumerate(self.st_block_3):
            x = module(x)
        output.append(x)
        x = self.downsample_3(x)
        for index, module in enumerate(self.st_block_4):
            x = module(x)
        output.append(x)
        return output

