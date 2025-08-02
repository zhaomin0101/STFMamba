import copy
import torch.nn as nn
from classification.models.vmamba import VSSBlock, Permute

class MambaSRmodel(nn.Module):
    def __init__(self, channel_first=False, norm_layer="LN", ssm_act_layer="silu", mlp_act_layer="gelu", **kwargs):
        super(MambaSRmodel, self).__init__()
        
        hidden_dims = [12,64,12] 
        self.conv_first = nn.Conv2d(6, hidden_dims[0], kernel_size=3, stride=2, padding=1)
    
        self.sr_block = nn.ModuleList([
            nn.Sequential(
                VSSBlock(
                    hidden_dim=hidden_dims[1], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                    ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'],
                    ssm_act_layer=ssm_act_layer, ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                    ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                    forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], 
                    mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                    gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']
                ),
                Permute(0, 2, 3, 1),
                VSSBlock(
                    hidden_dim=hidden_dims[2], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                    ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'],
                    ssm_act_layer=ssm_act_layer, ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                    ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                    forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], 
                    mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                    gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']
                ),
                Permute(0, 3, 1, 2)
            )
            for _ in range(2)
        ])
        self.conv_final = nn.ConvTranspose2d(hidden_dims[0], 6, kernel_size=4, stride=2, padding=1)
        self.sr_block_list = nn.ModuleList([copy.deepcopy(self.sr_block) for _ in range(4)])
    def forward(self, x):
        input_residual = x
        x = self.conv_first(x)
        for i in range(4):
            residual = x
            for block in self.sr_block_list[i]:
                x = block(x)
            x = x + residual
        x = self.conv_final(x)
        x = x + input_residual
        return x
