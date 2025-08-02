import torch.nn as nn
from models_mamba.model_VSS.vmamba import VSSBlock

class MambaSRmodel(nn.Module):
    def __init__(self, channel_first=False, norm_layer="LN", ssm_act_layer="silu", mlp_act_layer="gelu", **kwargs):
        super(MambaSRmodel, self).__init__()
        self.conv_first = nn.Conv2d(6, 12, 3, 2, 1)
        self.sr_block = nn.ModuleList()
        for _ in range(2):  
            self.sr_block.extend([
            nn.Identity(),
            VSSBlock(
            hidden_dim=64, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']
                ),
            Permute(0, 2, 3, 1),
            VSSBlock(
            hidden_dim=12, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']
                ),
            Permute(0, 3, 1, 2),
                ])
        self.conv_final = nn.ConvTranspose2d( in_channels=12, out_channels=6, kernel_size=4, 
    stride=2, padding=1)
        

    def forward(self, x):
        input = x
        x = self.conv_first(x)
        output = x
        for index, module in enumerate(self.sr_block):
            output = module(output)
        x = output+x
        output = x
        for index, module in enumerate(self.sr_block):
            output = module(output)
        x = output+x
        output = x
        for index, module in enumerate(self.sr_block):
            output = module(output)
        x = output+x
        output = x
        for index, module in enumerate(self.sr_block):
            output = module(output)
        output = output+x
        output = self.conv_final(output)
        output = input+output
        return output

