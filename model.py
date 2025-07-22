import torch.nn as nn
from models_mamba.MambaFusion import STFMamba
from models_mamba.MambaSR import MambaSR
from models_mamba.cross_attention import Cross_MultiAttention

class model_STF(nn.Module):
    def __init__(self):
        super(model_STF, self).__init__()
        self.model_SR = MambaSR(
            pretrained= None,
            patch_size=4, 
            in_chans= 6, 
            num_classes=1000, 
            depths=[2, 2, 9, 2], 
            dims=96, 
            # ===================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank=2,
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # ===================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v2",
            patchembed_version="v2",
            gmlp=False,
            use_checkpoint=False,
            )
        
        self.model_fusion = STFMamba(
            pretrained= None,
            patch_size=4, 
            in_chans= 6, 
            num_classes=1000, 
            depths=[2, 2, 9, 2], 
            dims=96, 
            # ===================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank=2,
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v2",
            patchembed_version="v2",
            gmlp=False,
            use_checkpoint=False,
            )
        
        self.cross_fusion = Cross_MultiAttention(in_channels=6, emb_dim=16, num_heads=4) 
        
        
    def forward(self, ref_lr, data, ref_target, def_device):
        SR1 = self.model_SR(ref_lr)
        SR2 = self.model_SR(data)
        out_1,out_2 = self.model_fusion(SR1, SR2, ref_target, def_device)
        fused_imgs = self.cross_fusion(out_1,out_2)
        return SR1,SR2,out_1,out_2,fused_imgs

