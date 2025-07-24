import torch.nn as nn
from classification.models.vmamba import LayerNorm2d
from models_mamba.Mamba_decoder import Decoder
from models_mamba.Mamba_encoder import Encoder
import torch.nn as nn
    
class STFMamba(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STFMamba, self).__init__()
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        
        self.encoder = Encoder(
            channel_first=False,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )
        self.encoder_2 = Encoder(
            channel_first=False,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_1 = Decoder(
            encoder_dims=[96,192,192*2,192*2*2],
            channel_first=False,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )
        self.decoder = Decoder(
            encoder_dims=[96,192,192*2,192*2*2],
            channel_first=False,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

    def forward(self, coarse_0, coarse_1, fine_0, def_device): 
        coarse0_fea = self.encoder(coarse_0) 
        coarse1_fea = self.encoder(coarse_1)
        fine0_fea = self.encoder(fine_0)
        #fine0_fea = self.encoder_2(fine_0) #We use shared network parameters for coarse and fine image feature extraction. If you have enough computing resources, two encoders can accelerate convergence and improve perfromence.
        
        output_1 = self.decoder_1(coarse0_fea, coarse1_fea, fine0_fea, def_device)+fine_0
        output_2 = self.decoder(coarse0_fea, coarse1_fea, fine0_fea, def_device)+coarse_1
        return output_1, output_2
