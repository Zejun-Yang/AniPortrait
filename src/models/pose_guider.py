import os
import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange
import numpy as np
from diffusers.models.modeling_utils import ModelMixin

from typing import Any, Dict, Optional
from src.models.attention import BasicTransformerBlock


class PoseGuider(ModelMixin):
    def __init__(self, noise_latent_channels=320, use_ca=True):
        super(PoseGuider, self).__init__()

        self.use_ca = use_ca

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Final projection layer
        self.final_proj = nn.Conv2d(in_channels=128, out_channels=noise_latent_channels, kernel_size=1)
        
        self.conv_layers_1 = nn.Sequential(
            nn.Conv2d(in_channels=noise_latent_channels, out_channels=noise_latent_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(noise_latent_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=noise_latent_channels, out_channels=noise_latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(noise_latent_channels),
            nn.ReLU(),
        )
        
        self.conv_layers_2 = nn.Sequential(
            nn.Conv2d(in_channels=noise_latent_channels, out_channels=noise_latent_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(noise_latent_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=noise_latent_channels, out_channels=noise_latent_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(noise_latent_channels*2),
            nn.ReLU(),
        )

        self.conv_layers_3 = nn.Sequential(
            nn.Conv2d(in_channels=noise_latent_channels*2, out_channels=noise_latent_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(noise_latent_channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=noise_latent_channels*2, out_channels=noise_latent_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(noise_latent_channels*4),
            nn.ReLU(),
        )
        
        self.conv_layers_4 = nn.Sequential(
            nn.Conv2d(in_channels=noise_latent_channels*4, out_channels=noise_latent_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(noise_latent_channels*4),
            nn.ReLU(),
        )
        
        if self.use_ca:
            self.cross_attn1 = Transformer2DModel(in_channels=noise_latent_channels)
            self.cross_attn2 = Transformer2DModel(in_channels=noise_latent_channels*2)
            self.cross_attn3 = Transformer2DModel(in_channels=noise_latent_channels*4)
            self.cross_attn4 = Transformer2DModel(in_channels=noise_latent_channels*4)

        # Initialize layers
        self._initialize_weights()

        self.scale = nn.Parameter(torch.ones(1) * 2)

    # def _initialize_weights(self):
    #     # Initialize weights with Gaussian distribution and zero out the final layer
    #     for m in self.conv_layers:
    #         if isinstance(m, nn.Conv2d):
    #             init.normal_(m.weight, mean=0.0, std=0.02)
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)

    #     init.zeros_(self.final_proj.weight)
    #     if self.final_proj.bias is not None:
    #         init.zeros_(self.final_proj.bias)
    
    def _initialize_weights(self):
        # Initialize weights with He initialization and zero out the biases
        conv_blocks = [self.conv_layers, self.conv_layers_1, self.conv_layers_2, self.conv_layers_3, self.conv_layers_4]
        for block_item in conv_blocks:
            for m in block_item:
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                    if m.bias is not None:
                        init.zeros_(m.bias)

        # For the final projection layer, initialize weights to zero (or you may choose to use He initialization here as well)
        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)

    def forward(self, x, ref_x):
        fea = []
        b = x.shape[0]
        
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.conv_layers(x)
        x = self.final_proj(x)
        x = x * self.scale
        # x = rearrange(x, "(b f) c h w -> b c f h w", b=b)
        fea.append(rearrange(x, "(b f) c h w -> b c f h w", b=b))
        
        x = self.conv_layers_1(x)
        if self.use_ca:
            ref_x = self.conv_layers(ref_x)
            ref_x = self.final_proj(ref_x)
            ref_x = ref_x * self.scale
            ref_x = self.conv_layers_1(ref_x)
            x = self.cross_attn1(x, ref_x)
        fea.append(rearrange(x, "(b f) c h w -> b c f h w", b=b))
        
        x = self.conv_layers_2(x)
        if self.use_ca:
            ref_x = self.conv_layers_2(ref_x)
            x = self.cross_attn2(x, ref_x)
        fea.append(rearrange(x, "(b f) c h w -> b c f h w", b=b))
        
        x = self.conv_layers_3(x)
        if self.use_ca:
            ref_x = self.conv_layers_3(ref_x)
            x = self.cross_attn3(x, ref_x)
        fea.append(rearrange(x, "(b f) c h w -> b c f h w", b=b))
        
        x = self.conv_layers_4(x)
        if self.use_ca:
            ref_x = self.conv_layers_4(ref_x)
            x = self.cross_attn4(x, ref_x)
        fea.append(rearrange(x, "(b f) c h w -> b c f h w", b=b))

        return fea

    @classmethod
    def from_pretrained(cls,pretrained_model_path):
        if not os.path.exists(pretrained_model_path):
            print(f"There is no model file in {pretrained_model_path}")
        print(f"loaded PoseGuider's pretrained weights from {pretrained_model_path} ...")

        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model = Hack_PoseGuider(noise_latent_channels=320)
                
        m, u = model.load_state_dict(state_dict, strict=True)
        # print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")        
        params = [p.numel() for n, p in model.named_parameters()]
        print(f"### PoseGuider's Parameters: {sum(params) / 1e6} M")
        
        return model
    

class Transformer2DModel(ModelMixin):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                )
                for d in range(num_layers)
            ]
        )

        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels)
        else:
            self.proj_out = nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0
            )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
    ):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, inner_dim
            )
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, inner_dim
            )
            hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )

        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        output = hidden_states + residual
        return output


if __name__ == '__main__':
    model = PoseGuider(noise_latent_channels=320).to(device="cuda")
    
    input_data = torch.randn(1,3,1,512,512).to(device="cuda")
    input_data1 = torch.randn(1,3,512,512).to(device="cuda")
    
    output = model(input_data, input_data1)
    for item in output:
        print(item.shape)
    
    # tf_model = Transformer2DModel(
    #     in_channels=320
    #     ).to('cuda')
    
    # input_data = torch.randn(4,320,32,32).to(device="cuda")
    # # input_emb = torch.randn(4,1,768).to(device="cuda")
    # input_emb = torch.randn(4,320,32,32).to(device="cuda")
    # o1 = tf_model(input_data, input_emb)
    # print(o1.shape)
