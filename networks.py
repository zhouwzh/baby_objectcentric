from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# from utils import get_activation_fn, init_parameters
import utils
import timm.layers.pos_embed

DEFAULT_WEIGHT_INIT = "default"

class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        initial_layer_norm: bool = False,
        activation: Union[str, nn.Module] = "relu",
        final_activation: Union[bool, str] = False,
        residual: bool = False,
        weight_init: str = DEFAULT_WEIGHT_INIT,
        frozen: bool = False,
    ):
        super().__init__()
        self.residual = residual
        if residual:
            assert inp_dim == outp_dim

        layers = []
        if initial_layer_norm:
            layers.append(nn.LayerNorm(inp_dim))

        cur_dim = inp_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(cur_dim, dim))
            layers.append(utils.get_activation_fn(activation))
            cur_dim = dim

        layers.append(nn.Linear(cur_dim, outp_dim))
        if final_activation:
            if isinstance(final_activation, bool):
                final_activation = "relu"
            layers.append(utils.get_activation_fn(final_activation))

        self.layers = nn.Sequential(*layers)
        utils.init_parameters(self.layers, weight_init)

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        outp = self.layers(inp)

        if self.residual:
            return inp + outp
        else:
            return outp
        
class MLPDecoder(nn.Module):
    """Decoder that reconstructs independently for every position and slot."""

    def __init__(
        self,
        inp_dim: int,                                       # input dimenstion of each slots
        outp_dim: int,                                      # output dimention of each output patch
        hidden_dims: List[int],
        n_patches: int, 
        activation: str = "relu",
        eval_output_size: Optional[Tuple[int]] = None,      # for resample 
        frozen: bool = False,
    ):
        super().__init__()
        self.outp_dim = outp_dim
        self.n_patches = n_patches
        self.eval_output_size = list(eval_output_size) if eval_output_size else None

        self.mlp = MLP(                            # last output_dim for attention
            inp_dim, outp_dim + 1, hidden_dims, activation=activation, frozen=frozen   
        )                                                   
        self.pos_emb = nn.Parameter(torch.randn(1, 1, n_patches, inp_dim) * inp_dim**-0.5)
        if frozen:
            self.pos_emb.requires_grad = False

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs, n_slots, dims = slots.shape

        if not self.training and self.eval_output_size is not None:  # evaluateing and testing
            pos_emb = timm.layers.pos_embed.resample_abs_pos_embed(  # 用于对VIT中APE absolute position embeddings在不同空间网格尺寸之间进行插值重采样
                self.pos_emb.squeeze(1),                             # (1,1,P,D) -> (1,P,D)
                new_size=self.eval_output_size,
                num_prefix_tokens=0,                                 # 多少个token不参与空间重采样
            ).unsqueeze(1)                                           # (1,H*W,D) -> (1,1,H*W,D)
        else:
            # print("<===Using training position embeddings===>")
            pos_emb = self.pos_emb

        slots = slots.view(bs, n_slots, 1, dims).expand(bs, n_slots, pos_emb.shape[2], dims)  #(B,K,D) -> (B,K,1,D) -> (B,K,P/H*W,D)
        slots = slots + pos_emb

        recons, alpha = self.mlp(slots).split((self.outp_dim, 1), dim=-1)

        masks = torch.softmax(alpha, dim=1)          # (B, K, P)
        recon = torch.sum(recons * masks, dim=1)     # (B, P, outp_dim)

        return {"reconstruction": recon, "masks": masks.squeeze(-1)}
    
