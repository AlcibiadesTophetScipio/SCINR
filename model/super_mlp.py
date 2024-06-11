import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Callable, Union

"""
Refer to https://github.com/LeeHW-THU/A-LIIF.
"""

class AdptiveMLP(nn.Module):
    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_layers: List[int],
            group_num: int = 10,
            bias: bool = True,
            act_func: Union[nn.Module, str] = 'relu',
            **kwargs
        ) -> None:
        super().__init__()

        if act_func == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_func == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif isinstance(act_func, nn.Module):
            self.act = act_func
        else:
            raise Exception("Invaild activation function.")

        layers_w, layers_b = [], []

        layers_w.append(nn.Parameter(
            torch.empty(group_num, in_features, hidden_layers[0]),
            requires_grad=True)
        )
        if len(hidden_layers)>1:
            for i in range(len(hidden_layers)-1):
                layers_w.append(nn.Parameter(
                        torch.empty(group_num, hidden_layers[i], hidden_layers[i+1]),
                        requires_grad=True)
                )
        layers_w.append(nn.Parameter(
                        torch.empty(group_num, hidden_layers[-1], out_features),
                        requires_grad=True)
        )

        if bias:
            layers_b.append(nn.Parameter(
                    torch.empty(group_num, hidden_layers[0]),
                    requires_grad=True)
            )
            if len(hidden_layers)>1:
                for i in range(len(hidden_layers)-1):
                    layers_b.append(nn.Parameter(
                            torch.empty(group_num, hidden_layers[i+1]),
                            requires_grad=True)
                    )
            layers_b.append(nn.Parameter(
                            torch.empty(group_num, out_features),
                            requires_grad=True)
            )

        self.layers_w, self.layers_b = nn.ParameterList(layers_w), nn.ParameterList(layers_b)
        self.bias = bias
        self.group_num = group_num
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for i in range(len(self.layers_w)):
            nn.init.kaiming_uniform_(self.layers_w[i])
            if self.bias:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.layers_w[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.layers_b[i], -bound, bound)

    def forward(
            self,
            input: torch.Tensor,
            co_mat: torch.Tensor,
            flag: str ='train',
            **kwargs
        ) -> dict:
        """
        Args:
            input: :math: `(*, C)`
            co_mat: :math: `(*, N)`
        """

        x = input
        for i in range(len(self.layers_w)):
            w = self.layers_w[i]
            dim_in, dim_out = w.shape[-2], w.shape[-1]
            w = w.reshape(self.group_num, -1)
            w = torch.matmul(co_mat, w).reshape(-1, dim_in, dim_out)
            x = torch.sum(x.unsqueeze(-1)*w, dim=-2)

            if self.bias:
                b = self.layers_b[i]
                dim_out = b.shape[-1]
                b = b.reshape(self.group_num, -1)
                b = torch.matmul(co_mat, b).reshape(-1, dim_out)
                x += b

            if i < len(self.layers_w)-1:
                x = self.act(x)

        return {'output': x}

"""
Refer to https://github.com/vsitzmann/siren/blob/master/modules.py
"""
class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        # return torch.sin(30 * input)
        return torch.sin(input)

class ModulatedMLP(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            z_length: int,
            hidden_layers: List[int],
            act_func: Union[nn.Module, str] = 'relu',
            **kwargs
        ) -> None:
        super().__init__()

        if act_func == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_func == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif isinstance(act_func, nn.Module):
            self.act = act_func
        else:
            raise Exception("Invaild activation function.")
        
        sin_act = Sine()
        mod_net = nn.ModuleList([nn.Sequential(nn.Linear(z_length, hidden_layers[0]), self.act)])
        sync_net = nn.ModuleList([nn.Sequential(nn.Linear(in_features, hidden_layers[0]), sin_act)])
        h_num_pre = hidden_layers[0]
        for h_num in hidden_layers[1:]:
                mod_net.append(
                    nn.Sequential(nn.Linear(h_num_pre+z_length, h_num), self.act)
                )
                sync_net.append(
                    nn.Sequential(nn.Linear(h_num_pre, h_num), sin_act)
                )
                h_num_pre = h_num
        sync_net.append(nn.Linear(h_num_pre, out_features))

        self.sync_net, self.mod_net = sync_net, mod_net
        assert len(self.sync_net) == len(self.mod_net)+1

    def forward(self, z_code, pos_embedding):
        a = self.mod_net[0](z_code)
        h = a*self.sync_net[0](pos_embedding)
        for i in range(len(self.mod_net[1:])):
            a = self.mod_net[i+1](torch.concat([a, z_code], dim=-1))
            h = a*self.sync_net[i+1](h)
        out = self.sync_net[-1](h)

        return out

if __name__ == '__main__':

    device = torch.device('cuda')

    input_data = torch.randn([16,60,80,3])
    co_mat = torch.rand(16,60,80,10)
    co_mat = F.normalize(co_mat, p=1, dim=-1)

    adptiveM = AdptiveMLP(
        3,3,[16,16,16]
    ).to(device)

    output = adptiveM(
        input_data.reshape(-1,3).to(device),
        co_mat.reshape(-1,10).to(device),
    )
    
    import sys
    sys.path.append('..')
    from SCINR.utils import get_norm_grid
    from SCINR.model.harmonic_embedding import HarmonicEmbedding

    z_code = torch.randn([16,512,60,80])
    norm_pos = get_norm_grid([60,80]).reshape(1,2,60,80).expand(16,2,60,80)
    posEmbedder = HarmonicEmbedding(16)
    num_pos = posEmbedder.get_output_dim(2)
    pos_embedding = posEmbedder(norm_pos.permute(0,2,3,1))
    modM = ModulatedMLP(
        num_pos,4,512,
        [512,512,512]
    )
    print(modM)
    output = modM(z_code=z_code.permute(0,2,3,1), pos_embedding=pos_embedding)

    pass