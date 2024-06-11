import torch
from torch import nn
from torchmeta.modules import MetaModule, MetaSequential
from collections import OrderedDict
from fastai.torch_core import apply_init


class BatchLinear(nn.Linear, MetaModule):
    '''
        Source: https://github.com/vsitzmann/light-field-networks/blob/master/custom_layers.py
        A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
        hypernetwork.
    '''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        if bias is not None:
            output += bias.unsqueeze(-2)
        return output

class FCBlocks(MetaModule):
    '''
        Modified from: https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py
        Basic block for hypernet / hyponet
    '''
    def __init__(self, in_features:int, out_features:int, hidden_size:int, 
                 num_hidden_layers:int, act_func='relu', 
                 outp_layer=False, init_func=None, **kwargs):
        super().__init__()
        if init_func is None:
            init_func = apply_init
        if act_func == 'relu':
            act_func = nn.ReLU(inplace=True)
        
        layers = [MetaSequential(BatchLinear(in_features, hidden_size), act_func)]
        
        for i in range(num_hidden_layers-1):
            layers.append(MetaSequential(BatchLinear(hidden_size,hidden_size), act_func))

        if outp_layer:
            layers.append(MetaSequential(BatchLinear(hidden_size, out_features)))

        self.model = MetaSequential(*layers)
        init_func(self.model, nn.init.kaiming_normal_)
        
    def forward(self, inputs, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        return self.model(inputs, params=self.get_subdict(params, 'model'))


class HypoNet(MetaModule):
    def __init__(self, in_features=3, out_features=3, hidden_size=256, 
                 num_hidden_layers=3, act_func='relu', **kwargs):
        super().__init__()
            
        self.model = FCBlocks(in_features, out_features, hidden_size, num_hidden_layers,
                              act_func=act_func, **kwargs)
    
    def forward(self, inputs, params=None):
        if params is None: # just copy torchmeta for non-hyper useage, it will just be a regular MLP
            params = OrderedDict(self.named_parameters())
        
        
        return self.model(inputs, self.get_subdict(params, 'model')).squeeze(1)


class FCLayers(nn.Module):
    def __init__(self, in_features:int, out_features:int, hidden_size:int, 
                 num_hidden_layers:int, act_func='relu', 
                 outp_layer=False, init_func=None, **kwargs):
        super().__init__()
        if init_func is None:
            init_func = apply_init
        if act_func == 'relu':
            act_func = nn.ReLU(inplace=True)
        
        layers = [nn.Sequential(nn.Linear(in_features, hidden_size), act_func)]
        
        for i in range(num_hidden_layers-1):
            layers.append(nn.Sequential(nn.Linear(hidden_size,hidden_size), act_func))

        if outp_layer:
            layers.append(nn.Sequential(nn.Linear(hidden_size, out_features)))

        self.model = nn.Sequential(*layers)
        init_func(self.model, nn.init.kaiming_normal_)
        
    def forward(self, inputs):
        
        return self.model(inputs)

class HyperNet(nn.Module):
    '''
        Take an embedding and generate weights for hyponet
        
        Parameters:
            embedding_size: size of latent vector caputured by encoder
            num_hidden_layers: number of hidden layers (level of heads that take emb to generate weights for hypo)
            hidden_dim: dim of hyper net
            hypo_arch: torchmeta HypoModule class
        
        Init:
            names: Parameter names, eg: model.model.0.0.weight
            param_shapes: shape of the current param matrix, according to names
            layers: each of which hypernet arch for "a" hyponet layer consist of 3 layers,
                    input layer with shape emb_size to hidden, hidden to hidden, hidden to output shape
                    eg: if you want a hypo layer with weight 256,256, bias 256, then you will have hyper net arch
                    Weight: Metalinear(emb_size, hidden), Relu, Metalinear(hidden,hidden), Relu, Metalinear(hidden, 256*256)
                    Bias: Metalinear(emb_size, hidden), Relu, Metalinear(hidden,hidden), Relu, Metalinear(hidden, 256)
        
        return:
            dict: with hyponet weight and bias
        
    '''
    def __init__(self, embedding_size, num_hidden_layers, hidden_dim, hypo_arch, init=None):
        super().__init__()
        
        self.names, self.param_shapes = [], [] #place holder for layer name and size
        self.layers = nn.ModuleList() # just like a python list
        for name, param in hypo_arch.meta_named_parameters(): #get hyponet structure
            self.names.append(name)
            self.param_shapes.append(param.shape)
            
            ### create hypernet arch based on hyponet ###
            
            '''
            output shape, if hyponet layer expect weight matrix [3,10], 
            then we need to outp shape 30 then reshape to fill the matrix
            ''' 
            outp_size = torch.numel(param) 
            self.layers.append(FCLayers(embedding_size, outp_size, hidden_dim, 
                                        num_hidden_layers, outp_layer=True))
            
            if init is None:
                apply_init(self.layers[-1].model[-1], nn.init.kaiming_normal_)
            else:
                raise NotImplementedError
    
    def forward(self, emb):
        params = OrderedDict() # holder for all hypo layers
        for name, layer, param_shape in zip(self.names, self.layers, self.param_shapes):
            '''
                reshape back to hypo layer, previously outp shape is r*c, now we need to flatten it back
                correct shape final: bs, row, col, row / col are shape of weight matrix in hyponet single layer
                for bias layer, the shape should be bs, col (single vector)
                
            ''' 
            params[name] = layer(emb).view([-1] + list(param_shape))#.squeeze()
        
        return params