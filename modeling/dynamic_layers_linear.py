"""
Dyanmically control layers for:
Conditional variational autoencoder model architecture for reconstructing fine scale solutions of 1D elliptic differential equations from coarse scale solutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# No of coarse elements is 21 while fine is 2000, so
in_features, out_features = 22, 2001

class CVAE(nn.Module):
    def __init__(self, in_features=in_features, out_features=out_features, layer_params=None):
        """
        in_features:int size of coarse scale
        out_features: int size of fine scale
        layer_params: a tuple containing in and out feature sizes for the network
        """
        super(CVAE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        ###### avoid testing if model is gabbage ######
        self.isnan = False

        
        ########## encoder automated layers ##########
        self.encoder = nn.ModuleList()
        assert layer_params != None, "Please provide the input and output sizes for the network: layer_params"
        in_list, out_list = layer_params
            
        # #####################
        # # final encoder output will split into mean and variance
        # # ensure that last encoder output is divisible by 2
        # if out_list[-1] % 2 == 1:
        #     out_list[-1] = out_list[-1] + 1
        # #####################
        inputs = [out_features] + out_list
        self.layer_inputs = inputs
        for inp, oupt in zip(in_list, out_list):
            self.encoder.append(nn.Linear(inp, oupt))
        # save latent dimension
        self.latent_dim = oupt//2

        ########## decoder automated layers ##########
        self.decoder = nn.ModuleList()
        # increment in-out to account for coarse scale input
        in_ls, out_ls = np.array(out_list[::-1])+in_features, np.array(in_list[::-1])+in_features
        # ensure last out has out_features 
        out_ls[-1] = out_features
        # ensure first decoder input size is half encoder last output size
        in_ls[0] = in_ls[0] - self.latent_dim

        for inp, oupt in zip(in_ls, out_ls):
            self.decoder.append(nn.Linear(inp, oupt))


    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, xf, xc):
        # encoding
        for layer in self.encoder[:-1]:
            xf = F.relu(layer(xf))
        x = self.encoder[-1](xf).view(-1, 2, self.latent_dim)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # decoding
        z = torch.cat((z, xc), dim=1)
        for layer in self.decoder[:-1]:
            z = F.relu(layer(z))
        # reconstruction = torch.sigmoid(self.dec2(x))
        reconstruction = self.decoder[-1](z)
        return reconstruction, mu, log_var
    
    

# def net_inp_out_sizes(no_layers=6):
#     """
#     accepts number of layers and generates VAE layers input and output integers for number of layers hyperparameter optimization
#     """
#     no_layers = no_layers
#     input_size = out_features

#     a = []
#     for i in range(no_layers):
#         a.append(input_size//2**(i+1))
#     a.append(1)
#     a = a[::-1]
#     left = [random.randint(a[i], a[i+1]) for i in range(len(a)-1)]
#     if len(left) == 1:
#         sometimes = [random.randint(a[-1], a[-1])]
#     else:
#         sometimes = [random.randint(a[i+1], a[i+2]) for i in range(len(a)-2)]
#     sometimes.append(random.randint(sometimes[-1], input_size))
#     # use sometimes with 2/10 probability to explore higher dimensions
#     choice = random.randint(0,10)
#     if choice < 2:
#         left = sometimes[:]
#     right = left[:]
#     left.pop(0)
#     left.append(input_size)
#     # ensure ouput match for no_layer=1
#     if no_layers==1 and len(left)>1:
#         return [left[-1]], [right[0]]
#     return left[::-1], right[::-1]

def net_inp_out_sizes_linear(layers=2, latent_bd=10, in_features=out_features):
    m = layers      # number of layers
    ell = latent_bd    # lower bound for latent dimension
    n = in_features    # input size
    if m == 1:
        inp = n
        out = np.random.randint(ell, n)
        # ensure the latent dimension is divisible by 2 for mean
        # and for variance
        if out % 2 == 1:
            out = out + 1
        return [n], [out]
    else:
        f = lambda x: ((n-ell)*x - n*m + ell)/(1-m)

        delm = (m-1)/m
        x = np.arange(1, m+1e-5, delm)
        # list of inputs to the encoder
        enc_dec_bds = f(x).astype('int')
        enc_dec_ls = [n]
        for i in range(len(enc_dec_bds)-1):
            inp = np.random.randint(enc_dec_bds[i+1], enc_dec_bds[i])
            enc_dec_ls.append(inp)
        # ensure the last output is divisible by two each for var and mean
        if enc_dec_ls[-1] % 2 == 1:
            enc_dec_ls[-1] = enc_dec_ls[-1] + 1
        # split into ins and outs
        enc_in, enc_out = enc_dec_ls[:-1], enc_dec_ls[1:]
        # enc_in, enc_out
    return enc_in, enc_out

params = net_inp_out_sizes_linear(layers=3)
if __name__ == '__main__':
    model = CVAE(in_features=in_features, out_features=out_features, layer_params=params)
    print(model)
