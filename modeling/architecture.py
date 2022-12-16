"""
Conditional variational autoencoder model architecture for reconstructing fine scale solutions of 1D elliptic differential equations from coarse scale solutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# No of coarse elements is 21 while fine is 2000, so
in_features, out_features = 22, 2001

class CVAE(nn.Module):
    def __init__(self, in_features=in_features, out_features=out_features):
        super(CVAE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # encoder
        self.enc1 = nn.Linear(in_features=out_features, out_features=out_features//2)
        self.enc2 = nn.Linear(in_features=out_features//2, out_features=out_features//4)
        self.enc3 = nn.Linear(in_features=out_features//4, out_features=in_features*2)
        

        # decoder
        self.dec1 = nn.Linear(in_features=in_features*2, out_features=out_features//4)
        self.dec2 = nn.Linear(in_features=out_features//4, out_features=out_features//4)
        self.dec3 = nn.Linear(in_features=out_features//4, out_features=out_features//2)
        self.dec4 = nn.Linear(in_features=out_features//2, out_features=out_features)

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
        x = F.relu(self.enc1(xf))
        x = F.relu(self.enc2(x))
        x = self.enc3(x).view(-1, 2, self.in_features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # decoding
        z = torch.cat((z, xc), dim=1)
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        # reconstruction = torch.sigmoid(self.dec2(x))
        reconstruction = self.dec4(x)
        return reconstruction, mu, log_var


if __name__ == '__main__':
    model = CVAE(in_features=in_features, out_features=out_features)
    print(model)
