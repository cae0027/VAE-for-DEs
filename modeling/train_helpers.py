"""
Contains final_loss, fit, and validate functions necessary to couple CVAE model for training
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def fit(model, dataloaderf, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, (dataf, datac) in enumerate(dataloaderf):
        dataf = dataf.to(device)
        datac = datac.to(device)
        dataf = dataf.view(dataf.size(0), -1)
        datac = datac.view(datac.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(dataf, datac)
        mse_loss = criterion(reconstruction, dataf)
        loss = final_loss(mse_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloaderf.dataset)
    return train_loss

def validate(model, dataloaderf, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (dataf, datac) in enumerate(dataloaderf):
            dataf = dataf.to(device)
            datac = datac.to(device)
            dataf = dataf.view(dataf.size(0), -1)
            datac = datac.view(datac.size(0), -1)
            reconstruction, mu, logvar = model(dataf, datac)
            mse_loss = criterion(reconstruction, dataf)
            loss = mse_loss + criterion(reconstruction, dataf)
            # loss = criterion(reconstruction, dataf)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            batch_size = dataloaderf.batch_size
            if i == int(len(dataloaderf.dataset)/dataloaderf.batch_size) - 1:
                num_rows = 8
                both = torch.cat((dataf.view(batch_size,  dataf.shape[1])[:1],
                                  reconstruction.view(batch_size,  dataf.shape[1])[:1]))
                # save_image(both.cpu(), f"./Images/output{epoch}.png", nrow=num_rows)
                both = both.cpu().numpy().T
                # plt.plot(x, both[:, 0], label='True soln')
                # plt.plot(x, both[:, 1], label='CVAE soln')
                # plt.legend()
                # plt.show()
    val_loss = running_loss/len(dataloaderf.dataset)
    return val_loss, both[:, 0], both[:, 1]

