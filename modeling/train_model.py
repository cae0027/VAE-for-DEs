
import sys
sys.path.append('../sources')

# Local import
# from architecture import CVAE
from dynamic_layers import CVAE
from train_helpers import final_loss, fit, validate, device
from load_data import load_data
from fem_error_1d import EvalCoarseSoln
# Built in import
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

in_features, out_features = 22, 2001
batch_size = 30
# load data
fine_path = '../data-gen/fine_scale_data_y.npy'
coarse_path = '../data-gen/coarse_scale_data_y.npy'
train_loaderf, val_loaderf = load_data(coarse_path, fine_path, split_frac=0.8, batch_size=batch_size)

def run_train(in_features=22, out_features=2001, epochs=200, lr=0.0005, batch_size=batch_size, no_layers=5):
    """
    Great for easy calls during hyperparameter tuning
    """
    model = CVAE(in_features=in_features, out_features=out_features, no_layers=no_layers).to(device)
    epochs = epochs
    lr = lr
    batch_size = batch_size
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # load x for plotting
    x = np.load('../data-gen/fine_scale_data_x.npy').astype(np.float32)

    train_loss = []
    val_loss = []
    # Initialize matrix to hold solutions from validation
    model_result = np.zeros((epochs, out_features))
    soln_true = np.zeros((epochs, out_features))
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = fit(model, train_loaderf, optimizer, criterion)
        val_epoch_loss, val_out_orig, val_out_model = validate(model, val_loaderf, criterion)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        model_result[epoch, :] = val_out_model
        soln_true[epoch, :] = val_out_orig
        print(f"Train Loss: {train_epoch_loss:.8f}")
        print(f"Val Loss: {val_epoch_loss:.8f}")
        ############## for some bad lr or other params, NANs are return for train and val losses, break model run to save compuational time ###############
        if np.isnan(val_epoch_loss):
            model.isnan = np.isnan(val_epoch_loss)
            print("Model is gabbage with the chosen parameters, next...")
            break
        ####### ensure avoid testing when NAN occurs #########
        else:
            plt.clf()
            plt.plot(x, val_out_orig, label='true soln')
            plt.plot(x, val_out_model, label='cvae soln')
            plt.xlabel(r"$x$")
            plt.ylabel(r'$u(x)$')
            plt.title("Reconstruction vs True Solution")
            plt.legend()
            # plt.ylim([-35, 3])   # use only when blow up occurs
            plt.pause(0.01)
    plt.show()


    if model.isnan:
        pass
    else:
        # plot validation reconstructions
        plt.plot(x, model_result.T)
        plt.title("Validation Reconstructions")
        plt.xlabel(r"$x$")
        plt.ylabel(r'$u(x)$')
        plt.show()

        # plot  true solns
        plt.plot(x, soln_true.T)
        plt.title("True Solutions")
        plt.xlabel(r"$x$")
        plt.ylabel(r'$u(x)$')
        plt.show()
    return x, model_result, soln_true, model


# load test data both coarse and fine scales
test_data_f = np.load('../data-gen/fine_scale_test_data_y.npy').T.astype(np.float32)
test_data_c = np.load('../data-gen/coarse_scale_test_data_y.npy').T.astype(np.float32)

criterion = nn.MSELoss()
comp = EvalCoarseSoln()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_test(model, optim_params=False):
    #### ensure model is not gabbage ####
    if model.isnan:
        return float('NAN')
    else:
        error = []
        model.eval()
        print(model)
        for yc, yf in  zip(test_data_c, test_data_f):
            yc = torch.from_numpy(yc)[None, :].to(device)
            z = torch.randn((1, model.latent_dim)).to(device)
            z = torch.cat((z, yc), dim=1)

            ###### if hyperparameter optimization is on ############
            if optim_params:
                for layer in model.decoder[:-1]:
                    z = F.relu(layer(z))
                y = model.decoder[-1](z)
            else:
                y = F.relu(model.dec1(z))
                y = F.relu(model.dec2(y))
                y = F.relu(model.dec3(y))
                y = model.dec4(y)

            y = y[0, :].detach().cpu().numpy()
            # compute the norm of y
            norm_yf = comp.error(np.zeros(len(yf)), yf)
            # compute the relative error in L2 norm
            err = comp.error(y, yf) / norm_yf

            # If you prefer RMSE
            # er = (y-yf)**2 
            # err = (er).mean() / (yf**2).mean()
            # RMSE emprirically way better than H1 norm

            error.append(err**0.5)
            # plt.plot(x, y[0, :].detach().numpy())
            # plt.plot(x, yf)
        error = np.array(error)
        av_error = np.mean(error)
        print("Average relative error is: ", av_error)
        plt.hist(error, bins=40, color=(0.2, 0.3, 0.7, 0.8))
        plt.xlabel("Errors")
        plt.ylabel("Frequency")
        plt.title(r"Histogram of relative error in $L^2$ norm")
        # plt.title(r"Histogram of Relative RMSE Error")
        plt.show()
    return error


if __name__ == '__main__':
    run_train()
