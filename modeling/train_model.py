# Local import
from architecture import CVAE
from train_helpers import final_loss, fit, validate, device
from load_data import load_data
# Built in import
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

in_features, out_features = 22, 2001
def run_model(in_features=22, out_features=2001, epochs=64, lr=0.0005, batch_size=64):
    """
    Great for easy calls during hyperparameter tuning
    """
    model = CVAE(in_features=in_features, out_features=out_features).to(device)
    epochs = epochs
    lr = lr
    batch_size = batch_size
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()


    # load data
    fine_path = '../data-gen/fine_scale_data_y.npy'
    coarse_path = '../data-gen/coarse_scale_data_y.npy'
    train_loaderf, val_loaderf, train_loaderc, val_loaderc = load_data(coarse_path, fine_path, split_frac=0.8, batch_size=batch_size)

    # load x for plotting
    x = np.load('../data-gen/fine_scale_data_x.npy').astype(np.float32)

    train_loss = []
    val_loss = []
    # Initialize matrix to hold solutions from validation
    model_result = np.zeros((epochs, out_features))
    soln_true = np.zeros((epochs, out_features))
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = fit(model, train_loaderf, train_loaderc, optimizer, criterion)
        val_epoch_loss, val_out_orig, val_out_model = validate(model, val_loaderf, val_loaderc, criterion)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        model_result[epoch, :] = val_out_model
        soln_true[epoch, :] = val_out_orig
        print(f"Train Loss: {train_epoch_loss:.8f}")
        print(f"Val Loss: {val_epoch_loss:.8f}")

        plt.clf()
        plt.plot(x, val_out_orig, label='true soln')
        plt.plot(x, val_out_model, label='cvae soln')
        plt.legend()
        plt.ylim([-36, 3])
        plt.pause(0.01)
        # plt.show()




    # plt.plot(x, model_result.T)
    plt.show()
    return x, model_result, soln_true