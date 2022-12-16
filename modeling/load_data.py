"""
Load coarse and fine scale data, transform to tensor and pass through pytorch data loader
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_data(coarse_path, fine_path, split_frac=0.8, batch_size=20):
    # load fine scale data
    dataf = np.load(fine_path).T.astype(np.float32)
    num_data = len(dataf)

    # load coarse scale data
    datac = np.load(coarse_path).T.astype(np.float32)

    split_frac = split_frac
    train_size = int(split_frac * num_data)

    train_dataf = dataf[:train_size]
    val_dataf = dataf[train_size:]
    train_datac = datac[:train_size]
    val_datac = datac[train_size:]
    print('Shape of training data (fine scale): ', train_dataf.shape)
    print('Shape of test data (fine scale): ', val_dataf.shape)
    print('Shape of training data (coarse scale): ', train_datac.shape)
    print('Shape of test data (coarse scale): ', val_datac.shape)

    # Prepare data for data loader
    trainf_y = TensorDataset(torch.from_numpy(train_dataf),
                        torch.from_numpy(train_datac))
    valf_y = TensorDataset(torch.from_numpy(val_dataf), torch.from_numpy(val_datac))

    trainc_y = TensorDataset(torch.from_numpy(train_datac),
                            torch.from_numpy(train_datac))
    valc_y = TensorDataset(torch.from_numpy(val_datac), torch.from_numpy(val_datac))

    # training and validation data loaders
    # fine scale loader
    train_loaderf = DataLoader(
        trainf_y,
        batch_size=batch_size,
        shuffle=True
    )
    val_loaderf = DataLoader(
        valf_y,
        batch_size=batch_size,
        shuffle=True
    )

    # coarse scale loader
    train_loaderc = DataLoader(
        trainf_y,
        batch_size=batch_size,
        shuffle=False
    )
    val_loaderc = DataLoader(
        valf_y,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loaderf, val_loaderf, train_loaderc, val_loaderc



if __name__ == '__main__':
    fine_path = '../data-gen/fine_scale_data_y.npy'
    coarse_path = '../data-gen/coarse_scale_data_y.npy'
    data = load_data(coarse_path, fine_path, split_frac=0.8, batch_size=20)
    print(data)


