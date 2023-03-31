"""
Random search to optimize:
    network layer size
    learning rate
    epoch
    batch size
    latent dimension
    coarse scale dimension - not yet implemented
"""

import sys
sys.path.append('../sources')

from train_model import run_train, run_test
from fem_error_1d import EvalCoarseSoln
from dynamic_layers import net_inp_out_sizes
import numpy as np
import pickle
import time
############## just for fun color printing to the terminal #######
from colorama import init as colorama_init
from colorama import Fore, Back
import colorama
#Initialize colorama
colorama.init(autoreset=True)
##################################################################

# # boundary points
# a = -1 
# b = 1
# # instantiate error class
# comp = EvalCoarseSoln(a=a, b=b)

search_result = {'num_layers':[], 'learning_rate':[], 'epoch':[], 'batch_size':[], 'latent_dim':[], 'layer_inputs':[], 'relativeRMSE':[], 'relativeL2error':[]}
# Parameters to optimize
num_layers = np.arange(1, 11)
now = time.time()
for i, layer in enumerate(num_layers):
    # run multiple times to explore different network input-output sizes or a given number of layers
    for _ in range(1000):
        params = net_inp_out_sizes(no_layers=layer)
        latent_dim = params[1][-1]
        batch = np.random.randint(5, 200)
        lr = round(np.random.uniform(1e-06, 0.1), 7)
        epoch = np.random.randint(2, 500)
        # run model at least two times to account for randomness in model weights
        for _ in range(3):
            _, _, _, model = run_train(epochs=epoch, lr=lr, batch_size=batch, layer_params=params)
            accRMSE, accL2 = run_test(model, optim_params=True)
            params_collect = [layer, lr, epoch, batch, latent_dim, model.layer_inputs, accRMSE, accL2]

            for j, param in enumerate(search_result.keys()):
                search_result[param].append(params_collect[j])
                # display current parameters
                # print(f"{param}:{params_collect[j]}", end=" ===> ")
                print(f"{param}:{params_collect[j]}")

            # save the results under the loop so you can view it even as the entire model runs
            with open("params_optim_new.pickle", 'wb') as f:
                pickle.dump(search_result, f, protocol=pickle.HIGHEST_PROTOCOL)

            current = time.time()
            print(Fore.RED+Back.GREEN+f"The time taken so far is: {round((current-now)/3600, 4)} hours")
