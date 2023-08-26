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
# from dynamic_layers import net_inp_out_sizes 
from dynamic_layers_linear import net_inp_out_sizes_linear
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

search_result = {'num_layers':[], 'learning_rate':[], 'epoch':[], 'batch_size':[], 'latent_dim':[], 'layer_inputs':[], 'relativeRMSE':[], 'relativeL2error':[]}
# Parameters to optimize
num_layers = np.arange(1, 20)
now = time.time()    
# run multiple times to explore different network input-output sizes or a given number of layers
for _ in range(5000):
    layer = np.random.choice(num_layers)
    # params = net_inp_out_sizes(no_layers=layer)
    # explore latent dimension for each layer
    latent_bd = np.random.randint(1, 1500)
    params = net_inp_out_sizes_linear(layers=layer, latent_bd=latent_bd)
    latent_dim = params[1][-1]
    batch = np.random.randint(5, 200)
    lr = round(np.random.uniform(1e-06, 0.0015), 7)
    epoch = np.random.randint(2, 500)
    # run model at least two times to account for randomness in model weights
    for _ in range(1):
        _, _, _, model = run_train(epochs=epoch, lr=lr, batch_size=batch, layer_params=params)
        accRMSE, accL2 = run_test(model, optim_params=True)
        params_collect = [layer, lr, epoch, batch, latent_dim//2, model.layer_inputs, accRMSE, accL2]

        for j, param in enumerate(search_result.keys()):
            search_result[param].append(params_collect[j])
            # display current parameters
            # print(f"{param}:{params_collect[j]}", end=" ===> ")
            print(f"{param}:{params_collect[j]}")

        # save the results under the loop so you can view it even as the entire model runs
        with open("params_optim_linear_new_ar.pickle", 'wb') as f:
            pickle.dump(search_result, f, protocol=pickle.HIGHEST_PROTOCOL)

        current = time.time()
        print(Fore.RED+Back.GREEN+f"The time taken so far is: {round((current-now)/3600, 4)} hours")


"""
###################### July 13, 2023 ###############################
1. As at July 13, model has run for 16 days yet only two layers (4 and 12) are been explored. The reason is cos
   of the for loop that runs over the layers before the main for loop that optimizes all other parameters. This 
   is not practical for exploring all potential layer sizes.
   As at this day, this for loop is removed and the number of layers choice is made within the main for loop.
   The goal is to explore all the layer sizes as time goes. 
###################### End July 13, 2023 Report ####################
"""