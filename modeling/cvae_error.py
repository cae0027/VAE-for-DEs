"""
Compute error between CVAE solution and true finite element solution using inner product induced by the stiffness matrix corresponding to the fine scale solutions
"""

import sys
sys.path.append('../sources')

from train_model import run_train, run_test
from fem_error_1d import EvalCoarseSoln
from dynamic_layers import net_inp_out_sizes
import matplotlib.pyplot as plt


# run CVAE to get reconstructed validation solutions, along with their corresponding true solns
params = net_inp_out_sizes(no_layers=3)
x, model_result, true_soln, model = run_train(epochs=50, lr=0.001, layer_params=params)

# boundary points
a = -1  
b = 1
# instantiate error class
comp = EvalCoarseSoln(a=a, b=b)

# compute error btw reconstucted solns and true solns for validation data
################################
# use this only if you want validation error
error = []
for i in range(len(model_result)):
    err = comp.error(model_result[i], true_soln[i])
    error.append(err)
plt.xlabel("Epochs")
plt.ylabel(r"$\Vert u - \hat{u}\Vert_{H^1(\Omega)}$")
plt.title("Validation Error")
plt.plot(error, color=(0.2, 0.3, 0.7, 0.8))
plt.show()
################################

# test the model, 
# turn on optim_params during hyperparameter optimization
print("Model RMSE and L2 error : ", run_test(model, optim_params=True))


