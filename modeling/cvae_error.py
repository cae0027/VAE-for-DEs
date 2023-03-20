"""
Compute error between CVAE solution and true finite element solution using inner product induced by the stiffness matrix corresponding to the fine scale solutions
"""

import sys
sys.path.append('../sources')

from train_model import run_train, run_test
from fem_error_1d import EvalCoarseSoln
import matplotlib.pyplot as plt


# run CVAE to get reconstructed solutions, along with true solns
x, model_result, true_soln, model = run_train(epochs=300)

# boundary points
a = -1  
b = 1
# instantiate error class
comp = EvalCoarseSoln(a=a, b=b)

# compute error btw reconstucted solns and true solns
error = []
for i in range(len(model_result)):
    err = comp.error(model_result[i], true_soln[i])
    error.append(err)
plt.plot(error)
plt.show()

run_test(model)
