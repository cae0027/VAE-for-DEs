"""
Run 1D finite element over different realizations of a random variable to generate data for fine and coarse scale elements
"""

import sys
sys.path.append('../sources')

from oned_fem_solver import forward_solver
import numpy as np
import matplotlib.pyplot as plt

# Generate fine and coarse scales solutions for some realizations of the random variable
# Interval end points
a = -1  
b = 1
# Parameter function
c = 21
x0 = 0.1
d = 0.01
def q(x): return d + np.exp(c*(np.cos(2*np.pi*x / x0 + eta)))
# def q(x): return x**2
# Forcing term
def f(x): return -(2.5 + 2*x)
u_a = 0
u_b = 1

m = 2000          # number of realizations
f_elements = 2000
c_elements = 21
sample_parameter = np.zeros(m)
eta = np.random.uniform(0, 2*np.pi)
xf, fua = forward_solver(a=a, b=b, u_a=u_a, u_b=u_b,n_elements=f_elements, eta=eta, q=q, f=f)
xc, cua = forward_solver(a=a, b=b, u_a=u_a, u_b=u_b,n_elements=c_elements, eta=eta, q=q, f=f)
fine_store = np.zeros((len(xf), m))
coarse_store = np.zeros((len(xc), m))

fig = plt.figure(figsize=(10, 5))
axf = plt.subplot(1, 2, 1)
axc = plt.subplot(1, 2, 2)
for i in range(m):
    eta = np.random.uniform(0, 2*np.pi)
    sample_parameter[i] = eta
    xf, fua = forward_solver(a=a, b=b, u_a=u_a, u_b=u_b,n_elements=f_elements, eta=eta, q=q, f=f)
    xc, cua = forward_solver(a=a, b=b, u_a=u_a, u_b=u_b,n_elements=c_elements, eta=eta, q=q, f=f)
    fine_store[:, i] = fua
    coarse_store[:, i] = cua
    axf.plot(xf, fua)       # fine soln
    axc.plot(xc, cua)       # coarse soln
    plt.pause(0.05)
    print(f"Iteration {i}/{m}")
np.save('fine_scale_data_y.npy', fine_store)    # save solution as np array
np.save('coarse_scale_data_y.npy', coarse_store)    # save solution as np array
np.save('fine_scale_data_x.npy', xf)    # save x fine values
np.save('coarse_scale_data_x.npy', xc)    # save x coarse values
# save the used uniform random sample
np.save('parameter_sample.npy', sample_parameter)
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x)$")
fig.tight_layout()
plt.show()
