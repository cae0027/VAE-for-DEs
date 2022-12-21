"""
Run 1D finite element forward solver over different mesh sizes and compute error between the fine and coarse scale solutions. Here, fine mesh is fixed at 2000 elements which has e-07 error
"""

import sys
sys.path.append('../../sources')

from oned_fem_solver import forward_solver
import numpy as np
import matplotlib.pyplot as plt

# Generate fine and coarse scales solutions for some realizations of the random variable
# Interval end points
def motivate_gen(c_elements):
    """
    Fix fine mesh at 2000 elements and vary the coarse elemts `c_elements` to generate solutions across scales
    """
    np.random.seed(4)
    a = -1  
    b = 1
    # Parameter function
    x0 = 0.1
    def q(x): return np.exp(5*(np.cos(2*np.pi*x / x0 + eta)))
    # Forcing term
    def f(x): return -(2.5 + 2*x)
    u_a = 0
    u_b = 1

    m = 1          # number of realizations
    f_elements = 2000
    c_elements = c_elements
    sample_parameter = np.zeros(m)
    eta = np.random.uniform(0, 2*np.pi)
    xf, fua = forward_solver(a=a, b=b, u_a=u_a, u_b=u_b,n_elements=f_elements, eta=eta, q=q, f=f)
    xc, cua = forward_solver(a=a, b=b, u_a=u_a, u_b=u_b,n_elements=c_elements, eta=eta, q=q, f=f)
    fine_store = np.zeros((len(xf), m))
    coarse_store = np.zeros((len(xc), m))

    # fig = plt.figure(figsize=(10, 5))
    # axf = plt.subplot(1, 2, 1)
    # axc = plt.subplot(1, 2, 2)
    for i in range(m):
        eta = np.random.uniform(0, 2*np.pi)
        sample_parameter[i] = eta
        xf, fua = forward_solver(a=a, b=b, u_a=u_a, u_b=u_b,n_elements=f_elements, eta=eta, q=q, f=f)
        xc, cua = forward_solver(a=a, b=b, u_a=u_a, u_b=u_b,n_elements=c_elements, eta=eta, q=q, f=f)
        fine_store[:, i] = fua
        coarse_store[:, i] = cua
        # axf.plot(xf, fua)       # fine soln
        # axc.plot(xc, cua)       # coarse soln
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"$u(x)$")
    # fig.tight_layout()
    # plt.show()

    return coarse_store, fine_store

# Vary c_elements from 1 to 2000 and run forward solver.  In each case, interpolate the coarse scale soln on fine mesh and compute the error with fine mesh.
result = []
from fem_error_1d import EvalCoarseSoln

if __name__ == '__main__':
    for i in range(2,2000+1):
        err = EvalCoarseSoln(a=-1, b=1, coarse_elements=i, fine_elements=2000)
        # generate forward solution
        coarse_soln, fine_soln =  motivate_gen(i)
        # interpolate coarse soln over fine mesh
        u_c = err.evalcoarsesoln(coarse_elements=i, fine_elements=2000, u_c=coarse_soln[:,0])
        # compute error between coarse and fine mesh
        err_curr = err.error(u_c, fine_soln[:,0])
        result.append(err_curr)
        # see order of error
        print(err_curr, "   ", i)
plt.plot(result)
plt.xlabel('Number of Elements')
plt.ylabel(r"$\Vert u_c - u_f \Vert_{L^2}$")
plt.show()


