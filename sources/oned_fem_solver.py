from oned_fem import  oned_mesh, oned_gauss, oned_shape, oned_linear, oned_bilinear
import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import linalg

def forward_solver(a=0., b=1.,u_a=0., u_b=1., n_elements=50, q=None, eta=1., f=None):
    """
    Description: 1D FEM solver for the elliptic problem -d/dx(q(x)du/dx) = f(x)
    Inputs:
        a: lower x boundary
        b: upper x boundary
        u_a: boundary condition at a
        u_b: boundary condition at b
        n_elements: number of finite elements
        q: the parameter, a function of x depending on eta
        eta: realization of a random variable
    Outputs:
        x: the mesh values
        ua: solution of differential equation
    """
    assert (q is not None) or (f is not None), "Please provide the parameter function, q or the load function, f"
    # Generate the computational mesh
    n_elements = n_elements  # specify number of elements

    # Compute nodes and connectivity matrix
    x, e_conn = oned_mesh(a, b, n_elements, 'linear')

    n_nodes = len(x)            # number of nodes
    n_dofs = len(e_conn[1, :])   # degrees of freedom per element

    # Index to keep track of equation numbers
    ide = np.zeros(n_nodes, dtype=int)  # initialize

    # Mark Dirichlet nodes by -1
    i_dir = [0, n_nodes-1]  # indices of dirichlet nodes
    ide[i_dir] = -1        # dirichlet nodes are marked by -1

    # Number remaining nodes from 0 to n_equations-1
    count = 0
    for i in range(n_nodes):
        if ide[i] == 0:
            ide[i] = count
            count = count + 1

    n_equations = count   # total number of equations

    # Initialize sparse stiffness matrix
    nnz = n_elements*n_dofs**2        # estimate the number of non-zero elements

    rows = np.zeros(nnz, dtype=int)   # row index
    cols = np.zeros(nnz, dtype=int)   # column index
    vals = np.zeros(nnz)              # matrix entries

    # Initialize the RHS
    c = np.zeros(n_equations)

    # boundary conditions
    def bc(x): return 0 if x == a else 1

    # Assembly
    # Gauss rule accurate to degree (2n-1), n is the input of oned_gauss
    r, w = oned_gauss(11)
    count = 0
    for i in range(n_elements):
        # local information
        i_loc = e_conn[i, :]  # local node indices
        x_loc = x[i_loc]    # local nodes

        # compute shape function on element
        x_g, w_g, phi, phi_x, phi_xx = oned_shape(x_loc, r, w)

        # compute local stiffnes matrix
        q_g = q(x_g)
        A_loc = oned_bilinear(q_g, phi_x, phi_x, w_g)

        # local RHS
        f_g = f(x_g)
        c_loc = oned_linear(f_g, phi, w_g)

        # global
        for j in range(n_dofs):
            # for each row
            j_test = i_loc[j]    # global node number
            j_eqn = ide[j_test]  # equation number

            if j_eqn >= 0:
                # update RHS
                c[j_eqn] = c[j_eqn] + c_loc[j]

                for m in range(n_dofs):
                    # for each column
                    i_trial = i_loc[m]  # global node number
                    i_col = ide[i_trial]  # equation number

                    if i_col >= 0:
                        # interio node: fill column
                        rows[count] = j_eqn
                        cols[count] = i_col
                        vals[count] = A_loc[j, m]
                        count = count + 1
                    else:
                        # Dirichlet node: apply dirichlet condition
                        u_dir = bc(x_loc[m])
                        # print("This is the Dirichlet nodes: ", u_dir, ", Location: ", x_loc[m])
                        c[j_eqn] = c[j_eqn] - A_loc[j, m] * u_dir

    # Delete entries that weren't filled
    noz = len(rows) - (count + 1)    # read # of rows not affected by count
    rows = rows[:-noz]               # delete the last 'noz' entries
    cols = cols[:-noz]               # delete the last 'noz' entries
    vals = vals[:-noz]               # delete the last 'noz' entries

    # Assemble sparse stiffness matrix
    A = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(n_equations, n_equations)).tocsr()

    # Compute finite element solution
    ua = np.zeros(n_nodes)
    # print([ua, ub])
    ua[i_dir] = [u_a, u_b]    # apply prescribed values at Dirichlet nodes
    ua[ide >= 0] = linalg.spsolve(A, c)  # solve the system at interior nodes
    return x, ua





if __name__ == "__main__":
    eta = 1
    q = lambda x: 1 + 0.5 * np.cos(2*np.pi*(x/1.2) + eta)
    f = lambda x: x**2
    x,u = forward_solver(a=0, b=1,u_a=0.0, u_b=1.0, n_elements=50, q=q, eta=1, f=f)
    print(x)