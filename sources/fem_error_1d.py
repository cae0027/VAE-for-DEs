

from oned_fem import  oned_mesh, oned_gauss, oned_shape, oned_linear, oned_bilinear
import numpy as np

class EvalCoarseSoln:
    """
    Evaluate the coarse solution at the fine mesh
    """
    def __init__(self, a=-1, b=1, coarse_elements=5, fine_elements=40):
        self.a = a
        self.b = b
        self.coarse_elements = coarse_elements
        self.fine_elements = fine_elements
        x_c, e_c = oned_mesh(a, b, coarse_elements, 'linear')
        x_f, e_f = oned_mesh(a, b, fine_elements, 'linear')
        r, w = oned_gauss(11)
        self.r = r
        self.w = w
        self.x_f = x_f
        self.e_f = e_f

    def evalcoarsesoln(self, coarse_elements=5, fine_elements=40, u_c=None):
        """
        evaluate the coarse solution u_c at the fine x_f
        """
        # Evaluate coarse function on the fine mesh
        u_catf = np.zeros(self.x_f.shape)
        for i in range(coarse_elements):
            # Local node indices
            i_loc = e_c[i,:]
            
            # Coarse element vertices
            x_loc = x_c[i_loc]
            
            # Ensure coarse solution is not None
            assert u_c is not None, "Please provide the coarse solution u_c"
            # Coarse function values at element vertices
            uc_loc = u_c[i_loc]
            
            # Determine what fine-scale vertices are contained in coarse element
            idx_f_in_el = ((self.x_f >= x_loc[0])*(self.x_f <= x_loc[1]))
            # print(idx_f_in_el)
            xf_in_el = self.x_f[idx_f_in_el]
            # print(xf_in_el)
            
            # Map fine-scale vertices to reference element [-1,1]
            xf_ref = 2/(x_loc[1]-x_loc[0])*(xf_in_el - x_loc[0]) - 1
            
            # Evaluate the coarse shape functions there
            _, _, phi, phi_x, _ = oned_shape(x_loc, xf_ref, np.ones(xf_ref.shape))
            
            # Linear combination of local coarse basis functions evaluated on fine mesh
            u_catf[idx_f_in_el] = phi.dot(uc_loc)
        return u_catf

    def error(self, u1, u2):
        """
        Compute the L2 inner product between u1 and u2 using the fine mesh bases functions
        """
        diff = np.abs(u1 - u2)
        result = []
        for i in range(self.fine_elements):
            # local information
            i_loc = self.e_f[i, :]
            x_loc = self.x_f[i_loc]
            
            # Error function values at element vertices
            diff_loc = diff[i_loc]
            
            # Determine what fine-scale vertices are contained in coarse element
            idx_f_in_el = ((self.x_f >= x_loc[0])*(self.x_f <= x_loc[1]))
            xf_in_el = self.x_f[idx_f_in_el]
            xf_in_el = np.linspace(xf_in_el[0], xf_in_el[1], 11)
            
            # Map fine-scale vertices to reference element [-1,1]
            xf_ref = 2/(x_loc[1]-x_loc[0])*(xf_in_el - x_loc[0]) - 1
            
            # Evaluate the coarse shape functions there
            dummy, dummy, phi, phi_x, dummy = oned_shape(x_loc, xf_ref, np.ones(xf_ref.shape))
            
            # Linear combination of local coarse basis functions evaluated on fine mesh
            diff_inter = phi.dot(diff_loc)


            # Compute shape function on element
            x_g, w_g, phi, phi_x, phi_xx = oned_shape(x_loc, self.r, self.w)

            # Compute the inner product on element
            phi = np.ones(phi.shape[0])
            inner = oned_linear(diff_inter**2, phi, w_g)
            result.append(inner)
        return np.sum(result)






if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = -5
    b = 5
    comp = EvalCoarseSoln(a=a, b=b)
    u = lambda x: np.sin(x)
    x_c, e_c = oned_mesh(a, b, 5, 'linear')
    u_c = u(x_c)

    u_cnew = comp.evalcoarsesoln(coarse_elements=5, u_c=u_c)
    print(comp.error(u(comp.x_f), u(comp.x_f)))

    plt.plot(x_c, u_c, label='coarse')
    plt.plot(comp.x_f, u_cnew, '.r', label='fine')
    plt.plot(comp.x_f, u(comp.x_f), label="True soln")
    plt.legend()
    plt.show()
