#!/usr/bin/env python3
"""
Module for solving one dimensional finite element problems
"""

import numpy as np

def oned_mesh(x_min, x_max, n_elements, element_type):
    """
    Description: Construct a one dimensional finite element mesh. 

    Usage: x, e_conn = oned_fem_mesh(x_min, x_max, n_elements, element_type)

    Inputs:

        x_min: double, left interval endpoint

        x_max: double, right interval endpoint

        n_elements: int, number of elements

        element_type: str, polynomial order 'linear', 'quadratic', or 'cubic'


    Outputs: 

        x: double, vector of finite element nodes

        e_conn: int, (n_elements, n_dof) connectivity matrix whose ith row
           contains the indices of the nodes x(i) contained in the ith element

    Last modified: 

        02/05/2021, Hans-Werner van Wyk

    """
    if element_type == 'linear':
        # Define mesh nodes
        x = np.linspace(x_min,x_max,n_elements+1);
        
        # Define connectivity matrix
        e_conn = (np.arange(n_elements),np.arange(1,n_elements+1))
        e_conn = np.stack(e_conn,axis=1)
        
    elif element_type == 'quadratic':
        # Define the mesh nodes
        x = np.linspace(x_min, x_max, 2*n_elements + 1)
        
        # Define the connectivity matrix
        e_conn = (np.arange(0,2*n_elements-1,2),
                  np.arange(1,2*n_elements,2),
                  np.arange(2,2*n_elements+1,2))
        e_conn = np.stack(e_conn,axis=1)
        
    elif element_type == 'cubic':
        # Define the mesh nodes
        x = np.linspace(x_min,x_max,3*n_elements+1)
        
        # Define the connectivity matrix
        e_conn = (np.arange(0,3*n_elements-2,3),
                  np.arange(1,3*n_elements-1,3),
                  np.arange(2,3*n_elements,3),
                  np.arange(3,3*n_elements+1,3))
        e_conn = np.stack(e_conn,axis=1)
     
    else: 
        # Error
        raise Exception('Use only "linear", "quadratic", '+\
                        'or "cubic" for element_type')
    
    return x,e_conn
    

def oned_gauss(rule):
    """
    Calculate Gauss integration points on (-1,1)

    
    Inputs: 
    
        rule: int, number of Gauss points (between 1 and 11). The precision of
            the Gauss rule is 2*rule - 1
            
            Int[-1,1] f(x) dx ~= sum_{i=1}^rule wi*f(ri)
        
    Outputs:
    
        r: double, (rule,) vector of Gauss points in (-1,1]
        
        w: double, (rule,) vector of Gauss weights
    """

    r = np.zeros(rule)
    w = np.zeros(rule)

    if rule == 1:      # up to order 1 polynomials exact
        r[0] = 0
        w[0] = 2
        
    elif rule == 2:   # up to order 3 polynomials exact
        r[0] =-1.0 / np.sqrt(3.0);
        r[1] =-r[0];
        w[0] = 1.0;
        w[1] = 1.0;
        
    elif rule == 3:  # up to order 5 polynomials exact
        r[0] =-np.sqrt(3.0/5.0);
        r[1] = 0.0;
        r[2] =-r[0];
        w[0] = 5.0 / 9.0;
        w[1] = 8.0 / 9.0;
        w[2] = w[0];
        
    elif rule == 4:  # up to order 7 polynomials exact
        r[0] =-np.sqrt((3.0+2.0*np.sqrt(6.0/5.0))/7.0);
        r[1] =-np.sqrt((3.0-2.0*np.sqrt(6.0/5.0))/7.0);
        r[2] =-r[1];
        r[3] =-r[0];
        w[0] = 0.5 - 1.0 / ( 6.0 * np.sqrt(6.0/5.0) );
        w[1] = 0.5 + 1.0 / ( 6.0 * np.sqrt(6.0/5.0) );
        w[2] = w[1];
        w[3] = w[0];
        
    elif rule == 5:  # up to order 9 polynomials exact
        r[0] =-np.sqrt(5.0+4.0*np.sqrt(5.0/14.0)) / 3.0;
        r[1] =-np.sqrt(5.0-4.0*np.sqrt(5.0/14.0)) / 3.0;
        r[2] = 0.0;
        r[3] =-r[1];
        r[4] =-r[0];
        w[0] = 161.0/450.0-13.0/(180.*np.sqrt(5.0/14.0));
        w[1] = 161.0/450.0+13.0/(180.*np.sqrt(5.0/14.0));
        w[2] = 128.0/225.0;
        w[3] = w[1];
        w[4] = w[0];
        
    elif rule == 6:
        r[0] = -0.2386191860831969;
        r[1] = -0.6612093864662645;
        r[2] = -0.9324695142031521;
        r[3] = - r[0];
        r[4] = - r[1];
        r[5] = - r[2];
        w[0] = 0.4679139345726910;
        w[1] = 0.3607615730481386;
        w[2] = 0.1713244923791704;
        w[3] = w[0];
        w[4] = w[1];
        w[5] = w[2];
        
    elif rule == 7:
        r[0] = -0.9491079123427585;
        r[1] = -0.7415311855993945;
        r[2] = -0.4058451513773972;
        r[3] =  0.0000000000000000;
        r[4] = - r[2];
        r[5] = - r[1];
        r[6] = - r[0];
        w[0] = 0.1294849661688697;
        w[1] = 0.2797053914892766;
        w[2] = 0.3818300505051189;
        w[3] = 0.4179591836734694;
        w[4] = w[2];
        w[5] = w[1];
        w[6] = w[0];
        
    elif rule == 8:
        r[0] = -0.9602898564975363;
        r[1] = -0.7966664774136267;
        r[2] = -0.5255324099163290;
        r[3] = -0.1834346424956498;
        r[4] = - r[3];
        r[5] = - r[2];
        r[6] = - r[1];
        r[7] = - r[0];
        w[0] = 0.1012285362903763;
        w[1] = 0.2223810344533745;
        w[2] = 0.3137066458778873;
        w[3] = 0.3626837833783620;
        w[4] = w[3];
        w[5] = w[2];
        w[6] = w[1];
        w[7] = w[0];

    elif rule == 9:
        r[0] = -0.9681602395076261;
        r[1] = -0.8360311073266358;
        r[2] = -0.6133714327005904;
        r[3] = -0.3242534234038089;
        r[4] =  0.0000000000000000;
        r[5] = - r[3];
        r[6] = - r[2];
        r[7] = - r[1];
        r[8] = - r[0];
        w[0] = 0.0812743883615744;
        w[1] = 0.1806481606948574;
        w[2] = 0.2606106964029354;
        w[3] = 0.3123470770400029;
        w[4] = 0.3302393550012598;
        w[5] = w[3];
        w[6] = w[2];
        w[7] = w[1];
        w[8] = w[0];
      
    elif rule == 10:
        r[0] = -0.9739065285171717;
        r[1] = -0.8650633666889845;
        r[2] = -0.6794095682990244;
        r[3] = -0.4333953941292472;
        r[4] = -0.1488743389816312;
        r[5] = - r[4];
        r[6] = - r[3];
        r[7] = - r[2];
        r[8] = - r[1];
        r[9] = - r[0];
        w[0] = 0.0666713443086881;
        w[1] = 0.1494513491505806;
        w[2] = 0.2190863625159820;
        w[3] = 0.2692667193099963;
        w[4] = 0.2955242247147529;
        w[5] = w[4];
        w[6] = w[3];
        w[7] = w[2];
        w[8] = w[1];
        w[9] = w[0];

    elif rule == 11:
        r[0] = -0.9782286581460570;
        r[1] = -0.8870625997680953;
        r[2] = -0.7301520055740494;
        r[3] = -0.5190961292068118;
        r[4] = -0.2695431559523450;
        r[5] =  0.0000000000000000;
        r[6] = - r[4];
        r[7] = - r[3];
        r[8] = - r[2];
        r[9] = - r[1];
        r[10] = - r[0];
        w[0] = 0.0556685671161737;
        w[1] = 0.1255803694649046;
        w[2] = 0.1862902109277343;
        w[3] = 0.2331937645919905;
        w[4] = 0.2628045445102467;
        w[5] = 0.2729250867779006;
        w[6] = w[4];
        w[7] = w[3];
        w[8] = w[2];
        w[9] = w[1];
        w[10] = w[0];

    else:
        raise Exception('Quadrature rule not supported')
        #Return computed quantities
        
    return r,w


def oned_shape(x,r,w):
    """
    computes test functions and derivatives for a Lagrange C0 element given
    element coordinates and Gauss points. (assumes all nodes are uniformly 
    distributed in the element).


    Usage:    x_g,w_g,phi,p_x,p_xx = oned_shape(x,r,w)

    Inputs:     
    
        x: double, coordinates of the element nodes
                 
        r: double, coordinates of Gauss points in (-1,1)
        
        w: double, Gauss weights associated with r
        
        
    Outputs:

        x_g: double, coordinates of Gauss points in the element
                 
        w_g: double, Gauss weights scaled by the element Jacobian
                 
        phi: double, value of element shape functions at x_g
        
        p_x: double, first spatial derivatives of phi
        
        p_xx: double, second spatial derivatives of phi
    
    
    Modified: 
    
        02/05/2021, Hans-Werner van Wyk 
    """
    n_dof = len(x)  # number of dofs
    n_g = len(r)  # number of Gaussian quadrature points
    
    if n_dof==2:
        # Transform coordinates for linear elements
        c0 = ( x[-1]-x[0] )/2
        c1 = ( x[-1]+x[0] )/2
    
        # Gaussian nodes
        x_g = c0*r + c1

        # Evaluate basis function at Gauss points
        phi = np.empty((n_g,n_dof))
        phi[:,1] = ( 1+r )/2
        phi[:,0] = ( 1-r )/2
        
        # Evaluate first derivative at Gauss points
        p_x = np.empty((n_g,n_dof))
        p_x[:,1] = 0.5*np.ones(n_g)/c0
        p_x[:,0] =-p_x[:,1]

        # Jacobian
        djac = c0
    
        # Compute physical weights
        w_g = djac*w

        # Second derivative 
        p_xx = np.zeros((n_g,n_dof))
    
    elif n_dof==3:
        
        # Transform coordinates for quadratic elements
        c0 = ( x[-1]-x[0] )/2
        c1 = ( x[-1]+x[0] )/2
        
        # Physical Gauss nodes
        x_g = c0*r + c1

        # defined backwards to help Matlab create the proper sized array
        phi = np.empty((n_g,n_dof))
        phi[:,2] = .5*r*( r+1 )
        phi[:,1] =-( r+1 )*( r-1 )
        phi[:,0] = .5*r*( r-1 )

        # First derivative 
        p_x = np.empty((n_g,n_dof))
        p_x[:,2] = ( r+.5 )/c0
        p_x[:,1] =-2*r/c0
        p_x[:,0] = ( r-.5 )/c0

        # Jacobian 
        djac = c0

        # Gaussian weight on physical element
        w_g = djac*w
        
        # Second derivative
        p_xx = np.empty((n_g,n_dof))
        p_xx[:,2] = np.ones(n_g)/c0**2
        p_xx[:,1] =-2*p_xx[:,2]
        p_xx[:,0] = p_xx[:,2]

    elif n_dof==4:
        # Transform coordinates for (nonconforming) cubic elements
        c0 = ( x[-1]-x[0] )/2
        c1 = ( x[-1]+x[0] )/2

        x_g = c0*r + c1

        r2  = r*r
        r3  = r*r2

        # defined backwards to help Matlab create the proper sized array
        phi = np.zeros((n_g,n_dof))
        phi[:,3] =  9*( r3+r2-r/9-1/9 )/16
        phi[:,2] =-27*( r3+r2/3-r-1/3 )/16
        phi[:,1] = 27*( r3-r2/3-r+1/3 )/16
        phi[:,0] =- 9*( r3-r2-r/9+1/9 )/16

        # Derivative on reference element
        p_r = np.zeros((n_g,n_dof))
        p_r[:,3] =  9*( 3*r2+2*r-1/9 )/16
        p_r[:,2] =-27*( 3*r2+2*r/3-1 )/16
        p_r[:,1] = 27*( 3*r2-2*r/3-1 )/16
        p_r[:,0] =- 9*( 3*r2-2*r-1/9 )/16
        
        # Second derivative on reference element
        p_rr = np.zeros((n_g,n_dof))
        p_rr[:,3] =  9*( 6*r+2   )/16
        p_rr[:,2] =-27*( 6*r+2/3 )/16
        p_rr[:,1] = 27*( 6*r-2/3 )/16
        p_rr[:,0] =- 9*( 6*r-2   )/16

        # Jacobian 
        dxdr = p_r.dot(x)
        drdx = 1/dxdr
        
        # Quadrature weights on physical element
        w_g = dxdr*w
        
        # Derivative on physical element
        p_x = np.zeros((n_g,n_dof))
        p_x[:,3] = p_r[:,3]*drdx
        p_x[:,2] = p_r[:,2]*drdx
        p_x[:,1] = p_r[:,1]*drdx
        p_x[:,0] = p_r[:,0]*drdx
        
        # Second derivative on physical element
        p_xx = np.zeros((n_g,n_dof))     
        p_xx[:,3] = p_rr[:,3]*drdx**2
        p_xx[:,2] = p_rr[:,2]*drdx**2
        p_xx[:,1] = p_rr[:,1]*drdx**2
        p_xx[:,0] = p_rr[:,0]*drdx**2
    else:
        raise Exception('Elements higher than cubic not currently supported')
    
    # Return computed quantities
    return x_g, w_g, phi, p_x, p_xx    

    
def oned_linear(kernel, test, w_g):
    """
    Compute the one-dimensional local linear form (f,phi)
    
    Inputs:
    
        kernel: double (n_gauss,) kernel evaluated at the Gauss nodes.
        
        test: double, (n_gauss, n_dofs) array of element test functions
            evaluated at the Gauss nodes.
            
        w_g: double, (n_gauss,) Gauss quadrature weights on element.     
        
        
    Output:
    
        L: double, (n_dofs,) vector with entries 
        
            L_i = int_Ke f(x) phi_i(x) dx ~= sum_j w_g(j) f(x_j) phi_i(x_j) 
    """
    b = np.dot(test.T, kernel*w_g)
    return b
    

def oned_bilinear(kernel,phi,test,w_g):
    """
    Compute the local bilinear form (kernel, phi, test)     
    
    
    Inputs:
    
        kernel: double (n_gauss,), kernel function in the integral evaluated at the
            Gauss points.
    
    
        phi: double, (n_gauss,n_dofs) matrix of trial functions evaluated at 
            Gauss points. 
            
            
        test: double, (n_gauss,n_dofs) matrix of test functions evaluated at 
            Gauss points. 
            
        w_g: double, (n_gauss,) Gauss quadrature weights.
        
        
    Output:
    
        B: double, (n_dofs, n_dofs) local blinear form over element Ke with entries
        
            B_ij = int_Ke f(x) phi_j(x) test_i(x) dx 
                 
                 ~= sum_{k} w_g(k) f(x_k) phi_j(x_k) test_i(x_k) 
        
    """
    M = np.dot(np.dot(test.T,np.diag(kernel*w_g)),phi)
    return M
    
    

    

