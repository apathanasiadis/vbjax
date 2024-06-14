import jax
import jax.numpy as np
import numpy
import scipy
# need to import ChCa functions from neural_mass.py

def find_intersections(y0, eta, params):  # solving only for dr=dV
    from scipy.optimize import fsolve
    intersections = []
    eigenvalues = []
    params = params._replace(Eta=eta)
    vars = y0[:2]
    
    # Define the intersection function as a system of equations
    def equations(vars, y0):
        r, V = vars[:,None]
        y = np.concatenate((r,V,y0[2:]))
        return [ChCa_dr(y, params), ChCa_dV(y, params)]
        
    #FOR TOMORROW: ADD HERE DIFFERENT INSTERSECTION COORDINATES OF THE FX POINT, FOR CALCULATING THE EIGENVALUE, AS RV0 COUPLED WITH THEIR PARAMETER WITH WHICH I OBTAINED IT        
    def eigen(ystar, y0, way=0):
        if not way:
            def wraper(ystar, y0, params):
                y = np.concatenate((ystar, y0[2:]))
                return ChCa_rV_dfun(y, params)
            J = jax.jacfwd(wraper, argnums=0)(ystar, y0, params)
            J = np.vstack((J[0], J[1]))
            return numpy.linalg.eigvals(J)[0].real
        else:
            y = np.concatenate((ystar, y0[2:]))
            A = np.vstack(jax.jit(jax.jacfwd(ChCa_dfun))(y, p))
            eival = scipy.linalg.eigvals(A)
            return A, eival.reshape(2,-1)[:,0].real

    # Guess initial points for root finding
    guess_points = [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)]

    # Use fsolve to find roots (intersection points)
    # for guess in guess_points:
    intersection = fsolve(equations, vars, args=y0, xtol=1e-16)
    A, eigenvalue = eigen(intersection, y0, 1)
    # intersections.append(intersection)
    # eigenvalues.append(eigenvalue)
    return intersection, A, eigenvalue