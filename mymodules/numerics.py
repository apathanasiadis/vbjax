import jax
import jax.numpy as np
from jax import grad, jacfwd
import vbjax as vb


def newton_step(x, func, tol):
    fx_shape = jax.eval_shape(func, x)
    fx = func(x)
    if fx_shape.size > 1:  
        J = jacfwd(func)(x)
        x_new = x - np.linalg.solve(J, fx)  # Solve J * delta_x = -fx for delta_x
        return x_new, np.linalg.norm(fx, ord=2) < tol
    else:  # Scalar output
        fprime = grad(func)
        x_new = x - fx / fprime(x)
        return x_new, np.abs(fx) < tol        

@jax.jit
def newton(func, x0, tol=1e-6, max_iter=100):
    def loop_body(carry, _):
        x, converged = carry
        x_new, stop_iteration = newton_step(x, func, tol)
        return (x_new, converged | stop_iteration), x_new
    
    carry = (x0, False)
    _, root = jax.lax.scan(loop_body, carry, np.arange(max_iter))    
    return root[-1]
