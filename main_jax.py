import jax
import jax.numpy as jnp
import time
from typing import Callable
from functools import partial

class OriginalImplementation:
    def __init__(self, theta: jnp.ndarray, x_initial: jnp.ndarray, x_target: jnp.ndarray, 
                 time_steps: int, f: Callable, g: Callable):
        self.theta = theta
        self.x_initial = x_initial
        self.x_target = x_target
        self.time_steps = time_steps
        self.f = f
        self.g = g
        
    def forward(self, theta, detach: bool = False):
        def step(x, _):
            x_input = jax.lax.stop_gradient(x) if detach else x
            c = self.f(x_input, theta)
            return self.g(x, c), None
            
        final_x, _ = jax.lax.scan(step, self.x_initial, jnp.arange(self.time_steps))
        loss = jnp.sum(0.5 * (final_x - self.x_target)**2)
        return loss, final_x

    def compute_gradients(self):
        start_time = time.time()
        
        loss_fn = lambda theta: self.forward(theta, detach=False)[0]
        loss_fn_detached = lambda theta: self.forward(theta, detach=True)[0]
            
        original_grad = jax.grad(loss_fn)(self.theta)
        detached_grad = jax.grad(loss_fn_detached)(self.theta)
        
        end_time = time.time()
        
        return original_grad, detached_grad, end_time - start_time

class EfficientImplementation:
    def __init__(self, theta: jnp.ndarray, x_initial: jnp.ndarray, x_target: jnp.ndarray, 
                 time_steps: int, f: Callable, g: Callable):
        self.theta = theta
        self.x_initial = x_initial
        self.x_target = x_target
        self.time_steps = time_steps
        self.f = f
        self.g = g
        
    def compute_ABCD(self, x: jnp.ndarray, c: jnp.ndarray):
        # A = dxn/dcn-1
        A = jax.jacrev(lambda c: self.g(x, c))(c)
        
        # B = dcn/dtheta
        B = jax.jacrev(lambda theta: self.f(x, theta))(self.theta)
        
        # C = dxn/dxn-1
        C = jax.jacrev(lambda x: self.g(x, c))(x)
        
        # D = dcn/dxn
        D = jax.jacrev(lambda x: self.f(x, self.theta))(x)
        
        return A.squeeze(), B.squeeze(), C.squeeze(), D.squeeze()

    def compute_gradients(self):
        start_time = time.time()
        
        # Forward pass
        x = self.x_initial
        derivatives = []
        xs = [x]
        cs = []
        
        for t in range(self.time_steps):
            c = self.f(x, self.theta)
            x_next = self.g(x, c)
            A, B, C, D = self.compute_ABCD(x, c)
            derivatives.append((A, B, C, D))
            cs.append(c)
            x = x_next
            xs.append(x)
            
        # Final loss gradient
        dL_dx = x - self.x_target
        
        def compute_path(step: int) -> jnp.ndarray:
            #Recursively compute gradient contribution for timestep n
            if step == 0:
                # Base case - first timestep
                A0, B0, C0, D0 = derivatives[0]
                return A0 * B0
            
            An, Bn, Cn, Dn = derivatives[step]
            
            # Direct path through control at this step
            direct = An * Bn
            
            # Path through state combines with all previous paths
            prev_paths = compute_path(step-1)
            state_effect = Cn + An * Dn  # dx{n}/dx{n-1} + dx{n}/dc{n} × dc{n}/dx{n-1}
            
            return direct + prev_paths * state_effect
        
        # Compute both gradients
        d_original = jnp.zeros_like(self.theta)
        d_detached = jnp.zeros_like(self.theta)
        
        if self.time_steps == 1:
            A0, B0, C0, D0 = derivatives[0]
            d_detached = A0 * B0 * dL_dx
            d_original = d_detached
        else:
            # Compute detached gradient using forward accumulation
            for m in range(self.time_steps):
                Am = derivatives[m][0]
                Bm = derivatives[m][1]
                
                term = Am * Bm
                if m+1 < self.time_steps:
                    for k in range(m+1, self.time_steps):
                        term = term * derivatives[k][2]
                d_detached += term
            
            # Compute original gradient using backward accumulation
            d_original = compute_path(self.time_steps-1)
            
            # Final multiplication with dL/dx for both gradients
            d_detached = d_detached * dL_dx
            d_original = d_original * dL_dx
        
        end_time = time.time()
        
        return d_original, d_detached, end_time - start_time

def compare_implementations():
    # Define custom functions
    def f(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        return theta * (x ** 2) / 1000 + 6*x
    
    def g(x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        return c + x + 32*x + c*c / 100
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    theta = jnp.array([0.5])
    x_initial = jnp.array([1.0])
    x_target = jnp.array([2.0])
    time_steps = 3
    
    # Create both implementations
    original_impl = OriginalImplementation(theta, x_initial, x_target, time_steps, f, g)
    efficient_impl = EfficientImplementation(theta, x_initial, x_target, time_steps, f, g)
    
    # Run and compare
    orig_grad, orig_detached, orig_time = original_impl.compute_gradients()
    eff_grad, eff_detached, eff_time = efficient_impl.compute_gradients()
    
    print("\nGradient Comparison:")
    print(f"Original Implementation Gradients:")
    print(f"  Normal: {orig_grad.item():.6f}")
    print(f"  Detached: {orig_detached.item():.6f}")
    print(f"\nEfficient Implementation Gradients:")
    print(f"  Normal: {eff_grad.item():.6f}")
    print(f"  Detached: {eff_detached.item():.6f}")
    
    print("\nGradients Match:")
    print(f"  Normal: {jnp.allclose(orig_grad, eff_grad)}")
    print(f"  Detached: {jnp.allclose(orig_detached, eff_detached)}")
    
    print("\nPerformance Comparison:")
    print(f"Original Implementation:")
    print(f"  Time: {orig_time:.4f} seconds")
    print(f"Efficient Implementation:")
    print(f"  Time: {eff_time:.4f} seconds")
    
    print(f"\nEfficiency Gains:")
    print(f"  Time: {(orig_time - eff_time) / orig_time * 100:.1f}% faster")

if __name__ == "__main__":
    print(jax.devices())
    compare_implementations()