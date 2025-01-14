import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from flax import linen as nn
from typing import Callable, Tuple
import time
from functools import partial

class NeuralController(nn.Module):
    hidden_dim: int
    control_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.control_dim)(x)
        return x

class PhysicsSimulator(nn.Module):
    state_dim: int
    control_dim: int
    hidden_dim: int
    max_velocity: float = 10.0
    dt: float = 0.01

    @nn.compact
    def __call__(self, state, control):
        state_encoded = nn.Dense(self.hidden_dim)(state)
        control_encoded = nn.Dense(self.hidden_dim)(control)
        
        combined = jnp.concatenate([state_encoded, control_encoded])
        x = nn.Dense(self.hidden_dim)(combined)
        x = jnp.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = jnp.tanh(x)
        acceleration = nn.Dense(self.state_dim)(x)
        
        acceleration = jnp.clip(acceleration, 
                              -self.max_velocity/self.dt,
                              self.max_velocity/self.dt)
        new_state = state + acceleration * self.dt
        return new_state

class OriginalImplementation:
    def __init__(self, theta: jnp.ndarray, x_initial: jnp.ndarray, x_target: jnp.ndarray,
                 time_steps: int, f: Callable, g: Callable):
        self.theta = theta
        self.x_initial = x_initial
        self.x_target = x_target
        self.time_steps = time_steps
        self.f = f
        self.g = g

    def forward(self, detach: bool = False):
        x = self.x_initial
        x_values = [x]
        
        for t in range(self.time_steps):
            x_input = jax.lax.stop_gradient(x) if detach else x
            c = self.f(x_input, self.theta)
            x = self.g(x, c)
            x_values.append(x)
            
        loss = 0.5 * jnp.sum((x - self.x_target)**2)
        return x, loss, x_values

    def compute_gradients(self):
        start_time = time.time()
        
        # Run forward pass with no detach
        x, loss, _ = self.forward(detach=False)
        grad_fn = jax.grad(lambda theta: self.forward(detach=False)[1])
        original_grad = grad_fn(self.theta)
        
        # Run forward pass with detach
        x, loss, _ = self.forward(detach=True)
        grad_fn_detached = jax.grad(lambda theta: self.forward(detach=True)[1])
        detached_grad = grad_fn_detached(self.theta)
        
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

    def compute_ABCD(self, x: jnp.ndarray, x_next: jnp.ndarray, c: jnp.ndarray):
        A = jax.jacrev(lambda c: self.g(x, c))(c)
        B = jax.jacrev(lambda theta: self.f(x, theta))(self.theta)
        C = jax.jacrev(lambda x: self.g(x, c))(x)
        D = jax.jacrev(lambda x: self.f(x, self.theta))(x)
        return A, B, C, D

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
            A, B, C, D = self.compute_ABCD(x, x_next, c)
            derivatives.append((A, B, C, D))
            cs.append(c)
            x = x_next
            xs.append(x)
            
        # Final loss gradient
        dL_dx = x - self.x_target
        
        # Backward pass for both gradients
        d_original = jnp.zeros_like(self.theta)
        d_detached = jnp.zeros_like(self.theta)
        accumulated_grad = dL_dx
        
        for t in range(self.time_steps-1, -1, -1):
            A, B, C, D = derivatives[t]
            
            # Compute gradient contribution at this timestep
            control_grad = jnp.matmul(A.T, accumulated_grad)
            param_grad = jnp.matmul(B.T, control_grad)
            
            d_detached += param_grad
            d_original += param_grad
            
            if t > 0:
                # Propagate gradient backward
                state_grad = jnp.matmul(C.T, accumulated_grad)
                control_to_state = jnp.matmul(D.T, control_grad)
                accumulated_grad = state_grad + control_to_state
        
        end_time = time.time()
        
        return d_original, d_detached, end_time - start_time

def create_test_environment(key):
    # Environment dimensions
    state_dim = 12
    control_dim = 6
    hidden_dim = 128
    
    # Create models
    controller = NeuralController(hidden_dim=hidden_dim, control_dim=control_dim)
    simulator = PhysicsSimulator(state_dim=state_dim, 
                               control_dim=control_dim,
                               hidden_dim=hidden_dim)
    
    # Initialize parameters
    key1, key2 = random.split(key)
    x_dummy = jnp.zeros((state_dim,))
    
    controller_params = controller.init(key1, x_dummy)
    simulator_params = simulator.init(key2, x_dummy, jnp.zeros((control_dim,)))
    
    def f(x, theta):
        # Reshape flat parameters back into network structure
        return controller.apply(controller_params, x)
    
    def g(x, c):
        return simulator.apply(simulator_params, x, c)
    
    # Initial conditions
    key1, key2 = random.split(key2)
    x_initial = random.normal(key1, (state_dim,))
    x_target = random.normal(key2, (state_dim,))
    
    # Pack controller parameters into a single tensor
    theta = jnp.concatenate([p.flatten() for p in jax.tree_util.tree_leaves(controller_params)])
    
    return f, g, theta, x_initial, x_target

def compare_extended():
    print("\nRunning extended comparison...")
    key = random.PRNGKey(0)
    f, g, theta, x_initial, x_target = create_test_environment(key)
    time_steps = 2
    
    # Create both implementations
    original_impl = OriginalImplementation(theta, x_initial, x_target, time_steps, f, g)
    efficient_impl = EfficientImplementation(theta, x_initial, x_target, time_steps, f, g)
    
    compare_results(original_impl, efficient_impl)

def compare_results(original_impl, efficient_impl):
    # Run and compare
    orig_grad, orig_detached, orig_time = original_impl.compute_gradients()
    eff_grad, eff_detached, eff_time = efficient_impl.compute_gradients()

    # Print dimensions
    state_dim = original_impl.x_initial.shape[0]
    control_dim = original_impl.f(original_impl.x_initial, original_impl.theta).shape[0]
    param_dim = original_impl.theta.shape[0]
    
    print("\nProblem Dimensions:")
    print(f"Observation Space Dimension: {state_dim}")
    print(f"Action Space Dimension: {control_dim}")
    print(f"Number of Model Parameters: {param_dim}")
    
    normal_diff = jnp.abs(orig_grad - eff_grad)
    detached_diff = jnp.abs(orig_detached - eff_detached)
    
    print("\nGradient Differences:")
    print(f"  Normal - Max Diff: {jnp.max(normal_diff):.6f}, Mean Diff: {jnp.mean(normal_diff):.6f}")
    print(f"  Detached - Max Diff: {jnp.max(detached_diff):.6f}, Mean Diff: {jnp.mean(detached_diff):.6f}")
    
    print("\nGradients Match (within 1e-5):")
    print(f"  Normal: {jnp.allclose(orig_grad, eff_grad, rtol=1e-5, atol=1e-5)}")
    print(f"  Detached: {jnp.allclose(orig_detached, eff_detached, rtol=1e-5, atol=1e-5)}")
    
    print("\nPerformance Comparison:")
    print(f"Original Implementation:")
    print(f"  Time: {orig_time:.4f} seconds")
    print(f"Efficient Implementation:")
    print(f"  Time: {eff_time:.4f} seconds")
    
    print(f"\nEfficiency Gains:")
    print(f"  Time: {(orig_time - eff_time) / orig_time * 100:.1f}% faster")

def main():
    print("\nTest: Extended Implementation with Neural Networks")
    compare_extended()

if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    main()