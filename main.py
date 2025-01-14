import torch
import time
from typing import Callable
import torch.autograd as autograd

class OriginalImplementation:
    def __init__(self, theta: torch.Tensor, x_initial: torch.Tensor, x_target: torch.Tensor, 
                 time_steps: int, f: Callable, g: Callable):
        self.theta = theta
        self.x_initial = x_initial
        self.x_target = x_target
        self.time_steps = time_steps
        self.f = f
        self.g = g
        
    def forward(self, detach: bool = False) -> tuple:
        x = self.x_initial.clone()
        x_values = [x.item()]
        
        for t in range(self.time_steps):
            x_input = x.detach() if detach else x
            c = self.f(x_input, self.theta)
            x = self.g(x, c)
            x_values.append(x.item())
            
        loss = 0.5 * (x - self.x_target)**2
        return x, loss, x_values

    def compute_gradients(self) -> tuple:
        torch.cuda.empty_cache()
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated()
        
        x, loss, _ = self.forward(detach=False)
        loss.backward()
        original_grad = self.theta.grad.clone()
        self.theta.grad.zero_()
        
        x, loss, _ = self.forward(detach=True)
        loss.backward()
        detached_grad = self.theta.grad.clone()
        self.theta.grad.zero_()
        
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated()
        
        return original_grad, detached_grad, end_time - start_time, initial_memory, peak_memory

class EfficientImplementation:
    def __init__(self, theta: torch.Tensor, x_initial: torch.Tensor, x_target: torch.Tensor, 
                 time_steps: int, f: Callable, g: Callable):
        self.theta = theta
        self.x_initial = x_initial
        self.x_target = x_target
        self.time_steps = time_steps
        self.f = f
        self.g = g
        
    def compute_ABCD(self, x: torch.Tensor, x_next: torch.Tensor, c: torch.Tensor) -> tuple:
        # Compute A = dxn/dcn-1 using autograd
        c_temp = c.clone().requires_grad_(True)
        x_next_temp = self.g(x, c_temp)
        A = autograd.grad(x_next_temp, c_temp, create_graph=True)[0]
        
        # Compute B = dcn/dtheta using autograd
        theta_temp = self.theta.clone().requires_grad_(True)
        c_temp = self.f(x, theta_temp)
        B = autograd.grad(c_temp, theta_temp, create_graph=True)[0]
        
        # Compute C = dxn/dxn-1 using autograd
        x_temp = x.clone().requires_grad_(True)
        x_next_temp = self.g(x_temp, c)
        C = autograd.grad(x_next_temp, x_temp, create_graph=True)[0]
        
        # Compute D = dcn/dxn using autograd
        x_temp = x.clone().requires_grad_(True)
        c_temp = self.f(x_temp, self.theta)
        D = autograd.grad(c_temp, x_temp, create_graph=True)[0]
        
        return A, B, C, D

    def compute_gradients(self) -> tuple:
        torch.cuda.empty_cache()
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        x = self.x_initial.clone()
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
        loss = 0.5 * (x - self.x_target)**2
        dL_dx = x - self.x_target
        
        def compute_path(step: int) -> torch.Tensor:
            """Recursively compute gradient contribution for timestep n"""
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
        d_original = torch.zeros_like(self.theta)
        d_detached = torch.zeros_like(self.theta)
        
        if self.time_steps == 1:
            A0, B0, C0, D0 = derivatives[0]
            d_detached = A0 * B0 * dL_dx
            d_original = d_detached
        else:
            # Compute detached gradient
            for m in range(0, self.time_steps):
                Am = derivatives[m][0]
                Bm = derivatives[m][1]
                
                term = Am * Bm
                if m+1 < self.time_steps:
                    for k in range(m+1, self.time_steps):
                        term = term * derivatives[k][2]
                d_detached += term

            # Compute original gradient recursively
            d_original = compute_path(self.time_steps-1)
            
            # Final multiplication with dL/dx for both gradients
            d_detached = d_detached * dL_dx
            d_original = d_original * dL_dx
        
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated()
        
        return d_original, d_detached, end_time - start_time, initial_memory, peak_memory

def compare_implementations():
    # Define custom functions
    def f(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return theta * (x ** 2) / 1000 + 6*x
    
    def g(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return c + x + 32*x + c*c / 100
    
    # Initialize parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    theta = torch.tensor([0.5], requires_grad=True, device=device)
    x_initial = torch.tensor([1.0], device=device)
    x_target = torch.tensor([2.0], device=device)
    time_steps = 4

    
    # Create both implementations
    original_impl = OriginalImplementation(theta, x_initial, x_target, time_steps, f, g)
    efficient_impl = EfficientImplementation(theta, x_initial, x_target, time_steps, f, g)
    
    # Run and compare
    orig_grad, orig_detached, orig_time, orig_init_mem, orig_peak_mem = original_impl.compute_gradients()
    eff_grad, eff_detached, eff_time, eff_init_mem, eff_peak_mem = efficient_impl.compute_gradients()
    
    print("\nGradient Comparison:")
    print(f"Original Implementation Gradients:")
    print(f"  Normal: {orig_grad.item():.6f}")
    print(f"  Detached: {orig_detached.item():.6f}")
    print(f"\nEfficient Implementation Gradients:")
    print(f"  Normal: {eff_grad.item():.6f}")
    print(f"  Detached: {eff_detached.item():.6f}")
    
    print("\nGradients Match:")
    print(f"  Normal: {torch.allclose(orig_grad, eff_grad)}")
    print(f"  Detached: {torch.allclose(orig_detached, eff_detached)}")
    
    print("\nPerformance Comparison:")
    print(f"Original Implementation:")
    print(f"  Time: {orig_time:.4f} seconds")
    print(f"  Memory Usage: {(orig_peak_mem - orig_init_mem) / 1024**2:.2f} MB")
    print(f"Efficient Implementation:")
    print(f"  Time: {eff_time:.4f} seconds")
    print(f"  Memory Usage: {(eff_peak_mem - eff_init_mem) / 1024**2:.2f} MB")
    
    print(f"\nEfficiency Gains:")
    print(f"  Time: {(orig_time - eff_time) / orig_time * 100:.1f}% faster")
    print(f"  Memory: {(orig_peak_mem - eff_peak_mem) / orig_peak_mem * 100:.1f}% less memory")

if __name__ == "__main__":
    compare_implementations()