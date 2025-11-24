import time
import torch
import math
import numpy as np
from scipy.stats import binom

# --- 1. The PyTorch Implementation (from previous file) ---
def calculate_torch(d: int, p_0: float, device='cpu'):
    if d <= 0: return 0.0
    
    # Move to device immediately
    r = torch.arange(1, d + 1, dtype=torch.float64, device=device)
    
    # Log-probability calculation (numerically stable)
    log_factorial_d = torch.lgamma(torch.tensor(d + 1.0, dtype=torch.float64, device=device))
    log_factorial_r = torch.lgamma(r + 1)
    log_factorial_d_minus_r = torch.lgamma(d - r + 1)
    
    log_combinations = log_factorial_d - log_factorial_r - log_factorial_d_minus_r
    
    p_0_clamped = max(1e-10, min(1.0 - 1e-10, p_0))
    log_probs = (
        log_combinations 
        + r * math.log(p_0_clamped) 
        + (d - r) * math.log(1 - p_0_clamped)
    )
    
    probs = torch.exp(log_probs)
    return torch.sum(probs * torch.sqrt(r)).item()

# --- 2. The SciPy / NumPy Implementation ---
def calculate_scipy(d: int, p_0: float):
    # SciPy's binom.pmf handles the heavy lifting (combinatorics) in C
    if d <= 0: return 0.0
    r = np.arange(1, d + 1)
    # binom.pmf is vectorized over r
    probs = binom.pmf(r, d, p_0)
    result = np.sum(probs * np.sqrt(r))
    return result

# --- 3. Benchmarking ---
def run_benchmark():
    # Parameters
    p_val = 0.65
    dimensions = [100, 5000, 20000, 50000, 100000] # Dimensions to test
    
    print(f"{'Dim (d)':<10} | {'SciPy (sec)':<15} | {'SciPy Result':<15} | {'PyTorch CPU (sec)':<18} | {'PyTorch CPU Result':<18} | {'Winner'}")
    print("-" * 60)

    for d in dimensions:
        # Measure SciPy
        start = time.time()
        res_scipy = calculate_scipy(d, p_val)
        time_scipy = time.time() - start
        
        # Measure PyTorch (CPU)
        start = time.time()
        res_torch = calculate_torch(d, p_val, device='cpu')
        time_torch = time.time() - start
        
        if time_scipy < time_torch:
            winner = "SciPy"
        else:
            winner = "PyTorch"
            
        print(f"{d:<10} | {time_scipy:.6f}        | {res_scipy:.6f}        | {time_torch:.6f}           | {res_torch:.6f}           | {winner}")

if __name__ == "__main__":
    print("Running Benchmark...\n")
    run_benchmark()
    
    print("\n--- Summary ---")
    print("1. SciPy is generally faster for pure math on CPU due to lower overhead.")
    print("2. PyTorch incurs overhead creating tensors and dispatching kernels.")
    print("3. HOWEVER: Use PyTorch if you need to backpropagate through p_0.")