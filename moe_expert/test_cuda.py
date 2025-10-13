#!/usr/bin/env python3
"""
CUDA Test Script for A40 Server
Run this to verify CUDA is working properly
"""

import torch
import time
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cuda_basic():
    """Basic CUDA availability test"""
    print("="*60)
    print("BASIC CUDA TEST")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - will use CPU (very slow!)")
        return False
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    print("✅ CUDA is available!")
    return True

def test_cuda_performance():
    """Test CUDA performance with matrix operations"""
    print("\n" + "="*60)
    print("CUDA PERFORMANCE TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ Skipping performance test - CUDA not available")
        return
    
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    
    # Test 1: Memory allocation
    print("\n1. Memory Test:")
    print(f"   Memory before: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # Create large tensors
    size = 5000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    print(f"   Memory after tensor creation: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Test 2: Matrix multiplication speed
    print("\n2. Matrix Multiplication Speed Test:")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(10):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    
    print(f"   Average time for {size}x{size} matrix multiplication: {avg_time:.4f} seconds")
    print(f"   Result device: {c.device}")
    print(f"   Result shape: {c.shape}")
    
    # Test 3: Neural network forward pass
    print("\n3. Neural Network Test:")
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 2000),
        torch.nn.ReLU(),
        torch.nn.Linear(2000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 10)
    ).to(device)
    
    x = torch.randn(100, 1000, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(100):
        y = model(x)
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    
    print(f"   Average forward pass time: {avg_time:.4f} seconds")
    print(f"   Output device: {y.device}")
    print(f"   Output shape: {y.shape}")
    
    # Cleanup
    del a, b, c, model, x, y
    torch.cuda.empty_cache()
    print(f"\n   Memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print("✅ CUDA performance test completed!")

def test_cuda_vs_cpu():
    """Compare CUDA vs CPU performance"""
    print("\n" + "="*60)
    print("CUDA vs CPU PERFORMANCE COMPARISON")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ Skipping comparison - CUDA not available")
        return
    
    # CPU test
    print("CPU Test:")
    device_cpu = torch.device("cpu")
    a_cpu = torch.randn(2000, 2000)
    b_cpu = torch.randn(2000, 2000)
    
    start_time = time.time()
    for i in range(5):
        c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"   CPU time: {cpu_time:.4f} seconds")
    
    # CUDA test
    print("CUDA Test:")
    device_cuda = torch.device("cuda:0")
    a_cuda = torch.randn(2000, 2000, device=device_cuda)
    b_cuda = torch.randn(2000, 2000, device=device_cuda)
    
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(5):
        c_cuda = torch.matmul(a_cuda, b_cuda)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    print(f"   CUDA time: {cuda_time:.4f} seconds")
    
    speedup = cpu_time / cuda_time
    print(f"   CUDA speedup: {speedup:.2f}x faster than CPU")
    
    if speedup < 2:
        print("⚠️  WARNING: CUDA speedup is low - check if CUDA is working properly")
    else:
        print("✅ CUDA is working efficiently!")

def main():
    print("CUDA DEBUGGING SCRIPT FOR A40 SERVER")
    print("="*60)
    
    # Basic test
    cuda_available = test_cuda_basic()
    
    if cuda_available:
        # Performance tests
        test_cuda_performance()
        test_cuda_vs_cpu()
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        print("1. If CUDA is working but training is slow:")
        print("   - Check GPU utilization with: nvidia-smi")
        print("   - Monitor memory usage during training")
        print("   - Ensure batch sizes are appropriate")
        print("2. If CUDA is not working:")
        print("   - Check CUDA installation")
        print("   - Verify PyTorch CUDA version matches CUDA version")
        print("   - Check if other processes are using GPU")
    else:
        print("\n❌ CUDA is not available. Training will be very slow on CPU.")
        print("Please check your CUDA installation and PyTorch setup.")

if __name__ == "__main__":
    main()
