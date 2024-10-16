import numpy as np
import time
import torch

def matrix_multiply_cpu(A, B):
    return np.dot(A, B)

def matrix_multiply_gpu(A, B):
    A_tensor = torch.tensor(A).to('cuda')
    B_tensor = torch.tensor(B).to('cuda')
    C_tensor = torch.matmul(A_tensor, B_tensor)
    return C_tensor.cpu().numpy()

def check_correctness(A, B, C):
    return np.allclose(C, np.dot(A, B))

def run_experiments(sizes):
    results = []
    for size in sizes:
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)

        start_time = time.time()
        C_cpu = matrix_multiply_cpu(A, B)
        cpu_time = time.time() - start_time

        start_time = time.time()
        C_gpu = matrix_multiply_gpu(A, B)
        gpu_time = time.time() - start_time

        is_correct = check_correctness(A, B, C_gpu)

        results.append((size, cpu_time, gpu_time, is_correct))

    return results

sizes = [100, 500, 1000, 1500, 2000]
results = run_experiments(sizes)

for size, cpu_time, gpu_time, is_correct in results:
    print(f"Размер матрицы: {size}x{size} | Время CPU: {cpu_time:.6f} сек | Время GPU: {gpu_time:.6f} сек | Корректность: {is_correct}")
