import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt

def sum_vector_cpu(vector):
    start_time = time.time()
    total = np.sum(vector)  
    elapsed_time = time.time() - start_time  
    return total, elapsed_time

def sum_vector_gpu(vector):
    start_time = time.time()
    total = cp.sum(vector)  
    cp.cuda.Stream.null.synchronize()  
    elapsed_time = time.time() - start_time  
    return total.get(), elapsed_time  

sizes = [1000, 10000, 100000, 500000, 1000000]  
results = []

for size in sizes:
    # Генерация случайного вектора
    vector_cpu = np.random.rand(size)
    vector_gpu = cp.asarray(vector_cpu)  

    sum_cpu, time_cpu = sum_vector_cpu(vector_cpu)
    print(f"Размер вектора: {size} | Сумма (CPU): {sum_cpu:.4f} | Время (CPU): {time_cpu:.6f} секунд")

    sum_gpu, time_gpu = sum_vector_gpu(vector_gpu)
    print(f"Размер вектора: {size} | Сумма (GPU): {sum_gpu:.4f} | Время (GPU): {time_gpu:.6f} секунд")

    results.append((size, time_cpu, time_gpu))

sizes, times_cpu, times_gpu = zip(*results)

plt.figure(figsize=(10, 6))
plt.plot(sizes, times_cpu, label='Время (CPU)', marker='o')
plt.plot(sizes, times_gpu, label='Время (GPU)', marker='s')
plt.xlabel('Размер вектора')
plt.ylabel('Время (секунды)')
plt.title('Сравнение времени сложения векторов на CPU и GPU')
plt.legend()
plt.grid()
plt.show()