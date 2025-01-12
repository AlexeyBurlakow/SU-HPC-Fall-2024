# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1e-w20Dytk_Lm8GtaQqLbTd1huHnrl7Qj
"""

import cupy as cp
import random
import matplotlib.pyplot as plt
import time

num_points = 800
x_values = cp.linspace(-10, 10, num_points)
noise = cp.random.normal(0, 10, num_points)
y_values = x_values ** 2 + noise
points = cp.stack((x_values, y_values), axis=1)

population_size = 1500
max_generations = 1000
Em = 1
Dm = 0.1
num_params = 5

def fitness(individuals, points):
    points_x, points_y = points[:, 0], points[:, 1]
    powers = cp.arange(individuals.shape[1])
    f_approx = cp.sum(individuals[:, cp.newaxis, :] * (points_x[:, cp.newaxis] ** powers), axis=2)
    errors = cp.sum((f_approx.T - points_y[:, cp.newaxis]) ** 2, axis=0)
    return errors
def crossover(individuals):
    new_population = []
    pop_size = len(individuals)
    while len(new_population) < pop_size:
        parent1 = individuals[random.randint(0, pop_size - 1)]
        parent2 = individuals[random.randint(0, pop_size - 1)]
        crosspoint = random.randint(1, len(parent1) - 1)
        child1 = cp.concatenate([parent1[:crosspoint], parent2[crosspoint:]])
        child2 = cp.concatenate([parent2[:crosspoint], parent1[crosspoint:]])
        new_population.append(child1)
        new_population.append(child2)
    return cp.array(new_population[:pop_size])

def mutate(individuals, Em, Dm):
    for i in range(1, len(individuals)):
        mut_number = max(1, int(cp.clip(cp.random.normal(Em, Dm), 1, len(individuals[i]))))
        for _ in range(mut_number):
            gene_to_mutate = random.randint(0, len(individuals[i]) - 1)
            individuals[i][gene_to_mutate] += cp.random.normal(0, 0.1)
    return individuals

def selection(individuals, fitnesses):
    sorted_indices = cp.argsort(fitnesses)
    return individuals[sorted_indices]

def initialize_population(pop_size, num_params):
    return cp.random.rand(pop_size, num_params)

def genetic_algorithm(points, population_size=1500, max_generations=1000,
                      Em=1, Dm=0.1, num_params=5, max_const_generations=50):
    start_time = time.time()

    points = cp.array(points)
    population = initialize_population(population_size, num_params)
    generation_number = 0
    best_fitness = float('inf')
    best_individual = None

    no_improvement_count = 0
    last_best_fitness = float('inf')

    while generation_number < max_generations:
        generation_number += 1
        current_fitnesses = fitness(population, points)
        best_fitness_current = cp.min(current_fitnesses)

        if best_fitness_current < best_fitness:
            best_fitness = best_fitness_current
            best_individual = population[cp.argmin(current_fitnesses)]
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= max_const_generations:
            print("Ранняя остановка из-за отсутствия улучшений.")
            break

        population = crossover(population)
        population = mutate(population, Em, Dm)
        population = selection(population, current_fitnesses)

        if generation_number % 50 == 0 or generation_number == max_generations:
            print(f"Generation {generation_number}/{max_generations}, Best Fitness: {best_fitness:.6f}")

    end_time = time.time()
    processing_time = end_time - start_time

    return best_individual, best_fitness, generation_number, processing_time

best_params, best_fitness, generations, processing_time = genetic_algorithm(
    points,
    population_size=population_size,
    max_generations=max_generations,
    Em=Em,
    Dm=Dm,
    num_params=num_params,
    max_const_generations=50
)

print(f"Время обработки на GPU: {processing_time:.2f} секунд")
print("Лучшие параметры многочлена:", best_params.get())
print("Лучшее значение пригодности:", best_fitness.get())
print("Число поколений:", generations)

import numpy as np

points_x = cp.linspace(-10, 10, num_points).get()
points_y = y_values.get()

best_params_np = best_params.get()
approx_y = sum(best_params_np[i] * (points_x ** i) for i in range(len(best_params_np)))

plt.figure(figsize=(10, 6))
plt.scatter(points_x, points_y, label="Original Points", color="blue", s=10)
plt.plot(points_x, approx_y, label="Approximated Polynomial", color="red", linewidth=2)
plt.title("Аппроксимация функции многочленом с помощью генетического алгоритма")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()