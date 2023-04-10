import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# Define the fitness function for genetic algorithm
def fitness_func(threshold, hist):
    # Compute the between-class variance for the given threshold
    w1 = np.sum(hist[:threshold])
    w2 = np.sum(hist[threshold:])
    mu1 = np.sum(np.arange(threshold) * hist[:threshold]) / w1
    mu2 = np.sum(np.arange(threshold, len(hist)) * hist[threshold:]) / w2
    variance = w1 * w2 * (mu1 - mu2) ** 2
    return variance

# Define the genetic algorithm function
def genetic_algorithm(hist, population_size=100, num_generations=20, mutation_rate=0.05):
    # Initialize the population
    population = np.random.randint(0, 256, (population_size,))

    # Loop through generations
    for i in range(num_generations):
        # Compute fitness scores for the population
        fitness_scores = np.array([fitness_func(threshold, hist) for threshold in population])

        # Select the fittest individuals
        fittest_indices = np.argsort(fitness_scores)[-int(population_size * 0.2):]
        fittest_individuals = population[fittest_indices]

        # Generate offspring
        offspring = []
        for j in range(population_size):
            parent1 = fittest_individuals[np.random.randint(len(fittest_individuals))]
            parent2 = fittest_individuals[np.random.randint(len(fittest_individuals))]
            offspring.append((parent1 + parent2) // 2)

        # Mutate offspring
        for j in range(len(offspring)):
            if np.random.rand() < mutation_rate:
                offspring[j] = np.random.randint(0, 256)

        # Replace population with offspring
        population = np.array(offspring)

    # Return the fittest individual
    fitness_scores = np.array([fitness_func(threshold, hist) for threshold in population])
    fittest_index = np.argmax(fitness_scores)
    fittest_threshold = population[fittest_index]
    return fittest_threshold

# Load the image and convert to grayscale
img = cv2.imread('/sam1.tif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute the histogram of the image
hist, _ = np.histogram(gray, bins=256)

# Apply the genetic algorithm to find the optimal threshold
threshold = genetic_algorithm(hist)

# Threshold the image and display the result
binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('binary image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
