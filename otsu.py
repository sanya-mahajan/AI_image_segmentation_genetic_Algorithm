import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# Define the fitness function using otus method
def fitness_func(threshold, hist):

    # Compute the between-class variance for the given threshold
    w1 = np.sum(hist[:threshold])
    w2 = np.sum(hist[threshold:])
    mu1 = np.sum(np.arange(threshold) * hist[:threshold]) / w1
    mu2 = np.sum(np.arange(threshold, len(hist)) * hist[threshold:]) / w2
    variance = w1 * w2 * (mu1 - mu2) ** 2
    return variance

def genetic_algorithm(hist, population_size=100, num_generations=20, mutation_rate=0.05):
    # initial  population
    population = np.random.randint(0, 256, (population_size,))

    # Loop through generations
    for i in range(num_generations):

        fitness_scores = np.array([fitness_func(threshold, hist) for threshold in population])

        # select the fittest 
        fittest_indices = np.argsort(fitness_scores)[-int(population_size * 0.2):]
        fittest_individuals = population[fittest_indices]

        # Generate offsp

        offspring = []
        for j in range(population_size):
            parent1 = fittest_individuals[np.random.randint(len(fittest_individuals))]
            parent2 = fittest_individuals[np.random.randint(len(fittest_individuals))]
            offspring.append((parent1 + parent2) // 2)

        # mutation acc to rate
        
        for j in range(len(offspring)):
            if np.random.rand() < mutation_rate:
                offspring[j] = np.random.randint(0, 256)

        # Replace population with offspring
        population = np.array(offspring)

    # Return the fittest
    fitness_scores = np.array([fitness_func(threshold, hist) for threshold in population])
    fittest_index = np.argmax(fitness_scores)
    fittest_threshold = population[fittest_index]
    return fittest_threshold

# convert to grayscale
img = cv2.imread('images/nuc.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist, _ = np.histogram(gray, bins=256)

# Apply the genetic algorithm to find the optimal threshold
threshold = genetic_algorithm(hist)

# Threshold the image and display the result
binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

#make window dimensions resizable
cv2.namedWindow('binary image', cv2.WINDOW_NORMAL)
cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
cv2.resizeWindow("binary image", 300, 700)
cv2.resizeWindow("original image", 300, 700)

cv2.imshow('binary image', binary)
cv2.imshow('original image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# display histogram with threshold
# fig, ax = plt.subplots()
# ax.hist(gray.ravel(), bins=256)
# ax.axvline(threshold, color='red', linestyle='dashed', linewidth=2)
# plt.show()


#can take user input for crossover and mutation rate 
# kapur;s entropy 