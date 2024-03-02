import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

# Section 1: Load and visualize weights from W1_output84.csv
def init_params_from_csv_W1():
    # Load weights from CSV files
    W1_data = []
    with open('W1_output84.csv', 'r') as file:
        for line in file:
            if line.strip().startswith('Weights'):
                continue  # Skip the line if it starts with 'neuron'
            W1_data.append(list(map(float, line.strip().split(','))))
    W1 = np.array(W1_data)
    
    # Divide into 10 separate weights
    weights1 = []
    for i in range(10):
        weight = W1[i*28:(i+1)*28, :]
        weights1.append(weight)
    
    return weights1

# Call the function to load parameters from CSV files for W1
weights1 = init_params_from_csv_W1()

# Visualize the weights for W1
plt.figure(figsize=(15, 7))
for i, weight1 in enumerate(weights1):
    plt.subplot(2, 5, i + 1)
    plt.imshow(weight1, cmap='gray')
    plt.title(f'Weight {i}')
    plt.axis('off')
plt.suptitle('Visualization of Weights')
plt.show()

# Section 2: Load and visualize weights from W2_output84.csv
def init_params_from_csv_W2():
    # Load weights from CSV files
    weights_data = {}
    with open('W2_output84.csv', 'r') as file:
        current_neuron = None
        for line in file:
            if line.strip().startswith('Weights for neuron'):
                current_neuron = int(line.strip().split()[-1])
                weights_data[current_neuron] = []
            else:
                weights_data[current_neuron].append(float(line.strip()))
    
    # Convert to numpy arrays
    weights2 = np.array([weights_data[i] for i in range(10)])
    
    return weights2

# Call the function to load parameters from CSV files for W2
weights2 = init_params_from_csv_W2()

# Visualize the weights for W2
plt.figure(figsize=(15, 7))
for i, weights_neuron in enumerate(weights2):
    plt.subplot(2, 5, i + 1)
    plt.plot(weights_neuron)
    plt.title(f'Neuron {i}')
    plt.xlabel('Weight Index')
    plt.ylabel('Weight Value')
plt.suptitle('Visualization of Weights W2')
plt.tight_layout()
plt.show()

# Section 3: Load and visualize biases from b1_output84.csv
def init_params_from_csv_b1():
    # Load biases from CSV file
    biases1_data = []
    with open('b1_output84.csv', 'r') as file:
        for line in file:
            if line.strip().startswith('Biases'):
                continue  # Skip the line if it starts with 'Biases'
            biases1_data.append(float(line.strip()))
    
    # Divide into 10 separate biases
    biases1 = []
    for i in range(10):
        bias1 = biases1_data[i]
        biases1.append(bias1)
    
    return biases1

# Call the function to load biases from CSV file for b1
biases1 = init_params_from_csv_b1()

# Visualize the biases for b1
plt.figure(figsize=(10, 5))
plt.bar(range(len(biases1)), biases1, color='skyblue')
plt.xlabel('Neuron')
plt.ylabel('Bias Value')
plt.title('Visualization of Biases b1')
plt.show()

# Section 4: Load and visualize biases from b2_output84.csv
def init_params_from_csv_b2():
    # Load biases from CSV file
    biases2_data = []
    with open('b2_output84.csv', 'r') as file:
        for line in file:
            if line.strip().startswith('Biases'):
                continue  # Skip the line if it starts with 'Biases'
            biases2_data.append(float(line.strip()))
    
    # Divide into 10 separate biases
    biases2 = []
    for i in range(10):
        bias2 = biases2_data[i]
        biases2.append(bias2)
    
    return biases2

# Call the function to load biases from CSV file for b2
biases2 = init_params_from_csv_b2()

# Visualize the biases for b2
plt.figure(figsize=(10, 5))
plt.bar(range(len(biases2)), biases2, color='skyblue')
plt.xlabel('Neuron')
plt.ylabel('Bias Value')
plt.title('Visualization of Biases b2')
plt.show()
