import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_params_from_csv():
    # Load weights from CSV files
    W_data = []
    with open('W1_output84.csv', 'r') as file:
        for line in file:
            if line.strip().startswith('Weights'):
                continue  # Skip the line if it starts with 'neuron'
            W_data.append(list(map(float, line.strip().split(','))))
    W = np.array(W_data)
    
    # Divide into 10 separate weights
    weights = []
    for i in range(10):
        weight = W[i*28:(i+1)*28, :]
        weights.append(weight)
    
    return weights

# Call the function to load parameters from CSV files
weights = init_params_from_csv()

# Visualize the weights
plt.figure(figsize=(15, 7))
for i, weight in enumerate(weights):
    plt.subplot(2, 5, i + 1)
    plt.imshow(weight, cmap='gray')
    plt.title(f'Weight {i}')
    plt.axis('off')
plt.suptitle('Visualization of Weights')
plt.show()