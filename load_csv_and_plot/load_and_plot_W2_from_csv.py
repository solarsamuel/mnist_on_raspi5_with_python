import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_params_from_csv():
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

# Call the function to load parameters from CSV files
weights2 = init_params_from_csv()

# Visualize the weights
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
