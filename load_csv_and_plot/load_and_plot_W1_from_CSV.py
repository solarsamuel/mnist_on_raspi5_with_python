import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import time
start_time = time.time()

def init_params_from_csv():
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

# Call the function to load parameters from CSV files
weights1 = init_params_from_csv()

print("--- %s seconds ---" % (time.time() - start_time))

# Visualize the weights
plt.figure(figsize=(15, 7))
for i, weight1 in enumerate(weights1):
    plt.subplot(2, 5, i + 1)
    plt.imshow(weight1, cmap='gray')
    plt.title(f'Weight {i}')
    plt.axis('off')
plt.suptitle('Visualization of Weights')
plt.show()