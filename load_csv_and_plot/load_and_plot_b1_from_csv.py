import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_params_from_csv():
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

# Call the function to load biases from CSV file
biases1 = init_params_from_csv()

# Visualize the biases
plt.figure(figsize=(10, 5))
plt.bar(range(len(biases1)), biases1, color='skyblue')
plt.xlabel('Neuron')
plt.ylabel('Bias Value')
plt.title('Visualization of Biases b1')
plt.show()