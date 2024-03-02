import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_params_from_csv():
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

# Call the function to load biases from CSV file
biases2 = init_params_from_csv()

# Visualize the biases
plt.figure(figsize=(10, 5))
plt.bar(range(len(biases2)), biases2, color='skyblue')
plt.xlabel('Neuron')
plt.ylabel('Bias Value')
plt.title('Visualization of Biases b2')
plt.show()
